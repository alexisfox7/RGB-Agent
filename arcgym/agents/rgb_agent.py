"""RGBAgent — the actual agent that thinks and acts.

Merges the analyzer (LLM calls via OpenCode in Docker) with the action queue.
The GameRunner drives this agent through the game loop.
"""
from __future__ import annotations

import atexit
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, IO, Literal, Optional

from arcgym.agents.base_agent import BaseArcAgent

log = logging.getLogger(__name__)


class QueueExhausted(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# ActionQueue
# ---------------------------------------------------------------------------

_VALID_ACTIONS = {"ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6", "RESET"}


class ActionQueue:
    """Parses analyzer [ACTIONS] JSON and serves actions one at a time."""

    def __init__(self) -> None:
        self._queue: deque[dict] = deque()
        self.plan_total: int = 0
        self.plan_index: int = 0

    def clear(self) -> None:
        self._queue.clear()
        self.plan_total = 0
        self.plan_index = 0

    def __len__(self) -> int:
        return len(self._queue)

    def __bool__(self) -> bool:
        return bool(self._queue)

    def pop(self) -> dict:
        action = self._queue.popleft()
        self.plan_index += 1
        return action

    def load(self, actions_text: str) -> bool:
        """Parse [ACTIONS] JSON and load the queue. Returns True on success."""
        clean = re.sub(r"```(?:json)?\s*", "", actions_text).strip()

        parsed = None
        decoder = json.JSONDecoder()
        for char in ("{", "["):
            idx = clean.find(char)
            if idx >= 0:
                try:
                    parsed, _ = decoder.raw_decode(clean, idx)
                    break
                except json.JSONDecodeError:
                    continue

        if parsed is None:
            log.warning("ActionQueue.load: could not parse: %s", actions_text[:200])
            return False

        if isinstance(parsed, list):
            parsed = {"plan": parsed, "reasoning": ""}

        plan = parsed.get("plan", parsed.get("actions", []))
        if not isinstance(plan, list) or not plan:
            log.warning("ActionQueue.load: empty or invalid plan")
            return False

        self._queue.clear()
        for step in plan:
            if isinstance(step, str):
                m = re.match(r"ACTION6\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)", step)
                if m:
                    name, data = "ACTION6", {"x": int(m.group(1)), "y": int(m.group(2))}
                else:
                    name, data = step, {}
            else:
                name = step.get("action")
                if not name:
                    log.warning("skipping step with no action key: %s", step)
                    continue
                data = (
                    {"x": int(step.get("x", 0)), "y": int(step.get("y", 0))}
                    if name == "ACTION6" else {}
                )
            if name not in _VALID_ACTIONS:
                log.warning("skipping unrecognized action: %s", name)
                continue
            self._queue.append({"name": name, "data": data, "obs_text": "", "action_text": ""})

        self.plan_total = len(self._queue)
        self.plan_index = 0
        reasoning = parsed.get("reasoning", "")
        log.info("loaded %d-step plan: %s — %s",
                 self.plan_total,
                 [s if isinstance(s, str) else s.get("action") for s in plan],
                 reasoning[:100])
        return True


# ---------------------------------------------------------------------------
# RGBAgent
# ---------------------------------------------------------------------------

class RGBAgent(BaseArcAgent):
    """The actual agent: manages an action queue, driven by an external analyzer."""

    def __init__(self, *, plan_size: int = 5, **kwargs: Any) -> None:
        self._queue = ActionQueue()
        self._last_score: int = 0
        self._score_changed: bool = False
        self._use_queued: bool = False
        self._plan_size = plan_size
        super().__init__(**kwargs)

    def reset(self) -> None:
        super().reset()
        self._queue.clear()
        self._last_score = 0
        self._score_changed = False
        self._use_queued = False

    @property
    def is_overhead_action(self) -> bool:
        return False

    @property
    def plan_index(self) -> int:
        return self._queue.plan_index

    @property
    def plan_total(self) -> int:
        return self._queue.plan_total

    def render_board(self) -> str | None:
        _, grid_text = self._process_frame(self._last_observation or {})
        return grid_text or None

    def set_action_plan(self, actions_text: str) -> bool:
        return self._queue.load(actions_text)

    # -- flush queue on score change (level transition) ------------------------

    def update_from_env(self, observation, reward, done, info=None):
        super().update_from_env(observation, reward, done, info)
        obs = observation if isinstance(observation, dict) else {}
        score = obs.get("score", 0)
        if score != self._last_score:
            if self._queue:
                log.info("score %d->%d: flushing %d queued actions",
                         self._last_score, score, len(self._queue))
                self._queue.clear()
            self._score_changed = True
            self._last_score = score

    # -- main loop -------------------------------------------------------------

    async def call_llm(self):
        self._use_queued = bool(self._queue and not self._score_changed)
        if not self._use_queued:
            self._score_changed = False
        return await super().call_llm()

    async def _call_observation_model(self, grid: str, score: int, grid_raw: list) -> str:
        history = self._format_step_history()
        tried = self._format_state_action_context(grid_raw)

        hint_block = ""
        if self._external_hint:
            hint_block = f"\n[STRATEGIC ANALYSIS FROM LOG REVIEW]\n{self._external_hint}\n"
            self._external_hint = None
        elif self._persistent_hint:
            hint_block = f"\n[CURRENT PLAN]\n{self._persistent_hint}\n"

        context = (
            f"{hint_block}"
            f"{history}"
            f"{tried}"
            f"**Current State:**\n"
            f"Score: {score}\n"
            f"Step: {self._action_counter}\n\n"
            f"**Current Matrix** 64x64 (ASCII characters):\n{grid}\n"
        )

        if self._use_queued:
            label = f"step {self._queue.plan_index + 1}/{self._queue.plan_total}"
            context += f"\n[Executing pre-planned action ({label}) — no model call]\n"
            self._last_observation_prompt = f"[Queued plan {label}]\n\n{context}"
            self._last_observation_response = f"[Pre-planned action {label}]"
        else:
            self._last_observation_prompt = f"[Observation context]\n\n{context}"
            self._last_observation_response = "[Observation model — context assembled]"

        return context

    async def _call_action_model(self, grid: str, last_obs: str) -> dict:
        if self._use_queued and self._queue:
            action = self._queue.pop()
            label = f"plan step {self._queue.plan_index}/{self._queue.plan_total}"

            action["obs_text"] = last_obs
            action["action_text"] = f"[queued {label}]"
            self._pending_action = action

            self._last_action_prompt = f"[Queued {label} — no model call]"
            self._last_action_response = (
                f"Tool Call: {action['name']}({json.dumps(action['data'])})\n"
                f"Content: Executing pre-planned action ({label})"
            )
            log.info("queue drain -> %s (%s, %d remaining)",
                     action.get("name"), label, len(self._queue))
            return action

        log.info("queue empty — ending episode")
        raise QueueExhausted("Queue empty, no actions from analyzer")


# ---------------------------------------------------------------------------
# Analyzer infrastructure (OpenCode in Docker)
# ---------------------------------------------------------------------------

_INITIAL_PROMPT = """\
You are a strategic advisor for an AI agent playing a grid-based puzzle game.
The agent's full prompt log for this run is at this ABSOLUTE path: {log_path}

You may only access this single file (use its absolute path directly with Read and Grep).

Most games have some form of timer mechanism. A score increase means a level was solved.

Deeply analyze this log to understand what the agent has been doing, what has worked,
what hasn't, and what patterns explain the game's behavior.

Your response MUST contain ALL sections below — the agent cannot act without [ACTIONS]:
1. A detailed strategic briefing (explain your reasoning, be specific with coordinates)
2. Followed by exactly this separator and a 2-3 sentence action plan:

[PLAN]
<concise action plan the agent should follow until the next analysis>
"""

_RESUME_PROMPT = """\
The prompt log has grown since your last analysis. The log file is at: {log_path}

Re-read the latest actions (from where you left off) and update your strategic briefing.
Focus on what changed: new moves, score transitions, and whether the agent followed
your previous plan or diverged. Parse the board programmatically from the file using
section markers ([POST-ACTION BOARD STATE], etc.) — do NOT visually copy the grid.

Your response MUST contain ALL three sections below — the agent cannot act without [ACTIONS]:
1. A detailed strategic briefing (explain your reasoning, be specific with coordinates)
2. Followed by exactly this separator and a 2-3 sentence action plan:

[PLAN]
<concise action plan the agent should follow until the next analysis>
"""

_ACTIONS_ADDENDUM = """
3. Followed by exactly this separator and a JSON action plan (REQUIRED — the agent cannot act without this):

[ACTIONS]
{{"plan": [{{"action": "ACTION1"}}, {{"action": "ACTION6", "x": 3, "y": 7}}, ...], "reasoning": "why these steps"}}

Available actions: ACTION1-4 (moves), ACTION6 (click at x,y), ACTION5 (no-op), RESET.
Each action MUST be a JSON object: {{"action": "ACTION6", "x": <row>, "y": <col>}} for clicks, {{"action": "ACTION1"}} for moves. Never use string shorthand like "ACTION6(x,y)".
Plan 1–{plan_size} actions. IMPORTANT: shorter plans (3-5 steps) are strongly preferred
because the agent can re-evaluate sooner. Only use more than 5 if you have very high
confidence AND the extra steps are critical. Even on a clear straight path, prefer
stopping early so the agent can observe the game's response and adapt.
\
"""

_PYTHON_ADDENDUM = (
    "\n\nBash (and therefore Python) is available to you. **Always** use Python to "
    "parse the board — do NOT try to visually read the ASCII grid.\n\n"
    "The log file uses section markers to delimit board grids:\n"
    "  [INITIAL BOARD STATE]   — the grid at the start (after Action 0 header)\n"
    "  [POST-ACTION BOARD STATE] — the grid after each action\n"
    "\n"
    "To extract the latest board into a matrix:\n"
    "```python\n"
    "import re\n"
    "data = open('{log_path}').read()\n"
    "# Find the last board state section\n"
    "boards = re.split(r'\\[(?:POST-ACTION|INITIAL) BOARD STATE\\]', data)\n"
    "last_board = boards[-1].strip()\n"
    "# Skip 'Score: N' line if present, then parse rows into a 2-D list\n"
    "lines = last_board.split('\\n')\n"
    "if lines[0].startswith('Score:'):\n"
    "    lines = lines[1:]\n"
    "grid = [list(row) for row in lines if row.strip()]\n"
    "# Now slice, count, compare programmatically\n"
    "```\n"
    "Run Python inline."
)

_DOCKER_IMAGE = os.environ.get("OPENCODE_DOCKER_IMAGE", "arcgym/opencode-sandbox:latest")


def _docker_image_exists(image: str) -> bool:
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", image],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10,
        )
        return result.returncode == 0
    except Exception:
        return False


class _EventStreamParser:
    """Parses nd-JSON events from opencode and writes to an analyzer log."""

    def __init__(self, f: IO[str]):
        self._f = f
        self.accumulated_text = ""
        self.session_id: str | None = None

    def _write(self, label: str, content: str) -> None:
        if content:
            self._f.write(f"[{label}]\n{content}\n\n")
            self._f.flush()

    def _write_tool(self, name: str, state: dict) -> None:
        status = state.get("status", "?")
        if status in ("running", "completed", "done"):
            input_data = state.get("input", {})
            input_str = json.dumps(input_data, indent=2) if isinstance(input_data, dict) else str(input_data)
            self._write(f"TOOL CALL: {name}", input_str)
        if status in ("completed", "done"):
            output = state.get("output", state.get("result", ""))
            is_error = state.get("is_error", False) or state.get("error", False)
            label = "TOOL RESULT ERROR" if is_error else "TOOL RESULT"
            self._write(label, str(output)[:4000])

    def handle(self, event: dict) -> None:
        etype = event.get("type")
        log.debug("event type=%s", etype)

        if etype == "step_start":
            sid = event.get("sessionID")
            if sid and not self.session_id:
                self.session_id = sid

        elif etype == "text":
            text = event.get("part", {}).get("text", "")
            if text:
                self.accumulated_text += text
                self._write("ASSISTANT", text)

        elif etype == "tool_use":
            part = event.get("part", {})
            self._write_tool(part.get("tool", "?"), part.get("state", {}))

        elif etype == "message.part.updated":
            part = event.get("part", {})
            ptype = part.get("type")
            if ptype in ("thinking", "reasoning"):
                self._write("THINKING", part.get("text", ""))
            elif ptype == "tool":
                name = part.get("name", "?")
                pstate = part.get("state", "?")
                if pstate == "running":
                    input_data = part.get("input", {})
                    input_str = json.dumps(input_data, indent=2) if isinstance(input_data, dict) else str(input_data)
                    self._write(f"TOOL CALL: {name}", input_str)
                elif pstate in ("completed", "done"):
                    result = part.get("result", part.get("output", ""))
                    text = result if isinstance(result, str) else str(result)
                    is_error = part.get("is_error", False) or part.get("error", False)
                    label = "TOOL RESULT ERROR" if is_error else "TOOL RESULT"
                    self._write(label, text[:4000])

        elif etype == "error":
            err = event.get("error", {})
            name = err.get("name", "UnknownError")
            msg = err.get("data", {}).get("message", str(err))
            self._write(f"ERROR: {name}", msg)
            log.error("API error: %s: %s", name, msg)
            if "overflow" in name.lower() or "too long" in msg.lower():
                self.session_id = None

        elif etype == "step_finish":
            cost = event.get("part", {}).get("cost")
            self._write("RESULT", f"cost=${cost}")

        elif etype == "assistant":
            for block in event.get("message", {}).get("content", []):
                btype = block.get("type")
                if btype == "thinking":
                    self._write("THINKING", block.get("thinking", ""))
                elif btype == "text":
                    text = block["text"]
                    self.accumulated_text += text
                    self._write("ASSISTANT", text)
                elif btype == "tool_use":
                    self._write(f"TOOL CALL: {block['name']}", json.dumps(block.get("input", {}), indent=2))

        elif etype == "user":
            for block in event.get("message", {}).get("content", []):
                if block.get("type") == "tool_result":
                    content = block.get("content", "")
                    if isinstance(content, list):
                        text = "\n".join(c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text")
                    elif isinstance(content, str):
                        text = content
                    else:
                        text = str(content)
                    is_error = block.get("is_error", False)
                    label = "TOOL RESULT ERROR" if is_error else "TOOL RESULT"
                    self._write(label, text[:4000])

        elif etype == "result":
            result_text = event.get("result", "").strip()
            if result_text and not self.accumulated_text.strip():
                self.accumulated_text = result_text
            cost = event.get("total_cost_usd")
            self._write("RESULT", f"cost=${cost}")

        else:
            self._f.write(f"[RAW:{etype}] {json.dumps(event)[:500]}\n")
            self._f.flush()


class _ContainerPool:
    """Manages persistent Docker containers running `opencode serve`."""

    def __init__(self, config_path: Path, permission: dict, docker_image: str, sandbox_prefix: str):
        self._config_path = config_path
        self._permission = permission
        self._image = docker_image
        self._prefix = sandbox_prefix
        self._containers: dict[str, dict] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> tuple[str, int, str]:
        with self._lock:
            if key in self._containers:
                info = self._containers[key]
                check = subprocess.run(
                    ["docker", "inspect", "-f", "{{.State.Running}}", info["name"]],
                    capture_output=True, text=True, timeout=5,
                )
                if check.returncode == 0 and "true" in check.stdout.lower():
                    return info["name"], info["port"], info["sandbox_dir"]
                log.warning("server container %s died, recreating", info["name"])
                subprocess.run(["docker", "rm", "-f", info["name"]], capture_output=True, timeout=10)
                del self._containers[key]

            return self._create(key)

    def _create(self, key: str) -> tuple[str, int, str]:
        sandbox = tempfile.mkdtemp(prefix=self._prefix)
        os.chmod(sandbox, 0o777)
        name = f"oc_{uuid.uuid4().hex[:12]}"
        port = 4096

        shutil.copy2(self._config_path, Path(sandbox) / "opencode.json")

        env_flags: list[str] = []
        for key_name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY", "OPENROUTER_API_KEY"):
            val = os.environ.get(key_name)
            if val:
                env_flags.extend(["-e", f"{key_name}={val}"])

        cmd = [
            "docker", "run", "-d",
            "--name", name,
            "--read-only",
            "--user", "1000:1000",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges:true",
            "--memory=4g", "--cpus=2",
            "--pids-limit=128",
            "--shm-size=8m",
            "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m,uid=1000,gid=1000",
            "--tmpfs", "/home/opencode:rw,noexec,nosuid,size=128m,uid=1000,gid=1000",
            "-v", f"{os.path.realpath(sandbox)}:/workspace:rw",
            "-e", "OPENCODE_CONFIG=/workspace/opencode.json",
            "-e", f"OPENCODE_PERMISSION={json.dumps(self._permission)}",
            *env_flags,
            self._image,
            "serve", "--port", str(port), "--hostname", "0.0.0.0",
        ]

        subprocess.run(cmd, check=True, capture_output=True, timeout=30)

        for _ in range(15):
            time.sleep(1)
            logs = subprocess.run(
                ["docker", "logs", name], capture_output=True, text=True, timeout=15,
            )
            if "listening" in logs.stdout or "listening" in logs.stderr:
                break
        else:
            log.warning("server %s may not be ready (timeout)", name)

        self._containers[key] = {"name": name, "port": port, "sandbox_dir": sandbox}
        log.info("container ready: %s", name)
        return name, port, sandbox

    def cleanup(self) -> None:
        with self._lock:
            for info in self._containers.values():
                try:
                    log.info("stopping container: %s", info["name"])
                    subprocess.run(["docker", "stop", "-t", "3", info["name"]], capture_output=True, timeout=10)
                    subprocess.run(["docker", "rm", "-f", info["name"]], capture_output=True, timeout=10)
                except Exception as e:
                    log.warning("failed to cleanup container %s: %s", info["name"], e)
                if info.get("sandbox_dir"):
                    shutil.rmtree(info["sandbox_dir"], ignore_errors=True)
            self._containers.clear()


# ---------------------------------------------------------------------------
# Analyzer factory
# ---------------------------------------------------------------------------

def make_analyzer(
    interval: int = 5,
    timeout: Optional[int] = None,
    use_subscription: bool = False,
    allow_bash: bool = False,
    action_mode: Optional[Literal["move", "click", "all"]] = None,
    plan_size: int = 5,
    allow_self_read: bool = False,
    model: str = "claude-opus-4-6",
    fast: bool = False,
    resume_session: bool = False,
) -> Callable[[Path, int], Optional[str]]:
    """Returns a hook that calls OpenCode to analyze the prompt log.

    Hook signature: hook(log_path, action_num, retry_nudge="") -> hint | None
    """
    if not shutil.which("docker"):
        raise FileNotFoundError("'docker' CLI not found. Install Docker Desktop to use the analyzer.")
    if not _docker_image_exists(_DOCKER_IMAGE):
        raise FileNotFoundError(
            f"Docker image '{_DOCKER_IMAGE}' not found. Build with:\n"
            f"  cd docker/opencode-sandbox && bash build.sh"
        )
    log.info("using Docker sandbox: %s", _DOCKER_IMAGE)

    oc_model = model if "/" in model else f"anthropic/{model}"
    oc_provider = oc_model.split("/")[0]

    permission: dict = {
        "*": "deny",
        "read": "allow",
        "grep": "allow",
        "bash": {
            "*": "deny",
            "python3 *": "allow",
            "python *": "allow",
        } if allow_bash else "deny",
        "external_directory": "deny",
        "doom_loop": "allow",
        "question": "deny",
        "edit": "deny",
        "write": "deny",
        "patch": "deny",
        "glob": "deny",
        "list": "deny",
        "lsp": "deny",
        "skill": "deny",
        "webfetch": "deny",
        "websearch": "deny",
        "todowrite": "deny",
        "todoread": "deny",
    }

    config = {
        "model": oc_model,
        "provider": {oc_provider: {}},
        "permission": permission,
        "agent": {"build": {"steps": 50}},
    }

    config_dir = tempfile.mkdtemp(prefix="opencode_analyzer_")
    config_path = Path(config_dir) / "opencode.json"
    config_path.write_text(json.dumps(config, indent=2))
    atexit.register(shutil.rmtree, config_dir, True)

    pool = _ContainerPool(config_path, permission, _DOCKER_IMAGE, f"oc_sandbox_{uuid.uuid4().hex[:8]}_")
    atexit.register(pool.cleanup)

    session_ids: dict[str, str] = {}
    session_lock = threading.Lock()

    def _build_prompt(log_name: str, analyzer_log_name: str, analyzer_log_exists: bool,
                      is_first: bool) -> str:
        if resume_session and not is_first:
            prompt = _RESUME_PROMPT.format(log_path=log_name)
        else:
            prompt = _INITIAL_PROMPT.format(log_path=log_name)
            if allow_self_read and analyzer_log_exists:
                prompt += (
                    f"\n\nYour previous analysis output is at: {analyzer_log_name}\n"
                    "Read it to see what you concluded last time and build on it. "
                    "Avoid repeating strategies that didn't work."
                )
        if allow_bash:
            prompt += _PYTHON_ADDENDUM.format(log_path=log_name)
        if action_mode:
            prompt += _ACTIONS_ADDENDUM.format(plan_size=plan_size)
        return prompt

    def _try_recover_text(container_name: str, sid: str, sandbox_dir: str) -> str:
        export_path = Path(sandbox_dir) / "_export.json"
        try:
            subprocess.run(
                ["docker", "exec", container_name, "sh", "-c",
                 f"opencode export {sid} > /workspace/_export.json 2>/dev/null"],
                capture_output=True, text=True, timeout=30,
            )
            if not export_path.exists():
                return ""
            data = json.loads(export_path.read_text())
            recovered = ""
            for msg in reversed(data.get("messages", [])):
                role = msg.get("info", {}).get("role")
                if role == "assistant":
                    for part in msg.get("parts", []):
                        if part.get("type") == "text":
                            candidate = part.get("text", "").strip()
                            if candidate and "[ACTIONS]" in candidate:
                                return candidate
                            if candidate and not recovered:
                                recovered = candidate
                    if recovered and "[ACTIONS]" in recovered:
                        return recovered
            return recovered
        except Exception as e:
            log.debug("export recovery failed: %s", e)
            return ""

    def hook(log_path: Path, action_num: int, retry_nudge: str = "") -> Optional[str]:
        if interval > 0 and action_num % interval != 0:
            return None
        if not log_path.exists():
            return None

        analyzer_log = log_path.parent / (log_path.stem + "_analyzer.txt")
        path_key = str(log_path)

        is_first = True
        current_sid = None
        if resume_session:
            with session_lock:
                if path_key in session_ids:
                    current_sid = session_ids[path_key]
                    is_first = False

        container_name, server_port, sandbox_dir = pool.get(path_key)
        sandbox = Path(sandbox_dir)

        try:
            shutil.copy2(log_path, sandbox / log_path.name)
            if allow_self_read and analyzer_log.exists():
                shutil.copy2(analyzer_log, sandbox / analyzer_log.name)

            prompt = _build_prompt(log_path.name, analyzer_log.name, analyzer_log.exists(), is_first)
            if retry_nudge:
                prompt += f"\n\n{retry_nudge}"

            oc_args = ["run", "--attach", f"http://127.0.0.1:{server_port}"]
            if resume_session and not is_first and current_sid:
                oc_args.extend(["--session", current_sid, "--continue"])
            oc_args.extend(["--model", oc_model])
            if fast:
                oc_args.extend(["--variant", "minimal"])
            oc_args.extend(["--format", "json", "--dir", "/workspace"])
            oc_args.append(prompt)

            cmd = ["docker", "exec", container_name, "opencode", *oc_args]
            log.info("exec %s model=%s%s", container_name, oc_model,
                     f" session={current_sid}" if current_sid else "")

            proc = subprocess.Popen(
                cmd, stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, bufsize=1,
            )

            stderr_lines: list[str] = []
            def drain_stderr():
                for line in proc.stderr:
                    stderr_lines.append(line.rstrip("\n"))
                    log.debug("STDERR: %s", line[:300].rstrip())

            stderr_thread = threading.Thread(target=drain_stderr, daemon=True)
            stderr_thread.start()

            with open(analyzer_log, "a", encoding="utf-8") as f:
                f.write(f"\n--- action={action_num} | {datetime.now().strftime('%H:%M:%S')} | opencode ---\n")
                if is_first or not resume_session:
                    f.write(f"[SYSTEM PROMPT]\n{prompt}\n\n")
                f.flush()

                parser = _EventStreamParser(f)
                deadline = time.monotonic() + timeout if timeout is not None else None

                while True:
                    line = proc.stdout.readline()
                    if not line:
                        break
                    if deadline is not None and time.monotonic() > deadline:
                        proc.kill()
                        f.write("[TIMEOUT]\n")
                        log.warning("timed out at action %d", action_num)
                        return None

                    line = line.rstrip("\n")
                    if not line.strip():
                        continue
                    try:
                        parser.handle(json.loads(line))
                    except json.JSONDecodeError:
                        f.write(f"[RAW] {line}\n")
                        f.flush()

                proc.wait()
                stderr_thread.join(timeout=5)
                if stderr_lines:
                    f.write(f"\n--- STDERR ---\n{''.join(l + chr(10) for l in stderr_lines)}")
                    f.flush()

                needs_recovery = (
                    not parser.accumulated_text.strip()
                    or (action_mode and "[ACTIONS]" not in parser.accumulated_text)
                )
                if needs_recovery and parser.session_id:
                    recovered = _try_recover_text(container_name, parser.session_id, sandbox_dir)
                    if recovered:
                        parser.accumulated_text = recovered
                        log.info("recovered %d chars via session export", len(recovered))

                if resume_session and parser.session_id is None and not is_first:
                    log.warning("context overflow — clearing session for %s", path_key)
                    with session_lock:
                        session_ids.pop(path_key, None)

                f.flush()

            hint = parser.accumulated_text.strip() or None

            if proc.returncode != 0 or not hint:
                log.warning("action=%d failed: rc=%d, hint_len=%d",
                            action_num, proc.returncode, len(hint) if hint else 0)
                if resume_session:
                    with session_lock:
                        session_ids.pop(path_key, None)
                return None

            if resume_session and parser.session_id:
                with session_lock:
                    session_ids[path_key] = parser.session_id

            log.info("action=%d OK (%d chars)", action_num, len(hint))
            return hint

        except Exception as e:
            log.error("unexpected error: %s", e, exc_info=True)
            return None

    return hook
