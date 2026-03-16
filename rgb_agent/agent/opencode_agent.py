"""OpenCodeAgent: runs OpenCode in a sandboxed Docker container to produce action plans."""
from __future__ import annotations

import atexit
import json
import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import IO, Optional

from rgb_agent.agent.prompts import (
    INITIAL_PROMPT,
    RESUME_PROMPT,
    ACTIONS_ADDENDUM,
    PYTHON_ADDENDUM,
)

log = logging.getLogger(__name__)

_DOCKER_IMAGE = os.environ.get("OPENCODE_DOCKER_IMAGE", "rgb-agent/opencode-sandbox:latest")


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
    """Parses nd-JSON events from OpenCode and writes to an analyzer log."""

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
        for key_name in ("CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY", "OPENROUTER_API_KEY"):
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


class OpenCodeAgent:
    """Runs OpenCode in a sandboxed Docker container to analyze game logs and produce action plans."""

    def __init__(
        self,
        *,
        model: str = "claude-opus-4-6",
        plan_size: int = 5,
        timeout: Optional[int] = None,
        resume_session: bool = True,
    ) -> None:
        if not shutil.which("docker"):
            raise FileNotFoundError("'docker' CLI not found. Install Docker Desktop to use the analyzer.")
        if not _docker_image_exists(_DOCKER_IMAGE):
            raise FileNotFoundError(
                f"Docker image '{_DOCKER_IMAGE}' not found. Build with:\n"
                f"  cd docker/opencode-sandbox && bash build.sh"
            )
        log.info("using Docker sandbox: %s", _DOCKER_IMAGE)

        self._oc_model = model if "/" in model else f"anthropic/{model}"
        self._plan_size = plan_size
        self._timeout = timeout
        self._resume_session = resume_session

        oc_provider = self._oc_model.split("/")[0]

        permission: dict = {
            "*": "deny",
            "read": "allow",
            "grep": "allow",
            "bash": {
                "*": "deny",
                "python3 *": "allow",
                "python *": "allow",
            },
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
            "model": self._oc_model,
            "provider": {oc_provider: {}},
            "permission": permission,
            "agent": {"build": {"steps": 50}},
        }

        config_dir = tempfile.mkdtemp(prefix="opencode_analyzer_")
        config_path = Path(config_dir) / "opencode.json"
        config_path.write_text(json.dumps(config, indent=2))
        atexit.register(shutil.rmtree, config_dir, True)

        self._pool = _ContainerPool(config_path, permission, _DOCKER_IMAGE, f"oc_sandbox_{uuid.uuid4().hex[:8]}_")
        atexit.register(self._pool.cleanup)

        self._session_ids: dict[str, str] = {}
        self._session_lock = threading.Lock()

    def _build_prompt(self, log_name: str, is_first: bool) -> str:
        if self._resume_session and not is_first:
            prompt = RESUME_PROMPT.format(log_path=log_name)
        else:
            prompt = INITIAL_PROMPT.format(log_path=log_name)
        prompt += PYTHON_ADDENDUM.format(log_path=log_name)
        prompt += ACTIONS_ADDENDUM.format(plan_size=self._plan_size)
        return prompt

    def _try_recover_text(self, container_name: str, sid: str, sandbox_dir: str) -> str:
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

    def analyze(self, log_path: Path, action_num: int, retry_nudge: str = "") -> Optional[str]:
        """Analyze the game log and return the agent's response text, or None on failure."""
        if not log_path.exists():
            return None

        analyzer_log = log_path.parent / (log_path.stem + "_analyzer.txt")
        path_key = str(log_path)

        is_first = True
        current_sid = None
        if self._resume_session:
            with self._session_lock:
                if path_key in self._session_ids:
                    current_sid = self._session_ids[path_key]
                    is_first = False

        container_name, server_port, sandbox_dir = self._pool.get(path_key)
        sandbox = Path(sandbox_dir)

        try:
            shutil.copy2(log_path, sandbox / log_path.name)

            prompt = self._build_prompt(log_path.name, is_first)
            if retry_nudge:
                prompt += f"\n\n{retry_nudge}"

            oc_args = ["run", "--attach", f"http://127.0.0.1:{server_port}"]
            if self._resume_session and not is_first and current_sid:
                oc_args.extend(["--session", current_sid, "--continue"])
            oc_args.extend(["--model", self._oc_model])
            oc_args.extend(["--format", "json", "--dir", "/workspace"])
            oc_args.append(prompt)

            cmd = ["docker", "exec", container_name, "opencode", *oc_args]
            log.info("exec %s model=%s%s", container_name, self._oc_model,
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
                if is_first or not self._resume_session:
                    f.write(f"[SYSTEM PROMPT]\n{prompt}\n\n")
                f.flush()

                parser = _EventStreamParser(f)
                deadline = time.monotonic() + self._timeout if self._timeout is not None else None

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
                    or "[ACTIONS]" not in parser.accumulated_text
                )
                if needs_recovery and parser.session_id:
                    recovered = self._try_recover_text(container_name, parser.session_id, sandbox_dir)
                    if recovered:
                        parser.accumulated_text = recovered
                        log.info("recovered %d chars via session export", len(recovered))

                if self._resume_session and parser.session_id is None and not is_first:
                    log.warning("context overflow — clearing session for %s", path_key)
                    with self._session_lock:
                        self._session_ids.pop(path_key, None)

                f.flush()

            hint = parser.accumulated_text.strip() or None

            if proc.returncode != 0 or not hint:
                log.warning("action=%d failed: rc=%d, hint_len=%d",
                            action_num, proc.returncode, len(hint) if hint else 0)
                if self._resume_session:
                    with self._session_lock:
                        self._session_ids.pop(path_key, None)
                return None

            if self._resume_session and parser.session_id:
                with self._session_lock:
                    self._session_ids[path_key] = parser.session_id

            log.info("action=%d OK (%d chars)", action_num, len(hint))
            return hint

        except Exception as e:
            log.error("unexpected error: %s", e, exc_info=True)
            return None
