"""ClaudeAgent: uses the Claude Agent SDK to analyze game logs and produce action plans."""
from __future__ import annotations

import asyncio
import atexit
import logging
import os
import shutil
import subprocess
import tempfile
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

# Allow launching Claude Code as a subprocess even when called from within
# a Claude Code session (e.g. during development/testing).
os.environ.pop("CLAUDECODE", None)

from claude_agent_sdk import query, ClaudeAgentOptions, AssistantMessage, ResultMessage, TextBlock, ToolUseBlock, ToolResultBlock, UserMessage

from rgb_agent.agent.prompts import (
    INITIAL_PROMPT,
    RESUME_PROMPT,
    ACTIONS_ADDENDUM,
    PYTHON_ADDENDUM,
)

log = logging.getLogger(__name__)

_DOCKER_IMAGE = os.environ.get("CLAUDE_DOCKER_IMAGE", "rgb-agent/claude-sandbox:latest")

SYSTEM_PROMPT = (
    "You are a strategic advisor for an AI agent playing a grid-based puzzle game. "
    "You have tools to read files and run Python code. Use them to analyze the game log."
)


class _ClaudeContainerPool:
    """Keeps persistent Docker containers alive so Claude Code sessions survive across calls."""

    def __init__(self) -> None:
        self._containers: dict[str, dict] = {}
        self._lock = threading.Lock()

    def get(self, key: str, workspace_dir: str) -> str:
        """Return a persistent container name for the given key, creating one if needed."""
        with self._lock:
            if key in self._containers:
                info = self._containers[key]
                # Check if container is still running
                check = subprocess.run(
                    ["docker", "inspect", "-f", "{{.State.Running}}", info["name"]],
                    capture_output=True, text=True, timeout=5,
                )
                if check.returncode == 0 and "true" in check.stdout.lower():
                    # Update workspace files
                    self._sync_workspace(info["name"], workspace_dir)
                    return info["name"]
                log.warning("container %s died, recreating", info["name"])
                subprocess.run(["docker", "rm", "-f", info["name"]], capture_output=True, timeout=10)
                del self._containers[key]

            return self._create(key, workspace_dir)

    def _sync_workspace(self, container_name: str, workspace_dir: str) -> None:
        """Copy latest workspace files into the running container."""
        subprocess.run(
            ["docker", "cp", f"{workspace_dir}/.", f"{container_name}:/workspace/"],
            capture_output=True, timeout=10,
        )

    def _create(self, key: str, workspace_dir: str) -> str:
        name = f"claude_{uuid.uuid4().hex[:12]}"

        env_flags: list[str] = []
        oauth_token = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")
        if oauth_token:
            env_flags.extend(["-e", f"CLAUDE_CODE_OAUTH_TOKEN={oauth_token}"])
        for key_name in ("ANTHROPIC_API_KEY",):
            val = os.environ.get(key_name)
            if val:
                env_flags.extend(["-e", f"{key_name}={val}"])

        cmd = [
            "docker", "run", "-d",
            "--name", name,
            "--entrypoint", "sleep",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges:true",
            "--tmpfs", "/tmp:rw,nosuid,size=256m",
            "--tmpfs", "/home/sandbox/.npm:rw,noexec,nosuid,size=64m",
            "-v", f"{os.path.realpath(workspace_dir)}:/workspace:rw",
            "-e", "HOME=/home/sandbox",
            "-e", "DISABLE_AUTOUPDATER=1",
            "-e", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1",
            *env_flags,
            _DOCKER_IMAGE,
            "infinity",
        ]

        subprocess.run(cmd, check=True, capture_output=True, timeout=30)
        self._containers[key] = {"name": name}
        log.info("claude container ready: %s", name)
        return name

    def cleanup(self) -> None:
        with self._lock:
            for info in self._containers.values():
                try:
                    log.info("stopping claude container: %s", info["name"])
                    subprocess.run(["docker", "stop", "-t", "3", info["name"]], capture_output=True, timeout=10)
                    subprocess.run(["docker", "rm", "-f", info["name"]], capture_output=True, timeout=10)
                except Exception as e:
                    log.warning("failed to cleanup container %s: %s", info["name"], e)
            self._containers.clear()


class ClaudeAgent:
    """Calls Claude Code via the Agent SDK to analyze game logs and produce action plans."""

    def __init__(
        self,
        *,
        model: str = "claude-opus-4-6",
        plan_size: int = 5,
        timeout: Optional[int] = None,
        use_docker: bool = False,
    ) -> None:
        self._model = model
        self._plan_size = plan_size
        self._timeout = timeout
        self._use_docker = use_docker
        self._call_count: dict[str, int] = {}
        self._session_ids: dict[str, str] = {}

        # Cumulative token usage across all analyze() calls
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.total_cache_create_tokens: int = 0
        self.total_cache_read_tokens: int = 0
        self.total_estimated_cost: float = 0.0
        self.total_calls: int = 0

        self._pool: _ClaudeContainerPool | None = None
        self._exec_scripts: dict[str, str] = {}  # key -> path to exec wrapper script
        if use_docker:
            self._pool = _ClaudeContainerPool()
            atexit.register(self._pool.cleanup)

    def _get_docker_cli(self, path_key: str, workspace_dir: str) -> str:
        """Get or create a docker exec wrapper script for this game."""
        container_name = self._pool.get(path_key, workspace_dir)

        if path_key not in self._exec_scripts:
            script_dir = tempfile.mkdtemp(prefix="claude_exec_")
            script_path = Path(script_dir) / "claude-exec.sh"
            # Rewrite --cwd from host path to /workspace inside the container
            script_path.write_text(
                f'#!/usr/bin/env bash\n'
                f'args=()\n'
                f'while [[ $# -gt 0 ]]; do\n'
                f'  case "$1" in\n'
                f'    --cwd) args+=("--cwd" "/workspace"); shift 2 ;;\n'
                f'    *) args+=("$1"); shift ;;\n'
                f'  esac\n'
                f'done\n'
                f'exec docker exec -i {container_name} claude "${{args[@]}}"\n'
            )
            script_path.chmod(0o755)
            self._exec_scripts[path_key] = str(script_path)
            atexit.register(shutil.rmtree, script_dir, True)

        return self._exec_scripts[path_key]

    def _build_prompt(self, log_path: str, is_first: bool) -> str:
        if not is_first:
            prompt = RESUME_PROMPT.format(log_path=log_path)
        else:
            prompt = INITIAL_PROMPT.format(log_path=log_path)
        prompt += PYTHON_ADDENDUM.format(log_path=log_path)
        prompt += ACTIONS_ADDENDUM.format(plan_size=self._plan_size)
        return prompt

    def analyze(self, log_path: Path, action_num: int, retry_nudge: str = "") -> Optional[str]:
        """Analyze the game log and return the agent's response text, or None on failure."""
        if not log_path.exists():
            return None

        path_key = str(log_path)
        is_first = path_key not in self._call_count
        self._call_count[path_key] = self._call_count.get(path_key, 0) + 1

        if self._use_docker:
            prompt_log_ref = f"/workspace/{log_path.name}"
        else:
            prompt_log_ref = str(log_path)
        prompt = self._build_prompt(prompt_log_ref, is_first)
        if retry_nudge:
            prompt += f"\n\n{retry_nudge}"

        analyzer_log = log_path.parent / (log_path.stem + "_analyzer.txt")
        with open(analyzer_log, "a", encoding="utf-8") as f:
            f.write(f"\n--- action={action_num} | {datetime.now().strftime('%H:%M:%S')} | claude-agent-sdk ---\n")
            if is_first:
                f.write(f"[USER PROMPT]\n{prompt}\n\n")
            f.flush()

        stderr_lines: list[str] = []

        # Remove ANTHROPIC_API_KEY so Claude Code uses Max plan auth.
        saved_api_key = os.environ.pop("ANTHROPIC_API_KEY", None)

        if self._use_docker:
            cli = self._get_docker_cli(path_key, str(log_path.parent.resolve()))
        else:
            cli = shutil.which("claude") or "claude"

        session_id = self._session_ids.get(path_key)
        resuming = not is_first and session_id is not None

        env = {"ANTHROPIC_API_KEY": ""}

        # Docker provides the sandbox; native sandbox for non-Docker.
        sandbox = (
            {"enabled": False}
            if self._use_docker
            else {
                "enabled": True,
                "autoAllowBashIfSandboxed": True,
                "allowUnsandboxedCommands": False,
            }
        )

        options = ClaudeAgentOptions(
            model=self._model,
            system_prompt=SYSTEM_PROMPT,
            allowed_tools=["Read", "Bash(python3 *)", "Bash(python *)"],
            disallowed_tools=["Edit", "Write", "WebSearch", "WebFetch"],
            permission_mode="bypassPermissions",
            cli_path=cli,
            cwd=str(log_path.parent),
            max_turns=50,
            stderr=lambda line: (stderr_lines.append(line), log.warning("claude stderr: %s", line)),
            env=env,
            sandbox=sandbox,
            resume=session_id if resuming else None,
            continue_conversation=resuming,
        )

        log.info("analyzer: model=%s, resume=%s, session=%s",
                 self._model, resuming, session_id or "new")

        try:
            accumulated_text = ""
            tool_log_parts: list[str] = []

            async def _run() -> str:
                nonlocal accumulated_text
                async for message in query(prompt=prompt, options=options):
                    log.debug("message type: %s", type(message).__name__)
                    if isinstance(message, AssistantMessage):
                        for block in message.content:
                            if isinstance(block, TextBlock):
                                accumulated_text += block.text
                            elif isinstance(block, ToolUseBlock):
                                tool_log_parts.append(
                                    f"\n[TOOL USE: {block.name}]\n{block.input}\n"
                                )
                    elif isinstance(message, UserMessage):
                        for block in message.content:
                            if isinstance(block, ToolResultBlock):
                                content = block.content
                                if isinstance(content, list):
                                    content = "\n".join(
                                        b.get("text", str(b)) if isinstance(b, dict) else str(b)
                                        for b in content
                                    )
                                prefix = "[TOOL ERROR]" if block.is_error else "[TOOL RESULT]"
                                content_str = str(content) if content else ""
                                if len(content_str) > 2000:
                                    content_str = content_str[:2000] + "\n... (truncated)"
                                tool_log_parts.append(f"\n{prefix}\n{content_str}\n")
                    elif isinstance(message, ResultMessage):
                        sid = getattr(message, "session_id", None)
                        if sid:
                            self._session_ids[path_key] = sid
                        if hasattr(message, "text") and message.text:
                            accumulated_text += message.text
                        cost = getattr(message, "total_cost_usd", None)
                        usage = getattr(message, "usage", None)
                        self.total_calls += 1
                        if cost is not None:
                            self.total_estimated_cost += cost
                        if usage:
                            inp = usage.get("input_tokens", 0) or 0
                            out = usage.get("output_tokens", 0) or 0
                            cc = usage.get("cache_creation_input_tokens", 0) or 0
                            cr = usage.get("cache_read_input_tokens", 0) or 0
                            self.total_input_tokens += inp
                            self.total_output_tokens += out
                            self.total_cache_create_tokens += cc
                            self.total_cache_read_tokens += cr
                            log.info(
                                "action=%d: tokens in=%d out=%d cache_create=%d cache_read=%d | "
                                "cumulative: calls=%d in=%d out=%d est_cost=$%.2f",
                                action_num, inp, out, cc, cr,
                                self.total_calls,
                                self.total_input_tokens,
                                self.total_output_tokens,
                                self.total_estimated_cost,
                            )
                return accumulated_text

            loop = asyncio.new_event_loop()
            try:
                result = loop.run_until_complete(
                    asyncio.wait_for(_run(), timeout=self._timeout)
                    if self._timeout
                    else _run()
                )
            finally:
                loop.close()

            with open(analyzer_log, "a", encoding="utf-8") as f:
                if tool_log_parts:
                    f.write("[TOOL CALLS]\n")
                    f.write("".join(tool_log_parts))
                    f.write("\n")
                f.write(f"[ASSISTANT]\n{result}\n\n")

            hint = result.strip() if result else None
            if not hint:
                log.warning("action=%d: empty response", action_num)
                return None

            log.info("action=%d OK (%d chars)", action_num, len(hint))
            return hint

        except asyncio.TimeoutError:
            log.warning("timed out at action %d — clearing session", action_num)
            self._session_ids.pop(path_key, None)
            with open(analyzer_log, "a", encoding="utf-8") as f:
                f.write("[TIMEOUT]\n")
            return None

        except Exception as e:
            log.error("claude agent error: %s — clearing session", e, exc_info=True)
            for attr in ("stderr", "stdout", "output", "message", "args"):
                val = getattr(e, attr, None)
                if val:
                    log.error("  error.%s: %s", attr, str(val)[:500])
            self._session_ids.pop(path_key, None)
            if stderr_lines:
                log.error("stderr: %s", "\n".join(stderr_lines[-20:]))
            return None

        finally:
            if saved_api_key is not None:
                os.environ["ANTHROPIC_API_KEY"] = saved_api_key
