"""GameState: grid processing, step history, state-action memory, and log formatting."""
from __future__ import annotations

import json
import logging
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from arcengine import GameAction
from rgb_agent.utils.grid_utils import (
    compute_grid_diff,
    format_grid_ascii,
    get_click_info,
    hash_grid_state,
)

log = logging.getLogger(__name__)


@dataclass
class Step:
    observation: Any = None
    action: Any = None
    model_response: str = ""
    chat_completions: list[dict[str, str]] = field(default_factory=list)
    reward: float = 0.0
    done: bool = False
    info: dict = field(default_factory=dict)


@dataclass
class Trajectory:
    uid: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "agent"
    steps: list[Step] = field(default_factory=list)


class GameState:
    """Tracks game state and formats the prompt log that the analyzer agent reads."""

    def __init__(
        self,
        *,
        name: str = "rgb_agent",
        game_id: str | None = None,
        context_window_size: int = 5,
        show_tried_actions: bool = True,
        include_strategy_in_context: bool = False,
        **_: Any,
    ) -> None:
        self.name = name
        self.game_id = game_id
        self.context_window_size = context_window_size
        self.show_tried_actions = show_tried_actions
        self.include_strategy_in_context = include_strategy_in_context

        self._step_history: deque = deque(maxlen=self.context_window_size)
        self._state_action_memory: dict[str, dict[str, dict[str, Any]]] = {}
        self.reset()

    def reset(self) -> None:
        self.trajectory = Trajectory(name=self.name)
        self.last_observation: dict[str, Any] | None = None
        self.action_counter: int = 0
        self.last_executed_action: str | None = None
        self._step_history = deque(maxlen=self.context_window_size)
        self._state_action_memory = {}
        self._external_hint: str | None = None
        self._persistent_hint: str | None = None
        self._pending_state_action: dict | None = None
        self._last_observation_prompt: str = ""
        self._last_observation_response: str = ""
        self._last_action_prompt: str = ""
        self._last_action_response: str = ""

    # --- Hints (set by runner from analyzer output) ---

    def set_external_hint(self, hint: str) -> None:
        """One-shot strategic hint for the next observation prompt."""
        self._external_hint = hint
        self._persistent_hint = None

    def set_persistent_hint(self, plan: str) -> None:
        """Short plan that persists on every prompt until the next analysis."""
        self._persistent_hint = plan

    # --- Grid processing ---

    def process_frame(self, obs: dict) -> tuple[list[list[int]], str]:
        frame_3d = obs.get("frame", [])
        grid_raw = [list(row) for row in frame_3d[-1]] if frame_3d else []
        return grid_raw, format_grid_ascii(grid_raw) if grid_raw else ""

    def render_board(self) -> str | None:
        _, grid_text = self.process_frame(self.last_observation or {})
        return grid_text or None

    # --- State-action memory ---

    def _record_state_action(self, state_hash: str, action_key: str, result: dict[str, Any]) -> None:
        self._state_action_memory.setdefault(state_hash, {})[action_key] = result

    def _get_tried_actions(self, state_hash: str) -> dict[str, dict[str, Any]]:
        return self._state_action_memory.get(state_hash, {})

    def format_state_action_context(self, grid: list[list[int]]) -> str:
        if not self.show_tried_actions:
            return ""
        tried = self._get_tried_actions(hash_grid_state(grid))
        if not tried:
            return ""
        lines = ["**From Current State, Already Tried:**\n"]
        for action_key, result in tried.items():
            changed = result.get("changed", False)
            diff = result.get("diff", "")
            marker = "changed" if changed else "no change"
            lines.append(f"- {action_key}: {marker}")
            if diff and changed:
                lines.append(f"    Diff: {diff}")
        lines.append("")
        return "\n".join(lines)

    # --- Step history ---

    def format_step_history(self, include_strategy: bool = True) -> str:
        if not self._step_history:
            return ""
        lines = ["**Recent History:**\n"]
        for entry in self._step_history:
            dup = " NO STATE CHANGE" if entry.get("no_state_change") else ""
            pre = entry.get("grid_raw", [])
            post = entry.get("post_grid_raw")
            diff = compute_grid_diff(pre, post) if post is not None else "(pending)"
            text = (
                f"Step {entry['step']}: {entry['action']}, Score={entry['score']}{dup}\n"
                f"  Changes: {diff}\n"
            )
            if include_strategy:
                obs_resp = entry.get("obs_response", "")
                if obs_resp:
                    text += f"  [Strategy]: {obs_resp}\n"
            lines.append(text)
        return "\n".join(lines) + "\n"

    # --- Build observation context (for prompt log) ---

    def build_observation_context(
        self, grid: str, score: int, grid_raw: list, *, use_queued: bool, queue: Any,
    ) -> str:
        history = self.format_step_history()
        tried = self.format_state_action_context(grid_raw)

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
            f"Step: {self.action_counter}\n\n"
            f"**Current Matrix** 64x64 (ASCII characters):\n{grid}\n"
        )

        if use_queued:
            label = f"step {queue.plan_index + 1}/{queue.plan_total}"
            context += f"\n[Executing pre-planned action ({label}) — no model call]\n"
            self._last_observation_prompt = f"[Queued plan {label}]\n\n{context}"
            self._last_observation_response = f"[Pre-planned action {label}]"
        else:
            self._last_observation_prompt = f"[Observation context]\n\n{context}"
            self._last_observation_response = "[Observation model — context assembled]"

        return context

    # --- Record environment update ---

    def record_env_update(self, observation: Any, reward: float, done: bool, info: dict = None) -> None:
        self.last_observation = observation

        prompts = []
        if self._last_observation_prompt:
            prompts.append({"role": "observation_phase", "content": self._last_observation_prompt})
        if self._last_observation_response:
            prompts.append({"role": "observation_response", "content": self._last_observation_response})
        if self._last_action_prompt:
            prompts.append({"role": "action_phase", "content": self._last_action_prompt})
        if self._last_action_response:
            prompts.append({"role": "action_response", "content": self._last_action_response})

        step = Step(observation=observation, reward=reward, done=done, info=info, chat_completions=prompts)
        self.trajectory.steps.append(step)

        if self._step_history:
            grid_raw, _ = self.process_frame(observation)
            pre_grid = self._step_history[-1].get("grid_raw", [])
            no_change = (pre_grid == grid_raw)
            self._step_history[-1]["no_state_change"] = no_change
            self._step_history[-1]["post_grid_raw"] = grid_raw

            if self._pending_state_action:
                diff = compute_grid_diff(pre_grid, grid_raw)
                record = {"changed": not no_change, "diff": diff if not no_change else ""}
                record.update(self._pending_state_action.get("extra", {}))
                self._record_state_action(
                    self._pending_state_action["state_hash"],
                    self._pending_state_action["action_key"],
                    record,
                )
                self._pending_state_action = None

    # --- Record model/action update ---

    def record_action(self, action_dict: dict) -> dict:
        """Record an action and return the GameAction-based result for env.step()."""
        obs_text = action_dict.get("obs_text", "")
        response_text = f"Observation: {obs_text}\nAction: {action_dict['name']}"

        if self.trajectory.steps:
            self.trajectory.steps[-1].model_response = response_text
            self.trajectory.steps[-1].action = action_dict

        obs = self.last_observation or {}
        grid_raw, _ = self.process_frame(obs)
        action_name = action_dict["name"]

        if action_name == "ACTION6":
            data = action_dict.get("data", {})
            x, y = data.get("x", 0), data.get("y", 0)
            label, comp_id = get_click_info(grid_raw, x, y)
            action_display = f"ACTION6(x={x}, y={y}, {label})"
            self._pending_state_action = {
                "state_hash": hash_grid_state(grid_raw),
                "action_key": f"click_{comp_id}",
                "extra": {"x": x, "y": y},
            }
        else:
            action_display = action_name
            self._pending_state_action = {
                "state_hash": hash_grid_state(grid_raw),
                "action_key": action_name,
                "extra": {},
            }

        self._step_history.append({
            "step": self.action_counter,
            "action": action_display,
            "score": obs.get("score", 0),
            "state": obs.get("state", "UNKNOWN"),
            "grid_raw": grid_raw,
            "no_state_change": False,
            "obs_response": self._last_observation_response if self.include_strategy_in_context else "",
        })

        self.action_counter += 1
        self.last_executed_action = action_name

        action = GameAction.from_name(action_name)
        result = {"action": action, "reasoning": response_text}
        if action == GameAction.ACTION6:
            x_pos = max(0, min(63, int(action_dict["data"].get("x", 0))))
            y_pos = max(0, min(63, int(action_dict["data"].get("y", 0))))
            result["x"] = y_pos
            result["y"] = x_pos
        return result
