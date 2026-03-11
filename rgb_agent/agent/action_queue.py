"""ActionQueue: parses JSON action plans and serves them one at a time."""
from __future__ import annotations

import json
import logging
import re
from collections import deque

log = logging.getLogger(__name__)


class QueueExhausted(RuntimeError):
    pass


_VALID_ACTIONS = {"ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6", "RESET"}


class ActionQueue:
    """Holds and serves a batch of parsed actions, with score-change flushing."""

    def __init__(self) -> None:
        self._queue: deque[dict] = deque()
        self.plan_total: int = 0
        self.plan_index: int = 0
        self._last_score: int = 0
        self.score_changed: bool = False

    def clear(self) -> None:
        self._queue.clear()
        self.plan_total = 0
        self.plan_index = 0

    def reset(self) -> None:
        self.clear()
        self._last_score = 0
        self.score_changed = False

    def __len__(self) -> int:
        return len(self._queue)

    def __bool__(self) -> bool:
        return bool(self._queue)

    def pop(self) -> dict:
        action = self._queue.popleft()
        self.plan_index += 1
        return action

    def check_score(self, score: int) -> None:
        """Flush the queue if the score changed."""
        if score != self._last_score:
            if self._queue:
                log.info("score %d->%d: flushing %d queued actions",
                         self._last_score, score, len(self._queue))
                self.clear()
            self.score_changed = True
            self._last_score = score

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
