"""GameRunner — orchestrator that drives the agent through a game loop.

The Swarm creates one per game thread and calls runner.run().
"""
import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional

import requests

from arcgym.agents.rgb_agent import RGBAgent, QueueExhausted
from arcgym.environments import ArcAgi3Env
from arcengine import GameState
from arcgym.metrics.structures import GameMetrics, LevelMetrics, AttemptMetrics, Status

log = logging.getLogger(__name__)

ROOT_URL = os.environ.get("ROOT_URL", "https://three.arcprize.org")
MAX_RETRIES = 5
INITIAL_BACKOFF = 1

_RETRY_NUDGE = (
    "CRITICAL: Your previous response was missing the [ACTIONS] section. "
    "You MUST end your response with an [ACTIONS] section containing a JSON action plan. "
    "Do NOT write actions to a file — output them directly in your response text."
)


def _run_with_retries(func: Callable, *args: Any, **kwargs: Any) -> Any:
    retries = 0
    backoff = INITIAL_BACKOFF
    while True:
        try:
            return func(*args, **kwargs)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            if retries >= MAX_RETRIES:
                log.error("Final attempt failed for %s after %d retries.", func.__name__, retries)
                raise
            log.warning("%s: %s. Retrying in %ds (%d/%d)",
                        func.__name__, type(e).__name__, backoff, retries + 1, MAX_RETRIES)
            time.sleep(backoff)
            retries += 1
            backoff *= 2


class GameRunner:
    """Drives the RGBAgent through a game loop.

    Not an agent — owns the env interaction loop,
    manages the analyzer, and collects metrics.
    """

    def __init__(
        self,
        *,
        env: ArcAgi3Env,
        game_id: str,
        agent_name: str,
        max_actions_per_game: int,
        run_index: int = 1,
        tags: Optional[list[str]] = None,
        prompts_log_path: Optional[Path] = None,
        analyzer=None,
        log_post_board: bool = False,
        analyzer_retries: int = 5,
        agent_kwargs: Optional[dict] = None,
    ) -> None:
        self.env = env
        self.game_id = game_id
        self.agent_name = agent_name
        self.max_actions_per_game = max_actions_per_game
        self.run_index = run_index
        self.tags = tags
        self.prompts_log_path = prompts_log_path
        self.analyzer = analyzer
        self.log_post_board = log_post_board
        self.analyzer_retries = analyzer_retries
        self._agent = RGBAgent(**(agent_kwargs or {}))

    def run(self) -> GameMetrics:
        metrics = GameMetrics(
            game_id=self.game_id,
            agent_name=self.agent_name,
            run_index=self.run_index,
            start_time=time.time(),
        )
        metrics.status = Status.IN_PROGRESS

        level_num = 1
        level_metrics = LevelMetrics(level_number=level_num)
        attempt_num = 1
        attempt_metrics = AttemptMetrics(attempt_number=attempt_num)
        attempt_start = metrics.start_time

        max_score = 0
        total_actions = 0
        arc_state: GameState | None = None
        arc_score = 0

        loop = asyncio.new_event_loop()

        try:
            self._agent.reset()

            # Initial reset
            observation = _run_with_retries(
                self.env.reset,
                task={"game_id": self.game_id, "max_actions": self.max_actions_per_game, "tags": self.tags},
            )
            arc_state = GameState[observation.get("state") or "NOT_PLAYED"]
            arc_score = observation.get("score", 0) or 0

            guid = observation.get("guid")
            if guid and not metrics.guid:
                metrics.guid = guid
                metrics.replay_url = f"{ROOT_URL}/replay/{self.game_id}/{guid}"
                log.info("[%s Run %d] Replay URL: %s", self.game_id, self.run_index, metrics.replay_url)
                if self.prompts_log_path:
                    info_path = self.prompts_log_path.parent / "run_info.txt"
                    info_path.write_text(
                        f"game_id: {self.game_id}\n"
                        f"guid: {guid}\n"
                        f"replay_url: {metrics.replay_url}\n"
                        f"scorecard_id: {getattr(self.env, '_scorecard_id', 'unknown')}\n"
                        f"command: {Path(sys.argv[0]).name} {' '.join(sys.argv[1:])}\n"
                    )

            self._agent.update_from_env(observation=observation, reward=0.0, done=False)

            # Log initial board
            if self.prompts_log_path:
                grid = self._agent.render_board()
                if grid:
                    with open(self.prompts_log_path, 'a', encoding='utf-8') as f:
                        f.write(f"\n{'='*80}\n")
                        f.write(f"Action 0 | Level {level_num} | Attempt {attempt_num} | INITIAL STATE\n")
                        f.write(f"Score: {arc_score} | State: {arc_state.name}\n")
                        f.write(f"{'='*80}\n\n")
                        f.write(f"[INITIAL BOARD STATE]\n{grid}\n\n")

            # Main game loop
            while total_actions < self.max_actions_per_game:
                try:
                    action_dict = loop.run_until_complete(self._agent.call_llm())
                except QueueExhausted:
                    log.info("queue exhausted at action %d — firing analyzer", total_actions)
                    loaded = False
                    for attempt in range(self.analyzer_retries):
                        nudge = _RETRY_NUDGE if attempt > 0 or total_actions > 0 else ""
                        log.info("analyzer attempt %d/%d action=%d nudge=%s",
                                 attempt + 1, self.analyzer_retries, total_actions, bool(nudge))
                        if self._fire_analyzer(total_actions, arc_score, retry_nudge=nudge):
                            loaded = True
                            break
                        log.warning("analyzer attempt %d/%d failed", attempt + 1, self.analyzer_retries)
                    if not loaded:
                        raise
                    action_dict = loop.run_until_complete(self._agent.call_llm())

                action_obj = self._agent.update_from_model()
                observation, reward, done = _run_with_retries(self.env.step, action_obj)

                total_actions += 1
                attempt_metrics.actions += 1

                prev_score = arc_score
                arc_state = GameState[observation.get("state") or "NOT_PLAYED"]
                arc_score = observation.get("score", 0) or 0
                max_score = max(max_score, arc_score)
                metrics.highest_level_reached = max(metrics.highest_level_reached, level_num)

                self._agent.update_from_env(observation=observation, reward=reward, done=done)

                self._log_action(total_actions, level_num, attempt_num, arc_score, arc_state)

                if self.log_post_board and self.prompts_log_path:
                    grid = self._agent.render_board()
                    if grid:
                        with open(self.prompts_log_path, 'a', encoding='utf-8') as f:
                            f.write(f"[POST-ACTION BOARD STATE]\nScore: {arc_score}\n{grid}\n\n")

                # Level completed
                if arc_score > prev_score and arc_state not in (GameState.WIN, GameState.GAME_OVER):
                    attempt_metrics.duration_seconds = time.time() - attempt_start
                    attempt_metrics.status = Status.COMPLETED
                    level_metrics.attempts.append(attempt_metrics)
                    level_metrics.status = Status.COMPLETED
                    metrics.level_metrics[level_num] = level_metrics

                    log.info("[%s Run %d] Level %d COMPLETED. Attempt %d actions: %d. Score: %d.",
                             self.game_id, self.run_index, level_num, attempt_num, attempt_metrics.actions, arc_score)

                    level_num += 1
                    metrics.highest_level_reached = max(metrics.highest_level_reached, level_num)
                    level_metrics = LevelMetrics(level_number=level_num)
                    attempt_num = 1
                    attempt_metrics = AttemptMetrics(attempt_number=attempt_num)
                    attempt_start = time.time()
                    continue

                if arc_state == GameState.GAME_OVER:
                    attempt_metrics.duration_seconds = time.time() - attempt_start
                    attempt_metrics.status = Status.GAME_OVER
                    attempt_metrics.game_overs += 1
                    level_metrics.attempts.append(attempt_metrics)
                    level_metrics.status = Status.GAME_OVER
                    metrics.level_metrics[level_num] = level_metrics
                    metrics.status = Status.TIMEOUT
                    log.warning("[%s Run %d] Game Over on Level %d, Attempt %d. Actions: %d.",
                                self.game_id, self.run_index, level_num, attempt_num, attempt_metrics.actions)
                    attempt_num += 1
                    attempt_metrics = AttemptMetrics(attempt_number=attempt_num)
                    attempt_start = time.time()

                if arc_state == GameState.WIN:
                    attempt_metrics.duration_seconds = time.time() - attempt_start
                    attempt_metrics.status = Status.COMPLETED
                    level_metrics.attempts.append(attempt_metrics)
                    level_metrics.status = Status.COMPLETED
                    metrics.level_metrics[level_num] = level_metrics
                    metrics.status = Status.COMPLETED_RUN
                    log.info("[%s Run %d] Game COMPLETED! Level %d actions: %d. Score: %d",
                             self.game_id, self.run_index, level_num, attempt_metrics.actions, arc_score)
                    break

        except QueueExhausted as e:
            log.info("[%s Run %d] Episode ended (queue exhausted): %s", self.game_id, self.run_index, e)
            metrics.status = Status.QUEUE_EXHAUSTED

        except Exception as e:
            metrics.status = Status.ERROR
            metrics.error_message = str(e)
            attempt_metrics.status = Status.ERROR
            level_metrics.status = Status.ERROR
            log.error("[%s Run %d] Exception: %s", self.game_id, self.run_index, e, exc_info=True)

        finally:
            metrics.end_time = time.time()
            metrics.run_duration_seconds = metrics.end_time - metrics.start_time

            if attempt_metrics.status == Status.IN_PROGRESS:
                attempt_metrics.duration_seconds = metrics.end_time - attempt_start
                if metrics.status == Status.ERROR:
                    attempt_metrics.status = Status.ERROR
                elif arc_state == GameState.WIN:
                    attempt_metrics.status = Status.COMPLETED
                    metrics.status = Status.COMPLETED_RUN
                else:
                    attempt_metrics.status = Status.TIMEOUT
                    if metrics.status == Status.IN_PROGRESS:
                        metrics.status = Status.TIMEOUT

            if (not level_metrics.attempts
                    or level_metrics.attempts[-1].attempt_number != attempt_metrics.attempt_number):
                level_metrics.attempts.append(attempt_metrics)
            if level_metrics.status == Status.IN_PROGRESS:
                level_metrics.status = attempt_metrics.status

            metrics.level_metrics[level_num] = level_metrics
            metrics.run_total_actions = sum(lm.total_actions for lm in metrics.level_metrics.values())
            metrics.total_game_overs_across_run = sum(lm.total_game_overs for lm in metrics.level_metrics.values())
            metrics.total_state_changes_across_run = sum(lm.total_state_changes for lm in metrics.level_metrics.values())
            metrics.final_score = max_score

            if metrics.guid and not metrics.replay_url:
                metrics.replay_url = f"{ROOT_URL}/replay/{self.game_id}/{metrics.guid}"

            loop.close()

        return metrics

    # -- private helpers -------------------------------------------------------

    def _fire_analyzer(self, action_num: int, arc_score: int, retry_nudge: str = "") -> bool:
        if not self.analyzer:
            return False
        if self.prompts_log_path and not self.log_post_board:
            grid = self._agent.render_board()
            if grid:
                with open(self.prompts_log_path, 'a', encoding='utf-8') as f:
                    f.write(f"[POST-ACTION BOARD STATE]\nScore: {arc_score}\n{grid}\n\n")

        hint = self.analyzer(self.prompts_log_path, action_num, retry_nudge=retry_nudge)
        if not hint:
            log.warning("analyzer returned None at action %d", action_num)
            return False

        hint = "\n".join(line.rstrip() for line in hint.split("\n"))

        actions_text = None
        if "\n[ACTIONS]\n" in hint:
            hint, actions_text = hint.split("\n[ACTIONS]\n", 1)
            actions_text = actions_text.strip()

        if "\n[PLAN]\n" in hint:
            full_hint, plan = hint.split("\n[PLAN]\n", 1)
            full_hint, plan = full_hint.strip(), plan.strip()
        else:
            full_hint = plan = hint

        self._agent.set_external_hint(full_hint)
        self._agent.set_persistent_hint(plan)

        if actions_text:
            if self._agent.set_action_plan(actions_text):
                log.info("analyzer at action %d: loaded action plan (%d chars)", action_num, len(actions_text))
                return True
            log.warning("analyzer at action %d: set_action_plan rejected the plan", action_num)
            return False

        log.warning("analyzer at action %d: hint received but NO [ACTIONS] section", action_num)
        return False

    def _log_action(self, action_num: int, level: int, attempt: int,
                    arc_score: int, arc_state: GameState) -> None:
        if not self.prompts_log_path or not self._agent.trajectory.steps:
            return
        last_step = self._agent.trajectory.steps[-1]
        with open(self.prompts_log_path, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*80}\n")
            plan_info = f" | Plan Step {self._agent.plan_index}/{self._agent.plan_total}" if self._agent.plan_total > 0 else ""
            f.write(f"Action {action_num} | Level {level} | Attempt {attempt}{plan_info}\n")
            f.write(f"Score: {arc_score} | State: {arc_state.name}\n")
            f.write(f"{'='*80}\n\n")
            if last_step.chat_completions:
                for msg in last_step.chat_completions:
                    role = msg.get('role', 'unknown')
                    content = msg.get('content', '')
                    tool_calls = msg.get('tool_calls', [])
                    f.write(f"[{role.upper()}]\n")
                    if content:
                        f.write(f"{content}\n")
                    for tc in tool_calls:
                        fn = tc.get('function', {}) if isinstance(tc, dict) else {}
                        f.write(f"Tool: {fn.get('name', tc)}({fn.get('arguments', '')})\n")
                    f.write("\n")
