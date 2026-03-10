"""Run one scorecard across multiple games in parallel threads.

Usage:
    arcgym-swarm --suite all --max-actions 500
    arcgym-swarm --game ls20,ft09
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

import arc_agi
from arc_agi import OperationMode

from arcgym.agents import AVAILABLE_AGENTS
from arcgym.evaluation.runner import GameRunner
from arcgym.environments import ArcAgi3Env
from arcgym.evaluation.config import EVALUATION_GAMES
from arcgym.metrics.structures import GameMetrics, Status
from arcgym.metrics.reporting import generate_console_report, save_summary_report, calculate_stats

log = logging.getLogger(__name__)

_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
load_dotenv(dotenv_path=_project_root / ".env.example")
load_dotenv(dotenv_path=_project_root / ".env", override=True)

ROOT_URL = os.environ.get("ROOT_URL", "https://three.arcprize.org")


class Swarm:
    """Manages a single scorecard and runs one agent per game in daemon threads."""

    def __init__(
        self,
        inner_agent_kwargs: dict[str, Any],
        arcade: arc_agi.Arcade,
        games: list[str],
        tags: list[str],
        max_actions: int = 500,
        analyzer_hook: Any = None,
        prompts_log_dir: Path | None = None,
        log_post_board: bool = True,
        analyzer_retries: int = 5,
    ) -> None:
        self.inner_agent_kwargs = inner_agent_kwargs
        self._arcade = arcade
        self.games = games
        self.tags = tags
        self.max_actions = max_actions
        self.analyzer_hook = analyzer_hook
        self.prompts_log_dir = prompts_log_dir
        self.log_post_board = log_post_board
        self.analyzer_retries = analyzer_retries

        self.card_id: str | None = None
        self.scorecard: Any = None
        self.results: dict[str, GameMetrics] = {}
        self._lock = threading.Lock()

    def run(self) -> dict[str, GameMetrics]:
        self.card_id = self._arcade.open_scorecard(tags=self.tags)
        log.info("Opened scorecard %s for %d game(s)", self.card_id, len(self.games))

        threads = [
            threading.Thread(target=self._run_game, args=(self.card_id, gid), daemon=True)
            for gid in self.games
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.scorecard = self._arcade.close_scorecard(self.card_id)
        log.info("Closed scorecard %s", self.card_id)
        return self.results

    def _run_game(self, card_id: str, game_id: str) -> None:
        try:
            env = ArcAgi3Env.from_arcade(
                arcade=self._arcade, game_id=game_id,
                scorecard_id=card_id, max_actions=self.max_actions,
            )

            prompts_log_path = None
            if self.prompts_log_dir:
                game_dir = self.prompts_log_dir / game_id.split("-")[0]
                game_dir.mkdir(parents=True, exist_ok=True)
                prompts_log_path = game_dir / "logs.txt"
                prompts_log_path.write_text("")

            runner = GameRunner(
                env=env,
                game_id=game_id,
                agent_name=self.inner_agent_kwargs.get("name", "swarm_agent"),
                max_actions_per_game=self.max_actions,
                tags=self.tags,
                prompts_log_path=prompts_log_path,
                analyzer=self.analyzer_hook,
                log_post_board=self.log_post_board,
                analyzer_retries=self.analyzer_retries,
                agent_kwargs=self.inner_agent_kwargs,
            )
            metrics = runner.run()

            with self._lock:
                self.results[game_id] = metrics

        except Exception as exc:
            log.error("Game %s failed: %s", game_id, exc, exc_info=True)
            with self._lock:
                self.results[game_id] = GameMetrics(
                    game_id=game_id,
                    agent_name=self.inner_agent_kwargs.get("name", "swarm_agent"),
                    start_time=time.time(),
                    status=Status.ERROR,
                    error_message=str(exc),
                )
        finally:
            try:
                env.close()
            except Exception:
                pass


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logging.getLogger("arc_agi").propagate = False

    parser = argparse.ArgumentParser(description="Run ARC-AGI-3 Swarm evaluation.")
    parser.add_argument("--agent", "-a", default="rgb_agent",
                        choices=list(AVAILABLE_AGENTS.keys()))
    parser.add_argument("--game", "-g",
                        help="Comma-separated game IDs (e.g. ls20-cb3b57cc,ft09-9ab2447a).")
    parser.add_argument("--suite", "-s", choices=list(EVALUATION_GAMES.keys()))
    parser.add_argument("--tags", "-t", help="Comma-separated tags.")
    parser.add_argument("--max-actions", type=int, default=500)
    parser.add_argument("--operation-mode", default="online", choices=["normal", "online", "offline"])
    parser.add_argument("--analyzer-interval", dest="analyzer_interval", type=int, default=10)
    parser.add_argument("--analyzer-model", dest="analyzer_model", default="claude-opus-4-6")
    parser.add_argument("--analyzer-retries", dest="analyzer_retries", type=int, default=5)

    args = parser.parse_args()

    # Resolve game list — support short names (e.g. "ls20" -> "ls20-cb3b57cc")
    all_known = {gid for ids in EVALUATION_GAMES.values() for gid in ids}
    prefix_map = {gid.split("-")[0]: gid for gid in all_known}

    games: list[str] = []
    if args.game:
        raw = [g.strip() for g in args.game.split(",") if g.strip()]
        games = [prefix_map.get(g, g) for g in raw]
    elif args.suite:
        games = EVALUATION_GAMES[args.suite]
    else:
        api_key = os.getenv("ARC_API_KEY", "")
        try:
            resp = requests.get(
                f"{ROOT_URL}/api/games",
                headers={"X-API-Key": api_key, "Accept": "application/json"},
                timeout=15,
            )
            resp.raise_for_status()
            games = [g["game_id"] for g in resp.json()]
            log.info("Fetched %d games from API", len(games))
        except Exception as exc:
            log.error("Failed to fetch games from API: %s", exc)
            sys.exit(1)

    if not games:
        log.error("No games to run. Provide --game, --suite, or set ARC_API_KEY.")
        sys.exit(1)

    tags = [t.strip() for t in (args.tags or "").split(",") if t.strip()]
    tags.append(f"swarm-{args.agent}")

    arcade = arc_agi.Arcade(
        arc_api_key=os.getenv("ARC_API_KEY", ""),
        arc_base_url=ROOT_URL,
        operation_mode=OperationMode(args.operation_mode),
    )

    from arcgym.agents.rgb_agent import make_analyzer

    analyzer_hook = make_analyzer(
        interval=0, use_subscription=False, allow_bash=True,
        action_mode="all", plan_size=args.analyzer_interval,
        allow_self_read=False, model=args.analyzer_model,
        fast=False, resume_session=True,
    )
    log.info("Analyzer enabled (interval=%d, model=%s)", args.analyzer_interval, args.analyzer_model)

    timestamp = datetime.now().strftime("%m%dT%H%M%S")
    run_dir = Path("evaluation_results") / f"{timestamp}_swarm_{args.agent}"
    run_dir.mkdir(parents=True, exist_ok=True)

    inner_agent_kwargs: dict[str, Any] = {
        "name": args.agent,
        "plan_size": args.analyzer_interval,
    }

    swarm = Swarm(
        inner_agent_kwargs=inner_agent_kwargs,
        arcade=arcade, games=games, tags=tags,
        max_actions=args.max_actions,
        analyzer_hook=analyzer_hook,
        prompts_log_dir=run_dir,
        log_post_board=True,
        analyzer_retries=args.analyzer_retries,
    )

    runner = threading.Thread(target=swarm.run, daemon=True)
    runner.start()

    def sigint_handler(sig: int, frame: Any) -> None:
        print("[Swarm] SIGINT received — cleaning up...", flush=True)
        sys.exit(1)

    signal.signal(signal.SIGINT, sigint_handler)

    while runner.is_alive():
        runner.join(timeout=1)

    results_list = list(swarm.results.values())

    print(f"\nScorecard ID: {swarm.card_id}")
    print(f"Results:      {run_dir}")
    for m in sorted(results_list, key=lambda r: r.game_id):
        if m.replay_url:
            print(f"  Replay:     {m.replay_url}")

    if swarm.scorecard:
        sc = swarm.scorecard
        print(f"\n{'='*60}")
        print(f"ARC Scorecard  —  overall score: {sc.score:.1f}")
        print(f"  Environments: {sc.total_environments_completed}/{sc.total_environments}")
        print(f"  Levels:       {sc.total_levels_completed}/{sc.total_levels}")
        print(f"  Actions:      {sc.total_actions}")
        for env in sc.environments:
            run = env.runs[0] if env.runs else None
            if not run:
                continue
            label = env.id or "unknown"
            state = run.state.name if run.state else "?"
            print(f"\n  {label}  score={run.score:.1f}  state={state}  actions={run.actions}")
            if run.level_scores:
                for i, (ls, la, lb) in enumerate(zip(
                    run.level_scores,
                    run.level_actions or [],
                    run.level_baseline_actions or [],
                )):
                    baseline = str(lb) if lb >= 0 else "n/a"
                    print(f"    Level {i+1}: efficiency={ls:.1f}  actions={la}  baseline={baseline}")
            if run.message:
                print(f"    Note: {run.message}")
        print(f"{'='*60}")

        scorecard_path = run_dir / "scorecard.json"
        scorecard_path.write_text(sc.model_dump_json(indent=2))
        log.info("Scorecard saved to %s", scorecard_path)

    if results_list:
        generate_console_report(results_list, "swarm", args.agent, 1, scorecard=swarm.scorecard)
        game_stats, overall = calculate_stats(results_list)
        summary_path = run_dir / "summary.txt"
        save_summary_report(
            str(summary_path), game_stats, overall, results_list,
            args.agent, "swarm", 1, scorecard=swarm.scorecard,
        )
        log.info("Summary saved to %s", summary_path)
    else:
        log.error("No results collected.")


if __name__ == "__main__":
    main()
