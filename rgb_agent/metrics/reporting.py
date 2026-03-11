"""Evaluation summary and statistics reporting."""
import textwrap
from collections import defaultdict
from typing import Any
from datetime import datetime
import statistics

from .structures import GameMetrics, Status


def calculate_stats(results: list[GameMetrics]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    game_level_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    game_summary = defaultdict(lambda: defaultdict(list))

    total_runs = 0
    total_completed = 0
    total_duration = 0.0

    for res in results:
        gid = res.game_id
        total_runs += 1
        total_duration += res.run_duration_seconds

        game_summary[gid]['status'].append(res.status)
        game_summary[gid]['final_score'].append(res.final_score)
        game_summary[gid]['run_total_actions'].append(res.run_total_actions)
        game_summary[gid]['run_duration'].append(res.run_duration_seconds)
        game_summary[gid]['total_game_overs'].append(res.total_game_overs_across_run)
        game_summary[gid]['highest_level'].append(res.highest_level_reached)

        if res.status == Status.COMPLETED_RUN:
            total_completed += 1

        for lnum, ldata in res.level_metrics.items():
            lvl = game_level_stats[gid][lnum]
            lvl['total_actions'].append(ldata.total_actions)
            lvl['total_game_overs'].append(ldata.total_game_overs)
            lvl['total_state_changes'].append(ldata.total_state_changes)
            lvl['status'].append(ldata.status)
            sa = ldata.actions_in_successful_attempt
            if sa is not None:
                lvl['success_actions'].append(sa)
            lvl['attempt_durations'].extend(a.duration_seconds for a in ldata.attempts)

    overall = {
        "total_runs": total_runs,
        "total_completed": total_completed,
        "overall_completion_rate": (total_completed / total_runs * 100.0) if total_runs else 0.0,
        "average_duration_all": total_duration / total_runs if total_runs else 0.0,
    }

    def _mean_or(lst, default=0.0):
        return statistics.mean(lst) if lst else default

    processed = {}
    for gid in set(game_summary) | set(game_level_stats):
        s = game_summary.get(gid, defaultdict(list))
        runs = len(s['status'])
        completed_runs = sum(1 for st in s['status'] if st == Status.COMPLETED_RUN)

        levels = {}
        raw_levels = game_level_stats.get(gid, {})
        if raw_levels:
            for lnum in range(1, max(raw_levels) + 1):
                ls = raw_levels.get(lnum, defaultdict(list))
                attempts = len(ls['status'])
                completed = sum(1 for st in ls['status'] if st == Status.COMPLETED)
                levels[lnum] = {
                    "attempts": attempts,
                    "completed": completed,
                    "avg_total_actions": _mean_or(ls['total_actions']),
                    "avg_success_actions": _mean_or(ls['success_actions']),
                    "avg_duration_per_attempt": _mean_or(ls['attempt_durations']),
                    "avg_total_state_changes": _mean_or(ls['total_state_changes']),
                    "avg_total_game_overs": _mean_or(ls['total_game_overs']),
                    "completion_rate": (completed / attempts * 100.0) if attempts else 0.0,
                }

        processed[gid] = {
            "num_runs": runs,
            "completed_runs": completed_runs,
            "avg_final_score": _mean_or(s['final_score']),
            "avg_run_total_actions": _mean_or(s['run_total_actions']),
            "avg_run_duration": _mean_or(s['run_duration']),
            "run_completion_rate": (completed_runs / runs * 100.0) if runs else 0.0,
            "avg_total_game_overs_per_run": _mean_or(s['total_game_overs']),
            "avg_highest_level": _mean_or(s['highest_level'], 1.0),
            "level_stats": levels,
        }

    return processed, overall


def _build_report_lines(
    game_stats: dict[str, dict[str, Any]],
    overall: dict[str, Any],
    results: list[GameMetrics],
    agent_name: str,
    suite_name: str,
    num_runs: int,
    scorecard: Any = None,
) -> list[str]:
    lines = [
        " Evaluation Report ",
        f"Agent: {agent_name}",
        f"Suite: {suite_name}",
        f"Requested Runs per Game: {num_runs}",
        f"Generated At: {datetime.now().isoformat()}",
        "-" * 50,
        "\n## Overall Summary",
        "-" * 50,
        f"Total Runs Attempted: {overall['total_runs']}",
        f"Total Runs Completed (Full Game Win): {overall['total_completed']}",
        f"Overall Game Completion Rate: {overall['overall_completion_rate']:.1f}%",
        f"Average Run Duration (all runs): {overall['average_duration_all']:.2f}s",
        "-" * 50,
        "\n## Per-Game Summary (Averaged Across Runs)",
    ]

    if not game_stats:
        lines.append("No game results to display.")
    else:
        for gid, stats in sorted(game_stats.items()):
            lines.extend([
                "\n" + "=" * 80,
                f"Game ID: {gid}",
                "=" * 80,
                "  Run Summary:",
                f"    Total Runs: {stats['num_runs']}",
                f"    Completed Runs: {stats['completed_runs']} ({stats['run_completion_rate']:.1f}%)",
                f"    Avg Final Score: {stats['avg_final_score']:.1f}",
                f"    Avg Highest Level: {stats['avg_highest_level']:.1f}",
                f"    Avg Total Actions: {stats['avg_run_total_actions']:.1f}",
                f"    Avg Run Duration: {stats['avg_run_duration']:.2f}s",
                f"    Avg Game Overs: {stats['avg_total_game_overs_per_run']:.1f}",
            ])

            if stats['level_stats']:
                header = (f"    {'Lvl':>3} | {'Avg Total Actions':>18} | {'Avg Success Actions':>20} | "
                          f"{'Avg Total GOs':>14} | {'Avg State D':>12} | {'Cmpl Rate':>11} | {'Attempts':>10}")
                lines.extend([
                    "\n  Level Statistics:",
                    "    " + "-" * (len(header) - 4),
                    header,
                    "    " + "-" * (len(header) - 4),
                ])
                for lnum, ls in sorted(stats['level_stats'].items()):
                    sa = f"{ls['avg_success_actions']:.1f}" if ls['avg_success_actions'] > 0 else "N/A"
                    lines.append(
                        f"    {lnum:>3} | {ls['avg_total_actions']:>18.1f} | {sa:>20} | "
                        f"{ls['avg_total_game_overs']:>14.1f} | {ls['avg_total_state_changes']:>12.1f} | "
                        f"{ls['completion_rate']:>10.1f}% | {ls['attempts']:>10}"
                    )
            else:
                lines.append("\n  No level statistics collected.")

    # Detailed run list
    lines.extend(["\n" + "=" * 80, "\n## Detailed Run List", "-" * 80])
    if not results:
        lines.append("No runs recorded.")
    else:
        current_gid = None
        for res in sorted(results, key=lambda r: (r.game_id, r.run_index)):
            if res.game_id != current_gid:
                lines.append(f"\nGame: {res.game_id}")
                current_gid = res.game_id
            detail = f"-> {res.replay_url or 'N/A'}"
            if res.status == Status.ERROR and res.error_message:
                detail = f"-> ERROR: {textwrap.shorten(res.error_message.replace(chr(10), ' '), width=70, placeholder='...')}"
            lines.append(
                f"  Run {res.run_index:>2}: {res.status.value:<15} Score={res.final_score:>4}, "
                f"HighestLvl={res.highest_level_reached:>2}, Actions={res.run_total_actions:>4}, "
                f"Dur={res.run_duration_seconds:.2f}s, GOs={res.total_game_overs_across_run:>3} {detail}"
            )
    lines.append("-" * 80)

    if scorecard:
        lines.extend([
            "\n" + "=" * 80,
            "## ARC Scorecard (Efficiency)",
            "=" * 80,
            f"  Overall Score: {scorecard.score:.1f}",
            f"  Environments:  {scorecard.total_environments_completed}/{scorecard.total_environments}",
            f"  Levels:        {scorecard.total_levels_completed}/{scorecard.total_levels}",
            f"  Total Actions: {scorecard.total_actions}",
        ])
        for env in scorecard.environments:
            run = env.runs[0] if env.runs else None
            if not run:
                continue
            label = env.id or "unknown"
            state = run.state.name if run.state else "?"
            lines.append(f"\n  {label}  score={run.score:.1f}  state={state}  actions={run.actions}")
            if run.level_scores:
                for i, (ls, la, lb) in enumerate(zip(
                    run.level_scores,
                    run.level_actions or [],
                    run.level_baseline_actions or [],
                )):
                    baseline = str(lb) if lb >= 0 else "n/a"
                    lines.append(f"    Level {i+1}: efficiency={ls:.1f}  actions={la}  baseline={baseline}")
            if run.message:
                lines.append(f"    Note: {run.message}")
        lines.append("-" * 80)

    lines.append("\n End of Report ")
    return lines


def generate_console_report(
    results_data: list[GameMetrics],
    suite_name: str,
    agent_name: str,
    num_runs_per_game: int,
    scorecard: Any = None,
):
    if not results_data:
        print("No results to report.")
        return
    game_stats, overall = calculate_stats(results_data)
    for line in _build_report_lines(game_stats, overall, results_data, agent_name, suite_name, num_runs_per_game, scorecard):
        print(line)


def save_summary_report(
    filepath: str,
    game_stats: dict[str, dict[str, Any]],
    overall: dict[str, Any],
    results_data: list[GameMetrics],
    agent_name: str,
    suite_name: str,
    num_runs_per_game: int,
    scorecard: Any = None,
):
    lines = _build_report_lines(game_stats, overall, results_data, agent_name, suite_name, num_runs_per_game, scorecard)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
