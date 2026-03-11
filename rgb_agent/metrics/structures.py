from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional
import time


class Status(str, Enum):
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    COMPLETED_RUN = "COMPLETED_RUN"
    TIMEOUT = "TIMEOUT"
    ERROR = "ERROR"
    GAME_OVER = "GAME_OVER"
    QUEUE_EXHAUSTED = "QUEUE_EXHAUSTED"


@dataclass
class AttemptMetrics:
    attempt_number: int
    actions: int = 0
    duration_seconds: float = 0.0
    state_changes: int = 0
    game_overs: int = 0
    status: Status = Status.IN_PROGRESS


@dataclass
class LevelMetrics:
    level_number: int
    attempts: List[AttemptMetrics] = field(default_factory=list)
    status: Status = Status.IN_PROGRESS

    @property
    def total_actions(self) -> int:
        return sum(a.actions for a in self.attempts)

    @property
    def total_game_overs(self) -> int:
        return sum(a.game_overs for a in self.attempts)

    @property
    def total_state_changes(self) -> int:
        return sum(a.state_changes for a in self.attempts)

    @property
    def actions_in_successful_attempt(self) -> Optional[int]:
        if self.status == Status.COMPLETED and self.attempts:
            return self.attempts[-1].actions
        return None

    @property
    def state_change_percentage(self) -> float:
        total = self.total_actions
        return (self.total_state_changes / total * 100.0) if total else 0.0


@dataclass
class GameMetrics:
    game_id: str
    agent_name: str
    run_index: int = 1
    guid: Optional[str] = None

    run_total_actions: int = 0
    final_score: int = 0
    highest_level_reached: int = 1
    status: Status = Status.PENDING
    error_message: Optional[str] = None

    level_metrics: Dict[int, LevelMetrics] = field(default_factory=dict)

    start_time: float = field(default_factory=time.time)
    end_time: float = field(default_factory=time.time)
    run_duration_seconds: float = 0.0

    replay_url: Optional[str] = None

    total_state_changes_across_run: int = 0
    total_game_overs_across_run: int = 0
    duplicate_actions: int = 0
