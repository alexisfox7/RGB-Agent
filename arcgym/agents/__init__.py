"""ArcGym agent package."""

from .rgb_agent import RGBAgent, QueueExhausted, ActionQueue

AVAILABLE_AGENTS = {
    "rgb_agent": RGBAgent,
}

__all__ = [
    "RGBAgent",
    "QueueExhausted",
    "ActionQueue",
    "AVAILABLE_AGENTS",
]
