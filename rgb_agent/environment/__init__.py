"""Environment package."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseEnv(ABC):
    @abstractmethod
    def reset(self) -> tuple[dict, dict]:
        pass

    @abstractmethod
    def step(self, action: Any) -> tuple[Any, float, bool, dict]:
        pass

    def close(self):
        return


from rgb_agent.environment.arcagi3 import ArcAgi3Env

__all__ = ["BaseEnv", "ArcAgi3Env"]
