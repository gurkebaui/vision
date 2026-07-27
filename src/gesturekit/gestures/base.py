"""Recogniser interface and the shared rolling context."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional

import numpy as np

from ..types import GestureEvent, HandState


@dataclass
class RecognizerContext:
    """Rolling per-hand history shared by every recogniser.

    Keeping one history (instead of each recogniser holding its own deque)
    means the trajectory is computed once per frame and stays consistent
    across recognisers.
    """

    maxlen: int = 48
    states: Deque[HandState] = field(default_factory=lambda: deque(maxlen=48))

    def push(self, state: HandState) -> None:
        self.states.append(state)

    @property
    def current(self) -> Optional[HandState]:
        return self.states[-1] if self.states else None

    def trajectory(self, seconds: float = 0.6) -> np.ndarray:
        """Palm centres from the last ``seconds``, shape ``(n, 2)``."""
        if not self.states:
            return np.zeros((0, 2), dtype=np.float32)
        t_end = self.states[-1].timestamp
        pts = [s.center for s in self.states if t_end - s.timestamp <= seconds]
        return np.asarray(pts, dtype=np.float32) if pts else np.zeros((0, 2), dtype=np.float32)

    def timestamps(self, seconds: float = 0.6) -> np.ndarray:
        if not self.states:
            return np.zeros((0,), dtype=np.float64)
        t_end = self.states[-1].timestamp
        ts = [s.timestamp for s in self.states if t_end - s.timestamp <= seconds]
        return np.asarray(ts, dtype=np.float64)

    def clear(self) -> None:
        self.states.clear()


class Recognizer:
    """Base class: consume a context, optionally emit events."""

    name = "recognizer"

    def update(self, ctx: RecognizerContext) -> list[GestureEvent]:
        raise NotImplementedError

    def reset(self) -> None:
        pass
