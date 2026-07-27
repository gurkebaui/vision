"""Shared dataclasses passed between the pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class Handedness(str, Enum):
    LEFT = "Left"
    RIGHT = "Right"
    UNKNOWN = "Unknown"


@dataclass(frozen=True)
class HandState:
    """Everything known about one hand in one frame.

    Geometry is computed once here and shared by every recogniser, rather than
    each recogniser recomputing curls and distances from raw landmarks.
    """

    points: np.ndarray                 # (21, 3) filtered, image-normalised
    raw_points: np.ndarray             # (21, 3) unfiltered
    normalized: np.ndarray             # (21, 3) translation/scale/rotation-free
    handedness: Handedness
    score: float
    curls: np.ndarray                  # (5,) 0=straight .. 1=folded
    extended: np.ndarray               # (5,) bool
    center: np.ndarray                 # (2,) palm centroid
    velocity: np.ndarray               # (2,) palm widths per second
    palm_size: float
    roll: float                        # degrees, 0 = fingers up
    pinch: float                       # thumb-index distance, palm widths
    timestamp: float
    label: Optional[str] = None        # canned MediaPipe label, if available
    label_score: float = 0.0

    @property
    def n_extended(self) -> int:
        return int(self.extended.sum())

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.velocity))


@dataclass(frozen=True)
class GestureEvent:
    """A recognised gesture ready to be mapped to an action."""

    name: str
    confidence: float
    hand: Handedness
    timestamp: float
    kind: str = "static"               # static | dynamic | continuous
    data: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.name}({self.confidence:.2f})"


@dataclass
class FrameResult:
    """Per-frame output handed to the renderer and the action router."""

    hands: list = field(default_factory=list)          # list[HandState]
    events: list = field(default_factory=list)         # list[GestureEvent]
    fps: float = 0.0
    latency_ms: float = 0.0
    inference_ms: float = 0.0
    dropped: int = 0
