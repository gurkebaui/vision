"""Gesture recognisers.

Three families, deliberately kept apart because they need different logic:

* :mod:`static`     -- hand *shapes* (fist, palm, peace, thumbs up, ...)
* :mod:`dynamic`    -- hand *motions* (swipes, circles) over a trajectory
* :mod:`continuous` -- streaming states (pinch-drag, cursor, scroll, zoom)
"""

from .base import Recognizer, RecognizerContext
from .continuous import ContinuousRecognizer
from .dynamic import DynamicRecognizer
from .engine import GestureEngine, GestureEngineConfig
from .static import TEMPLATES, StaticRecognizer

__all__ = [
    "Recognizer",
    "RecognizerContext",
    "StaticRecognizer",
    "DynamicRecognizer",
    "ContinuousRecognizer",
    "GestureEngine",
    "GestureEngineConfig",
    "TEMPLATES",
]
