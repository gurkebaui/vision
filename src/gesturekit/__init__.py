"""GestureKit -- hand-gesture control for presentations and the desktop.

Built on OpenCV and the modern MediaPipe Tasks API.

Quick start::

    from gesturekit import GestureApp, load_config

    app = GestureApp(load_config("presentation"))
    app.run()

Or from a shell::

    gesturekit run -p presentation
"""

from __future__ import annotations

__version__ = "2.0.0"

__all__ = [
    "__version__",
    "GestureApp",
    "AppConfig",
    "load_config",
    "list_profiles",
    "GestureEngine",
    "GestureEngineConfig",
    "HandTracker",
    "TrackerConfig",
    "Camera",
    "CameraConfig",
    "HandState",
    "GestureEvent",
    "Handedness",
    "geometry",
]

from . import geometry
from .types import GestureEvent, Handedness, HandState


def __getattr__(name: str):
    """Import the heavy pieces lazily.

    Keeps ``import gesturekit`` cheap (and possible without a camera present),
    which matters for the test-suite and for tooling that only needs the
    geometry helpers.
    """
    if name in ("GestureApp",):
        from .app import GestureApp

        return GestureApp
    if name in ("AppConfig", "load_config", "list_profiles"):
        from . import config

        return getattr(config, name)
    if name in ("GestureEngine", "GestureEngineConfig"):
        from .gestures.engine import GestureEngine, GestureEngineConfig

        return {"GestureEngine": GestureEngine, "GestureEngineConfig": GestureEngineConfig}[name]
    if name in ("HandTracker", "TrackerConfig"):
        from .tracker import HandTracker, TrackerConfig

        return {"HandTracker": HandTracker, "TrackerConfig": TrackerConfig}[name]
    if name in ("Camera", "CameraConfig"):
        from .camera import Camera, CameraConfig

        return {"Camera": Camera, "CameraConfig": CameraConfig}[name]
    raise AttributeError(f"module 'gesturekit' has no attribute '{name}'")
