"""Threaded camera capture with backend auto-selection.

Why a thread: ``VideoCapture.read()`` blocks until the driver hands over the
next frame.  Doing that on the main loop -- as v1 did -- means capture latency
and inference latency add up.  Here the grabber thread always holds the newest
frame and stale frames are dropped, so the pipeline reacts to *now* instead of
working through a backlog.  Under load you lose frames, never responsiveness.
"""

from __future__ import annotations

import platform
import threading
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class CameraConfig:
    index: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    fourcc: str = "MJPG"      # MJPG lets most webcams hit 30/60 fps at 720p+
    flip: bool = True         # mirror view feels natural to the user
    backend: str = "auto"
    warmup_frames: int = 5


def _backend_flag(name: str) -> int:
    name = (name or "auto").lower()
    explicit = {
        "v4l2": getattr(cv2, "CAP_V4L2", 0),
        "msmf": getattr(cv2, "CAP_MSMF", 0),
        "dshow": getattr(cv2, "CAP_DSHOW", 0),
        "avfoundation": getattr(cv2, "CAP_AVFOUNDATION", 0),
        "gstreamer": getattr(cv2, "CAP_GSTREAMER", 0),
        "any": cv2.CAP_ANY,
    }
    if name in explicit:
        return explicit[name]
    system = platform.system()
    if system == "Linux":
        return getattr(cv2, "CAP_V4L2", cv2.CAP_ANY)
    if system == "Windows":
        return getattr(cv2, "CAP_MSMF", cv2.CAP_ANY)
    if system == "Darwin":
        return getattr(cv2, "CAP_AVFOUNDATION", cv2.CAP_ANY)
    return cv2.CAP_ANY


class CameraError(RuntimeError):
    pass


class Camera:
    """Always-freshest-frame webcam reader."""

    def __init__(self, config: Optional[CameraConfig] = None):
        self.config = config or CameraConfig()
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray] = None
        self._stamp: float = 0.0
        self._seq: int = 0
        self._last_read_seq: int = -1
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = threading.Event()
        self.dropped = 0

    # -- lifecycle ---------------------------------------------------------
    def open(self) -> Camera:
        cfg = self.config
        cap = cv2.VideoCapture(cfg.index, _backend_flag(cfg.backend))
        if not cap.isOpened():  # retry with whatever backend OpenCV picks
            cap.release()
            cap = cv2.VideoCapture(cfg.index, cv2.CAP_ANY)
        if not cap.isOpened():
            raise CameraError(
                f"Cannot open camera index {cfg.index}.\n"
                f"  - Is another application using the webcam?\n"
                f"  - Try a different index: --camera 1\n"
                f"  - List devices with: gesturekit devices"
            )

        if cfg.fourcc:
            try:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cfg.fourcc))
            except Exception:
                pass
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
        cap.set(cv2.CAP_PROP_FPS, cfg.fps)
        # Shallow driver queue: we want the newest frame, not a smooth backlog.
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        self._cap = cap
        for _ in range(cfg.warmup_frames):  # let auto-exposure settle
            cap.read()

        self._running.set()
        self._thread = threading.Thread(target=self._loop, name="gesturekit-camera", daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        assert self._cap is not None
        failures = 0
        while self._running.is_set():
            ok, frame = self._cap.read()
            if not ok or frame is None:
                failures += 1
                if failures > 60:
                    self._running.clear()
                    break
                time.sleep(0.005)
                continue
            failures = 0
            if self.config.flip:
                frame = cv2.flip(frame, 1)
            with self._lock:
                if self._seq != self._last_read_seq:
                    self.dropped += 1  # previous frame was never consumed
                self._frame = frame
                self._stamp = time.perf_counter()
                self._seq += 1

    def read(self) -> tuple[bool, Optional[np.ndarray], float]:
        """Return ``(is_new, frame, capture_timestamp)``."""
        with self._lock:
            if self._frame is None:
                return False, None, 0.0
            is_new = self._seq != self._last_read_seq
            self._last_read_seq = self._seq
            return is_new, self._frame, self._stamp

    @property
    def is_open(self) -> bool:
        return self._running.is_set() and self._cap is not None

    @property
    def resolution(self) -> tuple[int, int]:
        if self._cap is None:
            return (self.config.width, self.config.height)
        return (
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    def release(self) -> None:
        self._running.clear()
        if self._thread is not None:
            self._thread.join(timeout=1.5)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> Camera:
        return self.open()

    def __exit__(self, *exc) -> None:
        self.release()


def list_devices(max_index: int = 8) -> list[dict]:
    """Probe camera indices and report the ones that deliver a frame.

    Probing missing indices makes OpenCV log a wall of V4L2 warnings, so the
    internal log level is muted for the duration of the scan.
    """
    found = []
    try:
        previous = cv2.utils.logging.getLogLevel()
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
    except Exception:
        previous = None

    for i in range(max_index):
        cap = cv2.VideoCapture(i, _backend_flag("auto"))
        try:
            if cap.isOpened():
                ok, frame = cap.read()
                if ok and frame is not None:
                    found.append(
                        {
                            "index": i,
                            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                            "fps": round(float(cap.get(cv2.CAP_PROP_FPS)), 1),
                        }
                    )
        except Exception:
            pass
        finally:
            cap.release()

    if previous is not None:
        try:
            cv2.utils.logging.setLogLevel(previous)
        except Exception:
            pass
    return found
