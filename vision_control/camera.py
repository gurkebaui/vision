"""
Non-blocking Threaded Camera Capture with USB MJPG high-FPS unlocking.

Decouples OS webcam driver I/O (V4L2 / DirectShow / AVFoundation) from the
computer-vision inference loop so CPU laptops never stall waiting on cap.read().
"""

from __future__ import annotations

import platform
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np


class ThreadedCamera:
    """Asynchronous camera reader that always surfaces the freshest frame."""

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        mirror: bool = True,
    ) -> None:
        self.camera_index = int(camera_index)
        self.requested_width = int(width)
        self.requested_height = int(height)
        self.requested_fps = int(fps)
        self.mirror = bool(mirror)

        self.cap: Optional[cv2.VideoCapture] = None
        self.actual_width: int = width
        self.actual_height: int = height
        self._running = False
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

        self._latest_frame: Optional[np.ndarray] = None
        self._latest_frame_id: int = 0
        self._latest_timestamp: float = 0.0
        self._capture_fps: float = 0.0

    def start(self) -> bool:
        if self._running:
            return True

        sys_name = platform.system()
        if sys_name == "Windows" and hasattr(cv2, "CAP_DSHOW"):
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(self.camera_index)
        elif sys_name == "Linux" and hasattr(cv2, "CAP_V4L2"):
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_V4L2)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(self.camera_index)
        else:
            self.cap = cv2.VideoCapture(self.camera_index)

        if self.cap is None or not self.cap.isOpened():
            return False

        # Request MJPG before setting resolution so USB 2.0 laptop cameras achieve 30-60 FPS at 720p/1080p
        if hasattr(cv2, "VideoWriter_fourcc"):
            self.cap.set(
                cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG")
            )
        if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.requested_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.requested_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.requested_fps)

        ret, first_frame = self.cap.read()
        if not ret or first_frame is None:
            self.cap.release()
            self.cap = None
            return False

        if self.mirror:
            first_frame = cv2.flip(first_frame, 1)

        self.actual_height, self.actual_width = first_frame.shape[:2]
        now = time.perf_counter()
        with self._lock:
            self._latest_frame = first_frame
            self._latest_frame_id = 1
            self._latest_timestamp = now

        self._running = True
        self._thread = threading.Thread(
            target=self._reader_loop, name="VisionCameraThread", daemon=True
        )
        self._thread.start()
        return True

    def _reader_loop(self) -> None:
        prev_t = time.perf_counter()
        ema_fps = float(self.requested_fps)
        while self._running and self.cap is not None:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                time.sleep(0.005)
                continue

            if self.mirror:
                frame = cv2.flip(frame, 1)

            now = time.perf_counter()
            dt = max(now - prev_t, 1e-4)
            prev_t = now
            inst_fps = min(120.0, 1.0 / dt)
            ema_fps = 0.9 * ema_fps + 0.1 * inst_fps

            with self._lock:
                self._latest_frame = frame
                self._latest_frame_id += 1
                self._latest_timestamp = now
                self._capture_fps = ema_fps

    def read_latest(
        self, last_seen_id: int = 0
    ) -> Tuple[bool, Optional[np.ndarray], int, float]:
        """Return (is_new, frame, frame_id, timestamp)."""
        with self._lock:
            if self._latest_frame is None:
                return False, None, 0, 0.0
            is_new = self._latest_frame_id != last_seen_id
            return (
                is_new,
                self._latest_frame,
                self._latest_frame_id,
                self._latest_timestamp,
            )

    @property
    def capture_fps(self) -> float:
        with self._lock:
            return self._capture_fps

    def reconfigure(self, width: int, height: int, mirror: bool) -> None:
        self.mirror = bool(mirror)
        if width != self.requested_width or height != self.requested_height:
            self.requested_width = int(width)
            self.requested_height = int(height)
            if self._running:
                self.stop()
                self.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
