"""Hand tracking on the MediaPipe Tasks API.

v1 used ``mp.solutions.hands``, the legacy API, which **no longer exists** in
MediaPipe 1.0 -- importing it raises ``ModuleNotFoundError``.  This module
targets ``mediapipe.tasks.python.vision`` instead and adds:

* LIVE_STREAM running mode, so inference happens on MediaPipe's own thread and
  never blocks the capture/render loop;
* One Euro filtering per hand, keyed by handedness so two hands keep separate
  filter state;
* velocity in palm-widths/second, which is what the dynamic recognisers need;
* optional use of Google's canned gesture classifier as an extra evidence
  channel alongside our own geometric recognisers.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from . import geometry as geo
from .filters import LandmarkFilter, OneEuroFilter
from .types import Handedness, HandState


@dataclass
class TrackerConfig:
    num_hands: int = 2
    min_detection_confidence: float = 0.6
    min_presence_confidence: float = 0.6
    min_tracking_confidence: float = 0.6
    use_canned_gestures: bool = True
    delegate: str = "cpu"            # "cpu" | "gpu"
    filter_min_cutoff: float = 1.2
    filter_beta: float = 0.06
    running_mode: str = "live_stream"  # live_stream | video


class TrackerError(RuntimeError):
    pass


class HandTracker:
    """Thin, fast wrapper around HandLandmarker / GestureRecognizer."""

    def __init__(self, model_path: Path, config: Optional[TrackerConfig] = None):
        self.config = config or TrackerConfig()
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise TrackerError(f"Model bundle not found: {self.model_path}")

        self._lock = threading.Lock()
        self._latest = None          # raw MediaPipe result
        self._latest_stamp = 0.0
        self._inference_ms = 0.0
        self._sent_at: dict[int, float] = {}
        self._last_ts_ms = -1

        self._pos_filters: dict[str, LandmarkFilter] = {}
        self._vel_filters: dict[str, OneEuroFilter] = {}
        self._prev_center: dict[str, tuple[np.ndarray, float]] = {}

        self._task = None
        self._is_recognizer = False
        self._build()

    # -- construction ------------------------------------------------------
    def _build(self) -> None:
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions, vision
        except ImportError as exc:  # pragma: no cover - import guard
            raise TrackerError(
                "MediaPipe is not installed. Install it with:\n"
                "    pip install 'mediapipe>=0.10.9'"
            ) from exc

        self._mp = mp
        cfg = self.config

        delegate = None
        if cfg.delegate.lower() == "gpu":
            delegate = BaseOptions.Delegate.GPU
        base = BaseOptions(model_asset_path=str(self.model_path), delegate=delegate)

        mode = (
            vision.RunningMode.LIVE_STREAM
            if cfg.running_mode == "live_stream"
            else vision.RunningMode.VIDEO
        )
        self._mode = mode
        self._live = mode == vision.RunningMode.LIVE_STREAM

        # A gesture_recognizer bundle also contains the landmarker, so we get
        # canned labels for free when the user supplies that bundle.
        wants_recognizer = cfg.use_canned_gestures and "gesture" in self.model_path.name.lower()

        common = dict(
            base_options=base,
            running_mode=mode,
            num_hands=cfg.num_hands,
            min_hand_detection_confidence=cfg.min_detection_confidence,
            min_hand_presence_confidence=cfg.min_presence_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence,
        )
        if self._live:
            common["result_callback"] = self._on_result

        try:
            if wants_recognizer:
                self._task = vision.GestureRecognizer.create_from_options(
                    vision.GestureRecognizerOptions(**common)
                )
                self._is_recognizer = True
            else:
                self._task = vision.HandLandmarker.create_from_options(
                    vision.HandLandmarkerOptions(**common)
                )
        except Exception as exc:
            if delegate is not None:  # GPU delegate unavailable -> fall back
                self.config.delegate = "cpu"
                self._build()
                return
            raise TrackerError(f"Failed to create MediaPipe task: {exc}") from exc

    # -- async plumbing ----------------------------------------------------
    def _on_result(self, result, output_image, timestamp_ms: int) -> None:
        sent = self._sent_at.pop(timestamp_ms, None)
        with self._lock:
            self._latest = result
            self._latest_stamp = time.perf_counter()
            if sent is not None:
                self._inference_ms = (self._latest_stamp - sent) * 1000.0
        # Drop bookkeeping for frames the callback skipped.
        if len(self._sent_at) > 16:
            for k in sorted(self._sent_at)[:-8]:
                self._sent_at.pop(k, None)

    def submit(self, rgb: np.ndarray, timestamp: float):
        """Feed a frame. Live-stream mode returns ``None`` (async)."""
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int(timestamp * 1000)
        if ts_ms <= self._last_ts_ms:   # MediaPipe demands strictly increasing
            ts_ms = self._last_ts_ms + 1
        self._last_ts_ms = ts_ms

        if self._live:
            self._sent_at[ts_ms] = time.perf_counter()
            if self._is_recognizer:
                self._task.recognize_async(mp_image, ts_ms)
            else:
                self._task.detect_async(mp_image, ts_ms)
            return None

        t0 = time.perf_counter()
        result = (
            self._task.recognize_for_video(mp_image, ts_ms)
            if self._is_recognizer
            else self._task.detect_for_video(mp_image, ts_ms)
        )
        self._inference_ms = (time.perf_counter() - t0) * 1000.0
        with self._lock:
            self._latest = result
            self._latest_stamp = time.perf_counter()
        return result

    def latest(self):
        with self._lock:
            return self._latest, self._latest_stamp

    @property
    def inference_ms(self) -> float:
        with self._lock:
            return self._inference_ms

    # -- conversion --------------------------------------------------------
    def to_states(self, result, timestamp: float) -> list[HandState]:
        """Turn a raw MediaPipe result into filtered :class:`HandState` objects."""
        if result is None or not getattr(result, "hand_landmarks", None):
            return []

        states: list[HandState] = []
        gestures = getattr(result, "gestures", None) or []

        for i, lms in enumerate(result.hand_landmarks):
            raw = geo.to_array(lms)
            if raw.shape[0] < 21:
                continue

            hand_label, score = Handedness.UNKNOWN, 0.0
            handedness = getattr(result, "handedness", None)
            if handedness and i < len(handedness) and handedness[i]:
                cat = handedness[i][0]
                name = (cat.category_name or "").capitalize()
                hand_label = (
                    Handedness.LEFT
                    if name.startswith("L")
                    else Handedness.RIGHT if name.startswith("R") else Handedness.UNKNOWN
                )
                score = float(cat.score)

            key = f"{hand_label.value}:{i if hand_label is Handedness.UNKNOWN else ''}"
            pos_f = self._pos_filters.get(key)
            if pos_f is None:
                pos_f = LandmarkFilter(self.config.filter_min_cutoff, self.config.filter_beta)
                self._pos_filters[key] = pos_f
                self._vel_filters[key] = OneEuroFilter(min_cutoff=1.5, beta=0.02)

            pts = pos_f(raw, timestamp)
            center = geo.hand_center(pts)
            psize = geo.palm_size(pts)

            prev = self._prev_center.get(key)
            if prev is not None:
                pc, pt = prev
                dt = max(timestamp - pt, 1e-3)
                vel = (center - pc) / (psize * dt)   # palm widths / second
                vel = self._vel_filters[key](vel.astype(np.float32), timestamp)
            else:
                vel = np.zeros(2, dtype=np.float32)
            self._prev_center[key] = (center.copy(), timestamp)

            label, label_score = None, 0.0
            if i < len(gestures) and gestures[i]:
                top = gestures[i][0]
                label = top.category_name
                label_score = float(top.score)

            curls = geo.finger_curl(pts)
            states.append(
                HandState(
                    points=pts,
                    raw_points=raw,
                    normalized=geo.normalize(pts),
                    handedness=hand_label,
                    score=score,
                    curls=curls,
                    extended=geo.fingers_extended(pts),
                    center=center,
                    velocity=np.asarray(vel, dtype=np.float32),
                    palm_size=psize,
                    roll=geo.hand_roll(pts),
                    pinch=geo.pinch_distance(pts),
                    timestamp=timestamp,
                    label=label,
                    label_score=label_score,
                )
            )
        return states

    def reset(self) -> None:
        for f in self._pos_filters.values():
            f.reset()
        for v in self._vel_filters.values():
            v.reset()
        self._prev_center.clear()

    def close(self) -> None:
        if self._task is not None:
            try:
                self._task.close()
            except Exception:
                pass
            self._task = None

    def __enter__(self) -> HandTracker:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
