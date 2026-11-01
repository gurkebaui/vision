"""
Tests for 3-Tier Long-Range Smart-Zoom ROI math, coordinate unprojection,
low-light CLAHE contrast boost, HUD sub-ROI rendering, and CPU frame latency.
"""

from __future__ import annotations

import math
import time

import cv2
import mediapipe as mp
import numpy as np

from vision_control.config import SystemConfig
from vision_control.gestures import GestureEvaluation
from vision_control.hud import ModernHUDRenderer
from vision_control.tracker import (
    HandTrackingResult,
    LongRangeHandTracker,
    enhance_low_light_crop,
    make_square_roi,
    unproject_crop_landmarks,
)


def test_make_square_roi_clamps_cleanly_at_borders():
    roi = make_square_roi(cx_px=20, cy_px=710, side_px=240, frame_w=1280, frame_h=720)
    assert roi.x1 == 0
    assert roi.x2 == 240
    assert roi.y2 == 720
    assert roi.y1 == 480
    assert roi.width == roi.height == 240
    assert math.isclose(roi.zoom_factor(1280, 720), 3.0, abs_tol=0.01)


def test_unproject_crop_landmarks_exact_inverse():
    fw, fh = 1920, 1080
    roi = make_square_roi(cx_px=960, cy_px=420, side_px=280, frame_w=fw, frame_h=fh)
    # Suppose true hand point is at pixel (1010, 390)
    true_fx, true_fy = 1010.0 / fw, 390.0 / fh
    crop_lx = (1010.0 - roi.x1) / roi.width
    crop_ly = (390.0 - roi.y1) / roi.height

    unproj = unproject_crop_landmarks([(crop_lx, crop_ly, -0.05)], roi, fw, fh)
    assert math.isclose(unproj[0][0], true_fx, abs_tol=1e-6)
    assert math.isclose(unproj[0][1], true_fy, abs_tol=1e-6)


def test_low_light_crop_enhancement_boosts_dim_contrast():
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(4, 4))
    dim_patch = np.full((256, 256, 3), 35, dtype=np.uint8)
    dim_patch[80:180, 90:170] = 55  # Low-contrast silhouette in dim room
    boosted = enhance_low_light_crop(dim_patch, clahe)
    assert boosted.shape == dim_patch.shape
    assert float(boosted.mean()) > float(dim_patch.mean())
    assert float(boosted.std()) > float(dim_patch.std())


def test_tracker_cpu_latency_and_hud_fast_render():
    cfg = SystemConfig(performance_preset="ultra_light")
    tracker = LongRangeHandTracker(cfg)
    hud = ModernHUDRenderer(cfg)

    try:
        frame = np.zeros((540, 960, 3), dtype=np.uint8)
        # Warmup
        tracker.process_frame(frame)

        t0 = time.perf_counter()
        res = tracker.process_frame(frame)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        assert dt_ms < 60.0  # Fast on CPU even in virtualized container

        # Test HUD rendering with simulated tracked hand + zoom PiP
        fake_track = HandTrackingResult(
            detected=True,
            landmarks=[(0.5 + (i % 5) * 0.01, 0.45 - (i // 5) * 0.02, 0.0) for i in range(21)],
            roi=make_square_roi(480, 240, 200, 960, 540),
            roi_crop_bgr=np.zeros((180, 180, 3), dtype=np.uint8),
            zoom_factor=2.7,
            estimated_distance_m=3.1,
            palm_px=42.0,
            inference_ms=7.8,
        )
        fake_eval = GestureEvaluation(
            gesture="pointing_up",
            gesture_name="Point Up",
            gesture_icon="☝️",
            confidence=0.92,
            clutch_unlocked=True,
            clutch_remaining_s=3.4,
            dwell_progress=0.65,
            palm_center_norm=(0.5, 0.45),
        )
        hud.trigger_toast("Next Slide")
        rendered = hud.render(frame.copy(), fake_track, fake_eval, fps=30.0)
        assert rendered.shape == (540, 960, 3)

        cfg.compact_hud = True
        compact = hud.render(frame.copy(), fake_track, fake_eval, fps=30.0)
        assert compact.shape == (236, 420, 3)
    finally:
        tracker.close()


def test_tasks_vision_backend_when_mp_solutions_removed(monkeypatch):
    """
    Simulate MediaPipe 0.10.32+ / 1.2.0+ on Fedora / Python 3.13+ where `mp.solutions`
    was completely removed: LongRangeHandTracker must initialize via `mp.tasks.vision`
    and process frames cleanly with zero AttributeError.
    """
    monkeypatch.delattr(mp, "solutions", raising=False)
    assert not hasattr(mp, "solutions")

    cfg = SystemConfig(performance_preset="balanced", pose_assist_enabled=False)
    tracker = LongRangeHandTracker(cfg)
    try:
        assert tracker._backend_type == "tasks_vision"
        assert tracker._tasks_hand is not None
        assert tracker._tasks_hand_track is not None

        frame = np.zeros((540, 960, 3), dtype=np.uint8)
        for _ in range(4):
            res = tracker.process_frame(frame)
            assert res.inference_ms >= 0.0

        # Also exercise continuous Tier-1 VIDEO tracking crop path directly
        roi = make_square_roi(480, 270, 224, 960, 540)
        lms, handedness, score, crop = tracker._infer_hand_on_crop(
            frame, roi, target_size=224, is_continuous_track=True, enable_clahe=True
        )
        assert lms is None
        assert crop.shape == (224, 224, 3)
    finally:
        tracker.close()

