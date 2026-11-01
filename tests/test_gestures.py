"""
Tests for Scale-Invariant 3D & Joint-Angle Gesture Engine across 0.5m – 5.0m distances,
Return-Stroke Swipe Suppression, and Presenter Clutch Gating.
"""

from __future__ import annotations

from typing import List, Tuple

import pytest

from vision_control.config import SystemConfig
from vision_control.gestures import ScaleInvariantGestureEngine


def build_synthetic_hand(
    pose: str,
    distance_m: float = 1.0,
    cx: float = 0.5,
    cy: float = 0.48,
    fw: int = 1280,
    fh: int = 720,
) -> List[Tuple[float, float, float]]:
    """
    Generate anatomically proportional 21 normalized landmarks at physical `distance_m`
    (from 0.5m close-up down to tiny 30px hands at 5.0m across an auditorium).
    """
    focal_px = 0.78 * fw
    palm_px = (0.088 * focal_px) / distance_m
    sx = palm_px / fw
    sy = palm_px / fh

    base_mcps = [
        (-0.38, -0.35),  # 5 index
        (-0.12, -0.44),  # 9 middle
        (0.14, -0.40),   # 13 ring
        (0.38, -0.30),   # 17 pinky
    ]
    curl_table = {
        "open_palm": [0, 0, 0, 0],
        "closed_fist": [1, 1, 1, 1],
        "pointing_up": [0, 1, 1, 1],
        "pointing_down": [0, 1, 1, 1],
        "peace_sign": [0, 0, 1, 1],
        "thumbs_up": [1, 1, 1, 1],
        "thumbs_down": [1, 1, 1, 1],
        "pinch": [0.35, 0, 0, 0],
        "shaka": [1, 1, 1, 0],
        "three_fingers": [0, 0, 0, 1],
    }
    curls = curl_table[pose]
    flip_down = pose in ("pointing_down", "thumbs_down")

    pts_pu: List[Tuple[float, float, float]] = [(0.0, 0.0, 0.0)] * 21
    pts_pu[0] = (0.0, -0.52 if flip_down else 0.52, 0.0)

    if pose == "thumbs_up":
        pts_pu[1] = (-0.24, 0.25, 0.0)
        pts_pu[2] = (-0.30, -0.02, 0.0)
        pts_pu[3] = (-0.30, -0.34, 0.0)
        pts_pu[4] = (-0.30, -0.66, 0.0)
    elif pose == "thumbs_down":
        pts_pu[1] = (-0.24, -0.25, 0.0)
        pts_pu[2] = (-0.30, 0.02, 0.0)
        pts_pu[3] = (-0.30, 0.34, 0.0)
        pts_pu[4] = (-0.30, 0.66, 0.0)
    elif pose == "pinch":
        pts_pu[1] = (-0.28, 0.30, 0.0)
        pts_pu[2] = (-0.42, 0.06, 0.0)
        pts_pu[3] = (-0.44, -0.24, 0.0)
        pts_pu[4] = (-0.36, -0.52, 0.0)
    elif pose == "closed_fist":
        pts_pu[1] = (-0.26, 0.28, 0.0)
        pts_pu[2] = (-0.28, 0.06, 0.0)
        pts_pu[3] = (-0.14, -0.04, 0.0)
        pts_pu[4] = (0.02, -0.02, 0.0)
    else:
        s = -1.0 if flip_down else 1.0
        pts_pu[1] = (-0.32, 0.30 * s, 0.0)
        pts_pu[2] = (-0.54, 0.08 * s, 0.0)
        pts_pu[3] = (-0.72, -0.14 * s, 0.0)
        pts_pu[4] = (-0.88, -0.34 * s, 0.0)

    lengths = [1.0, 1.08, 1.0, 0.82]
    spreads = (
        [-0.22, 0.20, 0.08, 0.14]
        if pose == "peace_sign"
        else [-0.10, -0.02, 0.06, 0.14]
    )

    for f in range(4):
        b = 5 + f * 4
        mx, my_raw = base_mcps[f]
        my = -my_raw if flip_down else my_raw
        pts_pu[b] = (mx, my, 0.0)
        dir_sign = 1.0 if flip_down else -1.0

        if pose == "pinch" and f == 0:
            pts_pu[6] = (-0.42, -0.58, 0.0)
            pts_pu[7] = (-0.40, -0.58, 0.0)
            pts_pu[8] = (-0.35, -0.53, 0.0)
        elif curls[f] < 0.5:
            sp = spreads[f]
            ln = lengths[f]
            pts_pu[b + 1] = (mx + sp * 0.35, my + dir_sign * 0.34 * ln, 0.0)
            pts_pu[b + 2] = (mx + sp * 0.70, my + dir_sign * 0.64 * ln, 0.0)
            pts_pu[b + 3] = (mx + sp * 1.05, my + dir_sign * 0.92 * ln, 0.0)
        else:
            pts_pu[b + 1] = (mx, my + dir_sign * 0.18, 0.04)
            pts_pu[b + 2] = (mx * 0.88, my - dir_sign * 0.06, 0.06)
            pts_pu[b + 3] = (mx * 0.82, my - dir_sign * 0.22, 0.04)

    return [(cx + ux * sx, cy + uy * sy, uz * sx) for (ux, uy, uz) in pts_pu]


@pytest.mark.parametrize(
    "pose",
    [
        "open_palm",
        "closed_fist",
        "pointing_up",
        "pointing_down",
        "peace_sign",
        "thumbs_up",
        "thumbs_down",
        "pinch",
        "shaka",
        "three_fingers",
    ],
)
@pytest.mark.parametrize("distance_m", [0.5, 1.2, 2.8, 4.8])
def test_scale_invariance_across_0_5m_to_4_8m(pose: str, distance_m: float):
    """All 10 static poses must classify identically at 0.5m, 1.2m, 2.8m, and 4.8m!"""
    engine = ScaleInvariantGestureEngine()
    lms = build_synthetic_hand(pose, distance_m=distance_m)
    detected_pose, conf, _, _, _ = engine.classify_static_pose(lms, aspect_ratio=1280 / 720)
    assert detected_pose == pose
    assert conf >= 0.78


def test_dynamic_swipe_and_return_stroke_suppression():
    """
    A flick right at 3.5m must fire 'swipe_right' -> 'next_slide', and immediately bringing
    the hand back left to center (recovery stroke) MUST NOT misfire 'swipe_left'!
    """
    cfg = SystemConfig(activation_mode="dwell", active_profile="presentation")
    engine = ScaleInvariantGestureEngine(cfg)

    triggered = []
    t = 1.0
    # 1. Flick right over 6 frames (at 3.5m distance!)
    for step in range(7):
        cx = 0.42 + step * 0.018  # ~2.5 Palm Units at 3.5m
        lms = build_synthetic_hand("open_palm", distance_m=3.5, cx=cx, cy=0.45)
        res = engine.update(lms, now=t, aspect_ratio=1280 / 720)
        if res.triggered_gesture:
            triggered.append((res.triggered_gesture, res.triggered_action))
        t += 0.033

    assert ("swipe_right", "next_slide") in triggered

    # 2. Immediately bring hand back left from 0.54 -> 0.42 (return stroke within 0.5s)
    return_triggers = []
    t += 0.45  # past basic refractory, still inside return-stroke suppression window (0.78s)
    for step in range(7):
        cx = 0.54 - step * 0.018
        lms = build_synthetic_hand("open_palm", distance_m=3.5, cx=cx, cy=0.45)
        res = engine.update(lms, now=t, aspect_ratio=1280 / 720)
        if res.triggered_gesture and res.triggered_gesture.startswith("swipe_"):
            return_triggers.append(res.triggered_gesture)
        t += 0.030

    assert "swipe_left" not in return_triggers


def test_clutch_unlock_and_intentional_lock_workflow():
    cfg = SystemConfig(
        activation_mode="clutch",
        active_profile="presentation",
        clutch_hold_seconds=0.35,
        clutch_window_seconds=4.0,
        gesture_cooldown_seconds=0.5,
    )
    engine = ScaleInvariantGestureEngine(cfg)

    t = 10.0
    # 1. Pointing up while LOCKED must NOT trigger next_slide
    for _ in range(15):
        lms = build_synthetic_hand("pointing_up", distance_m=2.5)
        res = engine.update(lms, now=t)
        assert not res.clutch_unlocked
        assert res.triggered_action is None
        t += 0.033

    # 2. Hold Open Palm for 0.4s -> unlocks Clutch WITHOUT immediately firing play_pause
    unlocked_seen = False
    for _ in range(15):
        lms = build_synthetic_hand("open_palm", distance_m=2.5)
        res = engine.update(lms, now=t)
        if res.clutch_unlocked:
            unlocked_seen = True
        assert res.triggered_action is None
        t += 0.033
    assert unlocked_seen

    # 3. Now perform Point Up -> triggers next_slide!
    fired_actions = []
    for _ in range(12):
        lms = build_synthetic_hand("pointing_up", distance_m=2.5)
        res = engine.update(lms, now=t)
        if res.triggered_action:
            fired_actions.append(res.triggered_action)
        t += 0.033
    assert "next_slide" in fired_actions

    # 4. Perform Closed Fist -> immediately locks the clutch window!
    t += 0.55
    for _ in range(12):
        lms = build_synthetic_hand("closed_fist", distance_m=2.5)
        res = engine.update(lms, now=t)
        t += 0.033
    assert not engine._clutch_unlocked
