"""Anatomically plausible synthetic hands.

The whole recognition stack is testable without a camera or a model file
because every recogniser consumes plain ``(21, 3)`` arrays.  This module
builds those arrays by forward-kinematically bending each finger, so a
"curled" finger really does fold toward the palm the way a real one does --
which is what the curl/angle maths is designed to measure.
"""

from __future__ import annotations

import math
import time
from typing import Optional, Sequence

import numpy as np

from gesturekit import geometry as geo
from gesturekit.types import Handedness, HandState

# Resting geometry in a hand-local frame: wrist at origin, fingers along +Y,
# roughly the proportions of a real hand.
_MCP = {
    "thumb": (-0.62, 0.30),
    "index": (-0.34, 0.98),
    "middle": (-0.11, 1.04),
    "ring": (0.12, 0.99),
    "pinky": (0.34, 0.88),
}
_SEG = {           # (proximal, middle, distal) segment lengths
    "thumb": (0.42, 0.34, 0.28),
    "index": (0.50, 0.32, 0.24),
    "middle": (0.55, 0.35, 0.25),
    "ring": (0.50, 0.32, 0.24),
    "pinky": (0.40, 0.26, 0.21),
}
_SPLAY = {"thumb": -0.85, "index": -0.18, "middle": 0.0, "ring": 0.16, "pinky": 0.34}


def _chain(origin, base_angle, segs, curl, extra_bend=0.0):
    """Walk a finger outward, adding curl at each joint."""
    pts = []
    x, y = origin
    ang = base_angle
    bends = (curl * 1.15 + extra_bend, curl * 1.35, curl * 1.15)
    for seg, bend in zip(segs, bends):
        ang += bend
        x += seg * math.sin(ang)
        y += seg * math.cos(ang)
        pts.append((x, y))
    return pts


def synth_hand(
    curls: Sequence[float] = (0, 0, 0, 0, 0),
    center: tuple[float, float] = (0.5, 0.55),
    scale: float = 0.30,
    rotation_deg: float = 0.0,
    spread: float = 1.0,
    thumb_across: bool = False,
) -> np.ndarray:
    """Build a ``(21, 3)`` landmark array.

    Args:
        curls: per-finger curl, 0.0 = straight, 1.0 = fully folded.
        center: where the wrist sits in normalised image coordinates.
        scale: hand size in normalised units.
        rotation_deg: in-plane rotation, 0 = fingers up.
        spread: lateral finger splay multiplier.
        thumb_across: fold the thumb over the palm (as in a real fist).
    """
    curls = list(curls) + [0.0] * (5 - len(curls))
    pts = np.zeros((21, 3), dtype=np.float32)
    local: list[tuple[float, float]] = [(0.0, 0.0)]  # wrist

    for i, name in enumerate(geo.FINGER_NAMES):
        mx, my = _MCP[name]
        mx *= spread
        base = _SPLAY[name] * spread
        c = float(np.clip(curls[i], 0.0, 1.0))
        if name == "thumb":
            if thumb_across:
                # A tucked thumb both swings across the palm and flexes, so the
                # tip ends up near the middle-finger knuckles.
                base += 1.85
                joints = _chain((mx, my), base, _SEG[name], max(c, 0.55) * 0.9)
            else:
                base += 0.55
                joints = _chain((mx, my), base, _SEG[name], c * 0.75)
        else:
            joints = _chain((mx, my), base, _SEG[name], c)
        local.append((mx, my))
        local.extend(joints)

    arr = np.asarray(local, dtype=np.float32)  # (21, 2), y up
    arr[:, 1] *= -1.0                          # image coords: y grows downward

    if rotation_deg:
        a = math.radians(rotation_deg)
        rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]], dtype=np.float32)
        arr = arr @ rot.T

    pts[:, :2] = arr * scale + np.asarray(center, dtype=np.float32)
    return pts


# Named poses used across the test-suite.
POSES: dict[str, dict] = {
    "open_palm": dict(curls=(0.0, 0.0, 0.0, 0.0, 0.0), spread=1.25),
    "closed_fist": dict(curls=(0.95, 1.0, 1.0, 1.0, 1.0), thumb_across=True),
    "point_up": dict(curls=(0.9, 0.0, 1.0, 1.0, 1.0), thumb_across=True),
    "point_down": dict(curls=(0.9, 0.0, 1.0, 1.0, 1.0), thumb_across=True, rotation_deg=180),
    "point_left": dict(curls=(0.9, 0.0, 1.0, 1.0, 1.0), thumb_across=True, rotation_deg=-90),
    "point_right": dict(curls=(0.9, 0.0, 1.0, 1.0, 1.0), thumb_across=True, rotation_deg=90),
    "peace": dict(curls=(0.9, 0.0, 0.0, 1.0, 1.0), thumb_across=True, spread=1.45),
    "thumbs_up": dict(curls=(0.0, 1.0, 1.0, 1.0, 1.0)),
    "thumbs_down": dict(curls=(0.0, 1.0, 1.0, 1.0, 1.0), rotation_deg=180),
    "rock": dict(curls=(0.9, 0.0, 1.0, 1.0, 0.0), thumb_across=True),
    "call_me": dict(curls=(0.0, 1.0, 1.0, 1.0, 0.0)),
    "three": dict(curls=(0.9, 0.0, 0.0, 0.0, 1.0), thumb_across=True),
    "four": dict(curls=(0.95, 0.0, 0.0, 0.0, 0.0), thumb_across=True),
}


def pinch_hand(closed: bool = True, others_open: bool = True, **kw) -> np.ndarray:
    """Pinch / OK sign: index curled round until its tip meets the thumb tip."""
    curls = (0.0, 0.55, 0.0 if others_open else 1.0, 0.0 if others_open else 1.0,
             0.0 if others_open else 1.0)
    pts = synth_hand(curls=curls, **kw)
    if closed:
        pts[geo.INDEX_TIP, :2] = pts[geo.THUMB_TIP, :2] + np.array([0.004, 0.004], np.float32)
        pts[geo.INDEX_DIP, :2] = (pts[geo.INDEX_PIP, :2] + pts[geo.INDEX_TIP, :2]) / 2 + \
            np.array([0.012, -0.008], np.float32)
    return pts


def make_state(
    pts: np.ndarray,
    timestamp: Optional[float] = None,
    velocity: tuple[float, float] = (0.0, 0.0),
    handedness: Handedness = Handedness.RIGHT,
    label: Optional[str] = None,
    label_score: float = 0.0,
) -> HandState:
    """Wrap raw landmarks into a fully populated :class:`HandState`."""
    t = time.perf_counter() if timestamp is None else timestamp
    return HandState(
        points=pts,
        raw_points=pts,
        normalized=geo.normalize(pts),
        handedness=handedness,
        score=0.98,
        curls=geo.finger_curl(pts),
        extended=geo.fingers_extended(pts),
        center=geo.hand_center(pts),
        velocity=np.asarray(velocity, dtype=np.float32),
        palm_size=geo.palm_size(pts),
        roll=geo.hand_roll(pts),
        pinch=geo.pinch_distance(pts),
        timestamp=t,
        label=label,
        label_score=label_score,
    )


def pose(name: str, **overrides) -> np.ndarray:
    """Build one of the named :data:`POSES`."""
    if name == "pinch":
        return pinch_hand(others_open=False, **overrides)
    if name == "ok":
        return pinch_hand(others_open=True, **overrides)
    kw = dict(POSES[name])
    kw.update(overrides)
    return synth_hand(**kw)


def swipe_frames(
    direction: str = "right",
    n: int = 14,
    duration: float = 0.30,
    distance: float = 0.42,
    t0: float = 1000.0,
    curls: Sequence[float] = (0.2, 0.1, 0.1, 0.1, 0.1),
) -> list[HandState]:
    """A straight, fast hand translation -- i.e. a swipe."""
    vec = {
        "right": (1.0, 0.0),
        "left": (-1.0, 0.0),
        "up": (0.0, -1.0),
        "down": (0.0, 1.0),
    }[direction]
    out = []
    for i in range(n):
        f = i / max(n - 1, 1)
        cx = 0.5 + vec[0] * distance * (f - 0.5)
        cy = 0.5 + vec[1] * distance * (f - 0.5)
        pts = synth_hand(curls=curls, center=(cx, cy), scale=0.26)
        out.append(make_state(pts, timestamp=t0 + f * duration))
    return out


#: A relaxed hand at rest: fingers loosely curled, not a deliberate pose.
#: Using a flat open palm here would literally *be* the "open_palm" gesture,
#: so a "does nothing happen when I stand still?" test needs this instead.
#: These bend values measure out at ~0.5 curl, i.e. genuinely in between
#: "extended" and "folded".
RESTING = (0.80, 0.86, 0.86, 0.86, 0.86)


def jitter_frames(
    n: int = 30,
    amplitude: float = 0.006,
    t0: float = 1000.0,
    seed: int = 0,
    curls: Sequence[float] = RESTING,
) -> list[HandState]:
    """A held-still, relaxed hand with realistic landmark noise.

    Nothing here should ever trigger an action -- this is the "user is just
    standing there talking" case that v1 kept misfiring on.
    """
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        off = rng.normal(0.0, amplitude, size=2)
        pts = synth_hand(curls=curls, center=(0.5 + off[0], 0.5 + off[1]), scale=0.26)
        out.append(make_state(pts, timestamp=t0 + i * 0.033))
    return out


def circle_frames(
    clockwise: bool = True, n: int = 30, radius: float = 0.13, t0: float = 1000.0
) -> list[HandState]:
    out = []
    for i in range(n):
        a = 2 * math.pi * i / n * (1 if clockwise else -1)
        cx = 0.5 + radius * math.cos(a)
        cy = 0.5 + radius * math.sin(a)
        pts = synth_hand(curls=(0.9, 0.0, 1.0, 1.0, 1.0), center=(cx, cy), scale=0.24)
        out.append(make_state(pts, timestamp=t0 + i * 0.04))
    return out
