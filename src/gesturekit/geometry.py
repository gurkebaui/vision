"""Vectorised hand geometry.

Everything downstream of the landmarker works on a single ``(21, 3)`` float32
array instead of a list of protobuf-ish landmark objects.  That one change is
responsible for most of the speed-up over the v1 code, which called
``landmarks[i].x`` hundreds of times per frame and re-derived the same
quantities in three different places.

The functions here are pure and NumPy-only, so they are cheap to unit test
without a camera, a model file, or MediaPipe installed.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Landmark indices (MediaPipe hand topology, 21 points)
# ---------------------------------------------------------------------------
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")

#: (mcp, pip, dip, tip) per finger, thumb first.
FINGER_CHAINS = np.array(
    [
        [THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP],
        [INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP],
        [MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP],
        [RING_MCP, RING_PIP, RING_DIP, RING_TIP],
        [PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP],
    ],
    dtype=np.int32,
)

TIPS = FINGER_CHAINS[:, 3]
PIPS = FINGER_CHAINS[:, 1]
MCPS = FINGER_CHAINS[:, 0]

#: Bones drawn by the HUD renderer.
HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


def to_array(landmarks: Iterable) -> np.ndarray:
    """Convert MediaPipe landmarks to a contiguous ``(21, 3)`` float32 array.

    Accepts anything with ``.x/.y/.z`` attributes, plain ``(x, y, z)`` tuples,
    or an array-like that is already the right shape.
    """
    if isinstance(landmarks, np.ndarray):
        return np.ascontiguousarray(landmarks, dtype=np.float32)

    seq: Sequence = list(landmarks)
    if not seq:
        return np.zeros((0, 3), dtype=np.float32)

    first = seq[0]
    if hasattr(first, "x"):
        return np.array([(p.x, p.y, getattr(p, "z", 0.0)) for p in seq], dtype=np.float32)
    return np.array(seq, dtype=np.float32)


def correct_aspect(pts: np.ndarray, aspect: float) -> np.ndarray:
    """Undo the non-square scaling in MediaPipe's normalised coordinates.

    MediaPipe divides x by the frame *width* and y by the frame *height*, so on
    a 16:9 camera one unit of x is 1.78x longer than one unit of y.  Any angle
    or distance computed on the raw values is therefore distorted: on a real
    webcam a closed fist can read as a thumbs-down, and finger splay is
    overstated by ~78%.

    Rescaling x into the same physical units as y makes all downstream geometry
    resolution- and aspect-independent.  Kept as a separate step so callers can
    pass already-corrected points (and so tests can use isotropic hands).
    """
    if pts.shape[0] == 0 or abs(aspect - 1.0) < 1e-6:
        return pts
    out = pts.copy()
    out[:, 0] *= aspect
    out[:, 2] *= aspect      # z shares the x normalisation in MediaPipe
    return out


def palm_size(pts: np.ndarray) -> float:
    """Scale reference: mean wrist->MCP distance in XY.

    Used to make every threshold resolution- and distance-independent, so a
    pinch means the same thing whether the hand is 40 cm or 2 m from the lens.
    """
    if pts.shape[0] < 21:
        return 1e-6
    d = np.linalg.norm(pts[MCPS, :2] - pts[WRIST, :2], axis=1)
    return max(float(d.mean()), 1e-6)


def hand_center(pts: np.ndarray) -> np.ndarray:
    """Stable palm centroid (wrist + the four MCP knuckles).

    v1 averaged all 21 points, which meant simply curling the fingers moved the
    "centre" far enough to register as a swipe.  Using only palm points keeps
    the anchor still while the fingers move.
    """
    if pts.shape[0] < 21:
        return np.zeros(2, dtype=np.float32)
    idx = np.array([WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP])
    return pts[idx, :2].mean(axis=0).astype(np.float32)


def normalize(pts: np.ndarray) -> np.ndarray:
    """Translation-, scale- and rotation-normalised copy of the landmarks.

    Wrist to the origin, palm size to 1.0, and the wrist->middle-MCP axis
    rotated to point "up".  Two people making the same sign with differently
    sized hands, at different distances, with heads tilted, produce nearly
    identical arrays -- which is what makes template matching viable.
    """
    if pts.shape[0] < 21:
        return pts.copy()

    out = pts - pts[WRIST]
    scale = palm_size(pts)
    out = out / scale

    axis = out[MIDDLE_MCP, :2]
    norm = float(np.linalg.norm(axis))
    if norm > 1e-6:
        cos_a, sin_a = axis[1] / norm, axis[0] / norm  # rotate axis onto -Y
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
        out[:, :2] = out[:, :2] @ rot.T
    return out.astype(np.float32)


def finger_curl(pts: np.ndarray) -> np.ndarray:
    """Per-finger curl in ``[0, 1]``: 0 = straight, 1 = fully folded.

    Computed from the angle at the PIP joint, which is robust to hand
    orientation.  v1 compared raw ``tip.y < pip.y``, so it silently broke the
    moment the hand was rotated, held sideways, or pointed at the camera.
    """
    if pts.shape[0] < 21:
        return np.ones(5, dtype=np.float32)

    a = pts[FINGER_CHAINS[:, 0]]  # mcp
    b = pts[FINGER_CHAINS[:, 1]]  # pip
    c = pts[FINGER_CHAINS[:, 3]]  # tip

    v1 = a - b
    v2 = c - b
    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)
    denom = np.maximum(n1 * n2, 1e-6)
    cos_ang = np.einsum("ij,ij->i", v1, v2) / denom
    ang = np.arccos(np.clip(cos_ang, -1.0, 1.0))  # pi = straight, small = folded
    curl = 1.0 - (ang / np.pi)
    return np.clip(curl * 1.35 - 0.15, 0.0, 1.0).astype(np.float32)


def fingers_extended(pts: np.ndarray, threshold: float = 0.45) -> np.ndarray:
    """Boolean ``(5,)`` mask of extended fingers.

    The thumb gets an extra spread test because a thumb folded across the palm
    can still look "straight" by joint angle alone.
    """
    curl = finger_curl(pts)
    ext = curl < threshold
    if pts.shape[0] >= 21:
        scale = palm_size(pts)
        spread = float(np.linalg.norm(pts[THUMB_TIP, :2] - pts[INDEX_MCP, :2])) / scale
        ext[0] = bool(ext[0] and spread > 0.55)
    return ext


def pinch_distance(pts: np.ndarray, finger: int = 1) -> float:
    """Scale-invariant thumb-tip to finger-tip distance (palm widths)."""
    if pts.shape[0] < 21:
        return 999.0
    tip = int(TIPS[finger])
    return float(np.linalg.norm(pts[THUMB_TIP, :2] - pts[tip, :2]) / palm_size(pts))


def pointing_direction(pts: np.ndarray) -> np.ndarray:
    """Unit vector along the index finger (PIP -> TIP), image coordinates."""
    if pts.shape[0] < 21:
        return np.zeros(2, dtype=np.float32)
    v = pts[INDEX_TIP, :2] - pts[INDEX_PIP, :2]
    n = float(np.linalg.norm(v))
    return (v / n).astype(np.float32) if n > 1e-6 else np.zeros(2, dtype=np.float32)


def hand_roll(pts: np.ndarray) -> float:
    """In-plane rotation of the palm in degrees; 0 = fingers up."""
    if pts.shape[0] < 21:
        return 0.0
    v = pts[MIDDLE_MCP, :2] - pts[WRIST, :2]
    return float(np.degrees(np.arctan2(v[0], -v[1])))


def palm_facing_camera(pts: np.ndarray, handedness: str = "Right") -> bool:
    """Rough palm/back-of-hand test via the sign of the palm-triangle normal."""
    if pts.shape[0] < 21:
        return True
    v1 = pts[INDEX_MCP] - pts[WRIST]
    v2 = pts[PINKY_MCP] - pts[WRIST]
    z = v1[0] * v2[1] - v1[1] * v2[0]
    return bool(z > 0) if handedness.lower().startswith("r") else bool(z < 0)


def bounding_box(pts: np.ndarray, pad: float = 0.04) -> tuple[float, float, float, float]:
    """Normalised ``(x0, y0, x1, y1)`` box around the hand."""
    if pts.shape[0] == 0:
        return (0.0, 0.0, 0.0, 0.0)
    xy = pts[:, :2]
    lo = np.clip(xy.min(axis=0) - pad, 0.0, 1.0)
    hi = np.clip(xy.max(axis=0) + pad, 0.0, 1.0)
    return (float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1]))
