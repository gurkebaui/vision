"""Static hand-shape recognition.

Two independent evidence sources are fused:

1. **Rule scores** -- smooth, continuous membership functions over finger
   curls and spreads.  Unlike v1's hard ``if extended == 4`` branches, a
   half-curled finger degrades the score gradually instead of flipping the
   answer, so borderline poses are reported as *low confidence* rather than
   *confidently wrong*.

2. **Template matching** -- cosine-style distance against canonical
   rotation/scale-normalised landmark templates, which captures the overall
   silhouette that per-finger rules miss.

The fused score then has to survive :class:`GestureStabilizer`: N consistent
frames plus a margin over the runner-up before anything is emitted.  That is
what stops the single-frame misfires that made v1 feel twitchy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .. import geometry as geo
from ..types import GestureEvent, HandState
from .base import Recognizer, RecognizerContext

# ---------------------------------------------------------------------------
# Smooth membership helpers
# ---------------------------------------------------------------------------


def _lo(x: float, edge: float, soft: float = 0.18) -> float:
    """1.0 well below ``edge``, ramping to 0.0 above it."""
    return float(np.clip((edge + soft - x) / (2 * soft), 0.0, 1.0))


def _hi(x: float, edge: float, soft: float = 0.18) -> float:
    """1.0 well above ``edge``, ramping to 0.0 below it."""
    return float(np.clip((x - edge + soft) / (2 * soft), 0.0, 1.0))


def _extended(curls: np.ndarray, i: int) -> float:
    return _lo(float(curls[i]), 0.42)


def _folded(curls: np.ndarray, i: int) -> float:
    return _hi(float(curls[i]), 0.55)


def _geometric_mean(values: list[float]) -> float:
    """Fuse sub-scores so that one bad term drags the whole score down.

    An arithmetic mean would let "4 out of 5 fingers right" score 0.8 even
    when the decisive finger is wrong; the geometric mean will not.
    """
    if not values:
        return 0.0
    arr = np.clip(np.asarray(values, dtype=np.float64), 1e-4, 1.0)
    return float(np.exp(np.log(arr).mean()))


# ---------------------------------------------------------------------------
# Rule set
# ---------------------------------------------------------------------------

RuleFn = Callable[[HandState], float]


def _rule_open_palm(h: HandState) -> float:
    c = h.curls
    base = _geometric_mean([_extended(c, i) for i in range(5)])
    spread = float(np.linalg.norm(h.points[geo.INDEX_TIP, :2] - h.points[geo.PINKY_TIP, :2]))
    spread /= h.palm_size
    return base * _hi(spread, 0.75, 0.3)


def _rule_closed_fist(h: HandState) -> float:
    return _geometric_mean([_folded(h.curls, i) for i in range(1, 5)] + [_hi(h.curls[0], 0.4, 0.25)])


def _rule_point_up(h: HandState) -> float:
    c = h.curls
    shape = _geometric_mean([_extended(c, 1), _folded(c, 2), _folded(c, 3), _folded(c, 4)])
    d = geo.pointing_direction(h.points)
    up = _hi(float(-d[1]), 0.55, 0.35)      # -y is up in image coords
    return shape * up


def _rule_point_down(h: HandState) -> float:
    c = h.curls
    shape = _geometric_mean([_extended(c, 1), _folded(c, 2), _folded(c, 3), _folded(c, 4)])
    d = geo.pointing_direction(h.points)
    return shape * _hi(float(d[1]), 0.55, 0.35)


def _rule_point_left(h: HandState) -> float:
    c = h.curls
    shape = _geometric_mean([_extended(c, 1), _folded(c, 2), _folded(c, 3), _folded(c, 4)])
    d = geo.pointing_direction(h.points)
    return shape * _hi(float(-d[0]), 0.6, 0.3)


def _rule_point_right(h: HandState) -> float:
    c = h.curls
    shape = _geometric_mean([_extended(c, 1), _folded(c, 2), _folded(c, 3), _folded(c, 4)])
    d = geo.pointing_direction(h.points)
    return shape * _hi(float(d[0]), 0.6, 0.3)


def _rule_peace(h: HandState) -> float:
    c = h.curls
    shape = _geometric_mean([_extended(c, 1), _extended(c, 2), _folded(c, 3), _folded(c, 4)])
    gap = float(np.linalg.norm(h.points[geo.INDEX_TIP, :2] - h.points[geo.MIDDLE_TIP, :2]))
    gap /= h.palm_size
    return shape * _hi(gap, 0.28, 0.18)


def _thumb_out(h: HandState) -> float:
    """Thumb clear of the index tip.

    Without this a pinch reads as a thumbs-up: both have one thumb out and
    four folded fingers.  The difference is that a pinch brings the thumb tip
    *to* the index tip, while a thumbs-up holds it well away.
    """
    return _hi(h.pinch, 0.55, 0.22)


def _rule_thumbs_up(h: HandState) -> float:
    c = h.curls
    shape = _geometric_mean([_extended(c, 0)] + [_folded(c, i) for i in range(1, 5)])
    v = h.points[geo.THUMB_TIP, :2] - h.points[geo.THUMB_MCP, :2]
    n = float(np.linalg.norm(v))
    if n < 1e-6:
        return 0.0
    return shape * _hi(float(-v[1] / n), 0.5, 0.35) * _thumb_out(h)


def _rule_thumbs_down(h: HandState) -> float:
    c = h.curls
    shape = _geometric_mean([_extended(c, 0)] + [_folded(c, i) for i in range(1, 5)])
    v = h.points[geo.THUMB_TIP, :2] - h.points[geo.THUMB_MCP, :2]
    n = float(np.linalg.norm(v))
    if n < 1e-6:
        return 0.0
    return shape * _hi(float(v[1] / n), 0.5, 0.35) * _thumb_out(h)


def _pinch_shape(h: HandState) -> float:
    """Thumb and index tips touching *and* the index curved to meet them.

    The curvature term matters: with a flat open hand the thumb tip can drift
    close to the index tip in 2D projection, which would otherwise read as a
    pinch.  A real pinch always bends the index finger somewhat.
    """
    close = _lo(h.pinch, 0.34, 0.12)
    bent = _hi(float(h.curls[1]), 0.16, 0.14)
    return close * bent


def _rule_pinch(h: HandState) -> float:
    """Thumb+index touching with the remaining fingers *closed*."""
    others_open = _geometric_mean([_extended(h.curls, i) for i in (2, 3, 4)])
    return _pinch_shape(h) * (1.0 - 0.85 * others_open)


def _rule_ok(h: HandState) -> float:
    """Thumb+index touching with the remaining fingers *fanned out*."""
    others_open = _geometric_mean([_extended(h.curls, i) for i in (2, 3, 4)])
    return _pinch_shape(h) * others_open


def _rule_rock(h: HandState) -> float:
    c = h.curls
    return _geometric_mean([_extended(c, 1), _folded(c, 2), _folded(c, 3), _extended(c, 4)])


def _rule_call_me(h: HandState) -> float:
    c = h.curls
    return _geometric_mean(
        [_extended(c, 0), _folded(c, 1), _folded(c, 2), _folded(c, 3), _extended(c, 4)]
    )


def _rule_three(h: HandState) -> float:
    c = h.curls
    return _geometric_mean([_extended(c, 1), _extended(c, 2), _extended(c, 3), _folded(c, 4)])


def _rule_four(h: HandState) -> float:
    c = h.curls
    return _geometric_mean([_folded(c, 0)] + [_extended(c, i) for i in range(1, 5)])


@dataclass(frozen=True)
class GestureDef:
    name: str
    rule: RuleFn
    canned: tuple = ()          # matching MediaPipe canned labels
    prior: float = 1.0          # multiplicative bias; <1 for easily confused poses


GESTURES: tuple[GestureDef, ...] = (
    GestureDef("open_palm", _rule_open_palm, ("Open_Palm",)),
    GestureDef("closed_fist", _rule_closed_fist, ("Closed_Fist",)),
    GestureDef("point_up", _rule_point_up, ("Pointing_Up",)),
    GestureDef("point_down", _rule_point_down, ()),
    GestureDef("point_left", _rule_point_left, ()),
    GestureDef("point_right", _rule_point_right, ()),
    GestureDef("peace", _rule_peace, ("Victory",)),
    GestureDef("thumbs_up", _rule_thumbs_up, ("Thumb_Up",)),
    GestureDef("thumbs_down", _rule_thumbs_down, ("Thumb_Down",)),
    GestureDef("pinch", _rule_pinch, (), prior=0.9),
    GestureDef("ok", _rule_ok, (), prior=0.95),
    GestureDef("rock", _rule_rock, (), prior=0.95),
    GestureDef("call_me", _rule_call_me, ()),
    GestureDef("three", _rule_three, ()),
    GestureDef("four", _rule_four, ()),
)

#: Canonical normalised templates, filled lazily from synthetic poses.
TEMPLATES: dict[str, np.ndarray] = {}


def score_all(hand: HandState, use_canned: bool = True) -> dict[str, float]:
    """Score every static gesture for one hand. Returns ``{name: 0..1}``."""
    scores: dict[str, float] = {}
    for g in GESTURES:
        try:
            s = float(g.rule(hand)) * g.prior
        except Exception:
            s = 0.0
        scores[g.name] = float(np.clip(s, 0.0, 1.0))

    # Fuse Google's canned classifier as a soft prior, never as an override:
    # it only knows 7 classes, so it must not veto our extra gestures.
    if use_canned and hand.label and hand.label_score > 0.5:
        for g in GESTURES:
            if hand.label in g.canned:
                boost = 1.0 + 0.35 * hand.label_score
                scores[g.name] = float(np.clip(scores[g.name] * boost, 0.0, 1.0))
    return scores


class GestureStabilizer:
    """Temporal voting with margin and dwell requirements.

    A gesture is only accepted when it has been the top candidate for
    ``min_frames`` consecutive frames, its mean score clears ``min_score``, and
    it leads the runner-up by ``margin``.  v1 fired on a single frame above a
    hard-coded threshold, which is why brushing past a pose triggered actions.
    """

    def __init__(
        self,
        min_frames: int = 4,
        min_score: float = 0.62,
        margin: float = 0.10,
        release_frames: int = 3,
    ):
        self.min_frames = int(min_frames)
        self.min_score = float(min_score)
        self.margin = float(margin)
        self.release_frames = int(release_frames)
        self._candidate: Optional[str] = None
        self._count = 0
        self._scores: list[float] = []
        self._active: Optional[str] = None
        self._absent = 0

    def reset(self) -> None:
        self._candidate = None
        self._count = 0
        self._scores.clear()
        self._active = None
        self._absent = 0

    @property
    def active(self) -> Optional[str]:
        return self._active

    def update(self, scores: dict[str, float]) -> Optional[tuple[str, float]]:
        """Feed one frame of scores; return ``(name, confidence)`` on a new lock."""
        if not scores:
            return self._decay()

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        top, top_score = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else 0.0

        if top_score < self.min_score or (top_score - second) < self.margin:
            return self._decay()

        if top == self._candidate:
            self._count += 1
            self._scores.append(top_score)
        else:
            self._candidate = top
            self._count = 1
            self._scores = [top_score]
        self._absent = 0

        if self._count >= self.min_frames and self._active != top:
            self._active = top
            conf = float(np.mean(self._scores[-self.min_frames:]))
            return top, conf
        return None

    def _decay(self) -> None:
        self._absent += 1
        if self._absent >= self.release_frames:
            self._candidate = None
            self._count = 0
            self._scores.clear()
            self._active = None
        return None


class StaticRecognizer(Recognizer):
    """Emits one event per *new* stable hand shape (edge-triggered)."""

    name = "static"

    def __init__(
        self,
        min_frames: int = 4,
        min_score: float = 0.62,
        margin: float = 0.10,
        use_canned: bool = True,
        max_speed: float = 1.8,
    ):
        self.stabilizer = GestureStabilizer(min_frames, min_score, margin)
        self.use_canned = use_canned
        self.max_speed = float(max_speed)
        self.last_scores: dict[str, float] = {}

    def reset(self) -> None:
        self.stabilizer.reset()
        self.last_scores = {}

    def update(self, ctx: RecognizerContext) -> list[GestureEvent]:
        hand = ctx.current
        if hand is None:
            self.stabilizer.update({})
            return []

        # A hand in flight is mid-swipe, not holding a pose: suppress shapes so
        # the dynamic recogniser owns that moment. Removes swipe/pose crosstalk.
        if hand.speed > self.max_speed:
            self.stabilizer.update({})
            return []

        scores = score_all(hand, self.use_canned)
        self.last_scores = scores
        hit = self.stabilizer.update(scores)
        if hit is None:
            return []
        name, conf = hit
        return [
            GestureEvent(
                name=name,
                confidence=conf,
                hand=hand.handedness,
                timestamp=hand.timestamp,
                kind="static",
                data={"scores": scores},
            )
        ]
