"""On-screen overlay.

Drawing is deliberately cheap: the previous version rebuilt a full-frame copy
for *every* translucent panel (two ``frame.copy()`` plus two ``addWeighted``
over the whole 1280x720 image, every frame).  Here blending is done on the
panel's own sub-rectangle only, which is roughly an order of magnitude less
pixel work and keeps the HUD off the critical path.

The overlay also drops emoji: OpenCV's Hershey fonts cannot render them, so v1
printed literal "?" boxes on screen for every label.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Optional, Sequence

import cv2
import numpy as np

from . import geometry as geo
from .types import FrameResult, GestureEvent, HandState

FONT = cv2.FONT_HERSHEY_SIMPLEX
LINE = cv2.LINE_AA

THEMES = {
    "dark": {
        "panel": (24, 24, 28),
        "panel_alpha": 0.72,
        "text": (238, 238, 238),
        "dim": (150, 150, 155),
        "accent": (120, 220, 80),
        "warn": (60, 190, 250),
        "bad": (70, 70, 240),
        "bone": (200, 200, 205),
        "joint": (120, 220, 80),
        "trail": (200, 160, 60),
    },
    "light": {
        "panel": (245, 245, 245),
        "panel_alpha": 0.82,
        "text": (30, 30, 30),
        "dim": (110, 110, 110),
        "accent": (40, 150, 40),
        "warn": (20, 130, 210),
        "bad": (40, 40, 200),
        "bone": (70, 70, 70),
        "joint": (40, 150, 40),
        "trail": (150, 110, 30),
    },
}


def panel(frame: np.ndarray, x0: int, y0: int, x1: int, y1: int, colour, alpha: float) -> None:
    """Alpha-blend a filled rectangle, touching only that rectangle."""
    h, w = frame.shape[:2]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return
    roi = frame[y0:y1, x0:x1]
    tint = np.empty_like(roi)
    tint[:] = colour
    cv2.addWeighted(tint, alpha, roi, 1.0 - alpha, 0.0, dst=roi)


def text(frame, s, org, colour, scale=0.5, weight=1, shadow=True) -> None:
    if shadow:
        cv2.putText(frame, s, (org[0] + 1, org[1] + 1), FONT, scale, (0, 0, 0), weight + 1, LINE)
    cv2.putText(frame, s, org, FONT, scale, colour, weight, LINE)


class HUD:
    """Renders landmarks, status panels and gesture feedback."""

    def __init__(self, theme: str = "dark", show_landmarks: bool = True,
                 show_scores: bool = False, show_trail: bool = True,
                 aspect: float = 1.0):
        self.colours = THEMES.get(theme, THEMES["dark"])
        self.show_landmarks = show_landmarks
        self.show_scores = show_scores
        self.show_trail = show_trail
        #: Landmarks arrive aspect-corrected; undo that to draw them back onto
        #: the video frame, or the skeleton will not line up with the hand.
        self.aspect = aspect
        self.toast: Optional[tuple[str, float, str]] = None
        self._trail: deque = deque(maxlen=28)
        self._recent: deque = deque(maxlen=6)

    # -- public ------------------------------------------------------------
    def notify(self, message: str, kind: str = "info", duration: float = 1.6) -> None:
        self.toast = (message, time.perf_counter() + duration, kind)

    def push_events(self, events: Sequence[GestureEvent], labels: dict) -> None:
        for ev in events:
            if ev.kind in ("continuous",):
                continue
            self._recent.appendleft((ev.name, labels.get(ev.name, ""), ev.confidence,
                                     time.perf_counter()))

    def draw(
        self,
        frame: np.ndarray,
        result: FrameResult,
        *,
        profile: str = "",
        backend: str = "",
        armed: bool = True,
        arm_remaining: float = 0.0,
        require_arm: bool = False,
        paused: bool = False,
        scores: Optional[dict] = None,
    ) -> np.ndarray:
        h, w = frame.shape[:2]

        if self.show_landmarks:
            for hand in result.hands:
                self._draw_hand(frame, hand, w, h)

        if self.show_trail and result.hands:
            self._draw_trail(frame, result.hands[0], w, h)

        self._draw_status(frame, result, profile, backend, paused, w, h)
        self._draw_arm(frame, armed, arm_remaining, require_arm, w, h)
        self._draw_recent(frame, w, h)

        if self.show_scores and scores:
            self._draw_scores(frame, scores, w, h)

        self._draw_toast(frame, w, h)
        self._draw_help(frame, w, h)
        return frame

    # -- pieces ------------------------------------------------------------
    def _to_pixels(self, xy: np.ndarray, w: int, h: int) -> np.ndarray:
        """Aspect-corrected normalised coords -> pixel coords."""
        out = np.asarray(xy, dtype=np.float32).reshape(-1, 2).copy()
        if abs(self.aspect - 1.0) > 1e-6:
            out[:, 0] /= self.aspect
        out[:, 0] *= w
        out[:, 1] *= h
        return out.astype(np.int32)

    def _draw_hand(self, frame, hand: HandState, w: int, h: int) -> None:
        c = self.colours
        pts = self._to_pixels(hand.points[:, :2], w, h)

        for a, b in geo.HAND_CONNECTIONS:
            cv2.line(frame, tuple(pts[a]), tuple(pts[b]), c["bone"], 2, LINE)

        for i, (x, y) in enumerate(pts):
            extended = bool(hand.extended[int(np.where(i == geo.TIPS)[0][0])]) \
                if i in geo.TIPS else False
            if i in geo.TIPS:
                colour = c["accent"] if extended else c["dim"]
                cv2.circle(frame, (x, y), 6, colour, -1, LINE)
                cv2.circle(frame, (x, y), 6, (20, 20, 20), 1, LINE)
            else:
                cv2.circle(frame, (x, y), 3, c["joint"], -1, LINE)

        # pinch indicator between thumb and index
        if hand.pinch < 0.45:
            t = tuple(pts[geo.THUMB_TIP])
            i = tuple(pts[geo.INDEX_TIP])
            strength = float(np.clip(1.0 - hand.pinch / 0.45, 0.0, 1.0))
            cv2.line(frame, t, i, c["warn"], 1 + int(2 * strength), LINE)

        x0, y0, x1, y1 = geo.bounding_box(hand.points, pad=0.03)
        (p0, p1) = [tuple(int(v) for v in pt)
                    for pt in self._to_pixels(np.array([[x0, y0], [x1, y1]]), w, h)]
        cv2.rectangle(frame, p0, p1, c["dim"], 1, LINE)
        label = f"{hand.handedness.value[:1]} {hand.score:.0%}"
        text(frame, label, (p0[0], max(14, p0[1] - 6)), c["dim"], 0.42)

    def _draw_trail(self, frame, hand: HandState, w: int, h: int) -> None:
        self._trail.append((float(hand.center[0]), float(hand.center[1])))
        if len(self._trail) < 2:
            return
        px = self._to_pixels(np.asarray(self._trail, dtype=np.float32), w, h)
        for i in range(1, len(px)):
            thickness = max(1, int(3 * i / len(px)))
            cv2.line(frame, tuple(px[i - 1]), tuple(px[i]),
                     self.colours["trail"], thickness, LINE)

    def _draw_status(self, frame, result: FrameResult, profile, backend, paused, w, h) -> None:
        c = self.colours
        pw, ph = 250, 92
        panel(frame, 10, 10, 10 + pw, 10 + ph, c["panel"], c["panel_alpha"])

        title = "GestureKit"
        text(frame, title, (22, 32), c["accent"], 0.62, 2)
        if paused:
            text(frame, "PAUSED", (140, 32), c["bad"], 0.55, 2)

        fps_colour = c["accent"] if result.fps >= 24 else (c["warn"] if result.fps >= 15 else c["bad"])
        text(frame, f"{result.fps:5.1f} fps", (22, 54), fps_colour, 0.48)
        text(frame, f"{result.latency_ms:4.0f} ms", (110, 54), c["dim"], 0.48)
        text(frame, f"inf {result.inference_ms:3.0f} ms", (180, 54), c["dim"], 0.44)

        text(frame, f"profile: {profile}", (22, 74), c["text"], 0.44)
        text(frame, f"in: {backend}", (22, 90), c["dim"], 0.42)

    def _draw_arm(self, frame, armed, remaining, require_arm, w, h) -> None:
        if not require_arm:
            return
        c = self.colours
        x0, y0 = 10, 112
        panel(frame, x0, y0, x0 + 250, y0 + 44, c["panel"], c["panel_alpha"])
        if armed:
            text(frame, f"ARMED  {remaining:.1f}s", (x0 + 12, y0 + 20), c["accent"], 0.5, 2)
            frac = float(np.clip(remaining / 3.0, 0.0, 1.0))
            bar_w = int(226 * frac)
            cv2.rectangle(frame, (x0 + 12, y0 + 28), (x0 + 238, y0 + 34), c["dim"], 1, LINE)
            if bar_w > 2:
                cv2.rectangle(frame, (x0 + 12, y0 + 28), (x0 + 12 + bar_w, y0 + 34),
                              c["accent"], -1, LINE)
        else:
            text(frame, "show arm gesture", (x0 + 12, y0 + 26), c["warn"], 0.48)

    def _draw_recent(self, frame, w, h) -> None:
        if not self._recent:
            return
        c = self.colours
        now = time.perf_counter()
        items = [r for r in self._recent if now - r[3] < 4.0]
        if not items:
            return
        pw, ph = 268, 22 * len(items) + 26
        x0 = w - pw - 10
        panel(frame, x0, 10, w - 10, 10 + ph, c["panel"], c["panel_alpha"])
        text(frame, "recent", (x0 + 12, 30), c["dim"], 0.44)
        for i, (name, label, conf, t) in enumerate(items):
            age = now - t
            colour = c["accent"] if age < 0.7 else c["text"]
            line = f"{name}" + (f" -> {label}" if label else "")
            text(frame, line[:30], (x0 + 12, 52 + i * 22), colour, 0.44)
            text(frame, f"{conf:.0%}", (w - 56, 52 + i * 22), c["dim"], 0.42)

    def _draw_scores(self, frame, scores: dict, w, h) -> None:
        c = self.colours
        top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:6]
        if not top:
            return
        pw = 230
        ph = 20 * len(top) + 26
        x0, y0 = 10, h - ph - 76
        panel(frame, x0, y0, x0 + pw, y0 + ph, c["panel"], c["panel_alpha"])
        text(frame, "scores", (x0 + 12, y0 + 18), c["dim"], 0.42)
        for i, (name, value) in enumerate(top):
            yy = y0 + 36 + i * 20
            text(frame, name[:14], (x0 + 12, yy), c["text"], 0.4)
            bar_x = x0 + 118
            cv2.rectangle(frame, (bar_x, yy - 8), (bar_x + 100, yy - 1), c["dim"], 1, LINE)
            fill = int(100 * float(np.clip(value, 0, 1)))
            if fill > 1:
                colour = c["accent"] if value > 0.62 else c["warn"]
                cv2.rectangle(frame, (bar_x, yy - 8), (bar_x + fill, yy - 1), colour, -1, LINE)

    def _draw_toast(self, frame, w, h) -> None:
        if not self.toast:
            return
        message, expiry, kind = self.toast
        if time.perf_counter() > expiry:
            self.toast = None
            return
        c = self.colours
        colour = {"info": c["text"], "good": c["accent"], "warn": c["warn"], "bad": c["bad"]}.get(
            kind, c["text"]
        )
        (tw, th), _ = cv2.getTextSize(message, FONT, 0.62, 2)
        x0 = (w - tw) // 2 - 16
        y0 = h // 2 - th - 14
        panel(frame, x0, y0, x0 + tw + 32, y0 + th + 26, c["panel"], 0.85)
        text(frame, message, (x0 + 16, y0 + th + 8), colour, 0.62, 2)

    def _draw_help(self, frame, w, h) -> None:
        c = self.colours
        panel(frame, 0, h - 26, w, h, c["panel"], 0.6)
        text(
            frame,
            "q quit   space pause   p profile   s scores   l landmarks   r reset   h help",
            (12, h - 8),
            c["dim"],
            0.42,
        )


def draw_help_screen(width: int, height: int, bindings: Sequence[tuple[str, str]],
                     profile: str, theme: str = "dark") -> np.ndarray:
    """Full-screen cheat sheet of the active bindings."""
    c = THEMES.get(theme, THEMES["dark"])
    img = np.zeros((height, width, 3), np.uint8)
    img[:] = c["panel"]
    text(img, f"GestureKit -- profile: {profile}", (28, 44), c["accent"], 0.8, 2)
    text(img, "press h to close", (28, 68), c["dim"], 0.46)

    col_h = 26
    per_col = max(1, (height - 130) // col_h)
    for i, (gesture, label) in enumerate(bindings):
        col = i // per_col
        row = i % per_col
        x = 28 + col * 330
        y = 106 + row * col_h
        if x > width - 200:
            break
        text(img, gesture[:18], (x, y), c["text"], 0.5)
        text(img, str(label)[:22], (x + 160, y), c["dim"], 0.48)
    return img
