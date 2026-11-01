"""
Scale-Invariant 3D & Joint-Angle Gesture Engine with Dynamic Velocity Swipes,
Return-Stroke Suppression, 1-Euro Air Cursor, and Intentional Clutch/Dwell Control.

Every geometric measurement is normalized into Palm Units (1 PU ≈ ~9 cm physical palm size),
making gesture accuracy and swipe feel identical at 0.4m, 2.0m, and 5.0m from the camera.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Sequence, Tuple

from .config import GESTURE_CATALOG, SystemConfig
from .filters import Point2DOneEuroFilter

OPPOSITE_SWIPE: Dict[str, str] = {
    "swipe_right": "swipe_left",
    "swipe_left": "swipe_right",
    "swipe_up": "swipe_down",
    "swipe_down": "swipe_up",
}


@dataclass
class GestureEvaluation:
    # Instantaneous & stabilized classification
    raw_gesture: Optional[str] = None
    gesture: Optional[str] = None
    gesture_name: str = "None"
    gesture_icon: str = "🤚"
    confidence: float = 0.0
    extended_fingers: List[str] = field(default_factory=list)
    extended_count: int = 0
    pinch_ratio: float = 1.0
    is_pinching: bool = False

    # Intentional activation & progress state (for HUD radial ring)
    clutch_unlocked: bool = True
    clutch_remaining_s: float = 0.0
    dwell_progress: float = 0.0  # [0.0 .. 1.0]

    # Triggered command on THIS frame (None unless newly confirmed & past cooldown)
    triggered_gesture: Optional[str] = None
    triggered_action: Optional[str] = None

    # Ergonomic 1-Euro smoothed normalized screen coordinates [0..1] for Air Mouse / Laser
    cursor_xy: Optional[Tuple[float, float]] = None
    palm_center_norm: Optional[Tuple[float, float]] = None
    palm_size_norm: float = 0.0


def _dist2d(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _dist3d(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def _angle_deg(
    v1: Tuple[float, float, float], v2: Tuple[float, float, float]
) -> float:
    n1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2 + v1[2] ** 2)
    n2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2 + v2[2] ** 2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cos_val = (v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]) / (n1 * n2)
    cos_val = max(-1.0, min(1.0, cos_val))
    return math.degrees(math.acos(cos_val))


def _vec(
    a: Tuple[float, float, float], b: Tuple[float, float, float]
) -> Tuple[float, float, float]:
    """Vector from a -> b."""
    return (b[0] - a[0], b[1] - a[1], b[2] - a[2])


class ScaleInvariantGestureEngine:
    """
    Distance-invariant biomechanical pose classifier + velocity swipe detector + clutch state machine.
    """

    def __init__(self, config: Optional[SystemConfig] = None) -> None:
        self.config = config or SystemConfig()

        # Trajectory history: (timestamp, iso_cx, iso_cy, palm_size)
        self._trajectory: Deque[Tuple[float, float, float, float]] = deque(maxlen=24)

        # Static pose stabilization buffer
        self._pose_history: Deque[Tuple[float, Optional[str], float]] = deque(maxlen=6)

        # Pinch hysteresis state
        self._pinch_active: bool = False

        # Swipe cooldown & return-stroke suppression
        self._last_swipe_time: float = -10.0
        self._last_swipe_dir: Optional[str] = None

        # Clutch & Dwell state
        self._clutch_unlocked: bool = False
        self._clutch_unlock_expiry: float = 0.0
        self._hold_candidate: Optional[str] = None
        self._hold_start_time: Optional[float] = None
        self._hold_fired_for_current_pose: bool = False
        self._last_trigger_time: float = -10.0
        self._last_trigger_gesture: Optional[str] = None

        # Smooth 1-Euro cursor filter
        self._cursor_filter = Point2DOneEuroFilter(
            min_cutoff=self.config.cursor_smoothing_min_cutoff,
            beta=self.config.cursor_smoothing_beta,
        )

    def reset(self) -> None:
        self._trajectory.clear()
        self._pose_history.clear()
        self._pinch_active = False
        self._last_swipe_time = -10.0
        self._last_swipe_dir = None
        self._clutch_unlocked = False
        self._clutch_unlock_expiry = 0.0
        self._hold_candidate = None
        self._hold_start_time = None
        self._hold_fired_for_current_pose = False
        self._cursor_filter.reset()

    def classify_static_pose(
        self,
        landmarks: Sequence[Tuple[float, float, float]],
        aspect_ratio: float = 16.0 / 9.0,
    ) -> Tuple[Optional[str], float, List[str], float, float]:
        """
        Classify static hand pose in aspect-equalized Palm Units.
        Returns (pose_key, confidence, extended_fingers, pinch_ratio_pu, palm_size_iso).
        """
        if len(landmarks) < 21:
            return None, 0.0, [], 1.0, 0.0

        # Convert to aspect-corrected isotropic space so rotation never stretches bones
        iso: List[Tuple[float, float, float]] = [
            (p[0] * aspect_ratio, p[1], p[2] * aspect_ratio) for p in landmarks
        ]

        wrist = iso[0]
        len_wm = _dist3d(iso[0], iso[9])
        wid_ip = _dist3d(iso[5], iso[17])
        palm_size = max(1e-4, 0.65 * len_wm + 0.50 * wid_ip)

        # Evaluate 4 non-thumb fingers using joint flexion angles + wrist reach
        finger_defs = [
            ("index", 5, 6, 7, 8),
            ("middle", 9, 10, 11, 12),
            ("ring", 13, 14, 15, 16),
            ("pinky", 17, 18, 19, 20),
        ]
        extended_fingers: List[str] = []
        curled_map: Dict[str, bool] = {}
        ext_map: Dict[str, bool] = {}

        for fname, mcp_i, pip_i, dip_i, tip_i in finger_defs:
            v1 = _vec(iso[mcp_i], iso[pip_i])
            v2 = _vec(iso[pip_i], iso[dip_i])
            v3 = _vec(iso[dip_i], iso[tip_i])
            ang_pip = _angle_deg(v1, v2)
            ang_dip = _angle_deg(v2, v3)
            reach = _dist2d(iso[tip_i], wrist) / max(1e-4, _dist2d(iso[pip_i], wrist))
            span_pu = _dist2d(iso[tip_i], iso[mcp_i]) / palm_size

            is_ext = (
                ang_pip < 44.0
                and ang_dip < 52.0
                and reach > 1.11
                and span_pu > 0.50
            )
            is_curled = (
                ang_pip > 56.0 or reach < 1.04 or span_pu < 0.44
            )
            ext_map[fname] = is_ext
            curled_map[fname] = is_curled
            if is_ext:
                extended_fingers.append(fname)

        # Evaluate Thumb
        thumb_ip_ang = _angle_deg(_vec(iso[2], iso[3]), _vec(iso[3], iso[4]))
        thumb_mcp_ang = _angle_deg(_vec(iso[1], iso[2]), _vec(iso[2], iso[3]))
        thumb_idx_mcp_pu = _dist2d(iso[4], iso[5]) / palm_size
        thumb_pinky_mcp_pu = _dist2d(iso[4], iso[17]) / palm_size

        thumb_ext = (
            thumb_ip_ang < 48.0
            and thumb_mcp_ang < 48.0
            and thumb_idx_mcp_pu > 0.48
            and thumb_pinky_mcp_pu > 0.78
        )
        if thumb_ext:
            extended_fingers.insert(0, "thumb")

        # Pinch ratio (thumb tip L[4] to index tip L[8] in Palm Units) with hysteresis
        pinch_pu = _dist2d(iso[4], iso[8]) / palm_size
        if self._pinch_active:
            if pinch_pu > 0.45:
                self._pinch_active = False
        else:
            if pinch_pu < 0.33:
                self._pinch_active = True

        non_thumb_ext_count = sum(
            1 for f in ("index", "middle", "ring", "pinky") if ext_map[f]
        )
        non_thumb_curled_count = sum(
            1 for f in ("index", "middle", "ring", "pinky") if curled_map[f]
        )

        # 1. Intentional Pinch (👌): thumb + index tips together while index isn't rolled into a fist
        index_wrist_pu = _dist2d(iso[8], wrist) / palm_size
        if self._pinch_active and (
            ext_map["middle"] or ext_map["ring"] or index_wrist_pu > 0.82
        ):
            conf = min(0.97, max(0.78, 1.0 - pinch_pu * 0.65))
            return "pinch", conf, extended_fingers, pinch_pu, palm_size

        # 2. Open Palm (✋): all 4 fingers extended
        if non_thumb_ext_count == 4 and pinch_pu > 0.42:
            conf = 0.95 if thumb_ext else 0.89
            return "open_palm", conf, extended_fingers, pinch_pu, palm_size

        # 3. Three Fingers (🤟 / 3️⃣): index + middle + ring extended, pinky curled
        if (
            ext_map["index"]
            and ext_map["middle"]
            and ext_map["ring"]
            and curled_map["pinky"]
        ):
            return "three_fingers", 0.90, extended_fingers, pinch_pu, palm_size

        # 4. Peace Sign (✌️): index + middle extended, ring + pinky curled, V-separated
        if (
            ext_map["index"]
            and ext_map["middle"]
            and curled_map["ring"]
            and curled_map["pinky"]
        ):
            v_spread_pu = _dist2d(iso[8], iso[12]) / palm_size
            if v_spread_pu > 0.20:
                conf = min(0.95, 0.80 + v_spread_pu * 0.25)
                return "peace_sign", conf, extended_fingers, pinch_pu, palm_size

        # 5. Single Index Pointing (☝️ Up / 👇 Down)
        if (
            ext_map["index"]
            and curled_map["middle"]
            and curled_map["ring"]
            and curled_map["pinky"]
        ):
            dx = iso[8][0] - iso[5][0]
            dy = iso[8][1] - iso[5][1]  # -y is UP, +y is DOWN
            mag = max(1e-4, math.hypot(dx, dy))
            vert_ratio = abs(dy) / mag
            if dy < 0 and vert_ratio >= 0.58:
                conf = min(0.94, 0.75 + vert_ratio * 0.20)
                return "pointing_up", conf, extended_fingers, pinch_pu, palm_size
            if dy > 0 and vert_ratio >= 0.55:
                conf = min(0.94, 0.75 + vert_ratio * 0.20)
                return "pointing_down", conf, extended_fingers, pinch_pu, palm_size

        # 6. Thumbs Up (👍) & Thumbs Down (👎)
        palm_cy = (iso[0][1] + iso[5][1] + iso[9][1] + iso[17][1]) * 0.25
        palm_cx = (iso[0][0] + iso[5][0] + iso[9][0] + iso[17][0]) * 0.25
        thumb_center_pu = math.hypot(iso[4][0] - palm_cx, iso[4][1] - palm_cy) / palm_size
        curled_tip_ys = [iso[8][1], iso[12][1], iso[16][1], iso[20][1]]

        if non_thumb_curled_count == 4 and thumb_ip_ang < 52.0:
            tdx = iso[4][0] - iso[2][0]
            tdy = iso[4][1] - iso[2][1]
            tmag = max(1e-4, math.hypot(tdx, tdy))
            if abs(tdy) / tmag >= 0.55 and (tmag / palm_size) > 0.36:
                conf = min(0.95, 0.79 + (abs(tdy) / tmag) * 0.16)
                if (
                    tdy < 0
                    and iso[4][1] < min(curled_tip_ys) - 0.08 * palm_size
                    and iso[4][1] < iso[5][1] - 0.12 * palm_size
                ):
                    return "thumbs_up", conf, extended_fingers, pinch_pu, palm_size
                if (
                    tdy > 0
                    and iso[4][1] > max(curled_tip_ys) + 0.08 * palm_size
                    and iso[4][1] > iso[5][1] + 0.12 * palm_size
                ):
                    return "thumbs_down", conf, extended_fingers, pinch_pu, palm_size

        # 7. Closed Fist (✊): all 4 fingers curled tightly and thumb tucked near fist
        if non_thumb_curled_count == 4 and thumb_center_pu < 0.82:
            return "closed_fist", 0.92, extended_fingers, pinch_pu, palm_size

        return None, 0.0, extended_fingers, pinch_pu, palm_size

    def _detect_dynamic_swipe(self, now: float) -> Tuple[Optional[str], float]:
        """
        Detect crisp directional hand flicks measured in Palm Units / sec (PU/s),
        with Return-Stroke Suppression so bringing your hand back never triggers the reverse action.
        """
        if len(self._trajectory) < 4:
            return None, 0.0

        # Global refractory period between any two swipes
        if (now - self._last_swipe_time) < max(0.42, self.config.gesture_cooldown_seconds * 0.65):
            return None, 0.0

        curr_t, curr_x, curr_y, curr_ps = self._trajectory[-1]

        # Find trajectory anchor ~0.13s–0.26s ago
        anchor = None
        idx_anchor = 0
        for i in range(len(self._trajectory) - 2, -1, -1):
            t_i = self._trajectory[i][0]
            age = curr_t - t_i
            if 0.10 <= age <= 0.28:
                anchor = self._trajectory[i]
                idx_anchor = i
                if age >= 0.18:
                    break

        if anchor is None:
            return None, 0.0

        dt = max(1e-3, curr_t - anchor[0])
        avg_ps = max(1e-4, 0.5 * (curr_ps + anchor[3]))
        dx_pu = (curr_x - anchor[1]) / avg_ps
        dy_pu = (curr_y - anchor[2]) / avg_ps
        disp_pu = math.hypot(dx_pu, dy_pu)

        # Compute path length over segment to ensure linear flick (not circular hand-talking!)
        path_pu = 0.0
        pts = list(self._trajectory)[idx_anchor:]
        for j in range(1, len(pts)):
            seg = math.hypot(pts[j][1] - pts[j - 1][1], pts[j][2] - pts[j - 1][2])
            path_pu += seg / avg_ps

        linearity = disp_pu / max(path_pu, 1e-4)
        vel_pu_s = disp_pu / dt

        sens = max(0.4, min(2.0, self.config.swipe_sensitivity))
        min_disp = 1.30 / sens
        min_vel = 5.0 / sens

        if disp_pu < min_disp or vel_pu_s < min_vel or linearity < 0.74:
            return None, 0.0

        swipe_key: Optional[str] = None
        if abs(dx_pu) > abs(dy_pu) * 1.55:
            swipe_key = "swipe_right" if dx_pu > 0 else "swipe_left"
        elif abs(dy_pu) > abs(dx_pu) * 1.55:
            swipe_key = "swipe_down" if dy_pu > 0 else "swipe_up"

        if swipe_key is None:
            return None, 0.0

        # Return-Stroke Suppression: ignore opposite-direction recovery motion for 0.75s
        if (
            self._last_swipe_dir is not None
            and OPPOSITE_SWIPE.get(self._last_swipe_dir) == swipe_key
            and (now - self._last_swipe_time) < 0.78
        ):
            return None, 0.0

        conf = min(0.98, 0.76 + min(0.22, (vel_pu_s - min_vel) * 0.03))
        self._last_swipe_time = now
        self._last_swipe_dir = swipe_key
        self._trajectory.clear()
        return swipe_key, conf

    def _compute_ergonomic_cursor(
        self,
        landmarks: Sequence[Tuple[float, float, float]],
        now: float,
    ) -> Tuple[float, float]:
        """
        Map comfortable shoulder-level hand sub-box into smooth [0..1] screen coordinates
        filtered through a 2D 1-Euro filter.
        """
        self._cursor_filter.update_params(
            self.config.cursor_smoothing_min_cutoff,
            self.config.cursor_smoothing_beta,
        )
        # Blend index MCP (stable) + index tip (responsive) so pinching doesn't yank the cursor!
        anchor_x = 0.65 * landmarks[5][0] + 0.35 * landmarks[8][0]
        anchor_y = 0.65 * landmarks[5][1] + 0.35 * landmarks[8][1]

        sens = max(0.6, min(2.5, self.config.cursor_sensitivity))
        half_span_x = 0.32 / sens
        half_span_y = 0.28 / sens
        center_x, center_y = 0.50, 0.46

        norm_x = 0.5 + (anchor_x - center_x) / max(0.10, 2.0 * half_span_x)
        norm_y = 0.5 + (anchor_y - center_y) / max(0.10, 2.0 * half_span_y)
        clamped_x = max(0.0, min(1.0, norm_x))
        clamped_y = max(0.0, min(1.0, norm_y))
        return self._cursor_filter.filter(clamped_x, clamped_y, now)

    def update(
        self,
        landmarks: Optional[Sequence[Tuple[float, float, float]]],
        now: float,
        aspect_ratio: float = 16.0 / 9.0,
    ) -> GestureEvaluation:
        """
        Process one frame of landmarks (or None if no hand detected) and return full evaluation.
        """
        mode = self.config.activation_mode
        profile_data = self.config.get_active_profile_data()
        mappings: Dict[str, str] = profile_data.get("mappings", {})

        # Expire clutch window if time elapsed
        if mode == "clutch" and self._clutch_unlocked and now > self._clutch_unlock_expiry:
            self._clutch_unlocked = False

        if not landmarks or len(landmarks) < 21:
            self._trajectory.clear()
            self._pose_history.clear()
            self._hold_candidate = None
            self._hold_start_time = None
            self._hold_fired_for_current_pose = False
            rem = max(0.0, self._clutch_unlock_expiry - now) if self._clutch_unlocked else 0.0
            return GestureEvaluation(
                clutch_unlocked=(mode != "clutch") or self._clutch_unlocked,
                clutch_remaining_s=round(rem, 1),
            )

        # Compute palm center & static pose
        palm_nx = (landmarks[0][0] + landmarks[5][0] + landmarks[9][0] + landmarks[17][0]) * 0.25
        palm_ny = (landmarks[0][1] + landmarks[5][1] + landmarks[9][1] + landmarks[17][1]) * 0.25

        static_pose, static_conf, ext_fingers, pinch_pu, palm_size_iso = (
            self.classify_static_pose(landmarks, aspect_ratio=aspect_ratio)
        )

        # Record trajectory for dynamic swipes
        self._trajectory.append(
            (now, palm_nx * aspect_ratio, palm_ny, palm_size_iso)
        )

        # Stabilize static pose over short 4-frame majority window
        self._pose_history.append((now, static_pose, static_conf))
        stable_pose: Optional[str] = None
        stable_conf: float = 0.0
        if static_pose is not None:
            recent = [
                (p, c)
                for (t_i, p, c) in self._pose_history
                if (now - t_i) < 0.25 and p == static_pose
            ]
            if len(recent) >= 2:
                stable_pose = static_pose
                stable_conf = sum(c for _, c in recent) / len(recent)

        # Check dynamic velocity swipes
        swipe_pose, swipe_conf = self._detect_dynamic_swipe(now)

        cursor_xy = self._compute_ergonomic_cursor(landmarks, now)

        # Determine active gesture & activation state machine
        triggered_gesture: Optional[str] = None
        triggered_action: Optional[str] = None
        dwell_progress: float = 0.0

        active_pose = swipe_pose if swipe_pose is not None else stable_pose
        active_conf = swipe_conf if swipe_pose is not None else stable_conf

        # =====================================================================
        # MODE 1: CLUTCH WINDOW MODE (Intentional Presenter Lock)
        # =====================================================================
        if mode == "clutch":
            wake_gesture = self.config.clutch_wake_gesture
            if not self._clutch_unlocked:
                # Waiting for intentional wake hold (e.g. steady Open Palm for ~0.38s)
                if stable_pose == wake_gesture:
                    if self._hold_candidate != wake_gesture or self._hold_start_time is None:
                        self._hold_candidate = wake_gesture
                        self._hold_start_time = now
                        self._hold_fired_for_current_pose = False
                    elapsed = now - self._hold_start_time
                    dwell_progress = min(1.0, elapsed / max(0.15, self.config.clutch_hold_seconds))
                    if elapsed >= self.config.clutch_hold_seconds:
                        self._clutch_unlocked = True
                        self._clutch_unlock_expiry = now + self.config.clutch_window_seconds
                        self._hold_fired_for_current_pose = True
                        dwell_progress = 1.0
                else:
                    self._hold_candidate = None
                    self._hold_start_time = None
            else:
                # Clutch is UNLOCKED!
                if swipe_pose is not None:
                    mapped = mappings.get(swipe_pose, "none")
                    if mapped and mapped != "none":
                        triggered_gesture = swipe_pose
                        triggered_action = mapped
                        self._last_trigger_time = now
                        self._last_trigger_gesture = swipe_pose
                        # Auto-extend active window so multi-slide navigation flows naturally
                        self._clutch_unlock_expiry = now + self.config.clutch_window_seconds
                elif stable_pose is not None:
                    if stable_pose != self._hold_candidate:
                        self._hold_candidate = stable_pose
                        self._hold_start_time = now
                        self._hold_fired_for_current_pose = False

                    # Don't immediately fire the wake gesture that JUST unlocked the window
                    # unless the user released and re-performed it!
                    if not self._hold_fired_for_current_pose and self._hold_start_time is not None:
                        req_hold = 0.28
                        elapsed = now - self._hold_start_time
                        dwell_progress = min(1.0, elapsed / req_hold)
                        cooldown_ok = (now - self._last_trigger_time) >= self.config.gesture_cooldown_seconds
                        if elapsed >= req_hold and cooldown_ok:
                            mapped = mappings.get(stable_pose, "none")
                            if mapped == "lock_clutch":
                                self._clutch_unlocked = False
                                self._clutch_unlock_expiry = 0.0
                                triggered_gesture = stable_pose
                                triggered_action = "lock_clutch"
                                self._hold_fired_for_current_pose = True
                            elif mapped and mapped != "none":
                                triggered_gesture = stable_pose
                                triggered_action = mapped
                                self._last_trigger_time = now
                                self._last_trigger_gesture = stable_pose
                                self._hold_fired_for_current_pose = True
                                self._clutch_unlock_expiry = now + self.config.clutch_window_seconds
                else:
                    self._hold_candidate = None
                    self._hold_start_time = None
                    self._hold_fired_for_current_pose = False

        # =====================================================================
        # MODE 2 & 3: DIRECT DWELL / INSTANT MODE
        # =====================================================================
        else:
            req_hold = (
                self.config.dwell_hold_seconds if mode == "dwell" else 0.16
            )
            if swipe_pose is not None:
                mapped = mappings.get(swipe_pose, "none")
                if mapped and mapped != "none":
                    triggered_gesture = swipe_pose
                    triggered_action = mapped
                    self._last_trigger_time = now
                    self._last_trigger_gesture = swipe_pose
            elif stable_pose is not None:
                if stable_pose != self._hold_candidate:
                    self._hold_candidate = stable_pose
                    self._hold_start_time = now
                    self._hold_fired_for_current_pose = False

                if self._hold_start_time is not None:
                    elapsed = now - self._hold_start_time
                    dwell_progress = min(1.0, elapsed / max(0.10, req_hold))
                    # Repeatable actions (volume/scroll) can repeat smoothly while held!
                    mapped = mappings.get(stable_pose, "none")
                    repeatable = mapped in ("volume_up", "volume_down", "scroll_up", "scroll_down")
                    cooldown = 0.35 if repeatable else self.config.gesture_cooldown_seconds

                    if (
                        elapsed >= req_hold
                        and (not self._hold_fired_for_current_pose or repeatable)
                        and (now - self._last_trigger_time) >= cooldown
                    ):
                        if mapped and mapped != "none":
                            triggered_gesture = stable_pose
                            triggered_action = mapped
                            self._last_trigger_time = now
                            self._last_trigger_gesture = stable_pose
                            self._hold_fired_for_current_pose = True
            else:
                self._hold_candidate = None
                self._hold_start_time = None
                self._hold_fired_for_current_pose = False

        meta = GESTURE_CATALOG.get(active_pose or "", {})
        rem_s = (
            max(0.0, self._clutch_unlock_expiry - now)
            if (mode == "clutch" and self._clutch_unlocked)
            else 0.0
        )

        return GestureEvaluation(
            raw_gesture=static_pose,
            gesture=active_pose,
            gesture_name=meta.get("name", "None") if active_pose else "None",
            gesture_icon=meta.get("icon", "🤚") if active_pose else "🤚",
            confidence=round(active_conf, 3),
            extended_fingers=ext_fingers,
            extended_count=len(ext_fingers),
            pinch_ratio=round(pinch_pu, 3),
            is_pinching=self._pinch_active,
            clutch_unlocked=(mode != "clutch") or self._clutch_unlocked,
            clutch_remaining_s=round(rem_s, 1),
            dwell_progress=round(dwell_progress, 3),
            triggered_gesture=triggered_gesture,
            triggered_action=triggered_action,
            cursor_xy=cursor_xy,
            palm_center_norm=(round(palm_nx, 4), round(palm_ny, 4)),
            palm_size_norm=round(palm_size_iso, 4),
        )
