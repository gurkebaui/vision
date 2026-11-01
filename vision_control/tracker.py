"""
3-Tier Long-Range CPU Hand Tracker (0.3m to 5.5m+).

Why legacy hand tracking stopped around ~1m:
  Standard MediaPipe downscales the entire 1280x720 / 1920x1080 camera frame into a
  192x192 tensor (a 6.7x–10x resolution crush!). At 2.5m–4.5m away, a presenter's hand
  is ~40–75px in the raw frame, collapsing to ~6–8px inside the palm detector.

How LongRangeHandTracker solves both range AND CPU usage simultaneously:
  1. Tier 1 — Velocity-Predicted Temporal Hand ROI Crop (Locked Path, ~5–8ms CPU):
     Once a hand is acquired, subsequent frames crop a tight square patch around the
     velocity-predicted hand position directly from native sensor pixels (zero full-frame
     downsampling!) and run lightweight landmark inference only on that 224–256px patch.
  2. Tier 2 — Upper-Body Pose Wrist-Guided Zoom (Presenter Acquisition, 1.2m–5.5m+):
     Even at 5m away, a human upper body is large in frame. Lightweight Pose locates
     shoulders, elbows, and raised wrists on a tiny 256x192 scout image, rejects resting
     arms hanging down at the presenter's waist, and extracts a native high-resolution
     zoom crop around the active palm.
  3. Tier 3 — Multi-Zone High-Res Scout Tiles + Local ROI CLAHE Contrast Enhancement:
     Catches distant hands even when the speaker's torso is behind a podium/desk or
     backlit by a bright projector screen, spending <0.2ms on local crop contrast boost.

Supports both MediaPipe Solutions Lite (0.10.x) and MediaPipe Tasks Vision (0.10.x / 1.x+).
"""

from __future__ import annotations

import math
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import mediapipe as mp
import numpy as np

from .config import SystemConfig
from .filters import LandmarkSmoother

MODEL_CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache" / "models"
HAND_TASK_URLS: Tuple[str, ...] = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "https://raw.githubusercontent.com/tarikurrahmann/Control-pc-using-Hand-Gesture/"
    "main/hand_landmarker.task",
)
POSE_LITE_TASK_URLS: Tuple[str, ...] = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
)


@dataclass
class ROIBox:
    """Square pixel crop region [x1, y1, x2, y2] in full-frame coordinates."""

    x1: int
    y1: int
    x2: int
    y2: int
    source: str = "full"  # "temporal_roi" | "pose_wrist" | "scout_tile" | "full_frame"

    @property
    def width(self) -> int:
        return max(1, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(1, self.y2 - self.y1)

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x1 + self.x2) * 0.5, (self.y1 + self.y2) * 0.5

    def zoom_factor(self, frame_w: int, frame_h: int) -> float:
        base = float(min(frame_w, frame_h))
        return round(max(1.0, base / float(max(1, self.width))), 2)

    def to_norm_tuple(self, frame_w: int, frame_h: int) -> Tuple[float, float, float, float]:
        return (
            self.x1 / max(1, frame_w),
            self.y1 / max(1, frame_h),
            self.x2 / max(1, frame_w),
            self.y2 / max(1, frame_h),
        )


@dataclass
class HandTrackingResult:
    detected: bool = False
    landmarks: List[Tuple[float, float, float]] = field(default_factory=list)
    raw_landmarks: List[Tuple[float, float, float]] = field(default_factory=list)
    handedness: str = "Right"
    score: float = 0.0
    roi: Optional[ROIBox] = None
    roi_crop_bgr: Optional[np.ndarray] = None
    zoom_factor: float = 1.0
    estimated_distance_m: float = 0.0
    palm_px: float = 0.0
    inference_ms: float = 0.0
    tracking_source: str = "none"
    pose_wrist_hint: Optional[Tuple[float, float]] = None


def make_square_roi(
    cx_px: float,
    cy_px: float,
    side_px: float,
    frame_w: int,
    frame_h: int,
    min_side: int = 128,
    source: str = "temporal_roi",
) -> ROIBox:
    """Create a clamped, guaranteed in-bounds square ROIBox."""
    max_side = min(frame_w, frame_h)
    side = int(max(min(min_side, max_side), min(round(side_px), max_side)))
    x1 = int(round(cx_px - side * 0.5))
    y1 = int(round(cy_px - side * 0.5))
    x1 = max(0, min(x1, frame_w - side))
    y1 = max(0, min(y1, frame_h - side))
    return ROIBox(x1=x1, y1=y1, x2=x1 + side, y2=y1 + side, source=source)


def unproject_crop_landmarks(
    crop_landmarks: Sequence[Tuple[float, float, float]],
    roi: ROIBox,
    frame_w: int,
    frame_h: int,
) -> List[Tuple[float, float, float]]:
    """Map crop-normalized [0..1] landmarks back into full-frame normalized [0..1] space."""
    rw = float(roi.width)
    rh = float(roi.height)
    fw = float(max(1, frame_w))
    fh = float(max(1, frame_h))
    z_scale = rw / fw
    out: List[Tuple[float, float, float]] = []
    for lx, ly, lz in crop_landmarks:
        fx = (roi.x1 + lx * rw) / fw
        fy = (roi.y1 + ly * rh) / fh
        fz = lz * z_scale
        out.append((fx, fy, fz))
    return out


def enhance_low_light_crop(
    crop_bgr: np.ndarray, clahe: cv2.CLAHE
) -> np.ndarray:
    """
    Fast (<0.2ms on 256x256) contrast/luminance boost ONLY when a crop is dim or washed out.
    Dramatically improves distant hand landmark contrast against bright projectors/windows.
    """
    gray_sample = crop_bgr[::4, ::4, 1]  # Green channel fast luma proxy
    mean_luma = float(gray_sample.mean())
    std_luma = float(gray_sample.std())
    if mean_luma >= 88.0 and std_luma >= 34.0:
        return crop_bgr

    lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    l_eq = clahe.apply(l_chan)
    if mean_luma < 65.0:
        # Gentle gamma lift for dark rooms
        l_eq = cv2.addWeighted(l_eq, 1.15, l_eq, 0.0, 12.0)
    merged = cv2.merge((l_eq, a_chan, b_chan))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def _ensure_task_model(
    filename: str, urls: Sequence[str], optional: bool = False
) -> Optional[Path]:
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_CACHE_DIR / filename
    if path.exists() and path.stat().st_size > 500_000:
        return path

    if not optional:
        print(f"📥 Caching MediaPipe model '{filename}' (one-time download -> .cache/models/)...")
    timeout_s = 4 if optional else 14
    for url in urls:
        tmp = path.with_suffix(".downloading")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "VisionControl/2.0"})
            with urllib.request.urlopen(req, timeout=timeout_s) as resp, open(tmp, "wb") as out:
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    out.write(chunk)
            if tmp.exists() and tmp.stat().st_size > 500_000:
                tmp.replace(path)
                if not optional:
                    print(f"✅ Cached '{filename}' ({path.stat().st_size / 1e6:.1f} MB)")
                return path
        except Exception:
            if tmp.exists():
                try:
                    tmp.unlink()
                except Exception:
                    pass
    return None


class LongRangeHandTracker:
    """
    CPU-optimized 3-tier hand & upper-body tracker extending effective range from <1m to 5m+.
    """

    # Scout zone centers in normalized (cx, cy, size_ratio_of_short_side)
    SCOUT_ZONES: Tuple[Tuple[float, float, float], ...] = (
        (0.50, 0.42, 0.44),  # Center presenter chest/shoulder zone (3m–4.5m)
        (0.32, 0.42, 0.45),  # Left presenter zone
        (0.68, 0.42, 0.45),  # Right presenter zone
        (0.50, 0.36, 0.30),  # Tight center auditorium zoom (4m–5.5m+)
        (0.28, 0.36, 0.32),  # Tight left auditorium zoom
        (0.72, 0.36, 0.32),  # Tight right auditorium zoom
    )

    def __init__(
        self,
        config: Optional[SystemConfig] = None,
        backend_preference: str = "auto",
    ) -> None:
        self.config = config or SystemConfig()
        self.backend_preference = backend_preference
        self.smoother = LandmarkSmoother(num_landmarks=21, min_cutoff=1.45, beta=1.8)
        self._clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(4, 4))

        # Temporal ROI state
        self._locked_roi: Optional[ROIBox] = None
        self._prev_palm_px_center: Optional[Tuple[float, float]] = None
        self._vel_px: Tuple[float, float] = (0.0, 0.0)
        self._prev_hand_span_px: float = 180.0
        self._missed_locked_frames: int = 0
        self._frame_counter: int = 0
        self._scout_idx: int = 0
        self._last_pose_wrist_norm: Optional[Tuple[float, float]] = None
        self._video_ts_ms: int = 0

        # Backend initialization: works seamlessly on BOTH MediaPipe 1.x+ (mp.tasks.vision,
        # where mp.solutions was removed) AND MediaPipe 0.10.x (mp.solutions Lite).
        self._backend_type: str = "none"
        self._mp_hands_track = None
        self._mp_hands_detect = None
        self._mp_pose = None
        self._tasks_hand = None
        self._tasks_hand_track = None
        self._tasks_pose = None
        self._init_backends()

    def _next_video_ts_ms(self) -> int:
        now_ms = int(time.monotonic() * 1000.0)
        self._video_ts_ms = max(self._video_ts_ms + 1, now_ms)
        return self._video_ts_ms

    def _init_backends(self) -> None:
        preset = self.config.get_preset()
        use_solutions = (
            self.backend_preference in ("auto", "solutions")
            and hasattr(mp, "solutions")
            and hasattr(mp.solutions, "hands")
        )
        if use_solutions:
            # MediaPipe 0.10.x path: uses bundled XNNPACK Lite models with zero network needed!
            self._backend_type = "solutions_lite"
            self._mp_hands_track = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                model_complexity=preset.model_complexity,
                min_detection_confidence=self.config.min_detection_confidence,
                min_tracking_confidence=self.config.min_tracking_confidence,
            )
            self._mp_hands_detect = mp.solutions.hands.Hands(
                static_image_mode=True,
                max_num_hands=1,
                model_complexity=preset.model_complexity,
                min_detection_confidence=max(0.28, self.config.min_detection_confidence - 0.05),
            )
            try:
                self._mp_pose = mp.solutions.pose.Pose(
                    static_image_mode=False,
                    model_complexity=1,
                    smooth_landmarks=True,
                    enable_segmentation=False,
                    min_detection_confidence=0.38,
                    min_tracking_confidence=0.35,
                )
            except Exception:
                self._mp_pose = None
            return

        # MediaPipe 0.10.30+ / 1.x+ Tasks Vision path (no mp.solutions required!)
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._backend_type = "tasks_vision"
        hand_model = _ensure_task_model("hand_landmarker.task", HAND_TASK_URLS)
        if hand_model is None:
            raise RuntimeError(
                "Could not download hand_landmarker.task on first run. "
                "Check internet access once to cache .cache/models/hand_landmarker.task."
            )

        # IMAGE runner for discontinuous scout / acquisition crops
        detect_opts = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(hand_model)),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=max(0.28, self.config.min_detection_confidence - 0.05),
            min_hand_presence_confidence=self.config.min_tracking_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
        )
        self._tasks_hand = vision.HandLandmarker.create_from_options(detect_opts)

        # VIDEO runner for continuous Tier-1 ROI tracking (skips palm detector on locked frames!)
        track_opts = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(hand_model)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=self.config.min_detection_confidence,
            min_hand_presence_confidence=self.config.min_tracking_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
        )
        self._tasks_hand_track = vision.HandLandmarker.create_from_options(track_opts)

        pose_model = (
            _ensure_task_model("pose_landmarker_lite.task", POSE_LITE_TASK_URLS, optional=True)
            if self.config.pose_assist_enabled
            else None
        )
        if pose_model is not None:
            try:
                pose_opts = vision.PoseLandmarkerOptions(
                    base_options=mp_python.BaseOptions(model_asset_path=str(pose_model)),
                    running_mode=vision.RunningMode.IMAGE,
                    num_poses=1,
                    min_pose_detection_confidence=0.38,
                    min_pose_presence_confidence=0.35,
                    min_tracking_confidence=0.35,
                )
                self._tasks_pose = vision.PoseLandmarker.create_from_options(pose_opts)
            except Exception:
                self._tasks_pose = None

    def reset(self) -> None:
        self._locked_roi = None
        self._prev_palm_px_center = None
        self._vel_px = (0.0, 0.0)
        self._missed_locked_frames = 0
        self._last_pose_wrist_norm = None
        self.smoother.reset()

    def _infer_hand_on_crop(
        self,
        frame_bgr: np.ndarray,
        roi: ROIBox,
        target_size: int,
        is_continuous_track: bool,
        enable_clahe: bool,
    ) -> Tuple[Optional[List[Tuple[float, float, float]]], str, float, np.ndarray]:
        """Extract ROI crop, optionally boost contrast, run inference, return crop landmarks."""
        crop_bgr = frame_bgr[roi.y1 : roi.y2, roi.x1 : roi.x2]
        ch, cw = crop_bgr.shape[:2]
        if cw != target_size or ch != target_size:
            interp = cv2.INTER_AREA if cw > target_size else cv2.INTER_LINEAR
            proc_bgr = cv2.resize(crop_bgr, (target_size, target_size), interpolation=interp)
        else:
            proc_bgr = crop_bgr

        if enable_clahe:
            proc_bgr = enhance_low_light_crop(proc_bgr, self._clahe)

        rgb = cv2.cvtColor(proc_bgr, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)

        if self._backend_type == "solutions_lite":
            engine = self._mp_hands_track if is_continuous_track else self._mp_hands_detect
            res = engine.process(rgb)
            if not res.multi_hand_landmarks:
                return None, "Right", 0.0, proc_bgr
            hlm = res.multi_hand_landmarks[0]
            lms = [(float(p.x), float(p.y), float(p.z)) for p in hlm.landmark]
            handedness = "Right"
            score = 0.88
            if res.multi_handedness:
                cls = res.multi_handedness[0].classification[0]
                handedness = cls.label or "Right"
                score = float(cls.score)
            return lms, handedness, score, proc_bgr

        # Tasks Vision API (VIDEO mode on continuous locked crops, IMAGE mode on scout jumps)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        if is_continuous_track and self._tasks_hand_track is not None:
            res = self._tasks_hand_track.detect_for_video(mp_img, self._next_video_ts_ms())
        else:
            res = self._tasks_hand.detect(mp_img)
        if not res.hand_landmarks:
            return None, "Right", 0.0, proc_bgr
        lms = [(float(p.x), float(p.y), float(p.z)) for p in res.hand_landmarks[0]]
        handedness = "Right"
        score = 0.88
        if res.handedness and res.handedness[0]:
            handedness = res.handedness[0][0].category_name or "Right"
            score = float(res.handedness[0][0].score)
        return lms, handedness, score, proc_bgr

    def _locate_active_wrist_rois(
        self, frame_bgr: np.ndarray
    ) -> List[ROIBox]:
        """
        Run lightweight upper-body Pose on a tiny 256x192 image to locate active/raised hands
        at 1.2m–5.5m+ and reject resting arms hanging down by the waist.
        """
        fh, fw = frame_bgr.shape[:2]
        small_bgr = cv2.resize(frame_bgr, (256, 192), interpolation=cv2.INTER_AREA)
        small_rgb = np.ascontiguousarray(cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB))

        pose_lms = None
        if self._backend_type == "solutions_lite" and self._mp_pose is not None:
            pres = self._mp_pose.process(small_rgb)
            if pres.pose_landmarks:
                pose_lms = [
                    (
                        float(p.x),
                        float(p.y),
                        float(getattr(p, "visibility", 1.0)),
                    )
                    for p in pres.pose_landmarks.landmark
                ]
        elif self._backend_type == "tasks_vision" and self._tasks_pose is not None:
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=small_rgb)
            pres = self._tasks_pose.detect(mp_img)
            if pres.pose_landmarks:
                pose_lms = [
                    (
                        float(p.x),
                        float(p.y),
                        float(getattr(p, "visibility", 1.0) or 1.0),
                    )
                    for p in pres.pose_landmarks[0]
                ]

        if not pose_lms or len(pose_lms) < 25:
            return []

        l_sh, r_sh = pose_lms[11], pose_lms[12]
        l_hip, r_hip = pose_lms[23], pose_lms[24]

        shoulder_px = math.hypot((l_sh[0] - r_sh[0]) * fw, (l_sh[1] - r_sh[1]) * fh)
        if shoulder_px < 18.0:
            shoulder_px = min(fw, fh) * 0.18

        mid_sh_y = (l_sh[1] + r_sh[1]) * 0.5
        if l_hip[2] > 0.35 and r_hip[2] > 0.35:
            torso_h = max(0.12, ((l_hip[1] + r_hip[1]) * 0.5) - mid_sh_y)
        else:
            torso_h = max(0.16, (shoulder_px / max(1.0, fh)) * 1.45)

        resting_cutoff_y = mid_sh_y + torso_h * 0.92

        candidates: List[Tuple[float, ROIBox, Tuple[float, float]]] = []
        # (elbow_idx, wrist_idx, pinky_idx, index_idx)
        arm_specs = [(13, 15, 17, 19), (14, 16, 18, 20)]
        for elb_i, wri_i, pnk_i, idx_i in arm_specs:
            elb = pose_lms[elb_i]
            wri = pose_lms[wri_i]
            pnk = pose_lms[pnk_i]
            idx = pose_lms[idx_i]

            if wri[2] < 0.32:
                continue
            if not (-0.05 <= wri[0] <= 1.05 and -0.05 <= wri[1] <= 1.05):
                continue

            # Resting-arm filter: ignore an arm hanging relaxed down by the waist/hip
            if self.config.resting_arm_rejection and wri[1] > resting_cutoff_y:
                continue

            if idx[2] > 0.35 and pnk[2] > 0.35:
                palm_nx = 0.45 * wri[0] + 0.275 * idx[0] + 0.275 * pnk[0]
                palm_ny = 0.45 * wri[1] + 0.275 * idx[1] + 0.275 * pnk[1]
            elif elb[2] > 0.30:
                palm_nx = wri[0] + 0.34 * (wri[0] - elb[0])
                palm_ny = wri[1] + 0.34 * (wri[1] - elb[1])
            else:
                palm_nx, palm_ny = wri[0], wri[1]

            cx_px = palm_nx * fw
            cy_px = palm_ny * fh
            side_px = max(128.0, min(min(fw, fh) * 0.65, shoulder_px * 1.48))
            roi = make_square_roi(
                cx_px, cy_px, side_px, fw, fh, min_side=120, source="pose_wrist"
            )
            # Prioritize higher raised hands (smaller y) and higher visibility
            priority = wri[1] - 0.15 * wri[2]
            candidates.append((priority, roi, (palm_nx, palm_ny)))

        candidates.sort(key=lambda item: item[0])
        if candidates:
            self._last_pose_wrist_norm = candidates[0][2]
        return [item[1] for item in candidates]

    def process_frame(
        self, frame_bgr: np.ndarray, timestamp: Optional[float] = None
    ) -> HandTrackingResult:
        """
        Track hand landmarks in `frame_bgr` across 0.3m–5.5m+ range with minimal CPU overhead.
        """
        t_start = time.perf_counter()
        now = timestamp if timestamp is not None else t_start
        fh, fw = frame_bgr.shape[:2]
        preset = self.config.get_preset()
        roi_target = preset.roi_size
        enable_clahe = preset.enable_clahe
        self._frame_counter += 1

        detected_crop_lms: Optional[List[Tuple[float, float, float]]] = None
        handedness = "Right"
        score = 0.0
        active_roi: Optional[ROIBox] = None
        crop_preview: Optional[np.ndarray] = None

        # =====================================================================
        # TIER 1: Velocity-Predicted Temporal Hand ROI (Fast Locked Path)
        # =====================================================================
        if self.config.smart_zoom_enabled and self._locked_roi is not None:
            if self._prev_palm_px_center is not None:
                pred_cx = self._prev_palm_px_center[0] + self._vel_px[0] * 0.8
                pred_cy = self._prev_palm_px_center[1] + self._vel_px[1] * 0.8
            else:
                pred_cx, pred_cy = self._locked_roi.center

            speed_px = math.hypot(self._vel_px[0], self._vel_px[1])
            pad_mult = min(
                3.9,
                2.45
                + (speed_px / max(28.0, self._prev_hand_span_px)) * 0.95
                + self._missed_locked_frames * 0.65,
            )
            desired_side = max(136.0, self._prev_hand_span_px * pad_mult)
            candidate_roi = make_square_roi(
                pred_cx,
                pred_cy,
                desired_side,
                fw,
                fh,
                min_side=128,
                source="temporal_roi",
            )
            detected_crop_lms, handedness, score, crop_preview = self._infer_hand_on_crop(
                frame_bgr,
                candidate_roi,
                target_size=roi_target,
                is_continuous_track=(self._missed_locked_frames == 0),
                enable_clahe=enable_clahe,
            )
            if detected_crop_lms is not None:
                active_roi = candidate_roi
                self._missed_locked_frames = 0
            else:
                self._missed_locked_frames += 1
                if self._missed_locked_frames >= 2:
                    self._locked_roi = None
                    self._vel_px = (0.0, 0.0)

        # =====================================================================
        # TIER 2 & 3: Long-Range Acquisition (when unlocked or lost)
        # =====================================================================
        if detected_crop_lms is None:
            # 2A: Quick downscaled square full-frame check for close/medium hands (0.3m–1.5m)
            if (not self.config.smart_zoom_enabled) or (self._frame_counter % 2 == 1):
                full_roi = ROIBox(0, 0, fw, fh, source="full_frame")
                small = cv2.resize(frame_bgr, (256, 256), interpolation=cv2.INTER_AREA)
                if enable_clahe:
                    small = enhance_low_light_crop(small, self._clahe)
                rgb_small = np.ascontiguousarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))

                if self._backend_type == "solutions_lite":
                    res = self._mp_hands_detect.process(rgb_small)
                    if res.multi_hand_landmarks:
                        detected_crop_lms = [
                            (float(p.x), float(p.y), float(p.z))
                            for p in res.multi_hand_landmarks[0].landmark
                        ]
                        if res.multi_handedness:
                            cls = res.multi_handedness[0].classification[0]
                            handedness = cls.label or "Right"
                            score = float(cls.score)
                        else:
                            score = 0.85
                        active_roi = full_roi
                else:
                    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_small)
                    res = self._tasks_hand.detect(mp_img)
                    if res.hand_landmarks:
                        detected_crop_lms = [
                            (float(p.x), float(p.y), float(p.z))
                            for p in res.hand_landmarks[0]
                        ]
                        if res.handedness and res.handedness[0]:
                            handedness = res.handedness[0][0].category_name or "Right"
                            score = float(res.handedness[0][0].score)
                        else:
                            score = 0.85
                        active_roi = full_roi

            # 2B: Upper-Body Pose Wrist-Guided Zoom (1.2m–5.5m+ Presenter Lock!)
            if (
                detected_crop_lms is None
                and self.config.smart_zoom_enabled
                and self.config.pose_assist_enabled
                and (self._frame_counter % max(1, preset.pose_scout_interval) == 0)
            ):
                wrist_rois = self._locate_active_wrist_rois(frame_bgr)
                for w_roi in wrist_rois[:2]:
                    lms, h_label, h_score, preview = self._infer_hand_on_crop(
                        frame_bgr,
                        w_roi,
                        target_size=roi_target,
                        is_continuous_track=False,
                        enable_clahe=enable_clahe,
                    )
                    if lms is not None:
                        detected_crop_lms = lms
                        handedness = h_label
                        score = h_score
                        active_roi = w_roi
                        crop_preview = preview
                        break

            # 2C: Multi-Zone High-Res Scout Tile (catches distant hands behind podiums/desks!)
            if (
                detected_crop_lms is None
                and self.config.smart_zoom_enabled
                and (self._frame_counter % max(1, preset.tile_scout_interval) == 0)
            ):
                zx, zy, zratio = self.SCOUT_ZONES[self._scout_idx % len(self.SCOUT_ZONES)]
                self._scout_idx += 1
                short_side = float(min(fw, fh))
                scout_roi = make_square_roi(
                    zx * fw,
                    zy * fh,
                    short_side * zratio,
                    fw,
                    fh,
                    min_side=128,
                    source="scout_tile",
                )
                lms, h_label, h_score, preview = self._infer_hand_on_crop(
                    frame_bgr,
                    scout_roi,
                    target_size=roi_target,
                    is_continuous_track=False,
                    enable_clahe=enable_clahe,
                )
                if lms is not None:
                    detected_crop_lms = lms
                    handedness = h_label
                    score = h_score
                    active_roi = scout_roi
                    crop_preview = preview

        inference_ms = (time.perf_counter() - t_start) * 1000.0

        if detected_crop_lms is None or active_roi is None:
            return HandTrackingResult(
                detected=False,
                inference_ms=inference_ms,
                tracking_source="searching",
                pose_wrist_hint=self._last_pose_wrist_norm,
            )

        # Unproject crop landmarks -> full-frame normalized [0..1]
        raw_full_lms = unproject_crop_landmarks(detected_crop_lms, active_roi, fw, fh)
        smoothed_lms = self.smoother.filter(raw_full_lms, now)

        # Update temporal ROI tracker state for next frame
        xs_px = [p[0] * fw for p in raw_full_lms]
        ys_px = [p[1] * fh for p in raw_full_lms]
        min_x, max_x = min(xs_px), max(xs_px)
        min_y, max_y = min(ys_px), max(ys_px)
        span_px = max(max_x - min_x, max_y - min_y, 24.0)
        cx_px = (min_x + max_x) * 0.5
        cy_px = (min_y + max_y) * 0.5

        if self._prev_palm_px_center is not None:
            inst_vx = cx_px - self._prev_palm_px_center[0]
            inst_vy = cy_px - self._prev_palm_px_center[1]
            self._vel_px = (
                0.55 * self._vel_px[0] + 0.45 * inst_vx,
                0.55 * self._vel_px[1] + 0.45 * inst_vy,
            )
        else:
            self._vel_px = (0.0, 0.0)

        self._prev_palm_px_center = (cx_px, cy_px)
        self._prev_hand_span_px = 0.35 * self._prev_hand_span_px + 0.65 * span_px

        tight_roi = make_square_roi(
            cx_px,
            cy_px,
            max(136.0, self._prev_hand_span_px * 2.45),
            fw,
            fh,
            min_side=128,
            source=active_roi.source,
        )
        self._locked_roi = tight_roi

        if crop_preview is None:
            sub = frame_bgr[tight_roi.y1 : tight_roi.y2, tight_roi.x1 : tight_roi.x2]
            if sub.size > 0:
                crop_preview = cv2.resize(sub, (180, 180), interpolation=cv2.INTER_LINEAR)

        # Estimate real-world distance in meters from palm size in pixels
        # Average adult wrist-to-middle-MCP is ~8.8 cm; typical 720p webcam horizontal FOV ~65 deg -> focal_px ~ 0.78 * fw
        w_pt = smoothed_lms[0]
        m_mcp = smoothed_lms[9]
        i_mcp = smoothed_lms[5]
        p_mcp = smoothed_lms[17]
        palm_len_px = math.hypot((w_pt[0] - m_mcp[0]) * fw, (w_pt[1] - m_mcp[1]) * fh)
        palm_wid_px = math.hypot((i_mcp[0] - p_mcp[0]) * fw, (i_mcp[1] - p_mcp[1]) * fh)
        effective_palm_px = max(8.0, 0.65 * palm_len_px + 0.45 * palm_wid_px)
        focal_px = 0.78 * fw
        est_dist_m = round(min(8.0, max(0.25, (0.088 * focal_px) / effective_palm_px)), 2)

        return HandTrackingResult(
            detected=True,
            landmarks=smoothed_lms,
            raw_landmarks=raw_full_lms,
            handedness=handedness,
            score=score,
            roi=tight_roi,
            roi_crop_bgr=crop_preview,
            zoom_factor=tight_roi.zoom_factor(fw, fh),
            estimated_distance_m=est_dist_m,
            palm_px=round(effective_palm_px, 1),
            inference_ms=inference_ms,
            tracking_source=active_roi.source,
            pose_wrist_hint=self._last_pose_wrist_norm,
        )

    def close(self) -> None:
        for obj in (
            self._mp_hands_track,
            self._mp_hands_detect,
            self._mp_pose,
            self._tasks_hand,
            self._tasks_hand_track,
            self._tasks_pose,
        ):
            if obj is not None and hasattr(obj, "close"):
                try:
                    obj.close()
                except Exception:
                    pass
