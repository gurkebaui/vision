"""
Configuration, CPU performance presets, control profiles, and JSON persistence.
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "settings.json"


@dataclass
class PerformancePresetConfig:
    name: str
    label: str
    capture_width: int
    capture_height: int
    target_fps: int
    roi_size: int
    model_complexity: int
    pose_scout_interval: int
    tile_scout_interval: int
    enable_clahe: bool
    description: str


PERFORMANCE_PRESETS: Dict[str, PerformancePresetConfig] = {
    "ultra_light": PerformancePresetConfig(
        name="ultra_light",
        label="Ultra-Light CPU",
        capture_width=960,
        capture_height=540,
        target_fps=30,
        roi_size=224,
        model_complexity=0,
        pose_scout_interval=10,
        tile_scout_interval=8,
        enable_clahe=True,
        description="Minimal CPU footprint (<8% on dual/quad-core laptops), up to ~3.2m range.",
    ),
    "balanced": PerformancePresetConfig(
        name="balanced",
        label="Balanced Long-Range",
        capture_width=1280,
        capture_height=720,
        target_fps=30,
        roi_size=256,
        model_complexity=0,
        pose_scout_interval=5,
        tile_scout_interval=4,
        enable_clahe=True,
        description="Recommended for CPU laptops: fast Lite inference + Smart Zoom up to ~4.5m.",
    ),
    "max_range": PerformancePresetConfig(
        name="max_range",
        label="Max Auditorium Range",
        capture_width=1920,
        capture_height=1080,
        target_fps=30,
        roi_size=320,
        model_complexity=0,
        pose_scout_interval=3,
        tile_scout_interval=2,
        enable_clahe=True,
        description="1080p capture + aggressive Smart Zoom & contrast boost for 4m–6m+ rooms.",
    ),
}


# Canonical action identifiers understood by OSActionController and the Web Stage
ACTION_CATALOG: Dict[str, Dict[str, str]] = {
    "next_slide": {"label": "Next Slide", "keys": "Right", "category": "Presentation"},
    "prev_slide": {"label": "Previous Slide", "keys": "Left", "category": "Presentation"},
    "play_pause": {"label": "Play / Pause", "keys": "Space", "category": "Presentation / Media"},
    "start_presentation": {"label": "Start Slideshow", "keys": "F5", "category": "Presentation"},
    "exit_presentation": {"label": "Exit / Escape", "keys": "Escape", "category": "Presentation"},
    "black_screen": {"label": "Blackout Screen", "keys": "B", "category": "Presentation"},
    "toggle_pointer": {"label": "Laser Pointer Mode", "keys": "Ctrl+L", "category": "Presentation"},
    "zoom_in": {"label": "Zoom In", "keys": "Ctrl++", "category": "View"},
    "zoom_out": {"label": "Zoom Out", "keys": "Ctrl+-", "category": "View"},
    "volume_up": {"label": "Volume Up", "keys": "VolumeUp", "category": "Media"},
    "volume_down": {"label": "Volume Down", "keys": "VolumeDown", "category": "Media"},
    "volume_mute": {"label": "Mute Audio", "keys": "VolumeMute", "category": "Media"},
    "next_track": {"label": "Next Track", "keys": "MediaNext", "category": "Media"},
    "prev_track": {"label": "Previous Track", "keys": "MediaPrev", "category": "Media"},
    "fullscreen": {"label": "Toggle Fullscreen", "keys": "F", "category": "Media"},
    "left_click": {"label": "Left Click / Drag", "keys": "MouseLeft", "category": "Air Mouse"},
    "right_click": {"label": "Right Click", "keys": "MouseRight", "category": "Air Mouse"},
    "scroll_up": {"label": "Scroll Up", "keys": "ScrollUp", "category": "Air Mouse"},
    "scroll_down": {"label": "Scroll Down", "keys": "ScrollDown", "category": "Air Mouse"},
    "browser_back": {"label": "Back", "keys": "Alt+Left", "category": "Air Mouse"},
    "browser_forward": {"label": "Forward", "keys": "Alt+Right", "category": "Air Mouse"},
    "cycle_profile": {"label": "Switch Control Mode", "keys": "ModeCycle", "category": "System"},
    "lock_clutch": {"label": "Lock Gesture Window", "keys": "Lock", "category": "System"},
    "none": {"label": "Disabled", "keys": "-", "category": "System"},
}


GESTURE_CATALOG: Dict[str, Dict[str, str]] = {
    "swipe_right": {"name": "Swipe Right", "icon": "👉", "kind": "dynamic"},
    "swipe_left": {"name": "Swipe Left", "icon": "👈", "kind": "dynamic"},
    "swipe_up": {"name": "Swipe Up", "icon": "⬆️", "kind": "dynamic"},
    "swipe_down": {"name": "Swipe Down", "icon": "⬇️", "kind": "dynamic"},
    "open_palm": {"name": "Open Palm", "icon": "✋", "kind": "static"},
    "closed_fist": {"name": "Closed Fist", "icon": "✊", "kind": "static"},
    "pointing_up": {"name": "Point Up", "icon": "☝️", "kind": "static"},
    "pointing_down": {"name": "Point Down", "icon": "👇", "kind": "static"},
    "peace_sign": {"name": "Peace Sign", "icon": "✌️", "kind": "static"},
    "thumbs_up": {"name": "Thumbs Up", "icon": "👍", "kind": "static"},
    "thumbs_down": {"name": "Thumbs Down", "icon": "👎", "kind": "static"},
    "pinch": {"name": "Thumb-Index Pinch", "icon": "👌", "kind": "static"},
    "three_fingers": {"name": "Three Fingers", "icon": "🤟", "kind": "static"},
}


DEFAULT_PROFILES: Dict[str, Dict[str, Any]] = {
    "presentation": {
        "id": "presentation",
        "name": "Presentation Pro",
        "subtitle": "PowerPoint, Keynote, Google Slides & PDFs",
        "cursor_enabled": False,
        "mappings": {
            "swipe_right": "next_slide",
            "swipe_left": "prev_slide",
            "pointing_up": "next_slide",
            "pointing_down": "prev_slide",
            "open_palm": "play_pause",
            "peace_sign": "toggle_pointer",
            "thumbs_up": "zoom_in",
            "thumbs_down": "black_screen",
            "closed_fist": "lock_clutch",
            "pinch": "left_click",
            "three_fingers": "cycle_profile",
            "swipe_up": "start_presentation",
            "swipe_down": "exit_presentation",
        },
    },
    "media": {
        "id": "media",
        "name": "Media & Couch Control",
        "subtitle": "Spotify, YouTube, VLC & System Audio",
        "cursor_enabled": False,
        "mappings": {
            "open_palm": "play_pause",
            "swipe_right": "next_track",
            "swipe_left": "prev_track",
            "thumbs_up": "volume_up",
            "thumbs_down": "volume_down",
            "pointing_up": "volume_up",
            "pointing_down": "volume_down",
            "closed_fist": "volume_mute",
            "peace_sign": "fullscreen",
            "three_fingers": "cycle_profile",
            "swipe_up": "volume_up",
            "swipe_down": "volume_down",
            "pinch": "play_pause",
        },
    },
    "air_mouse": {
        "id": "air_mouse",
        "name": "Air Mouse & Reader",
        "subtitle": "1-Euro Smooth Cursor, Pinch Click/Drag & Scroll",
        "cursor_enabled": True,
        "mappings": {
            "pinch": "left_click",
            "peace_sign": "right_click",
            "pointing_up": "scroll_up",
            "pointing_down": "scroll_down",
            "swipe_up": "scroll_up",
            "swipe_down": "scroll_down",
            "swipe_left": "browser_back",
            "swipe_right": "browser_forward",
            "thumbs_up": "zoom_in",
            "thumbs_down": "zoom_out",
            "three_fingers": "cycle_profile",
            "open_palm": "none",
            "closed_fist": "lock_clutch",
        },
    },
    "custom": {
        "id": "custom",
        "name": "Custom Bindings",
        "subtitle": "Personalized Gesture-to-Action Profile",
        "cursor_enabled": False,
        "mappings": {
            "swipe_right": "next_slide",
            "swipe_left": "prev_slide",
            "pointing_up": "next_slide",
            "pointing_down": "prev_slide",
            "open_palm": "play_pause",
            "peace_sign": "toggle_pointer",
            "thumbs_up": "volume_up",
            "thumbs_down": "volume_down",
            "closed_fist": "exit_presentation",
            "pinch": "left_click",
            "three_fingers": "cycle_profile",
            "swipe_up": "scroll_up",
            "swipe_down": "scroll_down",
        },
    },
}


PROFILE_ORDER: List[str] = ["presentation", "media", "air_mouse", "custom"]


@dataclass
class SystemConfig:
    # Tracking & Range
    performance_preset: str = "balanced"
    camera_index: int = 0
    mirror_camera: bool = True
    smart_zoom_enabled: bool = True
    pose_assist_enabled: bool = True
    resting_arm_rejection: bool = True
    min_detection_confidence: float = 0.35
    min_tracking_confidence: float = 0.30

    # Activation & Control
    active_profile: str = "presentation"
    activation_mode: str = "clutch"  # "clutch" | "dwell" | "instant"
    clutch_wake_gesture: str = "open_palm"
    clutch_window_seconds: float = 4.0
    clutch_hold_seconds: float = 0.38
    dwell_hold_seconds: float = 0.42
    gesture_cooldown_seconds: float = 0.75
    swipe_sensitivity: float = 1.0  # Multiplier around 1.0 (0.5 = stiffer, 1.5 = lighter flick)
    cursor_sensitivity: float = 1.35
    cursor_smoothing_min_cutoff: float = 1.15
    cursor_smoothing_beta: float = 1.45

    # OS Execution & UI
    os_control_enabled: bool = True
    compact_hud: bool = False
    show_zoom_pip: bool = True
    profiles: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: copy.deepcopy(DEFAULT_PROFILES)
    )

    def get_preset(self) -> PerformancePresetConfig:
        return PERFORMANCE_PRESETS.get(
            self.performance_preset, PERFORMANCE_PRESETS["balanced"]
        )

    def get_active_profile_data(self) -> Dict[str, Any]:
        return self.profiles.get(
            self.active_profile, self.profiles["presentation"]
        )

    def cycle_profile(self) -> str:
        idx = (
            PROFILE_ORDER.index(self.active_profile)
            if self.active_profile in PROFILE_ORDER
            else 0
        )
        self.active_profile = PROFILE_ORDER[(idx + 1) % len(PROFILE_ORDER)]
        return self.active_profile

    def update_from_dict(self, updates: Dict[str, Any]) -> None:
        scalar_fields = {
            "performance_preset": str,
            "camera_index": int,
            "mirror_camera": bool,
            "smart_zoom_enabled": bool,
            "pose_assist_enabled": bool,
            "resting_arm_rejection": bool,
            "min_detection_confidence": float,
            "min_tracking_confidence": float,
            "active_profile": str,
            "activation_mode": str,
            "clutch_wake_gesture": str,
            "clutch_window_seconds": float,
            "clutch_hold_seconds": float,
            "dwell_hold_seconds": float,
            "gesture_cooldown_seconds": float,
            "swipe_sensitivity": float,
            "cursor_sensitivity": float,
            "cursor_smoothing_min_cutoff": float,
            "cursor_smoothing_beta": float,
            "os_control_enabled": bool,
            "compact_hud": bool,
            "show_zoom_pip": bool,
        }
        for key, caster in scalar_fields.items():
            if key in updates and updates[key] is not None:
                val = caster(updates[key])
                setattr(self, key, val)

        if self.performance_preset not in PERFORMANCE_PRESETS:
            self.performance_preset = "balanced"
        if self.active_profile not in self.profiles:
            self.active_profile = "presentation"
        if self.activation_mode not in ("clutch", "dwell", "instant"):
            self.activation_mode = "clutch"

        if "profiles" in updates and isinstance(updates["profiles"], dict):
            for pid, pdata in updates["profiles"].items():
                if pid in self.profiles and isinstance(pdata, dict):
                    if "mappings" in pdata and isinstance(pdata["mappings"], dict):
                        self.profiles[pid]["mappings"].update(pdata["mappings"])
                    if "cursor_enabled" in pdata:
                        self.profiles[pid]["cursor_enabled"] = bool(
                            pdata["cursor_enabled"]
                        )

        if "mapping_update" in updates and isinstance(updates["mapping_update"], dict):
            mu = updates["mapping_update"]
            pid = mu.get("profile", self.active_profile)
            gesture = mu.get("gesture")
            action = mu.get("action")
            if (
                pid in self.profiles
                and gesture in GESTURE_CATALOG
                and action in ACTION_CATALOG
            ):
                self.profiles[pid]["mappings"][gesture] = action

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["preset_details"] = asdict(self.get_preset())
        data["presets_available"] = {
            k: asdict(v) for k, v in PERFORMANCE_PRESETS.items()
        }
        data["action_catalog"] = ACTION_CATALOG
        data["gesture_catalog"] = GESTURE_CATALOG
        return data

    def save(self, path: Optional[Path] = None) -> Path:
        target = Path(path) if path else DEFAULT_CONFIG_PATH
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        tmp_path = target.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        os.replace(tmp_path, target)
        return target

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "SystemConfig":
        target = Path(path) if path else DEFAULT_CONFIG_PATH
        cfg = cls()
        if target.exists():
            try:
                with open(target, "r", encoding="utf-8") as fh:
                    raw = json.load(fh)
                if isinstance(raw, dict):
                    cfg.update_from_dict(raw)
            except Exception:
                pass
        return cfg
