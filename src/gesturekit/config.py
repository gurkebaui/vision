"""Configuration and gesture profiles.

Everything tunable lives in YAML.  Profiles ship with the package, users can
drop overrides in ``~/.config/gesturekit/`` and switch between them at runtime
with a keypress or a gesture -- no code editing, which is what v1 required for
even a single shortcut change.
"""

from __future__ import annotations

import copy
import os
import sys
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Optional

import yaml

from .actions.router import Action
from .camera import CameraConfig
from .gestures.continuous import PointerConfig
from .gestures.dynamic import SwipeConfig
from .gestures.engine import GestureEngineConfig
from .tracker import TrackerConfig


def config_dir() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
        return base / "gesturekit"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "gesturekit"
    base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "gesturekit"


def builtin_profile_dir() -> Path:
    return Path(__file__).resolve().parent / "profiles"


@dataclass
class UIConfig:
    show_window: bool = True
    show_landmarks: bool = True
    show_hud: bool = True
    show_scores: bool = False
    show_trail: bool = True
    theme: str = "dark"
    window_title: str = "GestureKit"
    width: int = 0            # 0 = native camera size
    mirror_hint: bool = True
    fullscreen: bool = False


@dataclass
class AppConfig:
    """Top-level application configuration."""

    profile: str = "presentation"
    camera: CameraConfig = field(default_factory=CameraConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    engine: GestureEngineConfig = field(default_factory=GestureEngineConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    backend: str = "auto"
    dry_run: bool = False
    model: str = "hand_landmarker"
    model_path: Optional[str] = None
    allow_download: bool = True
    log_level: str = "INFO"

    # populated from the profile file
    bindings: dict = field(default_factory=dict)
    profile_description: str = ""

    def to_dict(self) -> dict:
        out = {}
        for f in fields(self):
            value = getattr(self, f.name)
            out[f.name] = asdict(value) if is_dataclass(value) else value
        return out


def _merge(base: dict, override: dict) -> dict:
    """Recursive dict merge; ``override`` wins."""
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def _apply(target: Any, data: dict) -> None:
    """Assign known dataclass fields from a dict, ignoring extras."""
    if not data:
        return
    valid = {f.name: f for f in fields(target)}
    for key, value in data.items():
        if key not in valid:
            continue
        current = getattr(target, key)
        if is_dataclass(current) and isinstance(value, dict):
            _apply(current, value)
        else:
            setattr(target, key, value)


def find_profile(name: str, relative_to: Optional[Path] = None) -> Optional[Path]:
    """Locate a profile by name or path.

    Search order: the directory of the profile that referenced it (so a user
    profile can ``extends:`` a sibling file), then the user profile dir, then
    the built-ins.
    """
    p = Path(name).expanduser()
    if p.suffix in (".yaml", ".yml") and p.is_file():
        return p

    directories = []
    if relative_to is not None:
        directories.append(Path(relative_to))
    directories += [config_dir() / "profiles", config_dir(), builtin_profile_dir()]

    for directory in directories:
        for ext in (".yaml", ".yml"):
            candidate = directory / f"{name}{ext}"
            if candidate.is_file():
                return candidate
    return None


def list_profiles() -> list[tuple[str, str, Path]]:
    """Every discoverable profile as ``(name, description, path)``."""
    seen: dict[str, tuple[str, Path]] = {}
    for directory in (builtin_profile_dir(), config_dir() / "profiles", config_dir()):
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.y*ml")):
            try:
                data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            except Exception:
                continue
            if "gestures" not in data and "extends" not in data:
                continue
            seen[path.stem] = (str(data.get("description", "")), path)
    return [(name, desc, path) for name, (desc, path) in sorted(seen.items())]


def load_profile(name: str, _depth: int = 0, _origin: Optional[Path] = None) -> dict:
    """Load a profile, following its ``extends:`` chain."""
    if _depth > 5:
        raise ValueError(
            f"Profile 'extends' chain is too deep at '{name}' -- is it circular?"
        )
    path = find_profile(name, relative_to=_origin)
    if path is None:
        available = ", ".join(n for n, _, _ in list_profiles()) or "(none found)"
        raise FileNotFoundError(f"Profile '{name}' not found. Available: {available}")

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parent = data.get("extends")
    if parent:
        base = load_profile(str(parent), _depth + 1, _origin=path.parent)
        data = _merge(base, data)
        data.pop("extends", None)
    data.setdefault("name", path.stem)
    return data


def parse_bindings(gestures: dict) -> dict[str, Action]:
    out: dict[str, Action] = {}
    for gesture, spec in (gestures or {}).items():
        try:
            out[str(gesture)] = Action.parse(spec)
        except Exception as exc:
            raise ValueError(f"Invalid binding for gesture '{gesture}': {exc}") from exc
    return out


def load_config(
    profile: str = "presentation",
    overrides: Optional[dict] = None,
    user_config: Optional[Path] = None,
) -> AppConfig:
    """Build an :class:`AppConfig` from profile + user config + CLI overrides."""
    cfg = AppConfig(profile=profile)

    # 1. user-wide config.yaml
    user_path = user_config or (config_dir() / "config.yaml")
    user_data: dict = {}
    if Path(user_path).is_file():
        user_data = yaml.safe_load(Path(user_path).read_text(encoding="utf-8")) or {}

    # 2. the profile itself
    profile_data = load_profile(profile)

    merged = _merge(user_data, {k: v for k, v in profile_data.items() if k != "gestures"})
    merged = _merge(merged, overrides or {})

    _apply(cfg, {k: v for k, v in merged.items() if k in {f.name for f in fields(cfg)}})
    for section, target in (
        ("camera", cfg.camera),
        ("tracker", cfg.tracker),
        ("ui", cfg.ui),
    ):
        _apply(target, merged.get(section, {}))

    engine_data = dict(merged.get("engine", {}))
    swipe_data = engine_data.pop("swipe", None)
    pointer_data = engine_data.pop("pointer", None)
    _apply(cfg.engine, engine_data)
    if swipe_data:
        cfg.engine.swipe = SwipeConfig(**{**asdict(cfg.engine.swipe), **swipe_data})
    if pointer_data:
        cfg.engine.pointer = PointerConfig(**{**asdict(cfg.engine.pointer), **pointer_data})

    cfg.profile = profile_data.get("name", profile)
    cfg.profile_description = str(profile_data.get("description", ""))
    cfg.bindings = parse_bindings(profile_data.get("gestures", {}))
    return cfg


def write_default_user_config(path: Optional[Path] = None) -> Path:
    """Write a commented starter config into the user config dir."""
    target = Path(path) if path else config_dir() / "config.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(_DEFAULT_USER_CONFIG, encoding="utf-8")
    return target


_DEFAULT_USER_CONFIG = """\
# GestureKit user configuration.
# Applies to every profile; a profile can still override any of it.
# Docs: https://github.com/gurkebaui/vision

camera:
  index: 0          # try 'gesturekit devices' to list cameras
  width: 1280
  height: 720
  fps: 30
  flip: true        # mirror the preview

tracker:
  num_hands: 2
  min_detection_confidence: 0.6
  delegate: cpu     # 'gpu' if your build supports it

engine:
  require_arm: false   # true = must show the arm gesture before commands fire
  arm_gesture: point_up
  arm_window: 3.0
  cooldown: 0.45       # seconds between repeats of the same gesture

ui:
  show_window: true
  show_landmarks: true
  show_scores: false   # live confidence bars, useful while tuning

backend: auto          # auto | pynput | ydotool | xdotool | pyautogui | dry-run
"""
