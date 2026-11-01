"""
Cross-Platform OS Action Controller ("PowerPoint and More").

Executes real system-level commands across Windows, macOS, and Linux:
  - Presentations (PowerPoint, Keynote, Google Slides, LibreOffice Impress, PDFs)
  - Media & System Volume (Play/Pause, Next/Prev Track, Volume Up/Down/Mute)
  - Air Mouse & Reader (Smooth 1-Euro cursor positioning, Pinch Click/Drag, Scroll, Browser Back/Fwd)

Lazily and safely initializes GUI automation backends so headless servers, CI tests,
and cloud Web Studio previews never crash when no X11/Wayland/Desktop display is attached.
"""

from __future__ import annotations

import os
import platform
import subprocess
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple

from .config import ACTION_CATALOG, SystemConfig


@dataclass
class ActionRecord:
    timestamp: float
    action: str
    label: str
    gesture: str
    profile: str
    executed_os: bool


class OSActionController:
    """Dispatches recognized gesture actions to the host OS safely and with minimal latency."""

    def __init__(self, config: Optional[SystemConfig] = None) -> None:
        self.config = config or SystemConfig()
        self.os_name = platform.system()
        self._pyautogui = None
        self._gui_available = False
        self._screen_size: Tuple[int, int] = (1920, 1080)
        self._laser_pointer_active = False
        self._mouse_dragging = False
        self.history: Deque[ActionRecord] = deque(maxlen=40)
        self._probe_gui_backend()

    def _probe_gui_backend(self) -> None:
        # Avoid importing pyautogui on headless Linux without DISPLAY / WAYLAND_DISPLAY
        if self.os_name == "Linux" and not (
            os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        ):
            self._gui_available = False
            return

        try:
            import pyautogui  # type: ignore

            pyautogui.FAILSAFE = True
            pyautogui.PAUSE = 0.01  # Minimal pause so main loop never stalls 100ms!
            w, h = pyautogui.size()
            self._screen_size = (int(w), int(h))
            self._pyautogui = pyautogui
            self._gui_available = True
        except Exception:
            self._pyautogui = None
            self._gui_available = False

    @property
    def gui_available(self) -> bool:
        return self._gui_available

    def _win_vk(self, vk_code: int) -> bool:
        if self.os_name != "Windows":
            return False
        try:
            import ctypes

            user32 = ctypes.windll.user32  # type: ignore[attr-defined]
            user32.keybd_event(vk_code, 0, 0, 0)
            user32.keybd_event(vk_code, 0, 2, 0)
            return True
        except Exception:
            return False

    def _mac_volume(self, delta: int = 0, mute_toggle: bool = False) -> bool:
        if self.os_name != "Darwin":
            return False
        try:
            if mute_toggle:
                script = (
                    "set curMuted to output muted of (get volume settings)\n"
                    "set volume output muted (not curMuted)"
                )
            else:
                script = (
                    f"set curVol to output volume of (get volume settings)\n"
                    f"set volume output volume (curVol + ({delta}))"
                )
            subprocess.Popen(
                ["osascript", "-e", script],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True
        except Exception:
            return False

    def update_cursor(
        self, cursor_xy: Optional[Tuple[float, float]], is_pinching: bool
    ) -> None:
        """
        Smoothly update OS mouse cursor when Air Mouse profile OR Laser Pointer mode is active.
        Supports continuous Pinch-and-Drag for drawing/highlighting on slides!
        """
        profile_data = self.config.get_active_profile_data()
        cursor_active = bool(profile_data.get("cursor_enabled", False)) or self._laser_pointer_active
        if not cursor_active or cursor_xy is None:
            if self._mouse_dragging and self._gui_available and self._pyautogui is not None:
                try:
                    self._pyautogui.mouseUp()
                except Exception:
                    pass
                self._mouse_dragging = False
            return

        if not (self.config.os_control_enabled and self._gui_available and self._pyautogui is not None):
            return

        sw, sh = self._screen_size
        margin = 6
        tx = int(max(margin, min(sw - margin, round(cursor_xy[0] * sw))))
        ty = int(max(margin, min(sh - margin, round(cursor_xy[1] * sh))))
        try:
            self._pyautogui.moveTo(tx, ty, _pause=False)
            if profile_data.get("cursor_enabled", False):
                if is_pinching and not self._mouse_dragging:
                    self._pyautogui.mouseDown(_pause=False)
                    self._mouse_dragging = True
                elif not is_pinching and self._mouse_dragging:
                    self._pyautogui.mouseUp(_pause=False)
                    self._mouse_dragging = False
        except Exception:
            pass

    def execute(
        self, action: str, gesture: str = "manual"
    ) -> Optional[Dict[str, Any]]:
        """Execute `action` and return a structured event dict for HUD & WebSocket clients."""
        if not action or action == "none":
            return None

        now = time.time()
        meta = ACTION_CATALOG.get(action, {"label": action, "keys": action})
        label = meta["label"]

        # Handle internal system/profile meta-actions first
        if action == "cycle_profile":
            new_prof = self.config.cycle_profile()
            prof_name = self.config.get_active_profile_data().get("name", new_prof)
            label = f"Mode -> {prof_name}"
            rec = ActionRecord(
                timestamp=now,
                action=action,
                label=label,
                gesture=gesture,
                profile=self.config.active_profile,
                executed_os=True,
            )
            self.history.appendleft(rec)
            return self._record_to_dict(rec)

        if action == "lock_clutch":
            rec = ActionRecord(
                timestamp=now,
                action=action,
                label="Gesture Window Locked",
                gesture=gesture,
                profile=self.config.active_profile,
                executed_os=True,
            )
            self.history.appendleft(rec)
            return self._record_to_dict(rec)

        if action == "toggle_pointer":
            self._laser_pointer_active = not self._laser_pointer_active
            label = (
                "Laser Pointer: ON"
                if self._laser_pointer_active
                else "Laser Pointer: OFF"
            )

        executed_os = False
        if self.config.os_control_enabled:
            executed_os = self._dispatch_os_command(action)

        rec = ActionRecord(
            timestamp=now,
            action=action,
            label=label,
            gesture=gesture,
            profile=self.config.active_profile,
            executed_os=executed_os,
        )
        self.history.appendleft(rec)
        return self._record_to_dict(rec)

    def _dispatch_os_command(self, action: str) -> bool:
        pag = self._pyautogui if self._gui_available else None

        try:
            # Hardware volume / media shortcuts with OS-native acceleration
            if action == "volume_up":
                if self._win_vk(0xAF) or self._mac_volume(delta=8):
                    return True
                if pag:
                    pag.press("volumeup")
                    return True
                return False

            if action == "volume_down":
                if self._win_vk(0xAE) or self._mac_volume(delta=-8):
                    return True
                if pag:
                    pag.press("volumedown")
                    return True
                return False

            if action == "volume_mute":
                if self._win_vk(0xAD) or self._mac_volume(mute_toggle=True):
                    return True
                if pag:
                    pag.press("volumemute")
                    return True
                return False

            if action == "next_track":
                if self._win_vk(0xB0):
                    return True
                if pag:
                    pag.press("nexttrack")
                    return True
                return False

            if action == "prev_track":
                if self._win_vk(0xB1):
                    return True
                if pag:
                    pag.press("prevtrack")
                    return True
                return False

            if pag is None:
                return False

            mod_key = "command" if self.os_name == "Darwin" else "ctrl"

            if action == "next_slide":
                pag.press("right")
            elif action == "prev_slide":
                pag.press("left")
            elif action == "play_pause":
                pag.press("space")
            elif action == "start_presentation":
                pag.press("f5")
            elif action == "exit_presentation":
                pag.press("esc")
            elif action == "black_screen":
                pag.press("b")
            elif action == "toggle_pointer":
                pag.hotkey(mod_key, "l")
            elif action == "zoom_in":
                pag.hotkey(mod_key, "+")
            elif action == "zoom_out":
                pag.hotkey(mod_key, "-")
            elif action == "fullscreen":
                pag.press("f")
            elif action == "left_click":
                if not self._mouse_dragging:
                    pag.click()
            elif action == "right_click":
                pag.rightClick()
            elif action == "scroll_up":
                pag.scroll(320)
            elif action == "scroll_down":
                pag.scroll(-320)
            elif action == "browser_back":
                pag.hotkey("command" if self.os_name == "Darwin" else "alt", "left")
            elif action == "browser_forward":
                pag.hotkey("command" if self.os_name == "Darwin" else "alt", "right")
            else:
                return False
            return True
        except Exception:
            return False

    @staticmethod
    def _record_to_dict(rec: ActionRecord) -> Dict[str, Any]:
        return {
            "timestamp": rec.timestamp,
            "time_str": time.strftime("%H:%M:%S", time.localtime(rec.timestamp)),
            "action": rec.action,
            "label": rec.label,
            "gesture": rec.gesture,
            "profile": rec.profile,
            "executed_os": rec.executed_os,
        }

    def recent_history(self, limit: int = 12) -> List[Dict[str, Any]]:
        return [self._record_to_dict(r) for r in list(self.history)[:limit]]
