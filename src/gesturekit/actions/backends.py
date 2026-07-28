"""Input backends.

v1 hard-depended on PyAutoGUI, which on Linux talks X11 only -- under a Wayland
session it either fails outright or silently does nothing.  Since the user's
target is Linux, this module ships four interchangeable backends and picks the
best available one at runtime:

===============  =========================================================
``pynput``       X11 + Windows + macOS. Best default; supports absolute
                 pointer moves and scrolling.
``ydotool``      Wayland-capable (uinput based). Needs the ydotool daemon.
``xdotool``      X11 CLI fallback, present on most desktops.
``pyautogui``    Legacy path, kept for compatibility with v1 setups.
``dry-run``      Logs what *would* happen. Always available, and what the
                 test-suite and ``--dry-run`` use.
===============  =========================================================

Every backend implements the same tiny interface, so nothing above this layer
knows or cares which one is live.
"""

from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess
from typing import Optional, Sequence

log = logging.getLogger("gesturekit.actions")


class InputBackend:
    """Abstract synthetic-input device."""

    name = "base"
    supports_absolute = False

    def available(self) -> bool:
        return False

    def key(self, key: str) -> None:
        raise NotImplementedError

    def hotkey(self, *keys: str) -> None:
        raise NotImplementedError

    def type_text(self, text: str) -> None:
        raise NotImplementedError

    def move_to(self, x: int, y: int) -> None:
        raise NotImplementedError

    def click(self, button: str = "left", count: int = 1) -> None:
        raise NotImplementedError

    def mouse_down(self, button: str = "left") -> None:
        raise NotImplementedError

    def mouse_up(self, button: str = "left") -> None:
        raise NotImplementedError

    def scroll(self, dx: int, dy: int) -> None:
        raise NotImplementedError

    def screen_size(self) -> tuple[int, int]:
        return (1920, 1080)

    def close(self) -> None:
        pass


class DryRunBackend(InputBackend):
    """Records intent instead of performing it. Safe everywhere."""

    name = "dry-run"
    supports_absolute = True

    def __init__(self, echo: bool = True, screen: tuple[int, int] = (1920, 1080)):
        self.echo = echo
        self.events: list[tuple] = []
        self._screen = screen

    def available(self) -> bool:
        return True

    def _record(self, *event) -> None:
        self.events.append(event)
        if self.echo:
            log.info("[dry-run] %s", " ".join(str(e) for e in event))

    def key(self, key: str) -> None:
        self._record("key", key)

    def hotkey(self, *keys: str) -> None:
        self._record("hotkey", "+".join(keys))

    def type_text(self, text: str) -> None:
        self._record("type", text)

    def move_to(self, x: int, y: int) -> None:
        self._record("move", x, y)

    def click(self, button: str = "left", count: int = 1) -> None:
        self._record("click", button, count)

    def mouse_down(self, button: str = "left") -> None:
        self._record("mouse_down", button)

    def mouse_up(self, button: str = "left") -> None:
        self._record("mouse_up", button)

    def scroll(self, dx: int, dy: int) -> None:
        self._record("scroll", dx, dy)

    def screen_size(self) -> tuple[int, int]:
        return self._screen


class PynputBackend(InputBackend):
    """pynput: the most capable cross-platform option."""

    name = "pynput"
    supports_absolute = True

    _KEYMAP = {
        "space": "space", "enter": "enter", "return": "enter", "tab": "tab",
        "esc": "esc", "escape": "esc", "backspace": "backspace", "delete": "delete",
        "up": "up", "down": "down", "left": "left", "right": "right",
        "home": "home", "end": "end", "pageup": "page_up", "pagedown": "page_down",
        "ctrl": "ctrl", "alt": "alt", "shift": "shift", "cmd": "cmd", "super": "cmd",
        "win": "cmd", "f1": "f1", "f2": "f2", "f3": "f3", "f4": "f4", "f5": "f5",
        "f6": "f6", "f7": "f7", "f8": "f8", "f9": "f9", "f10": "f10", "f11": "f11",
        "f12": "f12",
        "volumeup": "media_volume_up", "volumedown": "media_volume_down",
        "volumemute": "media_volume_mute", "playpause": "media_play_pause",
        "nexttrack": "media_next", "prevtrack": "media_previous",
    }

    def __init__(self):
        self._kb = None
        self._mouse = None
        self._Key = None

    def available(self) -> bool:
        try:
            from pynput import keyboard, mouse  # noqa: F401
        except Exception:
            return False
        try:
            from pynput.keyboard import Controller as KC
            from pynput.keyboard import Key
            from pynput.mouse import Controller as MC

            self._kb = KC()
            self._mouse = MC()
            self._Key = Key
            return True
        except Exception as exc:
            log.debug("pynput unavailable: %s", exc)
            return False

    def _resolve(self, key: str):
        k = key.strip().lower()
        mapped = self._KEYMAP.get(k, k)
        special = getattr(self._Key, mapped, None)
        if special is not None:
            return special
        return key if len(key) == 1 else mapped

    def key(self, key: str) -> None:
        k = self._resolve(key)
        self._kb.press(k)
        self._kb.release(k)

    def hotkey(self, *keys: str) -> None:
        resolved = [self._resolve(k) for k in keys]
        for k in resolved:
            self._kb.press(k)
        for k in reversed(resolved):
            self._kb.release(k)

    def type_text(self, text: str) -> None:
        self._kb.type(text)

    def move_to(self, x: int, y: int) -> None:
        self._mouse.position = (int(x), int(y))

    def _button(self, button: str):
        from pynput.mouse import Button

        return {"left": Button.left, "right": Button.right, "middle": Button.middle}.get(
            button, Button.left
        )

    def click(self, button: str = "left", count: int = 1) -> None:
        self._mouse.click(self._button(button), count)

    def mouse_down(self, button: str = "left") -> None:
        self._mouse.press(self._button(button))

    def mouse_up(self, button: str = "left") -> None:
        self._mouse.release(self._button(button))

    def scroll(self, dx: int, dy: int) -> None:
        self._mouse.scroll(int(dx), int(dy))

    def screen_size(self) -> tuple[int, int]:
        size = _screen_size_probe()
        return size or (1920, 1080)


class AutoGuiBackend(InputBackend):
    """PyAutoGUI, kept so existing v1 installs keep working."""

    name = "pyautogui"
    supports_absolute = True

    def __init__(self):
        self._pg = None

    def available(self) -> bool:
        try:
            import pyautogui

            pyautogui.FAILSAFE = False   # a corner-hit must not kill a live talk
            pyautogui.PAUSE = 0.0        # v1 paid 100 ms on every single call
            self._pg = pyautogui
            return True
        except Exception as exc:
            log.debug("pyautogui unavailable: %s", exc)
            return False

    def key(self, key: str) -> None:
        self._pg.press(key)

    def hotkey(self, *keys: str) -> None:
        self._pg.hotkey(*keys)

    def type_text(self, text: str) -> None:
        self._pg.typewrite(text)

    def move_to(self, x: int, y: int) -> None:
        self._pg.moveTo(int(x), int(y), _pause=False)

    def click(self, button: str = "left", count: int = 1) -> None:
        self._pg.click(button=button, clicks=count, _pause=False)

    def mouse_down(self, button: str = "left") -> None:
        self._pg.mouseDown(button=button, _pause=False)

    def mouse_up(self, button: str = "left") -> None:
        self._pg.mouseUp(button=button, _pause=False)

    def scroll(self, dx: int, dy: int) -> None:
        if dy:
            self._pg.scroll(int(dy))
        if dx:
            self._pg.hscroll(int(dx))

    def screen_size(self) -> tuple[int, int]:
        try:
            w, h = self._pg.size()
            return int(w), int(h)
        except Exception:
            return (1920, 1080)


class _CliBackend(InputBackend):
    """Shared plumbing for CLI-driven tools."""

    binary = ""

    def available(self) -> bool:
        return shutil.which(self.binary) is not None

    def _run(self, args: Sequence[str]) -> None:
        try:
            subprocess.run(
                [self.binary, *args],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=2.0,
            )
        except Exception as exc:
            log.debug("%s failed: %s", self.binary, exc)


class XdotoolBackend(_CliBackend):
    """X11 fallback via the xdotool binary."""

    name = "xdotool"
    binary = "xdotool"
    supports_absolute = True

    _BUTTONS = {"left": "1", "middle": "2", "right": "3"}

    def key(self, key: str) -> None:
        self._run(["key", "--clearmodifiers", _x_key(key)])

    def hotkey(self, *keys: str) -> None:
        self._run(["key", "--clearmodifiers", "+".join(_x_key(k) for k in keys)])

    def type_text(self, text: str) -> None:
        self._run(["type", "--clearmodifiers", text])

    def move_to(self, x: int, y: int) -> None:
        self._run(["mousemove", str(int(x)), str(int(y))])

    def click(self, button: str = "left", count: int = 1) -> None:
        self._run(["click", "--repeat", str(count), self._BUTTONS.get(button, "1")])

    def mouse_down(self, button: str = "left") -> None:
        self._run(["mousedown", self._BUTTONS.get(button, "1")])

    def mouse_up(self, button: str = "left") -> None:
        self._run(["mouseup", self._BUTTONS.get(button, "1")])

    def scroll(self, dx: int, dy: int) -> None:
        for _ in range(abs(int(dy))):
            self._run(["click", "4" if dy > 0 else "5"])
        for _ in range(abs(int(dx))):
            self._run(["click", "6" if dx < 0 else "7"])

    def screen_size(self) -> tuple[int, int]:
        return _screen_size_probe() or (1920, 1080)


class YdotoolBackend(_CliBackend):
    """Wayland-capable backend (uinput). Requires the ydotoold daemon."""

    name = "ydotool"
    binary = "ydotool"
    supports_absolute = True

    _BUTTONS = {"left": "0x40", "right": "0x41", "middle": "0x42"}

    def available(self) -> bool:
        if shutil.which(self.binary) is None:
            return False
        sock = os.environ.get("YDOTOOL_SOCKET", "/tmp/.ydotool_socket")
        return os.path.exists(sock) or os.access("/dev/uinput", os.W_OK)

    def key(self, key: str) -> None:
        self._run(["key", f"{_ydo_key(key)}:1", f"{_ydo_key(key)}:0"])

    def hotkey(self, *keys: str) -> None:
        codes = [_ydo_key(k) for k in keys]
        seq = [f"{c}:1" for c in codes] + [f"{c}:0" for c in reversed(codes)]
        self._run(["key", *seq])

    def type_text(self, text: str) -> None:
        self._run(["type", text])

    def move_to(self, x: int, y: int) -> None:
        self._run(["mousemove", "--absolute", "-x", str(int(x)), "-y", str(int(y))])

    def click(self, button: str = "left", count: int = 1) -> None:
        for _ in range(count):
            self._run(["click", self._BUTTONS.get(button, "0x40")])

    def mouse_down(self, button: str = "left") -> None:
        self._run(["click", self._BUTTONS.get(button, "0x40").replace("0x4", "0x8")])

    def mouse_up(self, button: str = "left") -> None:
        self._run(["click", self._BUTTONS.get(button, "0x40").replace("0x4", "0xC")])

    def scroll(self, dx: int, dy: int) -> None:
        self._run(["mousemove", "--wheel", "-x", str(int(dx)), "-y", str(int(dy))])


# ---------------------------------------------------------------------------
# Key-name translation
# ---------------------------------------------------------------------------

_X_KEYS = {
    "esc": "Escape", "escape": "Escape", "enter": "Return", "return": "Return",
    "space": "space", "tab": "Tab", "backspace": "BackSpace", "delete": "Delete",
    "up": "Up", "down": "Down", "left": "Left", "right": "Right",
    "pageup": "Prior", "pagedown": "Next", "home": "Home", "end": "End",
    "ctrl": "ctrl", "alt": "alt", "shift": "shift", "super": "super", "win": "super",
    "volumeup": "XF86AudioRaiseVolume", "volumedown": "XF86AudioLowerVolume",
    "volumemute": "XF86AudioMute", "playpause": "XF86AudioPlay",
    "nexttrack": "XF86AudioNext", "prevtrack": "XF86AudioPrev",
}

_YDO_KEYS = {
    "esc": "1", "escape": "1", "enter": "28", "return": "28", "space": "57",
    "tab": "15", "backspace": "14", "delete": "111", "up": "103", "down": "108",
    "left": "105", "right": "106", "pageup": "104", "pagedown": "109",
    "home": "102", "end": "107", "ctrl": "29", "alt": "56", "shift": "42",
    "super": "125", "win": "125", "f5": "63", "b": "48", "w": "17", "period": "52",
    "comma": "51", "volumeup": "115", "volumedown": "114", "volumemute": "113",
}


def _x_key(key: str) -> str:
    k = key.strip().lower()
    if k in _X_KEYS:
        return _X_KEYS[k]
    if len(k) == 1:
        return k
    if k.startswith("f") and k[1:].isdigit():
        return k.upper()
    return k


def _build_ydo_charmap() -> dict[str, str]:
    """Linux input-event keycodes for the printable keys, built once."""
    codes: dict[str, str] = {}
    for row, start in (("qwertyuiop", 16), ("asdfghjkl", 30), ("zxcvbnm", 44)):
        codes.update({c: str(start + i) for i, c in enumerate(row)})
    codes.update({c: str(2 + i) for i, c in enumerate("1234567890")})
    return codes


_YDO_CHARS = _build_ydo_charmap()


def _ydo_key(key: str) -> str:
    k = key.strip().lower()
    if k in _YDO_KEYS:
        return _YDO_KEYS[k]
    if len(k) == 1:
        return _YDO_CHARS.get(k, "57")
    if k.startswith("f") and k[1:].isdigit():
        n = int(k[1:])
        if 1 <= n <= 10:
            return str(58 + n)       # F1..F10 -> 59..68
        if n in (11, 12):
            return str(76 + n)       # F11, F12 -> 87, 88
    return "57"


def _screen_size_probe() -> Optional[tuple[int, int]]:
    """Best-effort screen size without importing a GUI toolkit."""
    system = platform.system()
    if system == "Linux":
        for cmd, parse in (
            (["xdpyinfo"], _parse_xdpyinfo),
            (["xrandr", "--current"], _parse_xrandr),
        ):
            if shutil.which(cmd[0]) is None:
                continue
            try:
                out = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=2.0
                ).stdout
                size = parse(out)
                if size:
                    return size
            except Exception:
                continue
    try:
        import tkinter

        root = tkinter.Tk()
        root.withdraw()
        size = (root.winfo_screenwidth(), root.winfo_screenheight())
        root.destroy()
        return size
    except Exception:
        return None


def _parse_xdpyinfo(text: str) -> Optional[tuple[int, int]]:
    for line in text.splitlines():
        if "dimensions:" in line:
            try:
                dims = line.split()[1]
                w, h = dims.split("x")
                return int(w), int(h)
            except Exception:
                return None
    return None


def _parse_xrandr(text: str) -> Optional[tuple[int, int]]:
    for line in text.splitlines():
        if " connected" in line and "x" in line:
            for tok in line.split():
                if "x" in tok and "+" in tok:
                    try:
                        wh = tok.split("+")[0]
                        w, h = wh.split("x")
                        return int(w), int(h)
                    except Exception:
                        continue
    return None


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

_ORDER = (PynputBackend, YdotoolBackend, XdotoolBackend, AutoGuiBackend)


def available_backends() -> list[str]:
    """Names of every backend usable on this machine right now."""
    names = [cls().name for cls in _ORDER if cls().available()]
    names.append("dry-run")
    return names


def select_backend(preferred: str = "auto", dry_run: bool = False) -> InputBackend:
    """Instantiate a backend by name, or auto-detect the best one.

    Wayland is detected explicitly: pynput's X11 path cannot inject there, so
    ydotool is tried first to avoid the "nothing happens and no error" trap.
    """
    if dry_run or preferred == "dry-run":
        return DryRunBackend()

    by_name = {
        "pynput": PynputBackend,
        "ydotool": YdotoolBackend,
        "xdotool": XdotoolBackend,
        "pyautogui": AutoGuiBackend,
        "autogui": AutoGuiBackend,
    }

    if preferred and preferred != "auto":
        cls = by_name.get(preferred.lower())
        if cls is None:
            raise ValueError(
                f"Unknown backend '{preferred}'. Choose from: "
                f"{', '.join(sorted(by_name))}, dry-run, auto"
            )
        backend = cls()
        if not backend.available():
            raise RuntimeError(
                f"Backend '{preferred}' is not available on this system.\n"
                f"Available: {', '.join(available_backends())}"
            )
        return backend

    order = list(_ORDER)
    is_wayland = (
        os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland"
        or bool(os.environ.get("WAYLAND_DISPLAY"))
    )
    if is_wayland:
        order = [YdotoolBackend, PynputBackend, XdotoolBackend, AutoGuiBackend]

    for cls in order:
        backend = cls()
        if backend.available():
            log.info("Using input backend: %s", backend.name)
            return backend

    log.warning(
        "No input backend available -- running in dry-run mode. "
        "Install one with: pip install 'gesturekit[input]'"
    )
    return DryRunBackend()
