"""Maps gesture names to actions via user-editable profiles.

v1 hard-coded the mapping in a chain of ``if gesture == ...`` branches inside
the main loop, so changing a shortcut meant editing Python.  Here the mapping
is data: a profile is a YAML file, gestures bind to typed actions, and the
whole thing can be reloaded at runtime with a keypress.

Supported action types::

    key: space                # single keystroke
    hotkey: [ctrl, shift, f5] # chord
    text: "hello"             # type a string
    click: left               # mouse button
    scroll: {dx: 0, dy: 3}
    cursor: absolute          # consume pointer stream
    drag: start | end
    mode: desktop             # switch profile
    command: ["notify-send", "hi"]   # run a program
    none:                     # explicitly do nothing
"""

from __future__ import annotations

import logging
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from ..types import GestureEvent
from .backends import InputBackend

log = logging.getLogger("gesturekit.actions")


@dataclass
class Action:
    """One bound action."""

    kind: str
    value: Any = None
    label: str = ""
    repeat: int = 1

    @classmethod
    def parse(cls, spec: Any) -> Action:
        """Build an Action from a YAML fragment (string or mapping)."""
        if spec is None:
            return cls("none", label="none")

        if isinstance(spec, str):
            text = spec.strip()
            if not text or text.lower() == "none":
                return cls("none", label="none")
            if "+" in text and " " not in text:     # "ctrl+shift+f5"
                keys = [k.strip() for k in text.split("+") if k.strip()]
                return cls("hotkey", keys, label=text)
            return cls("key", text, label=text)

        if isinstance(spec, dict):
            data = dict(spec)
            label = str(data.pop("label", "") or "")
            repeat = int(data.pop("repeat", 1) or 1)
            if not data:
                return cls("none", label=label or "none")
            kind, value = next(iter(data.items()))
            kind = str(kind).lower()
            if kind == "hotkey" and isinstance(value, str):
                value = [k.strip() for k in value.replace("+", " ").split() if k.strip()]
            return cls(kind, value, label=label or _default_label(kind, value), repeat=repeat)

        raise ValueError(f"Cannot parse action: {spec!r}")


def _default_label(kind: str, value: Any) -> str:
    if kind == "hotkey" and isinstance(value, (list, tuple)):
        return "+".join(str(v) for v in value)
    if isinstance(value, dict):
        return f"{kind}({', '.join(f'{k}={v}' for k, v in value.items())})"
    return f"{kind}:{value}" if value is not None else kind


@dataclass
class RouterStats:
    executed: int = 0
    suppressed: int = 0
    by_gesture: dict = field(default_factory=dict)
    history: list = field(default_factory=list)


class ActionRouter:
    """Executes the action bound to each gesture event."""

    def __init__(
        self,
        bindings: Optional[dict[str, Action]] = None,
        backend: Optional[InputBackend] = None,
        screen: Optional[tuple[int, int]] = None,
        enabled: bool = True,
    ):
        self.bindings = bindings or {}
        self.backend = backend
        self.enabled = enabled
        self._screen = screen
        self.stats = RouterStats()
        self._dragging = False
        self._last_cursor = (0.0, 0.0)
        self.on_mode_change = None      # callback(profile_name)

    # -- helpers -----------------------------------------------------------
    @property
    def screen(self) -> tuple[int, int]:
        if self._screen is None:
            self._screen = self.backend.screen_size() if self.backend else (1920, 1080)
        return self._screen

    def bind(self, gesture: str, action: Action) -> None:
        self.bindings[gesture] = action

    def describe(self) -> list[tuple[str, str]]:
        return sorted((g, a.label or a.kind) for g, a in self.bindings.items())

    # -- execution ---------------------------------------------------------
    def dispatch(self, events: list[GestureEvent]) -> list[tuple[GestureEvent, Action]]:
        """Run every bound event; returns the (event, action) pairs performed."""
        done: list[tuple[GestureEvent, Action]] = []
        for ev in events:
            action = self.bindings.get(ev.name)
            if action is None or action.kind == "none":
                continue
            if not self.enabled:
                self.stats.suppressed += 1
                continue
            try:
                self._execute(action, ev)
            except Exception as exc:                       # never kill the loop
                log.error("Action %s for '%s' failed: %s", action.kind, ev.name, exc)
                continue
            self.stats.executed += 1
            self.stats.by_gesture[ev.name] = self.stats.by_gesture.get(ev.name, 0) + 1
            self.stats.history.append((time.time(), ev.name, action.label))
            if len(self.stats.history) > 200:
                del self.stats.history[:-200]
            done.append((ev, action))
        return done

    def _execute(self, action: Action, ev: GestureEvent) -> None:
        be = self.backend
        if be is None:
            return
        kind = action.kind

        if kind == "key":
            for _ in range(action.repeat):
                be.key(str(action.value))

        elif kind == "hotkey":
            keys = action.value if isinstance(action.value, (list, tuple)) else [action.value]
            for _ in range(action.repeat):
                be.hotkey(*[str(k) for k in keys])

        elif kind == "text":
            be.type_text(str(action.value))

        elif kind == "click":
            button = str(action.value) if action.value else "left"
            be.click(button, action.repeat)

        elif kind == "cursor":
            x = float(ev.data.get("x", self._last_cursor[0]))
            y = float(ev.data.get("y", self._last_cursor[1]))
            self._last_cursor = (x, y)
            w, h = self.screen
            be.move_to(int(x * (w - 1)), int(y * (h - 1)))

        elif kind == "drag":
            mode = str(action.value or "").lower()
            if mode == "start" and not self._dragging:
                be.mouse_down("left")
                self._dragging = True
            elif mode == "end" and self._dragging:
                be.mouse_up("left")
                self._dragging = False

        elif kind == "scroll":
            value = action.value if isinstance(action.value, dict) else {}
            dx = int(value.get("dx", ev.data.get("dx", 0)) or 0)
            dy = int(value.get("dy", ev.data.get("dy", 0)) or 0)
            if "gain" in value:
                gain = float(value["gain"])
                dx = int(round(float(ev.data.get("dx", 0)) * gain))
                dy = int(round(float(ev.data.get("dy", 0)) * gain))
            if dx or dy:
                be.scroll(dx, dy)

        elif kind == "mode":
            if callable(self.on_mode_change):
                self.on_mode_change(str(action.value))

        elif kind == "command":
            cmd = action.value
            args = shlex.split(cmd) if isinstance(cmd, str) else [str(c) for c in cmd]
            subprocess.Popen(
                args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True
            )

        elif kind == "none":
            pass

        else:
            log.warning("Unknown action kind: %s", kind)

    def release(self) -> None:
        """Drop any held mouse button; called on shutdown so nothing sticks."""
        if self._dragging and self.backend:
            try:
                self.backend.mouse_up("left")
            except Exception:
                pass
            self._dragging = False
