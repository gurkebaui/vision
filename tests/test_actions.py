"""Action parsing, routing and backend selection."""

from __future__ import annotations

import pytest

from gesturekit.actions.backends import (
    DryRunBackend,
    _x_key,
    available_backends,
    select_backend,
)
from gesturekit.actions.router import Action, ActionRouter
from gesturekit.types import GestureEvent, Handedness


def ev(name, **data):
    return GestureEvent(name, 0.9, Handedness.RIGHT, 1000.0, "static", data)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def test_parse_plain_key():
    a = Action.parse("space")
    assert a.kind == "key" and a.value == "space"


def test_parse_plus_syntax_as_hotkey():
    a = Action.parse("ctrl+shift+f5")
    assert a.kind == "hotkey" and a.value == ["ctrl", "shift", "f5"]


def test_parse_mapping_forms():
    assert Action.parse({"key": "b"}).kind == "key"
    assert Action.parse({"hotkey": ["ctrl", "p"]}).value == ["ctrl", "p"]
    assert Action.parse({"hotkey": "ctrl+p"}).value == ["ctrl", "p"]
    assert Action.parse({"click": "right"}).value == "right"
    assert Action.parse({"scroll": {"gain": 3}}).value == {"gain": 3}


def test_parse_label_and_repeat():
    a = Action.parse({"key": "volumeup", "label": "Louder", "repeat": 4})
    assert a.label == "Louder" and a.repeat == 4 and a.value == "volumeup"


def test_parse_none_variants():
    assert Action.parse(None).kind == "none"
    assert Action.parse("none").kind == "none"
    assert Action.parse("").kind == "none"


def test_parse_generates_default_labels():
    assert Action.parse({"hotkey": ["ctrl", "a"]}).label == "ctrl+a"
    assert "key" in Action.parse({"key": "esc"}).label


def test_parse_rejects_garbage():
    with pytest.raises(ValueError):
        Action.parse(12345)


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def test_key_action_dispatch(dry_backend):
    router = ActionRouter({"peace": Action.parse("b")}, dry_backend)
    router.dispatch([ev("peace")])
    assert dry_backend.events == [("key", "b")]


def test_hotkey_dispatch(dry_backend):
    router = ActionRouter({"rock": Action.parse({"hotkey": ["ctrl", "p"]})}, dry_backend)
    router.dispatch([ev("rock")])
    assert dry_backend.events == [("hotkey", "ctrl+p")]


def test_repeat_sends_multiple(dry_backend):
    router = ActionRouter(
        {"thumbs_up": Action.parse({"key": "volumeup", "repeat": 3})}, dry_backend
    )
    router.dispatch([ev("thumbs_up")])
    assert len(dry_backend.events) == 3


def test_unbound_gesture_is_ignored(dry_backend):
    router = ActionRouter({}, dry_backend)
    router.dispatch([ev("peace")])
    assert dry_backend.events == []


def test_none_action_does_nothing(dry_backend):
    router = ActionRouter({"peace": Action.parse(None)}, dry_backend)
    router.dispatch([ev("peace")])
    assert dry_backend.events == []


def test_cursor_maps_to_screen_pixels(dry_backend):
    router = ActionRouter({"cursor_move": Action.parse({"cursor": "absolute"})},
                          dry_backend, screen=(1920, 1080))
    router.dispatch([ev("cursor_move", x=0.5, y=0.25)])
    kind, x, y = dry_backend.events[0]
    assert kind == "move"
    assert x == pytest.approx(959, abs=2)
    assert y == pytest.approx(269, abs=2)


def test_drag_start_and_end(dry_backend):
    router = ActionRouter(
        {"drag_start": Action.parse({"drag": "start"}),
         "drag_end": Action.parse({"drag": "end"})},
        dry_backend,
    )
    router.dispatch([ev("drag_start")])
    router.dispatch([ev("drag_end")])
    assert dry_backend.events == [("mouse_down", "left"), ("mouse_up", "left")]


def test_drag_is_not_started_twice(dry_backend):
    router = ActionRouter({"drag_start": Action.parse({"drag": "start"})}, dry_backend)
    router.dispatch([ev("drag_start")])
    router.dispatch([ev("drag_start")])
    assert len(dry_backend.events) == 1


def test_release_frees_a_held_button(dry_backend):
    router = ActionRouter({"drag_start": Action.parse({"drag": "start"})}, dry_backend)
    router.dispatch([ev("drag_start")])
    router.release()
    assert dry_backend.events[-1] == ("mouse_up", "left")


def test_release_is_idempotent(dry_backend):
    router = ActionRouter({}, dry_backend)
    router.release()
    router.release()
    assert dry_backend.events == []


def test_scroll_gain_scales_event_data(dry_backend):
    router = ActionRouter({"scroll": Action.parse({"scroll": {"gain": 3}})}, dry_backend)
    router.dispatch([ev("scroll", dx=0.0, dy=2.0)])
    assert dry_backend.events == [("scroll", 0, 6)]


def test_disabled_router_suppresses_everything(dry_backend):
    router = ActionRouter({"peace": Action.parse("b")}, dry_backend, enabled=False)
    router.dispatch([ev("peace")])
    assert dry_backend.events == []
    assert router.stats.suppressed == 1


def test_mode_action_triggers_callback(dry_backend):
    seen = []
    router = ActionRouter({"wave": Action.parse({"mode": "desktop"})}, dry_backend)
    router.on_mode_change = seen.append
    router.dispatch([ev("wave")])
    assert seen == ["desktop"]


def test_backend_exception_does_not_break_the_loop():
    class Exploding(DryRunBackend):
        def key(self, key):
            raise RuntimeError("boom")

    backend = Exploding(echo=False)
    router = ActionRouter({"peace": Action.parse("b"), "rock": Action.parse({"click": "left"})},
                          backend)
    router.dispatch([ev("peace"), ev("rock")])
    assert ("click", "left", 1) in backend.events     # second action still ran
    assert router.stats.executed == 1


def test_stats_track_usage(dry_backend):
    router = ActionRouter({"peace": Action.parse("b")}, dry_backend)
    router.dispatch([ev("peace")])
    router.dispatch([ev("peace")])
    assert router.stats.executed == 2
    assert router.stats.by_gesture["peace"] == 2


def test_describe_lists_bindings(dry_backend):
    router = ActionRouter(
        {"peace": Action.parse({"key": "b", "label": "Blackout"})}, dry_backend
    )
    assert router.describe() == [("peace", "Blackout")]


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

def test_dry_run_is_always_available():
    assert "dry-run" in available_backends()
    assert select_backend("dry-run").name == "dry-run"


def test_dry_run_flag_wins():
    assert select_backend("pynput", dry_run=True).name == "dry-run"


def test_auto_selection_always_returns_something():
    assert select_backend("auto") is not None


def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="Unknown backend"):
        select_backend("telepathy")


def test_dry_backend_records_all_primitives(dry_backend):
    dry_backend.key("a")
    dry_backend.hotkey("ctrl", "c")
    dry_backend.move_to(10, 20)
    dry_backend.click("right", 2)
    dry_backend.scroll(1, -1)
    dry_backend.type_text("hi")
    assert len(dry_backend.events) == 6


def test_x_key_translation():
    assert _x_key("esc") == "Escape"
    assert _x_key("pagedown") == "Next"
    assert _x_key("a") == "a"
    assert _x_key("f5") == "F5"
