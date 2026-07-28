"""Engine: arming state machine, cooldowns, multi-hand routing."""

from __future__ import annotations

from synth import make_state, pose, swipe_frames

from gesturekit.gestures.dynamic import SwipeConfig
from gesturekit.gestures.engine import GestureEngine, GestureEngineConfig
from gesturekit.types import Handedness


def hold(engine, name, frames=8, t0=1000.0, step=0.033, **kw):
    """Hold a pose for a while and collect everything the engine emits."""
    out = []
    for i in range(frames):
        state = make_state(pose(name, **kw), timestamp=t0 + i * step)
        out += engine.process([state], now=t0 + i * step)
    return out


def test_static_gesture_reaches_the_output():
    engine = GestureEngine(GestureEngineConfig(require_arm=False))
    names = [e.name for e in hold(engine, "peace")]
    assert "peace" in names


def test_cooldown_blocks_immediate_repeat():
    engine = GestureEngine(GestureEngineConfig(require_arm=False, cooldown=1.0))
    events = hold(engine, "peace", frames=10, t0=1000.0)
    events += hold(engine, "open_palm", frames=10, t0=1000.4)
    events += hold(engine, "peace", frames=10, t0=1000.7)
    assert [e.name for e in events].count("peace") == 1


def test_cooldown_expires():
    engine = GestureEngine(GestureEngineConfig(require_arm=False, cooldown=0.3))
    events = hold(engine, "peace", frames=8, t0=1000.0)
    events += hold(engine, "open_palm", frames=8, t0=1001.0)
    events += hold(engine, "peace", frames=8, t0=1002.0)
    assert [e.name for e in events].count("peace") == 2


# ---------------------------------------------------------------------------
# Arming -- v1's version was broken in three separate ways
# ---------------------------------------------------------------------------

def test_disabled_arming_lets_everything_through():
    engine = GestureEngine(GestureEngineConfig(require_arm=False))
    assert "peace" in [e.name for e in hold(engine, "peace")]


def test_commands_are_blocked_before_arming():
    engine = GestureEngine(GestureEngineConfig(require_arm=True, arm_gesture="point_up"))
    assert [e.name for e in hold(engine, "peace")] == []


def test_arm_gesture_opens_the_window():
    engine = GestureEngine(GestureEngineConfig(require_arm=True, arm_gesture="point_up"))
    events = hold(engine, "point_up", t0=1000.0)
    assert "armed" in [e.name for e in events]
    assert engine.armed


def test_command_passes_inside_the_window():
    cfg = GestureEngineConfig(require_arm=True, arm_gesture="point_up", arm_window=3.0)
    engine = GestureEngine(cfg)
    hold(engine, "point_up", t0=1000.0)
    events = hold(engine, "peace", t0=1001.0)
    assert "peace" in [e.name for e in events]


def test_window_expires_and_blocks_again():
    """v1 compared against a timestamp initialised to the integer 1."""
    cfg = GestureEngineConfig(require_arm=True, arm_gesture="point_up", arm_window=1.0)
    engine = GestureEngine(cfg)
    hold(engine, "point_up", t0=1000.0)
    events = hold(engine, "peace", t0=1005.0)
    assert "peace" not in [e.name for e in events]
    assert not engine.armed


def test_expiry_is_announced_once():
    cfg = GestureEngineConfig(require_arm=True, arm_gesture="point_up", arm_window=0.5)
    engine = GestureEngine(cfg)
    hold(engine, "point_up", t0=1000.0)
    events = hold(engine, "open_palm", t0=1002.0, frames=12)
    assert [e.name for e in events].count("arm_expired") == 1


def test_disarm_gesture_closes_the_window():
    cfg = GestureEngineConfig(
        require_arm=True, arm_gesture="point_up",
        disarm_gesture="closed_fist", arm_window=10.0, cooldown=0.1,
    )
    engine = GestureEngine(cfg)
    hold(engine, "point_up", t0=1000.0)
    assert engine.armed
    events = hold(engine, "closed_fist", t0=1001.0)
    assert "disarmed" in [e.name for e in events]
    assert not engine.armed


def test_arm_remaining_counts_down():
    cfg = GestureEngineConfig(require_arm=True, arm_window=3.0)
    engine = GestureEngine(cfg)
    assert engine.arm_remaining == 0.0
    engine.armed = True
    engine.armed_until = 10**12
    assert engine.arm_remaining > 0


def test_arm_remaining_is_infinite_when_disabled():
    engine = GestureEngine(GestureEngineConfig(require_arm=False))
    assert engine.arm_remaining == float("inf")


# ---------------------------------------------------------------------------
# Multi-hand
# ---------------------------------------------------------------------------

def test_hands_are_tracked_independently():
    engine = GestureEngine(GestureEngineConfig(require_arm=False, cooldown=0.0))
    events = []
    for i in range(10):
        t = 1000 + i * 0.033
        left = make_state(pose("peace", center=(0.3, 0.5)), timestamp=t,
                          handedness=Handedness.LEFT)
        right = make_state(pose("rock", center=(0.7, 0.5)), timestamp=t,
                           handedness=Handedness.RIGHT)
        events += engine.process([left, right], now=t)
    names = {e.name for e in events}
    assert "peace" in names and "rock" in names


def test_empty_frames_are_safe():
    engine = GestureEngine()
    for i in range(5):
        assert engine.process([], now=1000 + i * 0.03) == []


def test_reset_clears_state():
    engine = GestureEngine(GestureEngineConfig(require_arm=True))
    hold(engine, "point_up", t0=1000.0)
    engine.reset()
    assert not engine.armed
    assert engine.arm_remaining == 0.0


def test_swipe_flows_through_engine():
    cfg = GestureEngineConfig(require_arm=False, enable_swipes=True)
    engine = GestureEngine(cfg)
    events = []
    for state in swipe_frames("left"):
        events += engine.process([state], now=state.timestamp)
    assert "swipe_left" in [e.name for e in events]


def test_swipes_can_be_disabled():
    cfg = GestureEngineConfig(require_arm=False, enable_swipes=False,
                              enable_circles=False)
    engine = GestureEngine(cfg)
    events = []
    for state in swipe_frames("left"):
        events += engine.process([state], now=state.timestamp)
    assert "swipe_left" not in [e.name for e in events]


def test_continuous_events_bypass_cooldown():
    """The cursor must stream every frame or it stutters."""
    cfg = GestureEngineConfig(require_arm=False, enable_pointer=True, cooldown=5.0)
    engine = GestureEngine(cfg)
    events = []
    for i in range(12):
        t = 1000 + i * 0.03
        events += engine.process([make_state(pose("point_up"), timestamp=t)], now=t)
    assert len([e for e in events if e.name == "cursor_move"]) > 5


def test_continuous_events_bypass_arming():
    cfg = GestureEngineConfig(require_arm=True, enable_pointer=True)
    engine = GestureEngine(cfg)
    events = []
    for i in range(10):
        t = 1000 + i * 0.03
        events += engine.process([make_state(pose("point_up"), timestamp=t)], now=t)
    assert any(e.name == "cursor_move" for e in events)


def test_engine_accepts_profile_swipe_overrides():
    cfg = GestureEngineConfig(require_arm=False, swipe=SwipeConfig(min_distance=99.0))
    engine = GestureEngine(cfg)
    events = []
    for state in swipe_frames("left"):
        events += engine.process([state], now=state.timestamp)
    assert "swipe_left" not in [e.name for e in events]
