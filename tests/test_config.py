"""Profiles, config merging and the shipped bindings."""

from __future__ import annotations

import textwrap

import pytest
import yaml

from gesturekit.config import (
    AppConfig,
    _merge,
    find_profile,
    list_profiles,
    load_config,
    load_profile,
    parse_bindings,
)

SHIPPED = ["presentation", "desktop", "media", "accessibility"]


@pytest.mark.parametrize("name", SHIPPED)
def test_shipped_profiles_load(name):
    cfg = load_config(name)
    assert cfg.bindings
    assert cfg.profile_description


@pytest.mark.parametrize("name", SHIPPED)
def test_shipped_profiles_are_valid_yaml(name):
    path = find_profile(name)
    data = yaml.safe_load(path.read_text())
    assert "gestures" in data or "extends" in data


def test_list_profiles_finds_all_shipped():
    names = {n for n, _, _ in list_profiles()}
    assert set(SHIPPED) <= names


def test_missing_profile_raises_with_help():
    with pytest.raises(FileNotFoundError, match="Available"):
        load_config("no_such_profile")


def test_presentation_profile_has_slide_navigation():
    cfg = load_config("presentation")
    assert "swipe_left" in cfg.bindings
    assert "swipe_right" in cfg.bindings
    assert cfg.bindings["closed_fist"].kind == "key"


def test_desktop_profile_enables_pointer():
    cfg = load_config("desktop")
    assert cfg.engine.enable_pointer
    assert cfg.engine.enable_scroll
    assert "cursor_move" in cfg.bindings
    assert "click" in cfg.bindings


def test_accessibility_profile_is_more_forgiving():
    strict = load_config("presentation")
    easy = load_config("accessibility")
    assert easy.engine.static_min_frames > strict.engine.static_min_frames
    assert easy.engine.cooldown > strict.engine.cooldown
    assert easy.engine.static_min_score <= strict.engine.static_min_score


def test_profile_inheritance_merges_parent():
    """media extends desktop: inherited values survive, overrides win."""
    media = load_config("media")
    assert media.engine.scroll_sensitivity == load_config("desktop").engine.scroll_sensitivity
    assert not media.engine.enable_pointer         # overridden
    assert media.engine.swipe.min_distance == 1.0  # overridden


def test_profile_can_null_out_inherited_binding():
    media = load_config("media")
    assert media.bindings["cursor_move"].kind == "none"


def test_overrides_beat_profile_values():
    cfg = load_config("presentation", {"camera": {"index": 3, "width": 640},
                                       "engine": {"cooldown": 9.5}})
    assert cfg.camera.index == 3
    assert cfg.camera.width == 640
    assert cfg.engine.cooldown == 9.5


def test_nested_swipe_override():
    cfg = load_config("presentation", {"engine": {"swipe": {"min_speed": 7.5}}})
    assert cfg.engine.swipe.min_speed == 7.5
    assert cfg.engine.swipe.min_distance > 0     # untouched fields survive


def test_nested_pointer_override():
    cfg = load_config("desktop", {"engine": {"pointer": {"smoothing": 0.3}}})
    assert cfg.engine.pointer.smoothing == 0.3


def test_unknown_keys_are_ignored():
    cfg = load_config("presentation", {"camera": {"nonsense": 1}, "totally_bogus": 2})
    assert isinstance(cfg, AppConfig)


def test_merge_is_recursive_and_non_destructive():
    base = {"a": {"b": 1, "c": 2}, "d": 3}
    out = _merge(base, {"a": {"c": 99}})
    assert out == {"a": {"b": 1, "c": 99}, "d": 3}
    assert base["a"]["c"] == 2


def test_parse_bindings_reports_the_bad_gesture():
    with pytest.raises(ValueError, match="bad_one"):
        parse_bindings({"bad_one": 42})


def test_profile_from_explicit_path(tmp_path):
    path = tmp_path / "custom.yaml"
    path.write_text(
        textwrap.dedent(
            """
            name: custom
            description: test profile
            gestures:
              peace: {key: x, label: Test}
            """
        )
    )
    cfg = load_config(str(path))
    assert cfg.bindings["peace"].value == "x"


def test_extends_an_external_profile(tmp_path):
    path = tmp_path / "child.yaml"
    path.write_text(
        textwrap.dedent(
            """
            name: child
            extends: presentation
            description: child profile
            gestures:
              peace: {key: z}
            """
        )
    )
    cfg = load_config(str(path))
    assert cfg.bindings["peace"].value == "z"
    assert "swipe_left" in cfg.bindings      # inherited


def test_circular_extends_is_caught(tmp_path, monkeypatch):
    a = tmp_path / "a.yaml"
    a.write_text("name: a\nextends: a\ngestures: {}\n")
    with pytest.raises(ValueError, match="too deep"):
        load_profile(str(a))


@pytest.mark.parametrize("name", SHIPPED)
def test_every_binding_parses(name):
    for gesture, action in load_config(name).bindings.items():
        assert action.kind in {
            "key", "hotkey", "text", "click", "cursor", "drag",
            "scroll", "mode", "command", "none",
        }, f"{name}:{gesture} -> {action.kind}"


@pytest.mark.parametrize("name", SHIPPED)
def test_mode_bindings_point_at_real_profiles(name):
    known = {n for n, _, _ in list_profiles()}
    for gesture, action in load_config(name).bindings.items():
        if action.kind == "mode":
            assert action.value in known, f"{name}:{gesture} -> unknown profile {action.value}"
