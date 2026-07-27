"""CLI surface: argument parsing, subcommands and exit codes."""

from __future__ import annotations

import pytest

from gesturekit.cli import build_parser, main


def run(argv, capsys):
    code = main(argv)
    return code, capsys.readouterr()


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def test_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "gesturekit" in capsys.readouterr().out


def test_help_lists_subcommands(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    for command in ("run", "profiles", "bindings", "devices", "doctor", "download"):
        assert command in out


def test_run_defaults():
    args = build_parser().parse_args(["run"])
    assert args.profile == "presentation"
    assert args.require_arm is None        # tri-state: unset means "use profile"
    assert args.dry_run is False


def test_arm_flags_are_tri_state():
    parser = build_parser()
    assert parser.parse_args(["run", "--arm"]).require_arm is True
    assert parser.parse_args(["run", "--no-arm"]).require_arm is False


def test_run_accepts_tuning_flags():
    args = build_parser().parse_args(
        ["run", "-p", "desktop", "-c", "2", "--width", "640", "--fps", "60",
         "--hands", "1", "--sensitivity", "1.5", "--dry-run", "--scores",
         "--no-window", "--theme", "light"]
    )
    assert args.profile == "desktop"
    assert args.camera == 2
    assert args.width == 640
    assert args.hands == 1
    assert args.sensitivity == 1.5
    assert args.dry_run and args.scores and args.no_window


def test_invalid_hand_count_rejected():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["run", "--hands", "5"])


def test_invalid_backend_rejected():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["run", "--backend", "telepathy"])


# ---------------------------------------------------------------------------
# Read-only subcommands
# ---------------------------------------------------------------------------

def test_profiles_command(capsys):
    code, out = run(["profiles"], capsys)
    assert code == 0
    for name in ("presentation", "desktop", "media", "accessibility"):
        assert name in out.out


@pytest.mark.parametrize("profile", ["presentation", "desktop", "media", "accessibility"])
def test_bindings_command(profile, capsys):
    code, out = run(["bindings", "-p", profile], capsys)
    assert code == 0
    assert profile in out.out


def test_bindings_groups_output(capsys):
    _, out = run(["bindings", "-p", "presentation"], capsys)
    assert "HAND SHAPES" in out.out
    assert "MOTION" in out.out


def test_bindings_unknown_profile(capsys):
    code, out = run(["bindings", "-p", "nope"], capsys)
    assert code == 2
    assert "error" in out.err.lower()


def test_devices_command_runs(capsys):
    code, _ = run(["devices"], capsys)
    assert code in (0, 1)          # 1 when the machine has no camera


def test_doctor_command_runs(capsys):
    code, out = run(["doctor"], capsys)
    assert code in (0, 1)
    text = out.out
    assert "opencv" in text
    assert "mediapipe" in text
    assert "profiles" in text


def test_doctor_reports_versions(capsys):
    _, out = run(["doctor"], capsys)
    import cv2
    assert cv2.__version__ in out.out


# ---------------------------------------------------------------------------
# init / download
# ---------------------------------------------------------------------------

def test_init_writes_config(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    code, out = run(["init"], capsys)
    assert code == 0
    written = tmp_path / "gesturekit" / "config.yaml"
    assert written.is_file()

    import yaml
    data = yaml.safe_load(written.read_text())
    assert "camera" in data and "engine" in data


def test_init_refuses_to_clobber(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    run(["init"], capsys)
    code, out = run(["init"], capsys)
    assert code == 1
    assert "--force" in out.out


def test_init_force_overwrites(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    run(["init"], capsys)
    code, _ = run(["init", "--force"], capsys)
    assert code == 0


def test_download_reports_cached_model(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("GESTUREKIT_MODEL_DIR", str(tmp_path))
    (tmp_path / "hand_landmarker.task").write_bytes(b"x" * 4_000_000)
    code, out = run(["download"], capsys)
    assert code == 0
    assert "Already cached" in out.out


def test_download_failure_is_reported(tmp_path, monkeypatch, capsys):
    import urllib.request

    monkeypatch.setenv("GESTUREKIT_MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(OSError("offline"))
    )
    code, out = run(["download"], capsys)
    assert code == 2
    assert "error" in out.err.lower()


# ---------------------------------------------------------------------------
# run wiring (without starting a camera)
# ---------------------------------------------------------------------------

def test_run_builds_config_and_starts_app(monkeypatch, capsys):
    seen = {}

    class FakeApp:
        def __init__(self, config):
            seen["config"] = config

        def run(self):
            return 0

    import gesturekit.app as appmod
    monkeypatch.setattr(appmod, "GestureApp", FakeApp)

    code, out = run(["run", "-p", "desktop", "-c", "3", "--dry-run",
                     "--no-window", "--arm"], capsys)
    assert code == 0
    cfg = seen["config"]
    assert cfg.camera.index == 3
    assert cfg.dry_run is True
    assert cfg.ui.show_window is False
    assert cfg.engine.require_arm is True
    assert "DRY RUN" in out.out


def test_sensitivity_scales_thresholds_coherently(monkeypatch, capsys):
    seen = {}

    class FakeApp:
        def __init__(self, config):
            seen["config"] = config

        def run(self):
            return 0

    import gesturekit.app as appmod
    monkeypatch.setattr(appmod, "GestureApp", FakeApp)

    run(["run", "--sensitivity", "2.0"], capsys)
    twitchy = seen["config"].engine
    run(["run", "--sensitivity", "0.5"], capsys)
    strict = seen["config"].engine

    assert twitchy.static_min_frames < strict.static_min_frames
    assert twitchy.cooldown < strict.cooldown
    assert twitchy.swipe.min_distance < strict.swipe.min_distance


def test_run_with_unknown_profile(capsys):
    code, out = run(["run", "-p", "ghost"], capsys)
    assert code == 2
    assert "error" in out.err.lower()


def test_bare_invocation_defaults_to_run(monkeypatch):
    calls = []

    class FakeApp:
        def __init__(self, config):
            calls.append(config)

        def run(self):
            return 0

    import gesturekit.app as appmod
    monkeypatch.setattr(appmod, "GestureApp", FakeApp)
    assert main([]) == 0
    assert calls and calls[0].profile == "presentation"
