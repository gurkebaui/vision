"""Model resolution and camera configuration (no hardware needed)."""

from __future__ import annotations

import pytest

from gesturekit import models
from gesturekit.camera import CameraConfig, _backend_flag

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def test_known_models_declared():
    assert "hand_landmarker" in models.MODELS
    assert "gesture_recognizer" in models.MODELS
    for spec in models.MODELS.values():
        assert spec.url.startswith("https://")
        assert spec.filename.endswith(".task")
        assert spec.min_bytes > 0


def test_cache_dir_is_absolute():
    assert models.cache_dir().is_absolute()


def test_cache_dir_honours_env(monkeypatch, tmp_path):
    monkeypatch.setenv("GESTUREKIT_MODEL_DIR", str(tmp_path))
    assert models.cache_dir() == tmp_path


def test_find_model_locates_a_cached_file(monkeypatch, tmp_path):
    monkeypatch.setenv("GESTUREKIT_MODEL_DIR", str(tmp_path))
    blob = tmp_path / "hand_landmarker.task"
    blob.write_bytes(b"x" * 4_000_000)
    assert models.find_model("hand_landmarker") == blob


def test_find_model_rejects_truncated_file(monkeypatch, tmp_path):
    """A half-finished download must not be mistaken for a valid model."""
    monkeypatch.setenv("GESTUREKIT_MODEL_DIR", str(tmp_path))
    (tmp_path / "hand_landmarker.task").write_bytes(b"nope")
    assert models.find_model("hand_landmarker") is None


def test_explicit_path_is_used(tmp_path):
    blob = tmp_path / "custom.task"
    blob.write_bytes(b"x" * 100)
    assert models.ensure_model("hand_landmarker", explicit=blob) == blob


def test_explicit_missing_path_errors(tmp_path):
    with pytest.raises(models.ModelError, match="not found"):
        models.ensure_model("hand_landmarker", explicit=tmp_path / "nope.task")


def test_unknown_model_errors():
    with pytest.raises(models.ModelError, match="Unknown model"):
        models.ensure_model("banana")


def test_download_disabled_gives_actionable_error(monkeypatch, tmp_path):
    monkeypatch.setenv("GESTUREKIT_MODEL_DIR", str(tmp_path))
    with pytest.raises(models.ModelError) as exc:
        models.ensure_model("hand_landmarker", allow_download=False)
    assert "gesturekit download" in str(exc.value)


def test_download_failure_explains_offline_workaround(monkeypatch, tmp_path):
    import urllib.request

    monkeypatch.setenv("GESTUREKIT_MODEL_DIR", str(tmp_path))

    def boom(*a, **k):
        raise OSError("network unreachable")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with pytest.raises(models.ModelError) as exc:
        models.download_model("hand_landmarker")
    message = str(exc.value)
    assert "GESTUREKIT_MODEL_DIR" in message
    assert str(tmp_path) in message


def test_failed_download_leaves_no_partial_file(monkeypatch, tmp_path):
    import urllib.request

    monkeypatch.setenv("GESTUREKIT_MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(OSError("dead"))
    )
    with pytest.raises(models.ModelError):
        models.download_model("hand_landmarker")
    assert list(tmp_path.iterdir()) == []


def test_sha256(tmp_path):
    f = tmp_path / "a.bin"
    f.write_bytes(b"hello")
    assert models.sha256(f) == (
        "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    )


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

def test_camera_config_defaults():
    cfg = CameraConfig()
    assert cfg.width > 0 and cfg.height > 0
    assert cfg.flip is True          # mirrored preview is the intuitive default


def test_backend_flag_resolves_names():
    assert isinstance(_backend_flag("v4l2"), int)
    assert isinstance(_backend_flag("auto"), int)
    assert isinstance(_backend_flag("nonsense"), int)


def test_camera_open_failure_is_helpful(monkeypatch):
    import cv2

    from gesturekit.camera import Camera, CameraError

    class Dead:
        def isOpened(self):
            return False

        def release(self):
            pass

    monkeypatch.setattr(cv2, "VideoCapture", lambda *a, **k: Dead())
    with pytest.raises(CameraError) as exc:
        Camera(CameraConfig(index=7)).open()
    assert "index 7" in str(exc.value)
    assert "gesturekit devices" in str(exc.value)
