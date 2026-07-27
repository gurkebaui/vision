"""Model bundle resolution and download.

MediaPipe's Tasks API needs a ``.task`` bundle on disk.  v1 relied on the old
``mp.solutions`` API, which shipped its models inside the wheel -- that API no
longer exists in MediaPipe 1.0, so the bundle has to be fetched and cached.

Resolution order:

1. explicit path passed by the caller / config
2. ``GESTUREKIT_MODEL_DIR`` environment variable
3. the platform cache dir (``~/.cache/gesturekit`` on Linux)
4. next to the package, then the current working directory
5. download from Google's model CDN (only if allowed)

Downloads are atomic (temp file + rename) and size-checked, so an interrupted
download can never leave a half-written bundle that fails cryptically later.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

_BASE = "https://storage.googleapis.com/mediapipe-models"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    filename: str
    url: str
    min_bytes: int
    description: str


MODELS: dict[str, ModelSpec] = {
    "hand_landmarker": ModelSpec(
        name="hand_landmarker",
        filename="hand_landmarker.task",
        url=f"{_BASE}/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        min_bytes=3_000_000,
        description="21-point hand landmark detector (palm detector + landmark model).",
    ),
    "gesture_recognizer": ModelSpec(
        name="gesture_recognizer",
        filename="gesture_recognizer.task",
        url=f"{_BASE}/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task",
        min_bytes=7_000_000,
        description="Hand landmarker plus Google's 7-class canned gesture classifier.",
    ),
}


class ModelError(RuntimeError):
    """Raised when a model bundle cannot be located or downloaded."""


def cache_dir() -> Path:
    """Platform-appropriate cache directory for model bundles."""
    env = os.environ.get("GESTUREKIT_MODEL_DIR")
    if env:
        return Path(env).expanduser()
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return base / "gesturekit" / "models"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "gesturekit"
    base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "gesturekit"


def _candidate_paths(spec: ModelSpec) -> list[Path]:
    return [
        cache_dir() / spec.filename,
        Path(__file__).resolve().parent / "assets" / spec.filename,
        Path(__file__).resolve().parents[2] / "models" / spec.filename,
        Path.cwd() / spec.filename,
        Path.cwd() / "models" / spec.filename,
    ]


def find_model(name: str) -> Optional[Path]:
    """Return a cached bundle path, or ``None`` if it is not on disk yet."""
    spec = MODELS[name]
    for path in _candidate_paths(spec):
        try:
            if path.is_file() and path.stat().st_size >= spec.min_bytes:
                return path
        except OSError:
            continue
    return None


def download_model(
    name: str,
    dest: Optional[Path] = None,
    progress: Optional[Callable[[int, int], None]] = None,
    timeout: float = 60.0,
) -> Path:
    """Download a model bundle atomically and return its path."""
    spec = MODELS[name]
    target_dir = Path(dest) if dest else cache_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / spec.filename

    tmp_fd, tmp_name = tempfile.mkstemp(dir=str(target_dir), suffix=".part")
    os.close(tmp_fd)
    tmp_path = Path(tmp_name)

    try:
        req = urllib.request.Request(spec.url, headers={"User-Agent": "gesturekit/2.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            total = int(resp.headers.get("Content-Length") or 0)
            done = 0
            with tmp_path.open("wb") as fh:
                while True:
                    chunk = resp.read(1 << 16)
                    if not chunk:
                        break
                    fh.write(chunk)
                    done += len(chunk)
                    if progress:
                        progress(done, total)

        if tmp_path.stat().st_size < spec.min_bytes:
            raise ModelError(
                f"Downloaded {spec.filename} is only {tmp_path.stat().st_size} bytes "
                f"(expected >= {spec.min_bytes}); the download was likely truncated."
            )
        shutil.move(str(tmp_path), str(target))
        return target

    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        raise ModelError(
            f"Could not download '{spec.filename}'.\n"
            f"  URL: {spec.url}\n"
            f"  Reason: {exc}\n\n"
            f"If this machine has no internet access, download the file on another\n"
            f"machine and place it in: {target_dir}\n"
            f"or point GESTUREKIT_MODEL_DIR at the directory holding it."
        ) from exc
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def ensure_model(
    name: str = "hand_landmarker",
    explicit: Optional[os.PathLike] = None,
    allow_download: bool = True,
    progress: Optional[Callable[[int, int], None]] = None,
) -> Path:
    """Resolve a model bundle, downloading it on first run if permitted."""
    if name not in MODELS:
        raise ModelError(f"Unknown model '{name}'. Known: {', '.join(sorted(MODELS))}")

    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise ModelError(f"Model file not found at the given path: {path}")
        return path

    found = find_model(name)
    if found:
        return found

    if not allow_download:
        raise ModelError(
            f"Model '{name}' is not cached and downloads are disabled.\n"
            f"Run 'gesturekit download' or place {MODELS[name].filename} in {cache_dir()}."
        )
    return download_model(name, progress=progress)


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()
