"""Pytest fixtures. Adds ``tests/`` to the path so ``synth`` is importable."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))


@pytest.fixture
def dry_backend():
    from gesturekit.actions.backends import DryRunBackend

    return DryRunBackend(echo=False)
