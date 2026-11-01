#!/usr/bin/env python3
"""
Vision Control v2 — Comprehensive System Compatibility & Range Verification Runner.
Runs environment diagnostics, CPU inference latency benchmark, and automated pytest checks.
"""

from __future__ import annotations

import importlib
import platform
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


def check_imports() -> bool:
    print("📦 Checking Python dependencies...")
    pkgs = [
        ("numpy", "numpy"),
        ("cv2", "opencv-python"),
        ("mediapipe", "mediapipe"),
        ("PIL", "Pillow"),
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
    ]
    ok = True
    for mod, label in pkgs:
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", "installed")
            print(f"  ✅ {label:16s} ({ver})")
        except ImportError as exc:
            print(f"  ❌ {label:16s} ({exc})")
            ok = False
    return ok


def check_long_range_scale_invariance() -> bool:
    print("\n🔭 Verifying 0.5m – 4.8m Scale-Invariant Gesture Accuracy...")
    from tests.test_gestures import build_synthetic_hand
    from vision_control.gestures import ScaleInvariantGestureEngine

    engine = ScaleInvariantGestureEngine()
    poses = [
        "open_palm",
        "closed_fist",
        "pointing_up",
        "pointing_down",
        "peace_sign",
        "thumbs_up",
        "thumbs_down",
        "pinch",
        "three_fingers",
    ]
    distances = [0.5, 1.5, 3.0, 4.8]
    passed = 0
    total = 0
    for d in distances:
        for p in poses:
            total += 1
            lms = build_synthetic_hand(p, distance_m=d)
            det, conf, _, _, _ = engine.classify_static_pose(lms)
            if det == p and conf >= 0.78:
                passed += 1
    print(f"  ✅ {passed}/{total} pose-distance combinations passed (0.5m, 1.5m, 3.0m, 4.8m)")
    return passed == total


def check_cpu_latency() -> bool:
    print("\n⚡ Benchmarking CPU Long-Range Smart-Zoom Tracker...")
    import numpy as np
    from vision_control.config import SystemConfig
    from vision_control.tracker import LongRangeHandTracker

    cfg = SystemConfig(performance_preset="balanced")
    tracker = LongRangeHandTracker(cfg)
    try:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        for _ in range(3):
            tracker.process_frame(frame)
        iters = 15
        t0 = time.perf_counter()
        for _ in range(iters):
            tracker.process_frame(frame)
        avg_ms = (time.perf_counter() - t0) / iters * 1000.0
        fps_cap = 1000.0 / max(avg_ms, 0.1)
        print(f"  ✅ Avg CPU Inference: {avg_ms:.2f} ms/frame (~{fps_cap:.1f} FPS ceiling)")
        return avg_ms < 80.0
    finally:
        tracker.close()


def main() -> int:
    print("🧪 Vision Control v2 — Diagnostic & Verification Suite")
    print("=" * 58)
    print(f"Platform: {platform.system()} {platform.release()} | Python {sys.version.split()[0]}")

    if not check_imports():
        return 1
    if not check_long_range_scale_invariance():
        return 1
    if not check_cpu_latency():
        return 1

    print("\n🔬 Running full pytest suite...")
    rc = subprocess.call([sys.executable, "-m", "pytest", "-v", str(REPO_ROOT / "tests")])
    if rc == 0:
        print("\n🎉 All diagnostics and automated tests passed!")
    return rc


if __name__ == "__main__":
    sys.exit(main())
