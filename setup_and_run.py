#!/usr/bin/env python3
"""
Vision Control v2 — Automated Setup, Diagnostics & Launcher
Checks Python environment, installs dependencies from requirements.txt,
runs a quick CPU benchmark, and launches Desktop HUD or Web Control Studio.
"""

from __future__ import annotations

import importlib
import platform
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


def check_python() -> bool:
    v = sys.version_info
    print(f"Python Runtime : {v.major}.{v.minor}.{v.micro} ({platform.system()} {platform.machine()})")
    if (v.major, v.minor) < (3, 9):
        print("❌ Python 3.9+ is required.")
        return False
    return True


def ensure_dependencies() -> bool:
    required_imports = [
        ("numpy", "numpy"),
        ("cv2", "opencv-python"),
        ("mediapipe", "mediapipe"),
        ("PIL", "Pillow"),
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
    ]
    missing = []
    for mod_name, pkg_name in required_imports:
        try:
            importlib.import_module(mod_name)
        except ImportError:
            missing.append(pkg_name)

    if not missing:
        print("✅ All core Python packages are installed.")
        return True

    print(f"📦 Installing missing packages ({', '.join(missing)}) from requirements.txt...")
    req_file = REPO_ROOT / "requirements.txt"
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(req_file)])
        return True
    except subprocess.CalledProcessError as exc:
        print(f"❌ Package installation failed: {exc}")
        return False


def main() -> int:
    print("🚀 Vision Control v2 — Long-Range Gesture System Setup")
    print("=" * 56)
    if not check_python():
        return 1
    if not ensure_dependencies():
        return 1

    print("\nSelect launch mode:")
    print("  [1] Native Desktop Long-Range HUD  (Webcam -> Python -> OS)")
    print("  [2] Web Control Studio + OS Bridge (Browser UI + Interactive Stage)")
    print("  [3] Run CPU Benchmark & System Self-Test")
    print("  [q] Exit")

    try:
        choice = input("\nEnter choice [1/2/3/q] (default: 1): ").strip().lower()
    except EOFError:
        choice = "3"

    if choice in ("", "1"):
        from gesture_presentation_control import main as run_main

        return run_main([])
    if choice == "2":
        from gesture_presentation_control import main as run_main

        return run_main(["--studio", "--open-browser"])
    if choice == "3":
        return subprocess.call([sys.executable, str(REPO_ROOT / "test_gesture_system.py")])

    print("👋 Setup complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
