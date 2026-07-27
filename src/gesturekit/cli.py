"""Command-line interface.

    gesturekit run [--profile P]     start gesture control
    gesturekit profiles              list available profiles
    gesturekit bindings [-p P]       show what each gesture does
    gesturekit devices               list cameras
    gesturekit doctor                diagnose the installation
    gesturekit download              pre-fetch the model bundle
    gesturekit init                  write a starter user config
    gesturekit bench                 measure pipeline throughput
"""

from __future__ import annotations

import argparse
import logging
import sys

from . import __version__


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gesturekit",
        description="Hand-gesture control for presentations and the desktop.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  gesturekit run                       # presentation mode\n"
            "  gesturekit run -p desktop            # virtual mouse mode\n"
            "  gesturekit run --dry-run             # print actions, press nothing\n"
            "  gesturekit run -p desktop --arm      # require the arm gesture first\n"
            "  gesturekit doctor                    # check the setup\n"
        ),
    )
    p.add_argument("--version", action="version", version=f"gesturekit {__version__}")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    sub = p.add_subparsers(dest="command")

    run = sub.add_parser("run", help="start gesture control")
    run.add_argument("-p", "--profile", default="presentation",
                     help="profile name or path to a .yaml file")
    run.add_argument("-c", "--camera", type=int, help="camera index")
    run.add_argument("--width", type=int, help="capture width")
    run.add_argument("--height", type=int, help="capture height")
    run.add_argument("--fps", type=int, help="requested capture fps")
    run.add_argument("--hands", type=int, choices=[1, 2], help="number of hands to track")
    run.add_argument("--backend", default=None,
                     choices=["auto", "pynput", "ydotool", "xdotool", "pyautogui", "dry-run"],
                     help="input backend")
    run.add_argument("--dry-run", action="store_true",
                     help="log actions instead of performing them")
    run.add_argument("--arm", dest="require_arm", action="store_true",
                     help="require the arm gesture before commands fire")
    run.add_argument("--no-arm", dest="require_arm", action="store_false",
                     help="always-on control (default)")
    run.set_defaults(require_arm=None)
    run.add_argument("--arm-window", type=float, help="seconds the arm window stays open")
    run.add_argument("--cooldown", type=float, help="seconds between repeats of a gesture")
    run.add_argument("--sensitivity", type=float, metavar="X",
                     help="0.5 = strict, 1.0 = default, 2.0 = twitchy")
    run.add_argument("--no-window", action="store_true", help="headless: no preview window")
    run.add_argument("--fullscreen", action="store_true")
    run.add_argument("--scores", action="store_true", help="show live confidence bars")
    run.add_argument("--no-flip", action="store_true", help="do not mirror the preview")
    run.add_argument("--model", default=None, help="path to a .task bundle")
    run.add_argument("--gpu", action="store_true", help="try the GPU delegate")
    run.add_argument("--theme", choices=["dark", "light"], default=None)

    b = sub.add_parser("bindings", help="show the gesture map for a profile")
    b.add_argument("-p", "--profile", default="presentation")

    sub.add_parser("profiles", help="list available profiles")
    sub.add_parser("devices", help="list cameras")
    sub.add_parser("doctor", help="diagnose the installation")

    d = sub.add_parser("download", help="pre-fetch a model bundle")
    d.add_argument("--model", default="hand_landmarker",
                   choices=["hand_landmarker", "gesture_recognizer"])

    i = sub.add_parser("init", help="write a starter user config")
    i.add_argument("--force", action="store_true")

    bench = sub.add_parser("bench", help="measure pipeline throughput")
    bench.add_argument("-n", "--frames", type=int, default=300)
    bench.add_argument("-c", "--camera", type=int, default=0)
    return p


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------

def cmd_run(args) -> int:
    from .app import GestureApp
    from .config import load_config

    overrides: dict = {}
    camera: dict = {}
    tracker: dict = {}
    engine: dict = {}
    ui: dict = {}

    if args.camera is not None:
        camera["index"] = args.camera
    if args.width:
        camera["width"] = args.width
    if args.height:
        camera["height"] = args.height
    if args.fps:
        camera["fps"] = args.fps
    if args.no_flip:
        camera["flip"] = False
    if args.hands:
        tracker["num_hands"] = args.hands
    if args.gpu:
        tracker["delegate"] = "gpu"
    if args.require_arm is not None:
        engine["require_arm"] = args.require_arm
    if args.arm_window:
        engine["arm_window"] = args.arm_window
    if args.cooldown is not None:
        engine["cooldown"] = args.cooldown
    if args.no_window:
        ui["show_window"] = False
    if args.fullscreen:
        ui["fullscreen"] = True
    if args.scores:
        ui["show_scores"] = True
    if args.theme:
        ui["theme"] = args.theme

    if args.sensitivity:
        # One knob for "how eager should this be": scales the dwell/score/
        # distance thresholds together so they stay mutually consistent.
        s = max(0.25, min(4.0, args.sensitivity))
        engine["static_min_frames"] = max(2, int(round(4 / s)))
        engine["static_min_score"] = max(0.40, min(0.90, 0.62 / (0.6 + 0.4 * s)))
        engine["cooldown"] = engine.get("cooldown", 0.45) / s
        engine["swipe"] = {"min_distance": 1.15 / s, "min_speed": 2.2 / s}

    for key, value in (("camera", camera), ("tracker", tracker),
                       ("engine", engine), ("ui", ui)):
        if value:
            overrides[key] = value
    if args.backend:
        overrides["backend"] = args.backend
    if args.dry_run:
        overrides["dry_run"] = True
    if args.model:
        overrides["model_path"] = args.model

    try:
        config = load_config(args.profile, overrides)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        print("DRY RUN: gestures will be logged, no keys or clicks are sent.\n")

    return GestureApp(config).run()


def cmd_profiles(args) -> int:
    from .config import list_profiles

    rows = list_profiles()
    if not rows:
        print("No profiles found.")
        return 1
    width = max(len(n) for n, _, _ in rows)
    print(f"{'PROFILE'.ljust(width)}  DESCRIPTION")
    for name, desc, _path in rows:
        print(f"{name.ljust(width)}  {desc}")
    print("\nUse: gesturekit run -p <profile>")
    return 0


def cmd_bindings(args) -> int:
    from .config import load_config

    try:
        cfg = load_config(args.profile)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"profile: {cfg.profile}")
    if cfg.profile_description:
        print(f"         {cfg.profile_description}")
    print()

    groups = {"static": [], "motion": [], "pointer": [], "other": []}
    motion = {"swipe_left", "swipe_right", "swipe_up", "swipe_down",
              "circle_cw", "circle_ccw", "wave"}
    pointer = {"cursor_move", "click", "drag_start", "drag_end", "scroll",
               "zoom_in", "zoom_out"}
    for gesture, action in sorted(cfg.bindings.items()):
        if action.kind == "none":
            continue
        bucket = "motion" if gesture in motion else "pointer" if gesture in pointer else "static"
        groups[bucket].append((gesture, action))

    for title, items in (("HAND SHAPES", groups["static"]),
                         ("MOTION", groups["motion"]),
                         ("POINTER / ZOOM", groups["pointer"])):
        if not items:
            continue
        print(f"  {title}")
        width = max(len(g) for g, _ in items)
        for gesture, action in items:
            print(f"    {gesture.ljust(width)}   {action.label}")
        print()
    return 0


def cmd_devices(args) -> int:
    from .camera import list_devices

    print("Scanning camera indices 0-7 ...")
    found = list_devices()
    if not found:
        print("No cameras found. On Linux check: ls -l /dev/video*")
        return 1
    print(f"\n{'INDEX':<7}{'RESOLUTION':<14}FPS")
    for dev in found:
        resolution = f"{dev['width']}x{dev['height']}"
        print(f"{dev['index']:<7}{resolution:<14}{dev['fps']}")
    print("\nUse: gesturekit run -c <index>")
    return 0


def cmd_download(args) -> int:
    from . import models
    from .app import _progress

    try:
        existing = models.find_model(args.model)
        if existing:
            print(f"Already cached: {existing}")
            return 0
        print(f"Downloading {args.model} to {models.cache_dir()} ...")
        path = models.download_model(args.model, progress=_progress)
        size = path.stat().st_size / 1e6
        print(f"Saved {path} ({size:.1f} MB)")
        return 0
    except models.ModelError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def cmd_init(args) -> int:
    from .config import config_dir, write_default_user_config

    target = config_dir() / "config.yaml"
    if target.exists() and not args.force:
        print(f"{target} already exists. Use --force to overwrite.")
        return 1
    path = write_default_user_config()
    print(f"Wrote {path}")
    print("Edit it to change camera, thresholds or the default backend.")
    print(f"Custom profiles go in: {config_dir() / 'profiles'}")
    return 0


def cmd_doctor(args) -> int:
    """Check every moving part and say exactly what to do about failures."""
    import platform

    ok = True
    print("GestureKit doctor")
    print("=" * 58)
    print(f"python      {sys.version.split()[0]}  ({platform.system()} {platform.machine()})")

    try:
        import cv2
        print(f"opencv      {cv2.__version__}")
    except Exception as exc:
        print(f"opencv      MISSING ({exc})")
        print("            fix: pip install 'opencv-python>=4.8'")
        ok = False

    try:
        import mediapipe as mp
        version = getattr(mp, "__version__", "unknown")
        print(f"mediapipe   {version}")
        try:
            from mediapipe.tasks.python import vision  # noqa: F401
            print("            tasks API   available")
        except Exception:
            print("            tasks API   MISSING -- upgrade: pip install -U mediapipe")
            ok = False
    except Exception as exc:
        print(f"mediapipe   MISSING ({exc})")
        print("            fix: pip install 'mediapipe>=0.10.9'")
        ok = False

    try:
        import numpy
        print(f"numpy       {numpy.__version__}")
    except Exception:
        print("numpy       MISSING")
        ok = False

    from . import models
    cached = models.find_model("hand_landmarker")
    if cached:
        print(f"model       {cached} ({cached.stat().st_size / 1e6:.1f} MB)")
    else:
        print(f"model       not cached -- will download to {models.cache_dir()}")
        print("            pre-fetch with: gesturekit download")

    from .actions.backends import available_backends, select_backend
    backends = available_backends()
    print(f"input       available: {', '.join(backends)}")
    try:
        chosen = select_backend("auto")
        print(f"            selected:  {chosen.name}")
        if chosen.name == "dry-run":
            print("            fix: pip install 'gesturekit[input]'")
            ok = False
        else:
            print(f"            screen:    {chosen.screen_size()[0]}x{chosen.screen_size()[1]}")
    except Exception as exc:
        print(f"            selection failed: {exc}")
        ok = False

    import os
    session = os.environ.get("XDG_SESSION_TYPE", "")
    if session:
        print(f"session     {session}")
        if session.lower() == "wayland" and "ydotool" not in backends:
            print("            WARNING: Wayland blocks X11 input injection.")
            print("            fix: install ydotool and run ydotoold, or log into an X11 session.")

    try:
        from .camera import list_devices
        devices = list_devices(max_index=4)
        if devices:
            for d in devices:
                print(f"camera      index {d['index']}: {d['width']}x{d['height']} @ {d['fps']}")
        else:
            print("camera      none found")
            print("            fix: check ls -l /dev/video* and camera permissions")
            ok = False
    except Exception as exc:
        print(f"camera      probe failed: {exc}")
        ok = False

    from .config import config_dir, list_profiles
    print(f"profiles    {', '.join(n for n, _, _ in list_profiles())}")
    print(f"config      {config_dir()}")

    print("=" * 58)
    print("All good -- run: gesturekit run" if ok else "Problems found; see the fixes above.")
    return 0 if ok else 1


def cmd_bench(args) -> int:
    """Measure real end-to-end throughput on this machine."""
    import time

    import cv2
    import numpy as np

    from . import models
    from .camera import Camera, CameraConfig
    from .gestures.engine import GestureEngine
    from .tracker import HandTracker

    try:
        path = models.ensure_model("hand_landmarker")
    except models.ModelError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"Benchmarking {args.frames} frames ...")
    tracker = HandTracker(path)
    engine = GestureEngine()
    cam = Camera(CameraConfig(index=args.camera)).open()

    times, infer = [], []
    t_start = time.perf_counter()
    try:
        for _ in range(args.frames):
            t0 = time.perf_counter()
            is_new, frame, _ = cam.read()
            if frame is None:
                time.sleep(0.005)
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tracker.submit(rgb, time.perf_counter())
            result, _ = tracker.latest()
            hands = tracker.to_states(result, time.perf_counter())
            engine.process(hands)
            times.append((time.perf_counter() - t0) * 1000)
            infer.append(tracker.inference_ms)
    finally:
        cam.release()
        tracker.close()

    if not times:
        print("No frames captured.")
        return 1
    arr = np.array(times)
    total = time.perf_counter() - t_start
    print(f"\n  frames        {len(arr)}")
    print(f"  wall time     {total:.1f}s  ->  {len(arr) / total:.1f} fps")
    print(f"  loop  mean    {arr.mean():.2f} ms")
    print(f"  loop  p50     {np.percentile(arr, 50):.2f} ms")
    print(f"  loop  p95     {np.percentile(arr, 95):.2f} ms")
    print(f"  inference avg {np.mean([i for i in infer if i]) if any(infer) else 0:.2f} ms")
    print(f"  dropped       {cam.dropped}")
    return 0


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.log_level)

    if not args.command:
        # Bare `gesturekit` should do the obvious thing.
        args = parser.parse_args(["run"] + (argv or [])[0:0])
        args.command = "run"

    handlers = {
        "run": cmd_run,
        "profiles": cmd_profiles,
        "bindings": cmd_bindings,
        "devices": cmd_devices,
        "download": cmd_download,
        "init": cmd_init,
        "doctor": cmd_doctor,
        "bench": cmd_bench,
    }
    handler = handlers.get(args.command)
    if handler is None:
        parser.print_help()
        return 1
    try:
        return handler(args)
    except KeyboardInterrupt:
        print()
        return 130


if __name__ == "__main__":
    sys.exit(main())
