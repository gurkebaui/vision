#!/usr/bin/env python3
"""
Vision Control v2 — Long-Range, CPU-Optimized Hand Gesture Control System
Controls PowerPoint, Keynote, Google Slides, System Media/Volume, and Air Mouse
from 0.4m up to 5m+ away on standard laptops without a GPU.

Usage:
  python gesture_presentation_control.py                 # Native Desktop HUD + OS Control
  python gesture_presentation_control.py --studio        # Launch Web Control Studio + OS Bridge
  python gesture_presentation_control.py --preset ultra_light --profile media
  python gesture_presentation_control.py --benchmark     # Run CPU latency & FPS benchmark
"""

from __future__ import annotations

import argparse
import sys
import time
import webbrowser
from typing import Optional

import cv2
import numpy as np

from vision_control import __version__
from vision_control.actions import OSActionController
from vision_control.camera import ThreadedCamera
from vision_control.config import PERFORMANCE_PRESETS, PROFILE_ORDER, SystemConfig
from vision_control.gestures import ScaleInvariantGestureEngine
from vision_control.hud import ModernHUDRenderer
from vision_control.tracker import LongRangeHandTracker


class GesturePresentationController:
    """
    Main desktop runtime combining ThreadedCamera, LongRangeHandTracker,
    ScaleInvariantGestureEngine, OSActionController, and ModernHUDRenderer.
    """

    def __init__(self, config: Optional[SystemConfig] = None) -> None:
        self.config = config or SystemConfig.load()
        self.tracker = LongRangeHandTracker(self.config)
        self.gesture_engine = ScaleInvariantGestureEngine(self.config)
        self.actions = OSActionController(self.config)
        self.hud = ModernHUDRenderer(self.config)
        self.camera: Optional[ThreadedCamera] = None

        self.stats = {
            "total_frames": 0,
            "gestures_triggered": 0,
            "start_time": time.perf_counter(),
        }

    def start_camera(self) -> bool:
        preset = self.config.get_preset()
        self.camera = ThreadedCamera(
            camera_index=self.config.camera_index,
            width=preset.capture_width,
            height=preset.capture_height,
            fps=preset.target_fps,
            mirror=self.config.mirror_camera,
        )
        ok = self.camera.start()
        if not ok:
            print(
                f"❌ Cannot open camera index {self.config.camera_index}. "
                "Try --camera 1 or launch Web Studio with --studio."
            )
            return False
        print(
            f"✅ Camera active at {self.camera.actual_width}x{self.camera.actual_height} "
            f"(Preset: {preset.label})"
        )
        return True

    def run(self) -> int:
        print(f"🚀 Vision Gesture Control v{__version__}")
        print("=" * 56)
        print(f"  Profile     : {self.config.get_active_profile_data()['name']}")
        print(f"  CPU Preset  : {self.config.get_preset().label}")
        print(f"  Activation  : {self.config.activation_mode.upper()}")
        print(f"  Smart Zoom  : {'ON (0.4m - 5.0m+ Range)' if self.config.smart_zoom_enabled else 'OFF'}")
        print("=" * 56)

        if not self.start_camera() or self.camera is None:
            return 1

        win_name = "Vision Control v2 — Long-Range Gesture HUD"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

        last_frame_id = 0
        ema_fps = 30.0
        prev_loop_t = time.perf_counter()

        try:
            while True:
                is_new, frame, frame_id, frame_ts = self.camera.read_latest(last_frame_id)
                if not is_new or frame is None:
                    key = cv2.waitKey(2) & 0xFF
                    if not self._handle_key(key):
                        break
                    continue

                last_frame_id = frame_id
                work_frame = frame.copy()
                fh, fw = work_frame.shape[:2]
                aspect = float(fw) / float(max(1, fh))

                # 1. Long-Range 3-Tier Tracking
                track_res = self.tracker.process_frame(work_frame, timestamp=frame_ts)

                # 2. Scale-Invariant Biomechanical Gesture & Swipe Evaluation
                eval_res = self.gesture_engine.update(
                    track_res.landmarks if track_res.detected else None,
                    now=frame_ts,
                    aspect_ratio=aspect,
                )

                # 3. Continuous 1-Euro Air Mouse / Laser Pointer Cursor
                self.actions.update_cursor(eval_res.cursor_xy, eval_res.is_pinching)

                # 4. Triggered Command Execution
                if eval_res.triggered_action:
                    evt = self.actions.execute(
                        eval_res.triggered_action,
                        gesture=eval_res.triggered_gesture or "gesture",
                    )
                    if evt is not None:
                        self.stats["gestures_triggered"] += 1
                        self.hud.trigger_toast(f"{eval_res.gesture_icon}  {evt['label']}")
                        print(
                            f"[{evt['time_str']}] {evt['label']} "
                            f"({eval_res.gesture_name}, ~{track_res.estimated_distance_m:.1f}m)"
                        )

                # 5. Telemetry & Fast Sub-ROI HUD Render
                now_t = time.perf_counter()
                dt = max(1e-4, now_t - prev_loop_t)
                prev_loop_t = now_t
                ema_fps = 0.88 * ema_fps + 0.12 * min(120.0, 1.0 / dt)
                self.stats["total_frames"] += 1

                display_frame = self.hud.render(work_frame, track_res, eval_res, ema_fps)
                cv2.imshow(win_name, display_frame)

                key = cv2.waitKey(1) & 0xFF
                if not self._handle_key(key):
                    break

        except KeyboardInterrupt:
            print("\n👋 Interrupted by user.")
        finally:
            self.cleanup()
        return 0

    def _handle_key(self, key: int) -> bool:
        if key in (255, 0):
            return True
        if key == ord("q") or key == 27:
            return False
        if key == ord("r"):
            self.tracker.reset()
            self.gesture_engine.reset()
            self.hud.trigger_toast("Tracker & Calibration Reset")
        elif key == ord("m"):
            self.config.compact_hud = not self.config.compact_hud
            self.config.save()
        elif key == ord("p"):
            new_prof = self.config.cycle_profile()
            self.config.save()
            self.hud.trigger_toast(
                f"Profile: {self.config.get_active_profile_data()['name']}"
            )
        elif key == ord("c"):
            modes = ["clutch", "dwell", "instant"]
            idx = modes.index(self.config.activation_mode)
            self.config.activation_mode = modes[(idx + 1) % len(modes)]
            self.gesture_engine.reset()
            self.config.save()
            self.hud.trigger_toast(f"Activation: {self.config.activation_mode.upper()}")
        elif key in (ord("1"), ord("2"), ord("3")):
            preset_keys = ["ultra_light", "balanced", "max_range"]
            chosen = preset_keys[key - ord("1")]
            self.config.performance_preset = chosen
            preset = self.config.get_preset()
            if self.camera is not None:
                self.camera.reconfigure(
                    preset.capture_width, preset.capture_height, self.config.mirror_camera
                )
            self.config.save()
            self.hud.trigger_toast(f"CPU Preset: {preset.label}")
        return True

    def cleanup(self) -> None:
        if self.camera is not None:
            self.camera.stop()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        self.tracker.close()
        elapsed = max(0.1, time.perf_counter() - self.stats["start_time"])
        print(
            f"\n📊 Session Summary: {self.stats['total_frames']} frames in {elapsed:.1f}s "
            f"({self.stats['total_frames']/elapsed:.1f} avg FPS), "
            f"{self.stats['gestures_triggered']} commands executed."
        )


def run_benchmark(config: SystemConfig) -> int:
    print("⚡ Running CPU Long-Range Pipeline Benchmark...")
    tracker = LongRangeHandTracker(config)
    try:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        for _ in range(4):
            tracker.process_frame(frame)
        n = 25
        t0 = time.perf_counter()
        for _ in range(n):
            tracker.process_frame(frame)
        avg_ms = (time.perf_counter() - t0) / n * 1000.0
        print(f"  Backend          : {tracker._backend_type}")
        print(f"  Preset           : {config.get_preset().label}")
        print(f"  Avg Frame Latency: {avg_ms:.2f} ms")
        print(f"  Throughput Cap   : {1000.0 / max(avg_ms, 0.1):.1f} FPS on CPU")
        return 0
    finally:
        tracker.close()


def run_studio_server(
    config: SystemConfig, host: str = "0.0.0.0", port: int = 8000, open_browser: bool = False
) -> int:
    import uvicorn
    from vision_control.server import create_app

    app = create_app(config=config)
    url = f"http://localhost:{port}"
    print(f"🌐 Vision Control Web Studio running at {url} (bound to {host}:{port})")
    print("   Real-time WebSocket OS Bridge (/ws) active for PowerPoint, Media & Air Mouse.")
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    uvicorn.run(app, host=host, port=port, log_level="info")
    return 0


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vision Control v2 — Long-Range CPU Hand Gesture Control"
    )
    parser.add_argument(
        "--studio",
        action="store_true",
        help="Launch the modern Web Control Studio & WebSocket OS Bridge server",
    )
    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="Automatically open the Web Studio URL in default browser (with --studio)",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Server bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    parser.add_argument(
        "--preset",
        choices=list(PERFORMANCE_PRESETS.keys()),
        help="CPU performance & range preset",
    )
    parser.add_argument(
        "--profile",
        choices=PROFILE_ORDER,
        help="Active control profile (presentation, media, air_mouse, custom)",
    )
    parser.add_argument(
        "--mode",
        choices=["clutch", "dwell", "instant"],
        help="Gesture activation mode",
    )
    parser.add_argument("--camera", type=int, help="Camera device index (default: 0)")
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Start Desktop HUD in compact floating corner PiP mode",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run a headless CPU inference latency benchmark and exit",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    config = SystemConfig.load()

    if args.preset:
        config.performance_preset = args.preset
    if args.profile:
        config.active_profile = args.profile
    if args.mode:
        config.activation_mode = args.mode
    if args.camera is not None:
        config.camera_index = args.camera
    if args.compact:
        config.compact_hud = True

    if args.benchmark:
        return run_benchmark(config)

    if args.studio:
        return run_studio_server(
            config, host=args.host, port=args.port, open_browser=args.open_browser
        )

    controller = GesturePresentationController(config=config)
    return controller.run()


if __name__ == "__main__":
    sys.exit(main())
