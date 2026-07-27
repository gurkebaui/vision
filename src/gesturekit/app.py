"""Application loop.

Structure of one iteration:

1. grab the freshest camera frame (never a stale queued one)
2. hand it to MediaPipe *asynchronously* -- inference happens on its own thread
3. consume whatever result is ready, convert to :class:`HandState`
4. run the recognisers, route events to the input backend
5. draw the HUD and show the window

Because step 2 does not block, capture, inference and rendering overlap.  v1
ran all three serially, so its frame time was the *sum* of the three; here it
is roughly the slowest one.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import cv2

from . import models
from .actions.backends import InputBackend, select_backend
from .actions.router import ActionRouter
from .camera import Camera, CameraError
from .config import AppConfig, list_profiles, load_config
from .filters import EMA
from .gestures.engine import GestureEngine
from .hud import HUD, draw_help_screen
from .tracker import HandTracker, TrackerError
from .types import FrameResult

log = logging.getLogger("gesturekit")


class GestureApp:
    """Wires camera, tracker, engine, router and HUD together."""

    def __init__(self, config: Optional[AppConfig] = None, backend: Optional[InputBackend] = None):
        self.config = config or load_config()
        self.camera: Optional[Camera] = None
        self.tracker: Optional[HandTracker] = None
        self.engine = GestureEngine(self.config.engine)
        self.hud = HUD(
            theme=self.config.ui.theme,
            show_landmarks=self.config.ui.show_landmarks,
            show_scores=self.config.ui.show_scores,
            show_trail=self.config.ui.show_trail,
        )

        self.backend = backend or select_backend(self.config.backend, self.config.dry_run)
        self.router = ActionRouter(dict(self.config.bindings), self.backend)
        self.router.on_mode_change = self.switch_profile

        self.paused = False
        self.show_help = False
        self.running = False
        self._fps = EMA(0.12)
        self._latency = EMA(0.12)
        self._frames = 0
        self._started = 0.0
        self._last_result_id = None

    # -- setup -------------------------------------------------------------
    def prepare(self) -> None:
        cfg = self.config
        model_path = models.ensure_model(
            cfg.model,
            explicit=cfg.model_path,
            allow_download=cfg.allow_download,
            progress=_progress,
        )
        log.info("Model: %s", model_path)
        self.tracker = HandTracker(model_path, cfg.tracker)
        self.camera = Camera(cfg.camera).open()
        w, h = self.camera.resolution
        log.info("Camera: %dx%d (index %d)", w, h, cfg.camera.index)
        log.info("Input backend: %s", self.backend.name)
        log.info("Profile: %s -- %d bindings", cfg.profile, len(self.router.bindings))

    def switch_profile(self, name: str) -> None:
        """Hot-swap the active profile without restarting the camera."""
        try:
            new_cfg = load_config(name)
        except Exception as exc:
            log.error("Cannot load profile '%s': %s", name, exc)
            self.hud.notify(f"profile '{name}' failed", "bad")
            return

        self.config.profile = new_cfg.profile
        self.config.bindings = new_cfg.bindings
        self.config.engine = new_cfg.engine
        self.router.bindings = dict(new_cfg.bindings)
        self.engine = GestureEngine(new_cfg.engine)
        self.hud.show_scores = new_cfg.ui.show_scores
        log.info("Switched to profile: %s", new_cfg.profile)
        self.hud.notify(f"profile: {new_cfg.profile}", "good")

    def cycle_profile(self) -> None:
        names = [n for n, _, _ in list_profiles()]
        if not names:
            return
        try:
            idx = names.index(self.config.profile)
        except ValueError:
            idx = -1
        self.switch_profile(names[(idx + 1) % len(names)])

    # -- main loop ---------------------------------------------------------
    def run(self) -> int:
        try:
            self.prepare()
        except (models.ModelError, TrackerError, CameraError) as exc:
            log.error("%s", exc)
            return 2

        self.running = True
        self._started = time.perf_counter()
        window = self.config.ui.window_title
        show_window = self.config.ui.show_window

        if show_window:
            try:
                cv2.namedWindow(window, cv2.WINDOW_NORMAL)
                if self.config.ui.width:
                    cv2.resizeWindow(window, self.config.ui.width,
                                     int(self.config.ui.width * 9 / 16))
                if self.config.ui.fullscreen:
                    cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN,
                                          cv2.WINDOW_FULLSCREEN)
            except cv2.error:
                # opencv-python-headless has no GUI support. Carry on without a
                # preview rather than dying -- gesture control still works.
                log.warning(
                    "OpenCV has no GUI support (headless build): running without a "
                    "preview window. Install 'opencv-python' instead of "
                    "'opencv-python-headless' if you want one."
                )
                show_window = False
                self.config.ui.show_window = False

        last_tick = time.perf_counter()
        try:
            while self.running:
                if not self.camera.is_open:
                    log.error("Camera stopped delivering frames.")
                    break

                is_new, frame, stamp = self.camera.read()
                if frame is None:
                    time.sleep(0.004)
                    continue

                now = time.perf_counter()
                result = FrameResult()

                if is_new and not self.paused:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.tracker.submit(rgb, now)

                raw, raw_stamp = self.tracker.latest()
                if raw is not None and raw_stamp != self._last_result_id:
                    self._last_result_id = raw_stamp
                    hands = self.tracker.to_states(raw, now)
                    result.hands = hands
                    if not self.paused:
                        events = self.engine.process(hands, now)
                        result.events = events
                        performed = self.router.dispatch(events)
                        labels = {g: a.label for g, a in self.router.bindings.items()}
                        self.hud.push_events(events, labels)
                        for ev, action in performed:
                            log.info("%s -> %s (%.0f%%)", ev.name, action.label, ev.confidence * 100)
                        for ev in events:
                            if ev.kind == "meta":
                                self.hud.notify(ev.name.replace("_", " "), "good")
                else:
                    result.hands = []

                dt = now - last_tick
                last_tick = now
                if dt > 0:
                    result.fps = self._fps.update(1.0 / dt) or 0.0
                result.latency_ms = self._latency.update((now - stamp) * 1000.0) or 0.0
                result.inference_ms = self.tracker.inference_ms
                result.dropped = self.camera.dropped
                self._frames += 1

                if show_window:
                    if self.show_help:
                        h, w = frame.shape[:2]
                        canvas = draw_help_screen(
                            w, h, self.router.describe(), self.config.profile,
                            self.config.ui.theme
                        )
                    else:
                        canvas = self.hud.draw(
                            frame,
                            result,
                            profile=self.config.profile,
                            backend=self.backend.name,
                            armed=self.engine.armed,
                            arm_remaining=self.engine.arm_remaining,
                            require_arm=self.config.engine.require_arm,
                            paused=self.paused,
                            scores=self.engine.any_scores(),
                        )
                    cv2.imshow(window, canvas)

                    key = cv2.waitKey(1) & 0xFF
                    if key != 255 and not self._handle_key(key):
                        break
                    try:
                        if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                            break
                    except cv2.error:
                        break
                else:
                    time.sleep(0.001)

        except KeyboardInterrupt:
            log.info("Interrupted.")
        finally:
            self.shutdown()
        return 0

    def _handle_key(self, key: int) -> bool:
        """Return False to quit."""
        char = chr(key) if 32 <= key < 127 else ""
        if key == 27 or char == "q":
            return False
        if char == " ":
            self.paused = not self.paused
            self.hud.notify("paused" if self.paused else "resumed", "warn" if self.paused else "good")
        elif char == "p":
            self.cycle_profile()
        elif char == "s":
            self.hud.show_scores = not self.hud.show_scores
        elif char == "l":
            self.hud.show_landmarks = not self.hud.show_landmarks
        elif char == "t":
            self.hud.show_trail = not self.hud.show_trail
        elif char == "r":
            self.engine.reset()
            self.tracker.reset()
            self.hud.notify("reset", "good")
        elif char == "h":
            self.show_help = not self.show_help
        elif char == "d":
            self.router.enabled = not self.router.enabled
            self.hud.notify("actions on" if self.router.enabled else "actions off",
                            "good" if self.router.enabled else "warn")
        return True

    def shutdown(self) -> None:
        self.running = False
        self.router.release()
        if self.camera:
            self.camera.release()
        if self.tracker:
            self.tracker.close()
        if self.backend:
            self.backend.close()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            # Headless OpenCV builds (servers, minimal Linux, some WSL setups)
            # have no highgui at all. Shutting down must never fail on that.
            pass

        elapsed = max(time.perf_counter() - self._started, 1e-6)
        log.info(
            "Session: %d frames in %.1fs (%.1f fps avg), %d actions, %d dropped frames",
            self._frames, elapsed, self._frames / elapsed,
            self.router.stats.executed, self.camera.dropped if self.camera else 0,
        )
        if self.router.stats.by_gesture:
            top = sorted(self.router.stats.by_gesture.items(), key=lambda kv: -kv[1])
            log.info("Most used: %s", ", ".join(f"{g}={n}" for g, n in top[:5]))


def _progress(done: int, total: int) -> None:
    if not total:
        return
    pct = 100.0 * done / total
    bar = int(pct / 4)
    print(f"\r  downloading model [{'#' * bar}{'.' * (25 - bar)}] {pct:5.1f}%",
          end="", flush=True)
    if done >= total:
        print()
