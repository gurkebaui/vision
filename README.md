# GestureKit

Hand-gesture control for presentations and the desktop. Point at your screen to
move the cursor, pinch to click, swipe to change slides — all processed locally
on your machine.

Built on **OpenCV** and the **MediaPipe Tasks API**, with a recognition
pipeline designed around one goal: never fire an action you did not intend.

```bash
pip install -e ".[input]"
gesturekit doctor        # check the setup
gesturekit run           # presentation mode
gesturekit run -p desktop # virtual mouse mode
```

---

## Why this exists in this shape

Version 1 of this project used `mediapipe.solutions.hands`. That API **no
longer exists** — MediaPipe 1.0 removed it, so the old script now fails at
import with `ModuleNotFoundError`. Fixing that meant rebuilding the recognition
layer anyway, so the accuracy and latency problems got fixed at the same time.

| | v1 | v2 |
|---|---|---|
| MediaPipe API | `mp.solutions` (removed) | `mediapipe.tasks` (current) |
| Inference | blocking, in the capture loop | async LIVE_STREAM, own thread |
| Landmark smoothing | none | One Euro filter |
| Finger detection | `tip.y < pip.y` — breaks on a rotated hand | joint-angle curl, rotation invariant |
| Thresholds | pixel constants | palm-relative, distance invariant |
| Gesture decision | one frame over a threshold | N-frame vote + margin + dwell |
| Swipe detection | 2-frame delta > 0.05 | trajectory, straightness, speed, refractory |
| Gestures | 6 (2 unimplemented) | 15 shapes + swipes + circles + wave + pinch/drag/scroll/zoom |
| Key mapping | hard-coded `if` chain | YAML profiles, hot-swappable |
| Linux input | PyAutoGUI (X11 only) | pynput / ydotool (Wayland) / xdotool / PyAutoGUI |
| Tests | none | 425 |

### Concrete bugs that were fixed

- **The 3-second arming window never worked.** `self.finger_lift_time` was
  initialised to the integer `1`, so the first expiry check compared against
  1970. Arming also required a fist in the buffer *and* only ever triggered on
  `pointing_up`, contradicting its own instructions.
- **Swipes fired on noise.** A single-frame centroid delta over `0.05` counted
  as a swipe. Landmark jitter alone clears that.
- **Curling your fingers faked a swipe.** The "hand centre" averaged all 21
  landmarks, so closing your hand moved it far enough to trigger.
- **Peace sign was miscoded.** The branch checked `extended_count == 1 and
  'middle' in fingers` — that is one finger, not two, so a real peace sign
  never matched it.
- **The thumb was disabled.** Thumb detection was commented out, yet
  `thumbs_up` still tested for `'thumb' in extended_fingers` — permanently
  unreachable.
- **Confidences were fictional.** Hard-coded values like `0.25` and `0.45` sat
  below the `0.7` gate, so several gestures could never fire regardless of how
  well you performed them.
- **The UI printed `?` boxes.** OpenCV's Hershey fonts cannot render emoji, and
  every label used them.

---

## Install

Requires Python 3.9+ and a webcam.

```bash
git clone https://github.com/gurkebaui/vision
cd vision
pip install -e ".[input]"
```

The MediaPipe model (~7 MB) downloads automatically on first run and is cached
in `~/.cache/gesturekit`. To pre-fetch it, or for an offline machine:

```bash
gesturekit download
# offline: copy hand_landmarker.task there yourself, or set
# GESTUREKIT_MODEL_DIR=/path/to/models
```

### Linux notes

| Session | Backend | Setup |
|---|---|---|
| X11 | `pynput` | works out of the box |
| Wayland | `ydotool` | `sudo apt install ydotool`, then run `ydotoold` |

Wayland deliberately blocks applications from injecting input, so X11-based
tools silently do nothing there. `gesturekit doctor` detects this and tells you.

---

## Usage

```bash
gesturekit run                    # presentation profile
gesturekit run -p desktop         # virtual mouse
gesturekit run -p media           # video players
gesturekit run -p accessibility   # forgiving thresholds

gesturekit run --dry-run          # log gestures, press nothing — try this first
gesturekit run --scores           # live confidence bars, for tuning
gesturekit run --sensitivity 1.5  # one knob: higher = more eager
gesturekit run --arm              # require an arm gesture before commands fire

gesturekit profiles               # list profiles
gesturekit bindings -p desktop    # what each gesture does
gesturekit devices                # list cameras
gesturekit doctor                 # diagnose problems
gesturekit bench                  # measure throughput on your machine
```

### While it is running

| Key | Action |
|---|---|
| `q` / `Esc` | quit |
| `space` | pause detection |
| `p` | next profile |
| `h` | binding cheat sheet |
| `s` | toggle confidence bars |
| `l` / `t` | toggle landmarks / motion trail |
| `d` | disable actions (keep detecting) |
| `r` | reset the tracker |

---

## Gestures

**Shapes** — open palm, closed fist, point up/down/left/right, peace, thumbs
up/down, pinch, OK, rock, call me, three, four.

**Motion** — swipe left/right/up/down, circle clockwise/counter-clockwise, wave.

**Continuous** — cursor move, click, drag, two-finger scroll, two-hand zoom.

### Presentation profile

| Gesture | Action |
|---|---|
| Swipe left / right | Next / previous slide |
| Open palm | Advance |
| Peace | Blackout screen (`B`) |
| Thumbs up | Whiteout screen (`W`) |
| Point up | Laser pointer |
| Rock | Pen tool |
| Three / Four | Start slideshow / present from here |
| Closed fist | End slideshow |
| Wave | Switch to desktop mode |

### Desktop profile

| Gesture | Action |
|---|---|
| Point | Move cursor |
| Pinch | Click |
| Pinch and hold | Drag |
| Two fingers | Scroll |
| Two pinching hands | Zoom |
| Swipe left / right | Switch workspace |
| Thumbs up / down | Volume |
| Open palm | Alt-Tab |

Run `gesturekit bindings -p <profile>` for the full list.

---

## Configuration

```bash
gesturekit init     # writes ~/.config/gesturekit/config.yaml
```

Profiles are YAML and can extend each other. Drop your own in
`~/.config/gesturekit/profiles/`:

```yaml
name: my-setup
extends: presentation
description: My shortcuts

engine:
  cooldown: 0.8           # slower repeats
  static_min_frames: 6    # longer dwell before a gesture counts

gestures:
  peace:      { hotkey: [ctrl, shift, b], label: "My shortcut" }
  thumbs_up:  { command: "notify-send 'nice'" }
  closed_fist: null       # unbind
```

Action types: `key`, `hotkey`, `text`, `click`, `scroll`, `cursor`, `drag`,
`mode` (switch profile), `command` (run a program), `null` (unbind).

### Tuning

| Symptom | Fix |
|---|---|
| Gestures fire accidentally | `--sensitivity 0.7`, or raise `static_min_frames` |
| Gestures need holding too long | `--sensitivity 1.4`, or lower `static_min_frames` |
| Swipes are missed | lower `engine.swipe.min_distance` / `min_speed` |
| Swipes fire while repositioning | raise `min_speed` and `min_straightness` |
| Cursor is jittery | lower `engine.pointer.smoothing` |
| Cursor lags | raise `engine.pointer.beta` |

---

## How it works

```
camera thread ──► newest frame only (stale frames dropped)
                       │
                       ▼
          MediaPipe Tasks, LIVE_STREAM mode  ──► own thread, non-blocking
                       │
                       ▼
          One Euro filter ──► palm-relative geometry (curl, pinch, velocity)
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
    static rules   trajectory     continuous
    + stabiliser    analysis      state machines
        └──────────────┼──────────────┘
                       ▼
              arming ──► cooldowns ──► action router ──► input backend
```

**Why it is accurate.** Finger curl is computed from joint *angles*, so it
works with the hand rotated, tilted or sideways. Every threshold is expressed
in palm widths, so it behaves identically at 40 cm and 3 m. Scores are smooth
membership functions combined with a geometric mean, so an ambiguous pose comes
out as *low confidence* rather than *confidently wrong*. Nothing fires until a
gesture wins several consecutive frames by a clear margin.

**Why it is fast.** Capture, inference and rendering overlap instead of running
in series. Landmarks live in one `(21,3)` NumPy array. HUD panels blend only
their own rectangle rather than copying the whole frame.

Measured on the CPU side (excluding the neural network): geometry 0.14 ms,
all 15 gesture rules 0.52 ms, full engine 0.87 ms, HUD at 720p 1.37 ms —
about 2.9 ms per frame total. Run `gesturekit bench` for your own numbers.

---

## Browser demo

`web/` is a self-contained page that runs the **same recognition logic**
(ported to JS and verified against the Python implementation by
`tests/test_web_parity.py`) using MediaPipe's web build.

```bash
python -m http.server -d web 8000   # then open http://localhost:8000
```

It shows what the recogniser detects. It cannot control your computer —
browsers forbid that — so use the Python app for actual control.

---

## Development

```bash
pip install -e ".[dev,input]"
pytest                 # 425 tests, no camera or model needed
ruff check src tests
```

The test suite builds anatomically plausible synthetic hands
(`tests/synth.py`), so the whole pipeline — geometry, recognisers, engine,
routing, the app loop — is verified in CI on a machine with no webcam.

```
src/gesturekit/
  geometry.py      landmark maths (pure NumPy)
  filters.py       One Euro filter, hysteresis
  tracker.py       MediaPipe Tasks wrapper
  camera.py        threaded capture
  gestures/        static / dynamic / continuous recognisers + engine
  actions/         input backends + action router
  profiles/        YAML gesture maps
  hud.py           overlay rendering
  app.py           main loop
  cli.py           command line
```

## Privacy

Everything runs locally. No network access is used after the one-time model
download, nothing is recorded, and nothing is uploaded.

## License

MIT
