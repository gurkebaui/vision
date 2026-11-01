# ⌖ Vision Control v2 — Long-Range CPU Hand Gesture Control

Control **PowerPoint, Keynote, Google Slides, PDFs, System Media/Volume, and an Air Mouse / Laser Pointer** from **0.4m up to 5.0m+ away** — engineered specifically to run fast and cool on **standard laptops without a dedicated GPU**.

---

## 🚀 What Was Upgraded in v2.0

| Bottleneck in Legacy Repo | v2.0 Engineering Upgrade |
| :--- | :--- |
| **Stopped tracking past ~1m** (full 720p frame crushed into a `192×192` palm tensor turned distant hands into ~7px blobs; hardcoded `0.02` screen-fraction thresholds treated smaller hands as a closed fist) | **3-Tier Long-Range Smart-Zoom ROI Pipeline** (Temporal Velocity Crop + Upper-Body Pose Wrist Spotlight + Multi-Zone Auditorium Scout Tiles + local CLAHE contrast boost) feeds native sensor pixels to the hand landmarker at up to **4.5× optical-equivalent zoom**, paired with **Scale-Invariant Palm-Unit (PU) 3D Joint-Angle Geometry**. |
| **Heavy CPU load & low FPS on GPU-less laptops** (blocking USB reads, `model_complexity=1` full-frame inference, and two full-frame 720p `frame.copy()` alpha blends per frame) | **XNNPACK Lite (`model_complexity=0`) on 224–256px Smart Crops (~8–12ms CPU)**, **Asynchronous MJPG Threaded Capture** (unlocking 30–60 FPS over USB 2.0), and **Sub-ROI In-Place HUD Compositing (<0.4ms)**. |
| **Broken / misfiring controls** (thumb logic commented out, `right` arrow & zoom commented out, confidence bugs, rebound swipe reversals, talking-hand false triggers) | **13 Calibrated Static & Dynamic Gestures**, **Return-Stroke Swipe Suppression**, **1-Euro Adaptive Filter** (`1€ Filter`) for jitter-free Laser/Air-Mouse pointing, **Resting-Arm Rejection**, and **Presenter Clutch / Radial Dwell Gating**. |
| **Dated static UI & disconnected JS** | **Precision Dark-Glass Web Control Studio** + **Real-Time WebSocket OS Bridge (`/ws`)**, Live Smart-Zoom PiP Monitor, Interactive Presentation/Laser/Ink Stage, Range Simulator (`0.5m–5.0m`), and Custom Keymap Editor. |

---

## ⚡ Quick Start

### 1. Install Dependencies
```bash
python -m venv .venv
# Linux / macOS:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Run Native Desktop Long-Range HUD (Webcam → Python → OS)
```bash
python gesture_presentation_control.py
```
Optional flags:
```bash
# Ultra-light CPU mode (<8% CPU on thin & light laptops):
python gesture_presentation_control.py --preset ultra_light

# Maximum auditorium range (1080p Smart-Zoom up to 5m+):
python gesture_presentation_control.py --preset max_range

# Start directly in Media/Volume or Air-Mouse mode, or compact corner PiP window:
python gesture_presentation_control.py --profile media --mode dwell --compact
```

### 3. Launch the Web Control Studio + Real-Time OS Bridge
```bash
python gesture_presentation_control.py --studio --open-browser
```
Opens the **Vision Control Studio** at `http://localhost:8000` with:
- **Control Studio**: Live webcam Long-Range Smart-Zoom optics (or camera-free **0.5m–5.0m Range Simulator**), Smart-Zoom PiP inset, and live telemetry.
- **Presentation & Media Stage**: Fullscreen slide deck with **1-Euro Laser Pointer**, **Pinch-to-Draw Ink Annotations**, **Media Volume/Track Sandbox**, **Air-Scroll Teleprompter**, and custom Markdown slide import.
- **Keymap & Tuning**: Remap any of the **13 gestures** across **22+ OS & presentation commands**, persisted automatically to `config/settings.json`.
- **Range & CPU Diagnostics**: 3-zone distance calibration wizard (`0.8m`, `2.5m`, `4.0m+`) and live CPU inference benchmark.

---

## 🎮 Control Profiles ("PowerPoint and More")

Hold **🤟 Three Fingers** (`Index + Middle + Ring`) to cycle profiles hands-free from across the room, or press **`P`** in the Desktop HUD:

| Gesture | 📊 Presentation Pro | 🎵 Media & Audio | 🖱️ Air Mouse & Reader |
| :--- | :--- | :--- | :--- |
| **👉 Swipe Right** | Next Slide (`Right`) | Next Track | Browser Forward (`Alt+Right`) |
| **👈 Swipe Left** | Previous Slide (`Left`) | Previous Track | Browser Back (`Alt+Left`) |
| **☝️ Point Up** | Next Slide (`Right`) | Volume Up | Scroll Up |
| **👇 Point Down** | Previous Slide (`Left`) | Volume Down | Scroll Down |
| **✋ Open Palm** | Play / Pause (`Space`) | Play / Pause | *(Wake / Neutral)* |
| **✌️ Peace Sign** | Toggle Laser Pointer (`Ctrl+L`) | Fullscreen (`F`) | Right Click |
| **👍 Thumbs Up** | Zoom In (`Ctrl++`) | Volume Up | Zoom In |
| **👎 Thumbs Down** | Blackout Screen (`B`) | Volume Down | Zoom Out |
| **👌 Pinch (Thumb+Index)** | Click / Draw Laser Ink | Play / Pause | **Left Click / Hold to Drag** |
| **✊ Closed Fist** | Lock Gesture Window | Mute Audio | Lock Gesture Window |
| **⬆️ / ⬇️ Swipe Up/Down** | Start (`F5`) / Exit (`Esc`) | Volume Up / Down | Scroll Up / Down |
| **🤟 Three Fingers** | Cycle Control Mode | Cycle Control Mode | Cycle Control Mode |

---

## 🛡️ Intentional Activation Modes (Zero Talking-Hand Misfires)

- **Clutch Lock Mode (`--mode clutch`, Default)**: Hold **Open Palm (`✋`)** steady for `0.38s` (watch the radial ring fill around your palm) to unlock a **4-second Active Command Window** that automatically extends as you navigate slides. Make a **Closed Fist (`✊`)** or lower your hand to lock immediately so natural hand-talking never advances slides.
- **Direct Dwell Mode (`--mode dwell`)**: Crisp linear flick swipes fire immediately (with **Return-Stroke Suppression** so bringing your hand back never triggers the opposite slide), while static poses confirm via a smooth `0.42s` radial dwell ring.
- **Instant Mode (`--mode instant`)**: Fastest `0.16s` trigger for Air Mouse and power users.

---

## ⌨️ Desktop HUD Hotkeys

- **`Q` / `Esc`** — Quit application
- **`M`** — Toggle **Compact Corner PiP Mode** (`420×236` minimal monitor)
- **`P`** — Switch Control Profile (`Presentation` → `Media` → `Air Mouse` → `Custom`)
- **`C`** — Switch Activation Mode (`Clutch` → `Dwell` → `Instant`)
- **`1` / `2` / `3`** — Switch CPU Preset (`Ultra-Light` / `Balanced` / `Max-Range`)
- **`R`** — Reset ROI tracker & 1-Euro smoother

---

## 🧪 Testing & CPU Benchmarking

```bash
# Run full diagnostic suite + 0.5m–4.8m scale-invariance tests + pytest:
python test_gesture_system.py

# Run quick CLI CPU latency benchmark:
python gesture_presentation_control.py --benchmark
```

---

## 📁 Project Architecture

```text
vision/
├── gesture_presentation_control.py  # CLI entry point (Native Desktop HUD + --studio Web Server)
├── setup_and_run.py                 # Interactive installer & mode launcher
├── test_gesture_system.py           # Diagnostics, range verification & pytest runner
├── index.html                       # Unified Vision Control v2 Web Studio
├── vision_control/
│   ├── config.py                    # Presets, 4 control profiles, action catalog & JSON persistence
│   ├── filters.py                   # 1-Euro Filter (1D/2D/21-landmark) for jitter-free long range
│   ├── camera.py                    # Asynchronous MJPG ThreadedCamera reader
│   ├── tracker.py                   # 3-Tier Long-Range Smart-Zoom ROI + Pose Spotlight + CLAHE
│   ├── gestures.py                  # Palm-Unit 3D joint-angle classifier, swipes & clutch engine
│   ├── actions.py                   # Cross-platform OS keystrokes, media/volume & Air Mouse
│   ├── hud.py                       # Fast sub-ROI dark-glass OpenCV HUD & Smart-Zoom PiP
│   └── server.py                    # FastAPI + WebSocket (/ws) real-time studio & OS bridge
├── web/
│   ├── css/studio.css               # Precision dark-glass spatial optics UI
│   └── js/                          # Browser Long-Range Tracker, 1€ Filter, Gesture Engine & Stage
└── tests/                           # 48 unit & integration tests (range invariance, ROI, WS, filters)
```

## 📄 License

MIT License — all computer-vision inference runs 100% locally on your device.
