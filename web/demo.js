/**
 * Browser demo wiring: camera -> MediaPipe GestureRecognizer -> our scorer.
 *
 * Honest by design: if the camera or the model fails, the page says so. There
 * is no simulated fallback pretending to detect gestures, which is what the
 * previous version of this site did.
 */

import {
  FilesetResolver,
  GestureRecognizer,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/vision_bundle.mjs";

import { CONNECTIONS, GESTURE_INFO, Stabilizer, scoreAll } from "./gestures.js";

const WASM_ROOT = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm";
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task";

const el = (id) => document.getElementById(id);
const ui = {
  video: el("video"),
  overlay: el("overlay"),
  placeholder: el("placeholder"),
  loading: el("loading"),
  start: el("start"),
  stop: el("stop"),
  mirror: el("mirror"),
  gesture: el("gesture"),
  confidence: el("confidence"),
  confbar: el("confbar"),
  scorelist: el("scorelist"),
  events: el("events"),
  fps: el("fps"),
  inference: el("inference"),
  hands: el("hands"),
  grid: el("gesture-grid"),
};

const ctx = ui.overlay.getContext("2d");
const stabilizer = new Stabilizer();

let recognizer = null;
let stream = null;
let running = false;
let mirrored = true;
let lastVideoTime = -1;
let fpsEma = null;
let lastFrame = performance.now();

// --- static content ---------------------------------------------------------
for (const [id, label, action] of GESTURE_INFO) {
  const card = document.createElement("div");
  card.className = "card";
  card.innerHTML =
    `<div class="card-name">${label}</div>` +
    `<div class="card-id">${id}</div>` +
    `<div class="card-action muted small">${action}</div>`;
  ui.grid.appendChild(card);
}

// --- helpers ----------------------------------------------------------------
function say(message, kind = "info") {
  ui.placeholder.classList.remove("hidden");
  ui.placeholder.innerHTML =
    `<p class="${kind}">${message}</p>` +
    `<button id="retry" class="primary">Try again</button>`;
  document.getElementById("retry").addEventListener("click", start);
}

function logEvent(name, confidence) {
  const li = document.createElement("li");
  const time = new Date().toLocaleTimeString();
  li.innerHTML = `<span>${name}</span><span class="muted small">${(confidence * 100) | 0}% · ${time}</span>`;
  ui.events.prepend(li);
  while (ui.events.children.length > 8) ui.events.lastChild.remove();
}

function drawHand(landmarks, width, height) {
  ctx.lineWidth = 2;
  ctx.strokeStyle = "rgba(220,225,230,0.85)";
  for (const [a, b] of CONNECTIONS) {
    ctx.beginPath();
    ctx.moveTo(landmarks[a].x * width, landmarks[a].y * height);
    ctx.lineTo(landmarks[b].x * width, landmarks[b].y * height);
    ctx.stroke();
  }
  const tips = new Set([4, 8, 12, 16, 20]);
  for (let i = 0; i < landmarks.length; i++) {
    const p = landmarks[i];
    ctx.beginPath();
    ctx.arc(p.x * width, p.y * height, tips.has(i) ? 5.5 : 3, 0, Math.PI * 2);
    ctx.fillStyle = tips.has(i) ? "#78dc50" : "#8fb6d8";
    ctx.fill();
  }
}

function renderScores(scores) {
  const top = Object.entries(scores).sort((a, b) => b[1] - a[1]).slice(0, 5);
  ui.scorelist.innerHTML = top
    .map(([name, value]) => {
      const pct = Math.round(value * 100);
      const cls = value > 0.62 ? "good" : "";
      return `<li><span>${name}</span>
        <span class="mini-track"><span class="mini-fill ${cls}" style="width:${pct}%"></span></span>
        <span class="muted small">${pct}%</span></li>`;
    })
    .join("");
}

function clearReadout() {
  ui.gesture.textContent = "—";
  ui.confidence.textContent = "no hand";
  ui.confbar.style.width = "0%";
  ui.scorelist.innerHTML = "";
  ui.hands.textContent = "0";
}

// --- main loop --------------------------------------------------------------
function loop() {
  if (!running) return;

  const now = performance.now();
  const dt = now - lastFrame;
  lastFrame = now;
  if (dt > 0) {
    fpsEma = fpsEma === null ? 1000 / dt : fpsEma * 0.9 + (1000 / dt) * 0.1;
    ui.fps.textContent = fpsEma.toFixed(0);
  }

  const width = ui.video.videoWidth;
  const height = ui.video.videoHeight;
  if (width && height) {
    if (ui.overlay.width !== width) {
      ui.overlay.width = width;
      ui.overlay.height = height;
    }
    ctx.clearRect(0, 0, width, height);

    if (ui.video.currentTime !== lastVideoTime) {
      lastVideoTime = ui.video.currentTime;
      const t0 = performance.now();
      let result = null;
      try {
        result = recognizer.recognizeForVideo(ui.video, now);
      } catch (err) {
        console.error(err);
      }
      ui.inference.textContent = `${(performance.now() - t0).toFixed(1)} ms`;

      if (result?.landmarks?.length) {
        ui.hands.textContent = String(result.landmarks.length);
        for (const lm of result.landmarks) drawHand(lm, width, height);

        const primary = result.landmarks[0];
        const canned = result.gestures?.[0]?.[0];
        const scores = scoreAll(primary, canned?.categoryName, canned?.score ?? 0);
        renderScores(scores);

        const best = Object.entries(scores).sort((a, b) => b[1] - a[1])[0];
        ui.confidence.textContent = `${best[0]} at ${(best[1] * 100) | 0}%`;
        ui.confbar.style.width = `${Math.round(best[1] * 100)}%`;

        const hit = stabilizer.update(scores);
        if (hit) {
          ui.gesture.textContent = hit.name.replace(/_/g, " ");
          logEvent(hit.name, hit.confidence);
        } else if (!stabilizer.active) {
          ui.gesture.textContent = "—";
        }
      } else {
        stabilizer.update({});
        clearReadout();
      }
    }
  }
  requestAnimationFrame(loop);
}

// --- lifecycle --------------------------------------------------------------
async function ensureRecognizer() {
  if (recognizer) return recognizer;
  ui.loading.classList.remove("hidden");
  const fileset = await FilesetResolver.forVisionTasks(WASM_ROOT);
  recognizer = await GestureRecognizer.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: MODEL_URL, delegate: "GPU" },
    runningMode: "VIDEO",
    numHands: 2,
  });
  ui.loading.classList.add("hidden");
  return recognizer;
}

async function start() {
  ui.placeholder.classList.add("hidden");
  try {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error("This browser does not expose a camera API.");
    }
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
    });
  } catch (err) {
    say(
      err.name === "NotAllowedError"
        ? "Camera permission was denied. Allow it in the address bar, then try again."
        : `Camera unavailable: ${err.message}`,
      "error"
    );
    return;
  }

  try {
    await ensureRecognizer();
  } catch (err) {
    ui.loading.classList.add("hidden");
    say(`Could not load the model: ${err.message}. Check your connection.`, "error");
    stop();
    return;
  }

  ui.video.srcObject = stream;
  await ui.video.play();
  running = true;
  ui.stop.disabled = false;
  ui.start.disabled = true;
  stabilizer.reset();
  requestAnimationFrame(loop);
}

function stop() {
  running = false;
  if (stream) {
    for (const track of stream.getTracks()) track.stop();
    stream = null;
  }
  ui.video.srcObject = null;
  ui.stop.disabled = true;
  ui.start.disabled = false;
  clearReadout();
  ctx.clearRect(0, 0, ui.overlay.width, ui.overlay.height);
  ui.placeholder.classList.remove("hidden");
  ui.placeholder.innerHTML =
    `<p>Camera is off</p>` +
    `<button id="start2" class="primary">Enable camera</button>`;
  document.getElementById("start2").addEventListener("click", start);
}

function applyMirror() {
  const transform = mirrored ? "scaleX(-1)" : "none";
  ui.video.style.transform = transform;
  ui.overlay.style.transform = transform;
  ui.mirror.textContent = `Mirror: ${mirrored ? "on" : "off"}`;
}

ui.start.addEventListener("click", start);
ui.stop.addEventListener("click", stop);
ui.mirror.addEventListener("click", () => {
  mirrored = !mirrored;
  applyMirror();
});
applyMirror();
