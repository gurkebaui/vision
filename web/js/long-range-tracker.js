/**
 * Browser Long-Range Smart-Zoom Hand Tracker (0.3m – 5.5m+).
 *
 * Uses @mediapipe/tasks-vision (WASM SIMD / WebGL) with Offscreen/Canvas native-pixel
 * ROI cropping + Pose-Lite wrist spotlight + Multi-Zone Scout tiles, plus a realistic
 * Interactive Distance Simulator (0.5m – 5.0m) for camera-free testing and demos.
 */

import { LandmarkSmoother } from './filters.js';

const LOCAL_BUNDLE = '/vendor/mediapipe/vision_bundle.mjs';
const LOCAL_WASM = '/vendor/mediapipe/wasm';
const CDN_BUNDLE =
  'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/vision_bundle.mjs';
const CDN_WASM = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/wasm';

const LOCAL_HAND_MODEL = '/api/models/hand_landmarker.task';
const CDN_HAND_MODEL =
  'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';

const LOCAL_POSE_MODEL = '/api/models/pose_landmarker_lite.task';
const CDN_POSE_MODEL =
  'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task';

const SCOUT_ZONES = [
  [0.50, 0.42, 0.44],
  [0.32, 0.42, 0.45],
  [0.68, 0.42, 0.45],
  [0.50, 0.36, 0.30],
  [0.28, 0.36, 0.32],
  [0.72, 0.36, 0.32],
];

function makeSquareRoi(cx, cy, side, fw, fh, minSide = 128, source = 'temporal_roi') {
  const maxSide = Math.min(fw, fh);
  const clampedSide = Math.round(
    Math.max(Math.min(minSide, maxSide), Math.min(side, maxSide))
  );
  let x1 = Math.round(cx - clampedSide * 0.5);
  let y1 = Math.round(cy - clampedSide * 0.5);
  x1 = Math.max(0, Math.min(x1, fw - clampedSide));
  y1 = Math.max(0, Math.min(y1, fh - clampedSide));
  return {
    x1,
    y1,
    x2: x1 + clampedSide,
    y2: y1 + clampedSide,
    width: clampedSide,
    height: clampedSide,
    source,
  };
}

async function urlAvailable(url) {
  try {
    const res = await fetch(url, { method: 'HEAD' });
    return res.ok;
  } catch {
    return false;
  }
}

export class BrowserLongRangeTracker {
  constructor(configGetter) {
    this.getConfig = configGetter;
    this.smoother = new LandmarkSmoother(21, 1.45, 1.8);

    this.handLandmarker = null;
    this.poseLandmarker = null;
    this.delegateUsed = 'CPU';
    this.modelsReady = false;

    // Offscreen work canvases (tiny 256x256 for zero-copy speed!)
    this.mirrorCanvas = document.createElement('canvas');
    this.mirrorCtx = this.mirrorCanvas.getContext('2d', { willReadFrequently: false });

    this.cropCanvas = document.createElement('canvas');
    this.cropCtx = this.cropCanvas.getContext('2d', { willReadFrequently: true });

    this.poseCanvas = document.createElement('canvas');
    this.poseCanvas.width = 256;
    this.poseCanvas.height = 192;
    this.poseCtx = this.poseCanvas.getContext('2d', { willReadFrequently: false });

    // Temporal ROI tracking state
    this.lockedRoi = null;
    this.prevPalmCenterPx = null;
    this.velPx = { x: 0, y: 0 };
    this.prevHandSpanPx = 180;
    this.missedLockedFrames = 0;
    this.frameCounter = 0;
    this.scoutIdx = 0;
    this.lowLightDetected = false;

    // Interactive Simulator state
    this.simDistanceM = 2.6;
    this.simPose = 'open_palm';
    this.simSwipeAnim = null;
  }

  async initModels(onProgress) {
    if (this.modelsReady) return true;
    if (onProgress) onProgress('Loading Vision WASM runtime...');

    let mpVision = null;
    let wasmBase = CDN_WASM;
    if (await urlAvailable(LOCAL_BUNDLE)) {
      mpVision = await import(LOCAL_BUNDLE);
      wasmBase = LOCAL_WASM;
    } else {
      mpVision = await import(CDN_BUNDLE);
    }

    const { FilesetResolver, HandLandmarker, PoseLandmarker } = mpVision;
    const visionFileset = await FilesetResolver.forVisionTasks(wasmBase);

    const handModelUrl = (await urlAvailable(LOCAL_HAND_MODEL))
      ? LOCAL_HAND_MODEL
      : CDN_HAND_MODEL;

    if (onProgress) onProgress('Initializing Hand Landmarker...');

    const createHand = async (delegate) =>
      HandLandmarker.createFromOptions(visionFileset, {
        baseOptions: { modelAssetPath: handModelUrl, delegate },
        runningMode: 'IMAGE',
        numHands: 1,
        minHandDetectionConfidence: 0.34,
        minHandPresenceConfidence: 0.30,
        minTrackingConfidence: 0.30,
      });

    try {
      this.handLandmarker = await createHand('GPU');
      this.delegateUsed = 'WebGL / GPU';
    } catch {
      this.handLandmarker = await createHand('CPU');
      this.delegateUsed = 'WASM SIMD CPU';
    }

    // Optional Pose-Lite upper-body wrist spotlight for 1.5m–5.5m acquisition
    try {
      if (onProgress) onProgress('Initializing Long-Range Pose Spotlight...');
      const poseModelUrl = (await urlAvailable(LOCAL_POSE_MODEL))
        ? LOCAL_POSE_MODEL
        : CDN_POSE_MODEL;
      this.poseLandmarker = await PoseLandmarker.createFromOptions(visionFileset, {
        baseOptions: { modelAssetPath: poseModelUrl, delegate: 'CPU' },
        runningMode: 'IMAGE',
        numPoses: 1,
        minPoseDetectionConfidence: 0.36,
        minPosePresenceConfidence: 0.34,
      });
    } catch {
      this.poseLandmarker = null;
    }

    this.modelsReady = true;
    return true;
  }

  reset() {
    this.lockedRoi = null;
    this.prevPalmCenterPx = null;
    this.velPx = { x: 0, y: 0 };
    this.missedLockedFrames = 0;
    this.smoother.reset();
  }

  inferHandOnRoi(roi, roiTargetSize) {
    if (this.cropCanvas.width !== roiTargetSize) {
      this.cropCanvas.width = roiTargetSize;
      this.cropCanvas.height = roiTargetSize;
    }

    // Check low light every 15 frames on tiny sample
    if (this.frameCounter % 15 === 1) {
      this.cropCtx.filter = 'none';
      this.cropCtx.drawImage(
        this.mirrorCanvas,
        roi.x1,
        roi.y1,
        roi.width,
        roi.height,
        0,
        0,
        roiTargetSize,
        roiTargetSize
      );
      try {
        const sample = this.cropCtx.getImageData(
          roiTargetSize >> 2,
          roiTargetSize >> 2,
          64,
          64
        ).data;
        let sum = 0;
        for (let i = 1; i < sample.length; i += 16) sum += sample[i];
        const meanLuma = sum / (sample.length / 16);
        this.lowLightDetected = meanLuma < 86;
      } catch {
        this.lowLightDetected = false;
      }
    }

    this.cropCtx.filter = this.lowLightDetected
      ? 'contrast(1.18) brightness(1.12)'
      : 'none';
    this.cropCtx.drawImage(
      this.mirrorCanvas,
      roi.x1,
      roi.y1,
      roi.width,
      roi.height,
      0,
      0,
      roiTargetSize,
      roiTargetSize
    );
    this.cropCtx.filter = 'none';

    const res = this.handLandmarker.detect(this.cropCanvas);
    if (!res || !res.landmarks || res.landmarks.length === 0) {
      return null;
    }
    const lms = res.landmarks[0];
    let score = 0.88;
    if (res.handednesses && res.handednesses[0] && res.handednesses[0][0]) {
      score = res.handednesses[0][0].score || 0.88;
    }
    return { cropLandmarks: lms, score };
  }

  locateActiveWristRois(fw, fh, cfg) {
    if (!this.poseLandmarker) return [];
    this.poseCtx.drawImage(this.mirrorCanvas, 0, 0, 256, 192);
    const pres = this.poseLandmarker.detect(this.poseCanvas);
    if (!pres || !pres.landmarks || pres.landmarks.length === 0) return [];

    const plm = pres.landmarks[0];
    if (plm.length < 25) return [];

    const lSh = plm[11];
    const rSh = plm[12];
    const lHip = plm[23];
    const rHip = plm[24];

    const shoulderPx = Math.hypot((lSh.x - rSh.x) * fw, (lSh.y - rSh.y) * fh);
    const effShoulderPx = shoulderPx >= 18 ? shoulderPx : Math.min(fw, fh) * 0.18;
    const midShY = (lSh.y + rSh.y) * 0.5;
    const torsoH =
      (lHip.visibility || 0.5) > 0.35 && (rHip.visibility || 0.5) > 0.35
        ? Math.max(0.12, (lHip.y + rHip.y) * 0.5 - midShY)
        : Math.max(0.16, (effShoulderPx / Math.max(1, fh)) * 1.45);
    const restingCutoffY = midShY + torsoH * 0.92;

    const candidates = [];
    const arms = [
      [13, 15, 17, 19],
      [14, 16, 18, 20],
    ];
    for (const [elbI, wriI, pnkI, idxI] of arms) {
      const elb = plm[elbI];
      const wri = plm[wriI];
      const pnk = plm[pnkI];
      const idx = plm[idxI];
      const vis = wri.visibility ?? 0.8;
      if (vis < 0.32) continue;
      if (wri.x < -0.05 || wri.x > 1.05 || wri.y < -0.05 || wri.y > 1.05) continue;
      if (cfg.resting_arm_rejection !== false && wri.y > restingCutoffY) continue;

      let palmNx = wri.x;
      let palmNy = wri.y;
      if ((idx.visibility ?? 0.5) > 0.35 && (pnk.visibility ?? 0.5) > 0.35) {
        palmNx = 0.45 * wri.x + 0.275 * idx.x + 0.275 * pnk.x;
        palmNy = 0.45 * wri.y + 0.275 * idx.y + 0.275 * pnk.y;
      } else if ((elb.visibility ?? 0.5) > 0.30) {
        palmNx = wri.x + 0.34 * (wri.x - elb.x);
        palmNy = wri.y + 0.34 * (wri.y - elb.y);
      }

      const sidePx = Math.max(128, Math.min(Math.min(fw, fh) * 0.65, effShoulderPx * 1.48));
      const roi = makeSquareRoi(palmNx * fw, palmNy * fh, sidePx, fw, fh, 120, 'pose_wrist');
      candidates.push({ priority: wri.y - 0.15 * vis, roi });
    }
    candidates.sort((a, b) => a.priority - b.priority);
    return candidates.map((c) => c.roi);
  }

  processVideoFrame(videoEl, nowSec) {
    const t0 = performance.now();
    const fw = videoEl.videoWidth || 1280;
    const fh = videoEl.videoHeight || 720;
    const cfg = this.getConfig();
    const presetDetails = cfg.preset_details || { roi_size: 256, pose_scout_interval: 5, tile_scout_interval: 4 };
    const roiTarget = presetDetails.roi_size || 256;

    if (this.mirrorCanvas.width !== fw || this.mirrorCanvas.height !== fh) {
      this.mirrorCanvas.width = fw;
      this.mirrorCanvas.height = fh;
    }

    this.mirrorCtx.save();
    if (cfg.mirror_camera !== false) {
      this.mirrorCtx.translate(fw, 0);
      this.mirrorCtx.scale(-1, 1);
    }
    this.mirrorCtx.drawImage(videoEl, 0, 0, fw, fh);
    this.mirrorCtx.restore();

    this.frameCounter += 1;
    let detected = null;
    let activeRoi = null;

    // Tier 1: Velocity-Predicted Temporal Hand ROI
    if (cfg.smart_zoom_enabled !== false && this.lockedRoi) {
      const predCx = this.prevPalmCenterPx
        ? this.prevPalmCenterPx.x + this.velPx.x * 0.8
        : (this.lockedRoi.x1 + this.lockedRoi.x2) * 0.5;
      const predCy = this.prevPalmCenterPx
        ? this.prevPalmCenterPx.y + this.velPx.y * 0.8
        : (this.lockedRoi.y1 + this.lockedRoi.y2) * 0.5;
      const speed = Math.hypot(this.velPx.x, this.velPx.y);
      const pad = Math.min(
        3.9,
        2.45 + (speed / Math.max(28, this.prevHandSpanPx)) * 0.95 + this.missedLockedFrames * 0.65
      );
      const candidateRoi = makeSquareRoi(
        predCx,
        predCy,
        Math.max(136, this.prevHandSpanPx * pad),
        fw,
        fh,
        128,
        'temporal_roi'
      );
      detected = this.inferHandOnRoi(candidateRoi, roiTarget);
      if (detected) {
        activeRoi = candidateRoi;
        this.missedLockedFrames = 0;
      } else {
        this.missedLockedFrames += 1;
        if (this.missedLockedFrames >= 2) {
          this.lockedRoi = null;
          this.velPx = { x: 0, y: 0 };
        }
      }
    }

    // Tier 2 & 3: Acquisition when unlocked
    if (!detected) {
      if (cfg.smart_zoom_enabled === false || this.frameCounter % 2 === 1) {
        const fullRoi = { x1: 0, y1: 0, x2: fw, y2: fh, width: fw, height: fh, source: 'full_frame' };
        detected = this.inferHandOnRoi(fullRoi, 320);
        if (detected) activeRoi = fullRoi;
      }

      if (
        !detected &&
        cfg.smart_zoom_enabled !== false &&
        cfg.pose_assist_enabled !== false &&
        this.frameCounter % Math.max(1, presetDetails.pose_scout_interval || 5) === 0
      ) {
        const wristRois = this.locateActiveWristRois(fw, fh, cfg);
        for (const wRoi of wristRois.slice(0, 2)) {
          detected = this.inferHandOnRoi(wRoi, roiTarget);
          if (detected) {
            activeRoi = wRoi;
            break;
          }
        }
      }

      if (
        !detected &&
        cfg.smart_zoom_enabled !== false &&
        this.frameCounter % Math.max(1, presetDetails.tile_scout_interval || 4) === 0
      ) {
        const [zx, zy, zratio] = SCOUT_ZONES[this.scoutIdx % SCOUT_ZONES.length];
        this.scoutIdx += 1;
        const shortSide = Math.min(fw, fh);
        const scoutRoi = makeSquareRoi(
          zx * fw,
          zy * fh,
          shortSide * zratio,
          fw,
          fh,
          128,
          'scout_tile'
        );
        detected = this.inferHandOnRoi(scoutRoi, roiTarget);
        if (detected) activeRoi = scoutRoi;
      }
    }

    const inferenceMs = performance.now() - t0;

    if (!detected || !activeRoi) {
      return {
        detected: false,
        landmarks: [],
        roi: null,
        zoomFactor: 1.0,
        estimatedDistanceM: 0,
        palmPx: 0,
        inferenceMs,
        source: 'searching',
        lowLightBoost: this.lowLightDetected,
      };
    }

    const zScale = activeRoi.width / fw;
    const rawFull = detected.cropLandmarks.map((pt) => ({
      x: (activeRoi.x1 + pt.x * activeRoi.width) / fw,
      y: (activeRoi.y1 + pt.y * activeRoi.height) / fh,
      z: (pt.z || 0) * zScale,
    }));

    const smoothed = this.smoother.filter(rawFull, nowSec);

    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (const p of rawFull) {
      const px = p.x * fw;
      const py = p.y * fh;
      if (px < minX) minX = px;
      if (px > maxX) maxX = px;
      if (py < minY) minY = py;
      if (py > maxY) maxY = py;
    }
    const spanPx = Math.max(maxX - minX, maxY - minY, 24);
    const cxPx = (minX + maxX) * 0.5;
    const cyPx = (minY + maxY) * 0.5;

    if (this.prevPalmCenterPx) {
      this.velPx = {
        x: 0.55 * this.velPx.x + 0.45 * (cxPx - this.prevPalmCenterPx.x),
        y: 0.55 * this.velPx.y + 0.45 * (cyPx - this.prevPalmCenterPx.y),
      };
    }
    this.prevPalmCenterPx = { x: cxPx, y: cyPx };
    this.prevHandSpanPx = 0.35 * this.prevHandSpanPx + 0.65 * spanPx;

    const tightRoi = makeSquareRoi(
      cxPx,
      cyPx,
      Math.max(136, this.prevHandSpanPx * 2.45),
      fw,
      fh,
      128,
      activeRoi.source
    );
    this.lockedRoi = tightRoi;

    const palmLenPx = Math.hypot(
      (smoothed[0].x - smoothed[9].x) * fw,
      (smoothed[0].y - smoothed[9].y) * fh
    );
    const palmWidPx = Math.hypot(
      (smoothed[5].x - smoothed[17].x) * fw,
      (smoothed[5].y - smoothed[17].y) * fh
    );
    const effectivePalmPx = Math.max(8, 0.65 * palmLenPx + 0.45 * palmWidPx);
    const focalPx = 0.78 * fw;
    const estDistM = Math.min(8.0, Math.max(0.25, (0.088 * focalPx) / effectivePalmPx));
    const zoomFactor = Math.max(1.0, Math.min(fw, fh) / tightRoi.width);

    return {
      detected: true,
      landmarks: smoothed,
      score: detected.score,
      roi: tightRoi,
      zoomFactor,
      estimatedDistanceM: estDistM,
      palmPx: effectivePalmPx,
      inferenceMs,
      source: activeRoi.source,
      lowLightBoost: this.lowLightDetected,
      cropCanvas: this.cropCanvas,
    };
  }

  triggerSimulatedSwipe(direction = 'right', nowSec = performance.now() / 1000) {
    this.simSwipeAnim = { direction, startT: nowSec, duration: 0.22 };
  }

  /**
   * Generate anatomically accurate 21 landmarks scaled to `this.simDistanceM` (0.5m – 5.0m)
   * so users can test scale-invariant tracking, Smart Zoom ROI, and every gesture without a camera.
   */
  generateSimulatedFrame(fw, fh, nowSec) {
    const t0 = performance.now();
    const cfg = this.getConfig();
    const distM = Math.max(0.5, Math.min(5.0, this.simDistanceM));

    // If Smart Zoom is turned OFF in simulator and distance > 1.35m, demonstrate legacy drop-off!
    if (cfg.smart_zoom_enabled === false && distM > 1.35) {
      return {
        detected: false,
        landmarks: [],
        roi: null,
        zoomFactor: 1.0,
        estimatedDistanceM: distM,
        palmPx: 0,
        inferenceMs: 1.8,
        source: 'legacy_out_of_range',
        simulated: true,
      };
    }

    const focalPx = 0.78 * fw;
    const palmPx = (0.088 * focalPx) / distM;
    const scaleX = palmPx / fw;
    const scaleY = palmPx / fh;

    let cx = 0.52 + Math.sin(nowSec * 1.1) * 0.008;
    let cy = 0.46 + Math.cos(nowSec * 0.9) * 0.006;

    if (this.simSwipeAnim) {
      const p = (nowSec - this.simSwipeAnim.startT) / this.simSwipeAnim.duration;
      if (p <= 1.0) {
        const dir = this.simSwipeAnim.direction;
        const spanX = scaleX * 2.4;
        const spanY = scaleY * 2.4;
        if (dir === 'right') cx = 0.44 + p * spanX;
        else if (dir === 'left') cx = 0.58 - p * spanX;
        else if (dir === 'up') cy = 0.54 - p * spanY;
        else if (dir === 'down') cy = 0.38 + p * spanY;
      } else {
        this.simSwipeAnim = null;
      }
    }

    const pose = this.simPose;
    // Local palm-unit landmark offsets (wrist at 0, +0.55)
    const baseMcps = [
      [-0.38, -0.35], // index MCP 5
      [-0.12, -0.44], // middle MCP 9
      [0.14, -0.40],  // ring MCP 13
      [0.38, -0.30],  // pinky MCP 17
    ];

    const fingerCurl = {
      open_palm:     [0, 0, 0, 0],
      closed_fist:   [1, 1, 1, 1],
      pointing_up:   [0, 1, 1, 1],
      pointing_down: [0, 1, 1, 1],
      peace_sign:    [0, 0, 1, 1],
      thumbs_up:     [1, 1, 1, 1],
      thumbs_down:   [1, 1, 1, 1],
      pinch:         [0.35, 0, 0, 0],
      three_fingers: [0, 0, 0, 1],
    }[pose] || [0, 0, 0, 0];

    const flipDown = pose === 'pointing_down' || pose === 'thumbs_down';
    const ptsPu = new Array(21);
    ptsPu[0] = [0, flipDown ? -0.52 : 0.52, 0];

    // Thumb (1..4)
    if (pose === 'thumbs_up') {
      ptsPu[1] = [-0.24, 0.25, 0];
      ptsPu[2] = [-0.30, -0.02, 0];
      ptsPu[3] = [-0.30, -0.34, 0];
      ptsPu[4] = [-0.30, -0.66, 0];
    } else if (pose === 'thumbs_down') {
      ptsPu[1] = [-0.24, -0.25, 0];
      ptsPu[2] = [-0.30, 0.02, 0];
      ptsPu[3] = [-0.30, 0.34, 0];
      ptsPu[4] = [-0.30, 0.66, 0];
    } else if (pose === 'pinch') {
      ptsPu[1] = [-0.28, 0.30, 0];
      ptsPu[2] = [-0.42, 0.06, 0];
      ptsPu[3] = [-0.44, -0.24, 0];
      ptsPu[4] = [-0.36, -0.52, 0];
    } else if (pose === 'closed_fist') {
      ptsPu[1] = [-0.26, 0.28, 0];
      ptsPu[2] = [-0.28, 0.06, 0];
      ptsPu[3] = [-0.14, -0.04, 0];
      ptsPu[4] = [0.02, -0.02, 0];
    } else {
      const s = flipDown ? -1 : 1;
      ptsPu[1] = [-0.32, 0.30 * s, 0];
      ptsPu[2] = [-0.54, 0.08 * s, 0];
      ptsPu[3] = [-0.72, -0.14 * s, 0];
      ptsPu[4] = [-0.88, -0.34 * s, 0];
    }

    const lengths = [1.0, 1.08, 1.0, 0.82];
    const spreads = pose === 'peace_sign' ? [-0.22, 0.20, 0.08, 0.14] : [-0.10, -0.02, 0.06, 0.14];

    for (let f = 0; f < 4; f++) {
      const baseIdx = 5 + f * 4;
      const [mx, myRaw] = baseMcps[f];
      const my = flipDown ? -myRaw : myRaw;
      ptsPu[baseIdx] = [mx, my, 0];
      const curl = fingerCurl[f];
      const len = lengths[f];
      const dirSign = flipDown ? 1 : -1;

      if (pose === 'pinch' && f === 0) {
        ptsPu[6] = [-0.42, -0.58, 0];
        ptsPu[7] = [-0.40, -0.58, 0];
        ptsPu[8] = [-0.35, -0.53, 0]; // touches thumb tip!
      } else if (curl < 0.5) {
        const sp = spreads[f];
        ptsPu[baseIdx + 1] = [mx + sp * 0.35, my + dirSign * 0.34 * len, 0];
        ptsPu[baseIdx + 2] = [mx + sp * 0.70, my + dirSign * 0.64 * len, 0];
        ptsPu[baseIdx + 3] = [mx + sp * 1.05, my + dirSign * 0.92 * len, 0];
      } else {
        ptsPu[baseIdx + 1] = [mx, my + dirSign * 0.18, 0.04];
        ptsPu[baseIdx + 2] = [mx * 0.88, my - dirSign * 0.06, 0.06];
        ptsPu[baseIdx + 3] = [mx * 0.82, my - dirSign * 0.22, 0.04];
      }
    }

    const rawFull = ptsPu.map(([ux, uy, uz]) => ({
      x: cx + ux * scaleX,
      y: cy + uy * scaleY,
      z: uz * scaleX,
    }));

    const smoothed = this.smoother.filter(rawFull, nowSec);
    const sidePx = Math.max(136, palmPx * 3.1);
    const roi = makeSquareRoi(cx * fw, cy * fh, sidePx, fw, fh, 128, 'pose_wrist');
    const zoomFactor = Math.max(1.0, Math.min(fw, fh) / roi.width);

    return {
      detected: true,
      landmarks: smoothed,
      score: 0.95,
      roi,
      zoomFactor,
      estimatedDistanceM: distM,
      palmPx,
      inferenceMs: Math.max(2.1, performance.now() - t0 + 3.2),
      source: distM > 1.5 ? 'pose_wrist' : 'temporal_roi',
      simulated: true,
    };
  }
}
