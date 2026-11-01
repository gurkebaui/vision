/**
 * Scale-Invariant 3D & Joint-Angle Gesture Engine (Browser ES Module).
 *
 * Mirrors vision_control/gestures.py with Palm-Unit (PU) normalization,
 * bone-vector joint flexion angles, velocity swipes with return-stroke suppression,
 * and Clutch / Dwell intentional activation.
 */

import { Point2DOneEuroFilter } from './filters.js';

export const GESTURE_META = {
  swipe_right:   { name: 'Swipe Right',       icon: '👉', kind: 'dynamic' },
  swipe_left:    { name: 'Swipe Left',        icon: '👈', kind: 'dynamic' },
  swipe_up:      { name: 'Swipe Up',          icon: '⬆️', kind: 'dynamic' },
  swipe_down:    { name: 'Swipe Down',        icon: '⬇️', kind: 'dynamic' },
  open_palm:     { name: 'Open Palm',         icon: '✋', kind: 'static' },
  closed_fist:   { name: 'Closed Fist',       icon: '✊', kind: 'static' },
  pointing_up:   { name: 'Point Up',          icon: '☝️', kind: 'static' },
  pointing_down: { name: 'Point Down',        icon: '👇', kind: 'static' },
  peace_sign:    { name: 'Peace Sign',        icon: '✌️', kind: 'static' },
  thumbs_up:     { name: 'Thumbs Up',         icon: '👍', kind: 'static' },
  thumbs_down:   { name: 'Thumbs Down',       icon: '👎', kind: 'static' },
  pinch:         { name: 'Thumb-Index Pinch', icon: '👌', kind: 'static' },
  shaka:         { name: 'Shaka Sign',        icon: '🤙', kind: 'static' },
  three_fingers: { name: 'Three Fingers',     icon: '🤟', kind: 'static' },
};

const OPPOSITE_SWIPE = {
  swipe_right: 'swipe_left',
  swipe_left: 'swipe_right',
  swipe_up: 'swipe_down',
  swipe_down: 'swipe_up',
};

function dist2d(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function dist3d(a, b) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  const dz = (a.z || 0) - (b.z || 0);
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

function vec(a, b) {
  return { x: b.x - a.x, y: b.y - a.y, z: (b.z || 0) - (a.z || 0) };
}

function angleDeg(v1, v2) {
  const n1 = Math.sqrt(v1.x * v1.x + v1.y * v1.y + v1.z * v1.z);
  const n2 = Math.sqrt(v2.x * v2.x + v2.y * v2.y + v2.z * v2.z);
  if (n1 < 1e-6 || n2 < 1e-6) return 0;
  const cosVal = Math.max(-1, Math.min(1, (v1.x * v2.x + v1.y * v2.y + v1.z * v2.z) / (n1 * n2)));
  return (Math.acos(cosVal) * 180) / Math.PI;
}

export class ScaleInvariantGestureEngine {
  constructor(configGetter) {
    this.getConfig = configGetter;
    this.trajectory = [];
    this.poseHistory = [];
    this.pinchActive = false;
    this.lastSwipeTime = -10;
    this.lastSwipeDir = null;

    this.clutchUnlocked = false;
    this.clutchUnlockExpiry = 0;
    this.holdCandidate = null;
    this.holdStartTime = null;
    this.holdFiredForPose = false;
    this.lastTriggerTime = -10;

    this.cursorFilter = new Point2DOneEuroFilter(1.15, 1.45);
  }

  reset() {
    this.trajectory = [];
    this.poseHistory = [];
    this.pinchActive = false;
    this.lastSwipeTime = -10;
    this.lastSwipeDir = null;
    this.clutchUnlocked = false;
    this.clutchUnlockExpiry = 0;
    this.holdCandidate = null;
    this.holdStartTime = null;
    this.holdFiredForPose = false;
    this.cursorFilter.reset();
  }

  classifyStaticPose(landmarks, aspectRatio = 16 / 9) {
    if (!landmarks || landmarks.length < 21) {
      return { pose: null, confidence: 0, extendedFingers: [], pinchPu: 1.0, palmSize: 0 };
    }

    const iso = landmarks.map((p) => ({
      x: p.x * aspectRatio,
      y: p.y,
      z: (p.z || 0) * aspectRatio,
    }));

    const wrist = iso[0];
    const lenWm = dist3d(iso[0], iso[9]);
    const widIp = dist3d(iso[5], iso[17]);
    const palmSize = Math.max(1e-4, 0.65 * lenWm + 0.50 * widIp);

    const fingerDefs = [
      ['index', 5, 6, 7, 8],
      ['middle', 9, 10, 11, 12],
      ['ring', 13, 14, 15, 16],
      ['pinky', 17, 18, 19, 20],
    ];

    const extendedFingers = [];
    const extMap = {};
    const curledMap = {};

    for (const [name, mcp, pip, dip, tip] of fingerDefs) {
      const angPip = angleDeg(vec(iso[mcp], iso[pip]), vec(iso[pip], iso[dip]));
      const angDip = angleDeg(vec(iso[pip], iso[dip]), vec(iso[dip], iso[tip]));
      const reach = dist2d(iso[tip], wrist) / Math.max(1e-4, dist2d(iso[pip], wrist));
      const spanPu = dist2d(iso[tip], iso[mcp]) / palmSize;

      const isExt = angPip < 44 && angDip < 52 && reach > 1.11 && spanPu > 0.50;
      const isCurled = angPip > 56 || reach < 1.04 || spanPu < 0.44;
      extMap[name] = isExt;
      curledMap[name] = isCurled;
      if (isExt) extendedFingers.push(name);
    }

    const thumbIpAng = angleDeg(vec(iso[2], iso[3]), vec(iso[3], iso[4]));
    const thumbMcpAng = angleDeg(vec(iso[1], iso[2]), vec(iso[2], iso[3]));
    const thumbIdxMcpPu = dist2d(iso[4], iso[5]) / palmSize;
    const thumbPinkyMcpPu = dist2d(iso[4], iso[17]) / palmSize;

    const thumbExt =
      thumbIpAng < 48 && thumbMcpAng < 48 && thumbIdxMcpPu > 0.48 && thumbPinkyMcpPu > 0.78;
    if (thumbExt) extendedFingers.unshift('thumb');

    const pinchPu = dist2d(iso[4], iso[8]) / palmSize;
    if (this.pinchActive) {
      if (pinchPu > 0.45) this.pinchActive = false;
    } else if (pinchPu < 0.33) {
      this.pinchActive = true;
    }

    const nonThumbExt = ['index', 'middle', 'ring', 'pinky'].filter((f) => extMap[f]).length;
    const nonThumbCurled = ['index', 'middle', 'ring', 'pinky'].filter((f) => curledMap[f]).length;
    const indexWristPu = dist2d(iso[8], wrist) / palmSize;

    if (this.pinchActive && (extMap.middle || extMap.ring || indexWristPu > 0.82)) {
      const conf = Math.min(0.97, Math.max(0.78, 1.0 - pinchPu * 0.65));
      return { pose: 'pinch', confidence: conf, extendedFingers, pinchPu, palmSize };
    }

    if (nonThumbExt === 4 && pinchPu > 0.42) {
      return {
        pose: 'open_palm',
        confidence: thumbExt ? 0.95 : 0.89,
        extendedFingers,
        pinchPu,
        palmSize,
      };
    }

    if (extMap.index && extMap.middle && extMap.ring && curledMap.pinky) {
      return { pose: 'three_fingers', confidence: 0.90, extendedFingers, pinchPu, palmSize };
    }

    if (extMap.index && extMap.middle && curledMap.ring && curledMap.pinky) {
      const vSpreadPu = dist2d(iso[8], iso[12]) / palmSize;
      if (vSpreadPu > 0.20) {
        const conf = Math.min(0.95, 0.80 + vSpreadPu * 0.25);
        return { pose: 'peace_sign', confidence: conf, extendedFingers, pinchPu, palmSize };
      }
    }

    if (extMap.index && curledMap.middle && curledMap.ring && curledMap.pinky) {
      const dx = iso[8].x - iso[5].x;
      const dy = iso[8].y - iso[5].y;
      const mag = Math.max(1e-4, Math.hypot(dx, dy));
      const vertRatio = Math.abs(dy) / mag;
      if (dy < 0 && vertRatio >= 0.58) {
        return {
          pose: 'pointing_up',
          confidence: Math.min(0.94, 0.75 + vertRatio * 0.2),
          extendedFingers,
          pinchPu,
          palmSize,
        };
      }
      if (dy > 0 && vertRatio >= 0.55) {
        return {
          pose: 'pointing_down',
          confidence: Math.min(0.94, 0.75 + vertRatio * 0.2),
          extendedFingers,
          pinchPu,
          palmSize,
        };
      }
    }

    if (thumbExt && extMap.pinky && curledMap.index && curledMap.middle && curledMap.ring) {
      const shakaSpanPu = dist2d(iso[4], iso[20]) / palmSize;
      if (shakaSpanPu > 0.95) {
        const conf = Math.min(0.95, 0.80 + (shakaSpanPu - 0.95) * 0.20);
        return { pose: 'shaka', confidence: conf, extendedFingers, pinchPu, palmSize };
      }
    }

    const palmCx = (iso[0].x + iso[5].x + iso[9].x + iso[17].x) * 0.25;
    const palmCy = (iso[0].y + iso[5].y + iso[9].y + iso[17].y) * 0.25;
    const thumbCenterPu = Math.hypot(iso[4].x - palmCx, iso[4].y - palmCy) / palmSize;
    const curledTipYs = [iso[8].y, iso[12].y, iso[16].y, iso[20].y];

    if (nonThumbCurled === 4 && thumbIpAng < 52) {
      const tdx = iso[4].x - iso[2].x;
      const tdy = iso[4].y - iso[2].y;
      const tmag = Math.max(1e-4, Math.hypot(tdx, tdy));
      if (Math.abs(tdy) / tmag >= 0.55 && tmag / palmSize > 0.36) {
        const conf = Math.min(0.95, 0.79 + (Math.abs(tdy) / tmag) * 0.16);
        if (
          tdy < 0 &&
          iso[4].y < Math.min(...curledTipYs) - 0.08 * palmSize &&
          iso[4].y < iso[5].y - 0.12 * palmSize
        ) {
          return { pose: 'thumbs_up', confidence: conf, extendedFingers, pinchPu, palmSize };
        }
        if (
          tdy > 0 &&
          iso[4].y > Math.max(...curledTipYs) + 0.08 * palmSize &&
          iso[4].y > iso[5].y + 0.12 * palmSize
        ) {
          return { pose: 'thumbs_down', confidence: conf, extendedFingers, pinchPu, palmSize };
        }
      }
    }

    if (nonThumbCurled === 4 && thumbCenterPu < 0.82) {
      return { pose: 'closed_fist', confidence: 0.92, extendedFingers, pinchPu, palmSize };
    }

    return { pose: null, confidence: 0, extendedFingers, pinchPu, palmSize };
  }

  detectDynamicSwipe(nowSec, cfg) {
    if (this.trajectory.length < 4) return null;
    const minRefractory = Math.max(0.42, (cfg.gesture_cooldown_seconds || 0.75) * 0.65);
    if (nowSec - this.lastSwipeTime < minRefractory) return null;

    const curr = this.trajectory[this.trajectory.length - 1];
    let anchor = null;
    let anchorIdx = 0;
    for (let i = this.trajectory.length - 2; i >= 0; i--) {
      const age = curr.t - this.trajectory[i].t;
      if (age >= 0.10 && age <= 0.28) {
        anchor = this.trajectory[i];
        anchorIdx = i;
        if (age >= 0.18) break;
      }
    }
    if (!anchor) return null;

    const dt = Math.max(1e-3, curr.t - anchor.t);
    const avgPs = Math.max(1e-4, 0.5 * (curr.ps + anchor.ps));
    const dxPu = (curr.x - anchor.x) / avgPs;
    const dyPu = (curr.y - anchor.y) / avgPs;
    const dispPu = Math.hypot(dxPu, dyPu);

    let pathPu = 0;
    for (let j = anchorIdx + 1; j < this.trajectory.length; j++) {
      const prev = this.trajectory[j - 1];
      const pt = this.trajectory[j];
      pathPu += Math.hypot(pt.x - prev.x, pt.y - prev.y) / avgPs;
    }

    const linearity = dispPu / Math.max(pathPu, 1e-4);
    const velPuS = dispPu / dt;
    const sens = Math.max(0.4, Math.min(2.0, cfg.swipe_sensitivity || 1.0));
    const minDisp = 1.30 / sens;
    const minVel = 5.0 / sens;

    if (dispPu < minDisp || velPuS < minVel || linearity < 0.74) return null;

    let swipeKey = null;
    if (Math.abs(dxPu) > Math.abs(dyPu) * 1.55) {
      swipeKey = dxPu > 0 ? 'swipe_right' : 'swipe_left';
    } else if (Math.abs(dyPu) > Math.abs(dxPu) * 1.55) {
      swipeKey = dyPu > 0 ? 'swipe_down' : 'swipe_up';
    }
    if (!swipeKey) return null;

    if (
      this.lastSwipeDir &&
      OPPOSITE_SWIPE[this.lastSwipeDir] === swipeKey &&
      nowSec - this.lastSwipeTime < 0.78
    ) {
      return null;
    }

    const confidence = Math.min(0.98, 0.76 + Math.min(0.22, (velPuS - minVel) * 0.03));
    this.lastSwipeTime = nowSec;
    this.lastSwipeDir = swipeKey;
    this.trajectory = [];
    return { pose: swipeKey, confidence };
  }

  computeErgonomicCursor(landmarks, nowSec, cfg) {
    this.cursorFilter.updateParams(
      cfg.cursor_smoothing_min_cutoff || 1.15,
      cfg.cursor_smoothing_beta || 1.45
    );
    const anchorX = 0.65 * landmarks[5].x + 0.35 * landmarks[8].x;
    const anchorY = 0.65 * landmarks[5].y + 0.35 * landmarks[8].y;
    const sens = Math.max(0.6, Math.min(2.5, cfg.cursor_sensitivity || 1.35));
    const halfSpanX = 0.32 / sens;
    const halfSpanY = 0.28 / sens;
    const normX = 0.5 + (anchorX - 0.50) / Math.max(0.10, 2.0 * halfSpanX);
    const normY = 0.5 + (anchorY - 0.46) / Math.max(0.10, 2.0 * halfSpanY);
    const cx = Math.max(0, Math.min(1, normX));
    const cy = Math.max(0, Math.min(1, normY));
    return this.cursorFilter.filter(cx, cy, nowSec);
  }

  update(landmarks, nowSec, aspectRatio = 16 / 9) {
    const cfg = this.getConfig();
    const mode = cfg.activation_mode || 'clutch';
    const profileObj = (cfg.profiles && cfg.profiles[cfg.active_profile]) || {};
    const mappings = profileObj.mappings || {};

    if (mode === 'clutch' && this.clutchUnlocked && nowSec > this.clutchUnlockExpiry) {
      this.clutchUnlocked = false;
    }

    if (!landmarks || landmarks.length < 21) {
      this.trajectory = [];
      this.poseHistory = [];
      this.holdCandidate = null;
      this.holdStartTime = null;
      this.holdFiredForPose = false;
      const rem = this.clutchUnlocked ? Math.max(0, this.clutchUnlockExpiry - nowSec) : 0;
      return {
        gesture: null,
        gestureName: 'None',
        gestureIcon: '🤚',
        confidence: 0,
        extendedFingers: [],
        isPinching: false,
        clutchUnlocked: mode !== 'clutch' || this.clutchUnlocked,
        clutchRemainingS: rem,
        dwellProgress: 0,
        triggeredGesture: null,
        triggeredAction: null,
        cursorXY: null,
        palmCenter: null,
      };
    }

    const palmNx = (landmarks[0].x + landmarks[5].x + landmarks[9].x + landmarks[17].x) * 0.25;
    const palmNy = (landmarks[0].y + landmarks[5].y + landmarks[9].y + landmarks[17].y) * 0.25;

    const { pose: staticPose, confidence: staticConf, extendedFingers, pinchPu, palmSize } =
      this.classifyStaticPose(landmarks, aspectRatio);

    this.trajectory.push({ t: nowSec, x: palmNx * aspectRatio, y: palmNy, ps: palmSize });
    if (this.trajectory.length > 24) this.trajectory.shift();

    this.poseHistory.push({ t: nowSec, pose: staticPose, conf: staticConf });
    if (this.poseHistory.length > 6) this.poseHistory.shift();

    let stablePose = null;
    let stableConf = 0;
    if (staticPose) {
      const recent = this.poseHistory.filter(
        (item) => nowSec - item.t < 0.25 && item.pose === staticPose
      );
      if (recent.length >= 2) {
        stablePose = staticPose;
        stableConf = recent.reduce((acc, r) => acc + r.conf, 0) / recent.length;
      }
    }

    const swipeRes = this.detectDynamicSwipe(nowSec, cfg);
    const cursorXY = this.computeErgonomicCursor(landmarks, nowSec, cfg);

    let triggeredGesture = null;
    let triggeredAction = null;
    let dwellProgress = 0;

    const activePose = swipeRes ? swipeRes.pose : stablePose;
    const activeConf = swipeRes ? swipeRes.confidence : stableConf;

    if (mode === 'clutch') {
      const wakeGesture = cfg.clutch_wake_gesture || 'open_palm';
      const wakeHold = cfg.clutch_hold_seconds || 0.38;
      const winDur = cfg.clutch_window_seconds || 4.0;

      if (!this.clutchUnlocked) {
        if (stablePose === wakeGesture) {
          if (this.holdCandidate !== wakeGesture || this.holdStartTime === null) {
            this.holdCandidate = wakeGesture;
            this.holdStartTime = nowSec;
            this.holdFiredForPose = false;
          }
          const elapsed = nowSec - this.holdStartTime;
          dwellProgress = Math.min(1, elapsed / Math.max(0.15, wakeHold));
          if (elapsed >= wakeHold) {
            this.clutchUnlocked = true;
            this.clutchUnlockExpiry = nowSec + winDur;
            this.holdFiredForPose = true;
            dwellProgress = 1;
          }
        } else {
          this.holdCandidate = null;
          this.holdStartTime = null;
        }
      } else {
        if (swipeRes) {
          const mapped = mappings[swipeRes.pose] || 'none';
          if (mapped !== 'none') {
            triggeredGesture = swipeRes.pose;
            triggeredAction = mapped;
            this.lastTriggerTime = nowSec;
            this.clutchUnlockExpiry = nowSec + winDur;
          }
        } else if (stablePose) {
          if (stablePose !== this.holdCandidate) {
            this.holdCandidate = stablePose;
            this.holdStartTime = nowSec;
            this.holdFiredForPose = false;
          }
          if (!this.holdFiredForPose && this.holdStartTime !== null) {
            const reqHold = 0.28;
            const elapsed = nowSec - this.holdStartTime;
            dwellProgress = Math.min(1, elapsed / reqHold);
            const cooldownOk =
              nowSec - this.lastTriggerTime >= (cfg.gesture_cooldown_seconds || 0.75);
            if (elapsed >= reqHold && cooldownOk) {
              const mapped = mappings[stablePose] || 'none';
              if (mapped === 'lock_clutch') {
                this.clutchUnlocked = false;
                this.clutchUnlockExpiry = 0;
                triggeredGesture = stablePose;
                triggeredAction = 'lock_clutch';
                this.holdFiredForPose = true;
              } else if (mapped !== 'none') {
                triggeredGesture = stablePose;
                triggeredAction = mapped;
                this.lastTriggerTime = nowSec;
                this.holdFiredForPose = true;
                this.clutchUnlockExpiry = nowSec + winDur;
              }
            }
          }
        } else {
          this.holdCandidate = null;
          this.holdStartTime = null;
          this.holdFiredForPose = false;
        }
      }
    } else {
      const reqHold = mode === 'dwell' ? cfg.dwell_hold_seconds || 0.42 : 0.16;
      if (swipeRes) {
        const mapped = mappings[swipeRes.pose] || 'none';
        if (mapped !== 'none') {
          triggeredGesture = swipeRes.pose;
          triggeredAction = mapped;
          this.lastTriggerTime = nowSec;
        }
      } else if (stablePose) {
        if (stablePose !== this.holdCandidate) {
          this.holdCandidate = stablePose;
          this.holdStartTime = nowSec;
          this.holdFiredForPose = false;
        }
        if (this.holdStartTime !== null) {
          const elapsed = nowSec - this.holdStartTime;
          dwellProgress = Math.min(1, elapsed / Math.max(0.1, reqHold));
          const mapped = mappings[stablePose] || 'none';
          const repeatable = ['volume_up', 'volume_down', 'scroll_up', 'scroll_down'].includes(
            mapped
          );
          const cooldown = repeatable ? 0.35 : cfg.gesture_cooldown_seconds || 0.75;
          if (
            elapsed >= reqHold &&
            (!this.holdFiredForPose || repeatable) &&
            nowSec - this.lastTriggerTime >= cooldown
          ) {
            if (mapped !== 'none') {
              triggeredGesture = stablePose;
              triggeredAction = mapped;
              this.lastTriggerTime = nowSec;
              this.holdFiredForPose = true;
            }
          }
        }
      } else {
        this.holdCandidate = null;
        this.holdStartTime = null;
        this.holdFiredForPose = false;
      }
    }

    const meta = (activePose && GESTURE_META[activePose]) || null;
    const remS =
      mode === 'clutch' && this.clutchUnlocked
        ? Math.max(0, this.clutchUnlockExpiry - nowSec)
        : 0;

    return {
      gesture: activePose,
      gestureName: meta ? meta.name : 'None',
      gestureIcon: meta ? meta.icon : '🤚',
      confidence: activeConf,
      extendedFingers,
      pinchRatio: pinchPu,
      isPinching: this.pinchActive,
      clutchUnlocked: mode !== 'clutch' || this.clutchUnlocked,
      clutchRemainingS: remS,
      dwellProgress,
      triggeredGesture,
      triggeredAction,
      cursorXY,
      palmCenter: { x: palmNx, y: palmNy },
    };
  }
}
