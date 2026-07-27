/**
 * Gesture recognition — a faithful JavaScript port of the Python engine.
 *
 * The scoring rules, thresholds and stabiliser logic mirror
 * `src/gesturekit/geometry.py` and `src/gesturekit/gestures/static.py`, so the
 * browser demo shows the same answers as the desktop app rather than a
 * separate, hand-waved implementation.
 */

export const WRIST = 0;
export const THUMB_MCP = 2, THUMB_TIP = 4;
export const INDEX_MCP = 5, INDEX_PIP = 6, INDEX_TIP = 8;
export const MIDDLE_MCP = 9, MIDDLE_TIP = 12;
export const RING_MCP = 13, RING_TIP = 16;
export const PINKY_MCP = 17, PINKY_TIP = 20;

// (mcp, pip, dip, tip) per finger, thumb first.
const CHAINS = [
  [1, 2, 3, 4],
  [5, 6, 7, 8],
  [9, 10, 11, 12],
  [13, 14, 15, 16],
  [17, 18, 19, 20],
];

export const CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [5,9],[9,10],[10,11],[11,12],
  [9,13],[13,14],[14,15],[15,16],
  [13,17],[17,18],[18,19],[19,20],
  [0,17],
];

const sub = (a, b) => [a.x - b.x, a.y - b.y, (a.z ?? 0) - (b.z ?? 0)];
const norm2 = (v) => Math.hypot(v[0], v[1]);

/** Mean wrist→MCP distance: the scale reference for every threshold. */
export function palmSize(lm) {
  const mcps = [INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP];
  let total = 0;
  for (const i of mcps) total += norm2(sub(lm[i], lm[WRIST]));
  return Math.max(total / mcps.length, 1e-6);
}

/** Palm centroid — deliberately excludes fingers so curling them moves nothing. */
export function handCenter(lm) {
  const idx = [WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP];
  let x = 0, y = 0;
  for (const i of idx) { x += lm[i].x; y += lm[i].y; }
  return [x / idx.length, y / idx.length];
}

/** Per-finger curl in [0,1] from the PIP joint angle: orientation independent. */
export function fingerCurls(lm) {
  return CHAINS.map(([mcp, pip, , tip]) => {
    const v1 = sub(lm[mcp], lm[pip]);
    const v2 = sub(lm[tip], lm[pip]);
    const n1 = Math.hypot(v1[0], v1[1], v1[2]);
    const n2 = Math.hypot(v2[0], v2[1], v2[2]);
    const denom = Math.max(n1 * n2, 1e-6);
    const dot = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2];
    const ang = Math.acos(Math.min(1, Math.max(-1, dot / denom)));
    return Math.min(1, Math.max(0, (1 - ang / Math.PI) * 1.35 - 0.15));
  });
}

export function pinchDistance(lm) {
  return norm2(sub(lm[THUMB_TIP], lm[INDEX_TIP])) / palmSize(lm);
}

export function pointingDirection(lm) {
  const v = sub(lm[INDEX_TIP], lm[INDEX_PIP]);
  const n = norm2(v);
  return n > 1e-6 ? [v[0] / n, v[1] / n] : [0, 0];
}

// --- smooth membership helpers (mirror the Python `_lo` / `_hi`) ------------
const lo = (x, edge, soft = 0.18) => clamp((edge + soft - x) / (2 * soft));
const hi = (x, edge, soft = 0.18) => clamp((x - edge + soft) / (2 * soft));
const clamp = (v) => Math.min(1, Math.max(0, v));
const ext = (c, i) => lo(c[i], 0.42);
const fold = (c, i) => hi(c[i], 0.55);

/** Geometric mean: one wrong finger drags the whole score down. */
function gmean(values) {
  if (!values.length) return 0;
  let sum = 0;
  for (const v of values) sum += Math.log(Math.min(1, Math.max(1e-4, v)));
  return Math.exp(sum / values.length);
}

/** Build the derived features every rule reads. */
export function handFeatures(lm) {
  const curls = fingerCurls(lm);
  const scale = palmSize(lm);
  return {
    lm,
    curls,
    scale,
    pinch: pinchDistance(lm),
    dir: pointingDirection(lm),
    center: handCenter(lm),
    spread: norm2(sub(lm[INDEX_TIP], lm[PINKY_TIP])) / scale,
    gap: norm2(sub(lm[INDEX_TIP], lm[MIDDLE_TIP])) / scale,
    thumbSpread: norm2(sub(lm[THUMB_TIP], lm[INDEX_MCP])) / scale,
  };
}

function pinchShape(f) {
  return lo(f.pinch, 0.34, 0.12) * hi(f.curls[1], 0.16, 0.14);
}

function thumbOut(f) {
  return hi(f.pinch, 0.55, 0.22);
}

export const RULES = {
  open_palm: (f) => gmean([0,1,2,3,4].map(i => ext(f.curls, i))) * hi(f.spread, 0.75, 0.3),

  closed_fist: (f) =>
    gmean([1,2,3,4].map(i => fold(f.curls, i)).concat([hi(f.curls[0], 0.4, 0.25)])),

  point_up: (f) =>
    gmean([ext(f.curls,1), fold(f.curls,2), fold(f.curls,3), fold(f.curls,4)]) *
    hi(-f.dir[1], 0.55, 0.35),

  point_down: (f) =>
    gmean([ext(f.curls,1), fold(f.curls,2), fold(f.curls,3), fold(f.curls,4)]) *
    hi(f.dir[1], 0.55, 0.35),

  point_left: (f) =>
    gmean([ext(f.curls,1), fold(f.curls,2), fold(f.curls,3), fold(f.curls,4)]) *
    hi(-f.dir[0], 0.6, 0.3),

  point_right: (f) =>
    gmean([ext(f.curls,1), fold(f.curls,2), fold(f.curls,3), fold(f.curls,4)]) *
    hi(f.dir[0], 0.6, 0.3),

  peace: (f) =>
    gmean([ext(f.curls,1), ext(f.curls,2), fold(f.curls,3), fold(f.curls,4)]) *
    hi(f.gap, 0.28, 0.18),

  thumbs_up: (f) => {
    const shape = gmean([ext(f.curls,0)].concat([1,2,3,4].map(i => fold(f.curls,i))));
    const v = sub(f.lm[THUMB_TIP], f.lm[THUMB_MCP]);
    const n = norm2(v);
    return n < 1e-6 ? 0 : shape * hi(-v[1] / n, 0.5, 0.35) * thumbOut(f);
  },

  thumbs_down: (f) => {
    const shape = gmean([ext(f.curls,0)].concat([1,2,3,4].map(i => fold(f.curls,i))));
    const v = sub(f.lm[THUMB_TIP], f.lm[THUMB_MCP]);
    const n = norm2(v);
    return n < 1e-6 ? 0 : shape * hi(v[1] / n, 0.5, 0.35) * thumbOut(f);
  },

  pinch: (f) => pinchShape(f) * (1 - 0.85 * gmean([2,3,4].map(i => ext(f.curls, i)))),
  ok:    (f) => pinchShape(f) * gmean([2,3,4].map(i => ext(f.curls, i))),

  rock: (f) => gmean([ext(f.curls,1), fold(f.curls,2), fold(f.curls,3), ext(f.curls,4)]),

  call_me: (f) =>
    gmean([ext(f.curls,0), fold(f.curls,1), fold(f.curls,2), fold(f.curls,3), ext(f.curls,4)]),

  three: (f) =>
    gmean([ext(f.curls,1), ext(f.curls,2), ext(f.curls,3), fold(f.curls,4)]),

  four: (f) =>
    gmean([fold(f.curls,0)].concat([1,2,3,4].map(i => ext(f.curls, i)))),
};

const PRIORS = { pinch: 0.9, ok: 0.95, rock: 0.95 };

const CANNED = {
  Open_Palm: "open_palm",
  Closed_Fist: "closed_fist",
  Pointing_Up: "point_up",
  Victory: "peace",
  Thumb_Up: "thumbs_up",
  Thumb_Down: "thumbs_down",
};

/** Score every gesture for one hand. */
export function scoreAll(lm, cannedLabel = null, cannedScore = 0) {
  const f = handFeatures(lm);
  const scores = {};
  for (const [name, rule] of Object.entries(RULES)) {
    let value = 0;
    try { value = rule(f) * (PRIORS[name] ?? 1); } catch { value = 0; }
    scores[name] = clamp(value);
  }
  // Canned label is a soft prior, never an override.
  if (cannedLabel && cannedScore > 0.5) {
    const mapped = CANNED[cannedLabel];
    if (mapped) scores[mapped] = clamp(scores[mapped] * (1 + 0.35 * cannedScore));
  }
  return scores;
}

/**
 * Temporal voting: N consistent frames, a minimum score and a margin over the
 * runner-up before a gesture is accepted. Edge-triggered.
 */
export class Stabilizer {
  constructor({ minFrames = 4, minScore = 0.62, margin = 0.1, releaseFrames = 3 } = {}) {
    Object.assign(this, { minFrames, minScore, margin, releaseFrames });
    this.reset();
  }

  reset() {
    this.candidate = null;
    this.count = 0;
    this.scores = [];
    this.active = null;
    this.absent = 0;
  }

  update(scores) {
    const ranked = Object.entries(scores).sort((a, b) => b[1] - a[1]);
    if (!ranked.length) return this._decay();

    const [top, topScore] = ranked[0];
    const second = ranked[1]?.[1] ?? 0;
    if (topScore < this.minScore || topScore - second < this.margin) return this._decay();

    if (top === this.candidate) {
      this.count += 1;
      this.scores.push(topScore);
    } else {
      this.candidate = top;
      this.count = 1;
      this.scores = [topScore];
    }
    this.absent = 0;

    if (this.count >= this.minFrames && this.active !== top) {
      this.active = top;
      const recent = this.scores.slice(-this.minFrames);
      return { name: top, confidence: recent.reduce((a, b) => a + b, 0) / recent.length };
    }
    return null;
  }

  _decay() {
    this.absent += 1;
    if (this.absent >= this.releaseFrames) this.reset();
    return null;
  }
}

export const GESTURE_INFO = [
  ["open_palm",   "Open palm",     "Advance / play"],
  ["closed_fist", "Closed fist",   "End slideshow"],
  ["point_up",    "Point up",      "Laser pointer"],
  ["point_down",  "Point down",    "Directional"],
  ["point_left",  "Point left",    "Directional"],
  ["point_right", "Point right",   "Directional"],
  ["peace",       "Peace / V",     "Blackout screen"],
  ["thumbs_up",   "Thumbs up",     "Whiteout / volume up"],
  ["thumbs_down", "Thumbs down",   "Volume down"],
  ["pinch",       "Pinch",         "Click / grab"],
  ["ok",          "OK sign",       "Normal cursor"],
  ["rock",        "Rock sign",     "Pen tool"],
  ["call_me",     "Call me",       "Eraser"],
  ["three",       "Three fingers", "Start slideshow"],
  ["four",        "Four fingers",  "Present from here"],
];
