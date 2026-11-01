/**
 * 1-Euro Filter (Casiez et al., CHI 2012) for jitter-free long-range landmarks & Air Cursor.
 */

function smoothingFactor(dt, cutoffHz) {
  const tau = 1.0 / (2.0 * Math.PI * Math.max(cutoffHz, 1e-4));
  return 1.0 / (1.0 + tau / Math.max(dt, 1e-4));
}

export class OneEuroFilter {
  constructor(minCutoff = 1.25, beta = 1.6, dCutoff = 1.0) {
    this.minCutoff = minCutoff;
    this.beta = beta;
    this.dCutoff = dCutoff;
    this.xPrev = null;
    this.dxPrev = 0.0;
    this.tPrev = null;
  }

  reset() {
    this.xPrev = null;
    this.dxPrev = 0.0;
    this.tPrev = null;
  }

  filter(x, tSec) {
    if (this.tPrev === null || this.xPrev === null) {
      this.xPrev = x;
      this.dxPrev = 0.0;
      this.tPrev = tSec;
      return x;
    }
    const dt = Math.max(tSec - this.tPrev, 1e-4);
    if (dt > 0.45) {
      this.xPrev = x;
      this.dxPrev = 0.0;
      this.tPrev = tSec;
      return x;
    }
    const aD = smoothingFactor(dt, this.dCutoff);
    const dx = (x - this.xPrev) / dt;
    const dxHat = aD * dx + (1.0 - aD) * this.dxPrev;

    const cutoff = this.minCutoff + this.beta * Math.abs(dxHat);
    const a = smoothingFactor(dt, cutoff);
    const xHat = a * x + (1.0 - a) * this.xPrev;

    this.xPrev = xHat;
    this.dxPrev = dxHat;
    this.tPrev = tSec;
    return xHat;
  }
}

export class Point2DOneEuroFilter {
  constructor(minCutoff = 1.15, beta = 1.45, deadzone = 0.0012) {
    this.fx = new OneEuroFilter(minCutoff, beta, 1.0);
    this.fy = new OneEuroFilter(minCutoff, beta, 1.0);
    this.deadzone = deadzone;
    this.lastOut = null;
  }

  updateParams(minCutoff, beta) {
    this.fx.minCutoff = minCutoff;
    this.fy.minCutoff = minCutoff;
    this.fx.beta = beta;
    this.fy.beta = beta;
  }

  reset() {
    this.fx.reset();
    this.fy.reset();
    this.lastOut = null;
  }

  filter(x, y, tSec) {
    const sx = this.fx.filter(x, tSec);
    const sy = this.fy.filter(y, tSec);
    if (this.lastOut && this.deadzone > 0) {
      if (Math.hypot(sx - this.lastOut.x, sy - this.lastOut.y) < this.deadzone) {
        return this.lastOut;
      }
    }
    this.lastOut = { x: sx, y: sy };
    return this.lastOut;
  }
}

export class LandmarkSmoother {
  constructor(count = 21, minCutoff = 1.45, beta = 1.8) {
    this.filters = Array.from({ length: count }, () => ({
      x: new OneEuroFilter(minCutoff, beta),
      y: new OneEuroFilter(minCutoff, beta),
      z: new OneEuroFilter(minCutoff * 0.85, beta),
    }));
  }

  reset() {
    for (const f of this.filters) {
      f.x.reset();
      f.y.reset();
      f.z.reset();
    }
  }

  filter(landmarks, tSec) {
    return landmarks.map((pt, i) => {
      const f = this.filters[i];
      if (!f) return { x: pt.x, y: pt.y, z: pt.z || 0 };
      return {
        x: f.x.filter(pt.x, tSec),
        y: f.y.filter(pt.y, tSec),
        z: f.z.filter(pt.z || 0, tSec),
      };
    });
  }
}
