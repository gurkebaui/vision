/**
 * Interactive Presentation Deck, Laser Pointer / Highlighter Canvas,
 * Media Synthesizer Player, and Air-Scroll Reader Stage.
 */

export const DEFAULT_SLIDES = [
  {
    tag: 'VISION CONTROL v2.0',
    title: 'Long-Range Hand Gesture Control',
    subtitle: 'Stand 0.4m to 5.0m+ away. Zero clickers. Zero GPU required.',
    bullets: [
      '3-Tier Smart Zoom ROI crops distant hands at native sensor resolution before inference.',
      'Scale-invariant Palm-Unit (PU) joint geometry works identically at 0.5m and 4.8m.',
      '1-Euro adaptive filtering eliminates pixel chatter while keeping flick swipes instant.',
    ],
    metric: '5.0m+',
    metricLabel: 'Effective Range',
  },
  {
    tag: 'OPTICS & RANGE ARCHITECTURE',
    title: 'Why Legacy Tracking Stopped at 1 Meter',
    subtitle: 'Solving the 192x192 full-frame downsampling bottleneck',
    bullets: [
      'Legacy pipelines shrink an entire 1280x720 frame to 192x108 — crushing a 3m hand to 7 pixels.',
      'Tier 2 Upper-Body Pose locates raised wrists across the room and rejects resting arms.',
      'Tier 1 Velocity-Predicted Temporal Crop locks onto the hand at up to 4.5x optical-equivalent zoom.',
    ],
    metric: '4.5x',
    metricLabel: 'Smart ROI Zoom',
  },
  {
    tag: 'CPU-FIRST ENGINEERING',
    title: 'Built for Thin Laptops Without a GPU',
    subtitle: 'XNNPACK Lite inference + Threaded MJPG capture + Sub-ROI HUD',
    bullets: [
      'Processing a 256x256 smart crop uses 33x less memory bandwidth than full-frame 1080p uploads.',
      'Local LAB-L CLAHE contrast boost runs only on the hand crop (<0.2ms) for backlit rooms.',
      'Sub-region HUD compositing eliminates full-frame buffer copies (~8–12ms total CPU budget).',
    ],
    metric: '~9ms',
    metricLabel: 'CPU Frame Time',
  },
  {
    tag: 'INTENTIONAL CONTROL',
    title: 'Talk Naturally Without False Triggers',
    subtitle: 'Presenter Clutch Window + Return-Stroke Swipe Suppression',
    bullets: [
      'Clutch Mode: Hold Open Palm for 0.4s to unlock an auto-extending 4s command window.',
      'Make a Closed Fist anytime to lock immediately while gesturing to your audience.',
      'Return-Stroke Suppression ignores recovery hand motion after a directional swipe.',
    ],
    metric: '0%',
    metricLabel: 'Rebound Misfires',
  },
  {
    tag: 'BEYOND POWERPOINT',
    title: 'Presentations, Media & Air Mouse in One',
    subtitle: 'Hold 3 Fingers (🤟) to cycle control modes from across the room',
    bullets: [
      'Presentation Pro: PowerPoint, Keynote, Google Slides, PDFs, Laser Pointer & Zoom.',
      'Media & Couch Mode: Spotify, YouTube, VLC Play/Pause, Track Skip & System Volume.',
      'Air Mouse & Reader: 1-Euro cursor, Thumb-Index Pinch Click/Drag & hands-free scrolling.',
    ],
    metric: '4 Modes',
    metricLabel: 'Instant Switch',
  },
  {
    tag: 'LIVE INTERACTIVE TEST',
    title: 'Try Laser Pointer & Ink Annotations Now',
    subtitle: 'Show Peace Sign (✌️) to toggle Laser Mode, then Pinch (👌) to draw!',
    bullets: [
      'Swipe Right (👉) or Point Up (☝️) to advance to the next slide.',
      'Swipe Left (👈) or Point Down (👇) to return to the previous slide.',
      'Thumbs Up (👍) to Zoom In, Thumbs Down (👎) to toggle Blackout Curtain.',
    ],
    metric: 'Live',
    metricLabel: 'Laser & Ink Ready',
  },
];

export class StageController {
  constructor() {
    this.slides = [...DEFAULT_SLIDES];
    this.currentSlide = 0;
    this.isPlaying = false;
    this.blackout = false;
    this.zoomLevel = 1.0;
    this.laserEnabled = false;
    this.playTimer = null;

    // Media deck state
    this.mediaPlaying = false;
    this.mediaVolume = 72;
    this.mediaMuted = false;
    this.mediaTrackIndex = 0;
    this.mediaTracks = [
      { title: 'Keynote Ambient Opening', artist: 'Studio Synth Ensemble', duration: '03:42' },
      { title: 'Deep Focus Telemetry', artist: 'Neural Acoustic Lab', duration: '04:15' },
      { title: 'Spatial Horizon Suite', artist: 'Hannover Electronic', duration: '02:58' },
    ];

    // Drawing ink strokes on main stage slide: array of [{x, y}]
    this.inkStrokes = [];
    this.activeStroke = null;
    this.lastCursor = { x: 0.5, y: 0.5 };
  }

  initDOM() {
    this.miniTitle = document.getElementById('mini-slide-title');
    this.miniSub = document.getElementById('mini-slide-sub');
    this.miniCount = document.getElementById('mini-slide-count');
    this.miniLaserDot = document.getElementById('mini-laser-dot');
    this.miniVolFill = document.getElementById('mini-vol-fill');
    this.miniVolText = document.getElementById('mini-vol-text');

    this.stageSlideEl = document.getElementById('stage-slide-card');
    this.stageTag = document.getElementById('stage-slide-tag');
    this.stageTitle = document.getElementById('stage-slide-title');
    this.stageSub = document.getElementById('stage-slide-sub');
    this.stageBullets = document.getElementById('stage-slide-bullets');
    this.stageMetric = document.getElementById('stage-slide-metric');
    this.stageMetricLabel = document.getElementById('stage-slide-metric-label');
    this.stageCounter = document.getElementById('stage-slide-counter');
    this.stageBlackoutEl = document.getElementById('stage-blackout-overlay');
    this.stageLaserDot = document.getElementById('stage-laser-dot');
    this.inkCanvas = document.getElementById('stage-ink-canvas');
    this.inkCtx = this.inkCanvas ? this.inkCanvas.getContext('2d') : null;

    this.mediaTitleEl = document.getElementById('media-track-title');
    this.mediaArtistEl = document.getElementById('media-track-artist');
    this.mediaPlayStateEl = document.getElementById('media-play-badge');
    this.mediaVolBarEl = document.getElementById('media-vol-bar');
    this.mediaVolValEl = document.getElementById('media-vol-val');
    this.readerScrollBox = document.getElementById('air-reader-scroll');

    this.renderAll();
  }

  handleAction(action) {
    switch (action) {
      case 'next_slide':
      case 'browser_forward':
        this.nextSlide();
        break;
      case 'prev_slide':
      case 'browser_back':
        this.prevSlide();
        break;
      case 'play_pause':
        this.togglePlay();
        this.mediaPlaying = !this.mediaPlaying;
        break;
      case 'start_presentation':
        this.currentSlide = 0;
        this.blackout = false;
        break;
      case 'exit_presentation':
        this.isPlaying = false;
        this.blackout = false;
        this.zoomLevel = 1.0;
        if (this.playTimer) clearInterval(this.playTimer);
        break;
      case 'black_screen':
        this.blackout = !this.blackout;
        break;
      case 'toggle_pointer':
        this.laserEnabled = !this.laserEnabled;
        break;
      case 'zoom_in':
        this.zoomLevel = Math.min(1.45, +(this.zoomLevel + 0.12).toFixed(2));
        break;
      case 'zoom_out':
        this.zoomLevel = Math.max(0.85, +(this.zoomLevel - 0.12).toFixed(2));
        break;
      case 'volume_up':
        this.mediaMuted = false;
        this.mediaVolume = Math.min(100, this.mediaVolume + 8);
        break;
      case 'volume_down':
        this.mediaVolume = Math.max(0, this.mediaVolume - 8);
        break;
      case 'volume_mute':
        this.mediaMuted = !this.mediaMuted;
        break;
      case 'next_track':
        this.mediaTrackIndex = (this.mediaTrackIndex + 1) % this.mediaTracks.length;
        this.mediaPlaying = true;
        break;
      case 'prev_track':
        this.mediaTrackIndex =
          (this.mediaTrackIndex - 1 + this.mediaTracks.length) % this.mediaTracks.length;
        this.mediaPlaying = true;
        break;
      case 'scroll_up':
        if (this.readerScrollBox) {
          this.readerScrollBox.scrollBy({ top: -140, behavior: 'smooth' });
        }
        break;
      case 'scroll_down':
        if (this.readerScrollBox) {
          this.readerScrollBox.scrollBy({ top: 140, behavior: 'smooth' });
        }
        break;
      default:
        break;
    }
    this.renderAll();
  }

  nextSlide() {
    this.blackout = false;
    this.currentSlide = (this.currentSlide + 1) % this.slides.length;
    this.clearInk();
    this.renderAll();
  }

  prevSlide() {
    this.blackout = false;
    this.currentSlide = (this.currentSlide - 1 + this.slides.length) % this.slides.length;
    this.clearInk();
    this.renderAll();
  }

  togglePlay() {
    this.isPlaying = !this.isPlaying;
    if (this.playTimer) {
      clearInterval(this.playTimer);
      this.playTimer = null;
    }
    if (this.isPlaying) {
      this.playTimer = setInterval(() => this.nextSlide(), 4000);
    }
    this.renderAll();
  }

  clearInk() {
    this.inkStrokes = [];
    this.activeStroke = null;
    this.redrawInk();
  }

  importCustomDeck(rawText) {
    if (!rawText || !rawText.trim()) return false;
    const blocks = rawText
      .split(/\n\s*---\s*\n|\n{2,}(?=#)/g)
      .map((b) => b.trim())
      .filter(Boolean);
    if (!blocks.length) return false;

    const parsed = blocks.map((block, idx) => {
      const lines = block
        .split('\n')
        .map((l) => l.trim())
        .filter(Boolean);
      const title = (lines[0] || `Slide ${idx + 1}`).replace(/^#+\s*/, '');
      const bullets = lines
        .slice(1)
        .map((l) => l.replace(/^[-*•\d.)]+\s*/, ''))
        .slice(0, 5);
      return {
        tag: `CUSTOM SLIDE ${idx + 1}`,
        title,
        subtitle: bullets[0] || 'Gesture-controlled custom deck',
        bullets: bullets.length > 1 ? bullets.slice(1) : bullets,
        metric: `${idx + 1}/${blocks.length}`,
        metricLabel: 'Deck Progress',
      };
    });

    this.slides = parsed;
    this.currentSlide = 0;
    this.clearInk();
    this.renderAll();
    return true;
  }

  updateLaserAndInk(cursorXY, isPinching, cursorModeEnabled) {
    const showLaser = this.laserEnabled || cursorModeEnabled;
    if (!cursorXY || !showLaser) {
      if (this.miniLaserDot) this.miniLaserDot.style.opacity = '0';
      if (this.stageLaserDot) this.stageLaserDot.style.opacity = '0';
      this.activeStroke = null;
      return;
    }

    this.lastCursor = cursorXY;
    const leftPct = `${(cursorXY.x * 100).toFixed(2)}%`;
    const topPct = `${(cursorXY.y * 100).toFixed(2)}%`;

    if (this.miniLaserDot) {
      this.miniLaserDot.style.opacity = '1';
      this.miniLaserDot.style.left = leftPct;
      this.miniLaserDot.style.top = topPct;
    }
    if (this.stageLaserDot) {
      this.stageLaserDot.style.opacity = '1';
      this.stageLaserDot.style.left = leftPct;
      this.stageLaserDot.style.top = topPct;
      this.stageLaserDot.classList.toggle('pinching', !!isPinching);
    }

    if (isPinching) {
      if (!this.activeStroke) {
        this.activeStroke = [];
        this.inkStrokes.push(this.activeStroke);
      }
      this.activeStroke.push({ x: cursorXY.x, y: cursorXY.y });
      this.redrawInk();
    } else {
      this.activeStroke = null;
    }
  }

  redrawInk() {
    if (!this.inkCanvas || !this.inkCtx) return;
    const rect = this.inkCanvas.getBoundingClientRect();
    if (rect.width > 0 && (this.inkCanvas.width !== Math.round(rect.width))) {
      this.inkCanvas.width = Math.round(rect.width);
      this.inkCanvas.height = Math.round(rect.height);
    }
    const w = this.inkCanvas.width;
    const h = this.inkCanvas.height;
    const ctx = this.inkCtx;
    ctx.clearRect(0, 0, w, h);
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.lineWidth = 4;
    ctx.strokeStyle = '#22d3ee';
    ctx.shadowColor = 'rgba(34, 211, 238, 0.65)';
    ctx.shadowBlur = 10;

    for (const stroke of this.inkStrokes) {
      if (stroke.length < 2) continue;
      ctx.beginPath();
      ctx.moveTo(stroke[0].x * w, stroke[0].y * h);
      for (let i = 1; i < stroke.length; i++) {
        ctx.lineTo(stroke[i].x * w, stroke[i].y * h);
      }
      ctx.stroke();
    }
  }

  renderAll() {
    const s = this.slides[this.currentSlide] || DEFAULT_SLIDES[0];

    if (this.miniTitle) this.miniTitle.textContent = s.title;
    if (this.miniSub) this.miniSub.textContent = s.subtitle;
    if (this.miniCount) {
      this.miniCount.textContent = `Slide ${this.currentSlide + 1} / ${this.slides.length}${
        this.blackout ? ' · BLACKOUT' : ''
      }`;
    }
    const effVol = this.mediaMuted ? 0 : this.mediaVolume;
    if (this.miniVolFill) this.miniVolFill.style.width = `${effVol}%`;
    if (this.miniVolText) {
      this.miniVolText.textContent = this.mediaMuted ? 'Muted' : `${this.mediaVolume}%`;
    }

    if (this.stageTag) this.stageTag.textContent = s.tag;
    if (this.stageTitle) this.stageTitle.textContent = s.title;
    if (this.stageSub) this.stageSub.textContent = s.subtitle;
    if (this.stageMetric) this.stageMetric.textContent = s.metric;
    if (this.stageMetricLabel) this.stageMetricLabel.textContent = s.metricLabel;
    if (this.stageCounter) {
      this.stageCounter.textContent = `Slide ${this.currentSlide + 1} of ${this.slides.length}`;
    }
    if (this.stageBullets) {
      this.stageBullets.innerHTML = (s.bullets || [])
        .map((b) => `<li>${b}</li>`)
        .join('');
    }
    if (this.stageSlideEl) {
      this.stageSlideEl.style.transform = `scale(${this.zoomLevel})`;
    }
    if (this.stageBlackoutEl) {
      this.stageBlackoutEl.style.opacity = this.blackout ? '1' : '0';
      this.stageBlackoutEl.style.pointerEvents = this.blackout ? 'auto' : 'none';
    }

    const track = this.mediaTracks[this.mediaTrackIndex];
    if (this.mediaTitleEl) this.mediaTitleEl.textContent = track.title;
    if (this.mediaArtistEl) {
      this.mediaArtistEl.textContent = `${track.artist} · ${track.duration}`;
    }
    if (this.mediaPlayStateEl) {
      this.mediaPlayStateEl.textContent = this.mediaPlaying ? 'PLAYING' : 'PAUSED';
      this.mediaPlayStateEl.className = `pill ${this.mediaPlaying ? 'pill-emerald' : 'pill-muted'}`;
    }
    if (this.mediaVolBarEl) this.mediaVolBarEl.style.width = `${effVol}%`;
    if (this.mediaVolValEl) {
      this.mediaVolValEl.textContent = this.mediaMuted ? 'MUTED' : `${this.mediaVolume}%`;
    }
  }
}
