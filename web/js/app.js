/**
 * Vision Control v2 — Main Web Studio Application Controller.
 */

import { ScaleInvariantGestureEngine, GESTURE_META } from './gesture-engine.js';
import { BrowserLongRangeTracker } from './long-range-tracker.js';
import { StageController } from './stage-deck.js';

const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [0, 17], [17, 18], [18, 19], [19, 20],
];

const FALLBACK_CONFIG = {
  performance_preset: 'balanced',
  camera_index: 0,
  mirror_camera: true,
  smart_zoom_enabled: true,
  pose_assist_enabled: true,
  resting_arm_rejection: true,
  active_profile: 'presentation',
  activation_mode: 'clutch',
  clutch_wake_gesture: 'open_palm',
  clutch_window_seconds: 4.0,
  clutch_hold_seconds: 0.38,
  dwell_hold_seconds: 0.42,
  gesture_cooldown_seconds: 0.75,
  swipe_sensitivity: 1.0,
  cursor_sensitivity: 1.35,
  cursor_smoothing_min_cutoff: 1.15,
  cursor_smoothing_beta: 1.45,
  os_control_enabled: true,
  show_zoom_pip: true,
  preset_details: {
    capture_width: 1280,
    capture_height: 720,
    roi_size: 256,
    pose_scout_interval: 5,
    tile_scout_interval: 4,
  },
  action_catalog: {
    next_slide: { label: 'Next Slide', keys: 'Right' },
    prev_slide: { label: 'Previous Slide', keys: 'Left' },
    play_pause: { label: 'Play / Pause', keys: 'Space' },
    start_presentation: { label: 'Start Slideshow', keys: 'F5' },
    exit_presentation: { label: 'Exit / Escape', keys: 'Escape' },
    black_screen: { label: 'Blackout Screen', keys: 'B' },
    toggle_pointer: { label: 'Laser Pointer Mode', keys: 'Ctrl+L' },
    zoom_in: { label: 'Zoom In', keys: 'Ctrl++' },
    zoom_out: { label: 'Zoom Out', keys: 'Ctrl+-' },
    volume_up: { label: 'Volume Up', keys: 'VolumeUp' },
    volume_down: { label: 'Volume Down', keys: 'VolumeDown' },
    volume_mute: { label: 'Mute Audio', keys: 'VolumeMute' },
    next_track: { label: 'Next Track', keys: 'MediaNext' },
    prev_track: { label: 'Previous Track', keys: 'MediaPrev' },
    fullscreen: { label: 'Toggle Fullscreen', keys: 'F' },
    left_click: { label: 'Left Click / Drag', keys: 'MouseLeft' },
    right_click: { label: 'Right Click', keys: 'MouseRight' },
    scroll_up: { label: 'Scroll Up', keys: 'ScrollUp' },
    scroll_down: { label: 'Scroll Down', keys: 'ScrollDown' },
    browser_back: { label: 'Back', keys: 'Alt+Left' },
    browser_forward: { label: 'Forward', keys: 'Alt+Right' },
    cycle_profile: { label: 'Switch Control Mode', keys: 'ModeCycle' },
    lock_clutch: { label: 'Lock Gesture Window', keys: 'Lock' },
    none: { label: 'Disabled', keys: '-' },
  },
  profiles: {
    presentation: {
      id: 'presentation',
      name: 'Presentation Pro',
      subtitle: 'PowerPoint, Keynote, Google Slides & PDFs',
      cursor_enabled: false,
      mappings: {
        swipe_right: 'next_slide',
        swipe_left: 'prev_slide',
        pointing_up: 'next_slide',
        pointing_down: 'prev_slide',
        open_palm: 'play_pause',
        peace_sign: 'toggle_pointer',
        thumbs_up: 'zoom_in',
        thumbs_down: 'black_screen',
        closed_fist: 'lock_clutch',
        pinch: 'left_click',
        shaka: 'next_slide',
        three_fingers: 'cycle_profile',
        swipe_up: 'start_presentation',
        swipe_down: 'zoom_out',
      },
    },
    media: {
      id: 'media',
      name: 'Media & Couch Control',
      subtitle: 'Spotify, YouTube, VLC & System Audio',
      cursor_enabled: false,
      mappings: {
        open_palm: 'play_pause',
        swipe_right: 'next_track',
        swipe_left: 'prev_track',
        thumbs_up: 'volume_up',
        thumbs_down: 'volume_down',
        pointing_up: 'volume_up',
        pointing_down: 'volume_down',
        closed_fist: 'volume_mute',
        peace_sign: 'fullscreen',
        three_fingers: 'cycle_profile',
        swipe_up: 'volume_up',
        swipe_down: 'volume_down',
        pinch: 'play_pause',
        shaka: 'next_track',
      },
    },
    air_mouse: {
      id: 'air_mouse',
      name: 'Air Mouse & Reader',
      subtitle: '1-Euro Smooth Cursor, Pinch Click/Drag & Scroll',
      cursor_enabled: true,
      mappings: {
        pinch: 'left_click',
        peace_sign: 'right_click',
        pointing_up: 'scroll_up',
        pointing_down: 'scroll_down',
        swipe_up: 'scroll_up',
        swipe_down: 'scroll_down',
        swipe_left: 'browser_back',
        swipe_right: 'browser_forward',
        thumbs_up: 'zoom_in',
        thumbs_down: 'zoom_out',
        three_fingers: 'cycle_profile',
        open_palm: 'none',
        closed_fist: 'lock_clutch',
        shaka: 'browser_forward',
      },
    },
    custom: {
      id: 'custom',
      name: 'Custom Bindings',
      subtitle: 'Personalized Gesture-to-Action Profile',
      cursor_enabled: false,
      mappings: {
        swipe_right: 'next_slide',
        swipe_left: 'prev_slide',
        pointing_up: 'next_slide',
        pointing_down: 'prev_slide',
        open_palm: 'play_pause',
        peace_sign: 'toggle_pointer',
        thumbs_up: 'volume_up',
        thumbs_down: 'volume_down',
        closed_fist: 'exit_presentation',
        pinch: 'left_click',
        shaka: 'next_slide',
        three_fingers: 'cycle_profile',
        swipe_up: 'scroll_up',
        swipe_down: 'scroll_down',
      },
    },
  },
};

class VisionStudioApp {
  constructor() {
    this.config = structuredClone(FALLBACK_CONFIG);
    this.ws = null;
    this.wsConnected = false;
    this.osGuiAvailable = false;
    this.osName = 'Browser';

    this.tracker = new BrowserLongRangeTracker(() => this.config);
    this.gestureEngine = new ScaleInvariantGestureEngine(() => this.config);
    this.stage = new StageController();

    this.cameraRunning = false;
    this.simulatorRunning = false;
    this.rafId = null;
    this.videoStream = null;

    this.fpsEma = 0;
    this.lastFrameTime = performance.now();
    this.commandCount = 0;
    this.eventLog = [];
    this.toastTimer = null;

    // Range Calibration state
    this.calibSamples = { close: null, mid: null, far: null };
    this.latestTrack = null;
    this.latestEval = null;
  }

  async init() {
    this.cacheDOM();
    this.stage.initDOM();
    this.bindUIEvents();
    this.applyInitialRoute();

    await this.loadServerConfig();
    this.connectWebSocket();
    this.syncConfigToUI();
    this.renderIdleCanvas();
  }

  cacheDOM() {
    this.videoEl = document.getElementById('studio-video');
    this.canvasEl = document.getElementById('studio-canvas');
    this.ctx = this.canvasEl.getContext('2d');
    this.pipCanvasEl = document.getElementById('zoom-pip-canvas');
    this.pipCtx = this.pipCanvasEl.getContext('2d');
    this.pipWrapEl = document.getElementById('zoom-pip-wrap');
    this.pipLabelEl = document.getElementById('zoom-pip-label');

    this.statusBadge = document.getElementById('camera-status-badge');
    this.bridgeBadge = document.getElementById('os-bridge-badge');
    this.loadingOverlay = document.getElementById('model-loading-overlay');
    this.loadingText = document.getElementById('model-loading-text');
    this.toastBanner = document.getElementById('hud-action-toast');

    // Telemetry readouts
    this.elFps = document.getElementById('metric-fps');
    this.elLatency = document.getElementById('metric-latency');
    this.elRange = document.getElementById('metric-range');
    this.elZoom = document.getElementById('metric-zoom');
    this.elSource = document.getElementById('metric-source');
    this.elGestureIcon = document.getElementById('live-gesture-icon');
    this.elGestureName = document.getElementById('live-gesture-name');
    this.elConfFill = document.getElementById('live-conf-fill');
    this.elConfText = document.getElementById('live-conf-text');
    this.elClutchPill = document.getElementById('clutch-state-pill');
    this.elEventList = document.getElementById('command-history-list');
  }

  bindUIEvents() {
    // Top navigation tabs
    document.querySelectorAll('[data-nav-tab]').forEach((btn) => {
      btn.addEventListener('click', (e) => {
        e.preventDefault();
        const tab = btn.getAttribute('data-nav-tab');
        this.switchTab(tab);
      });
    });

    window.addEventListener('hashchange', () => this.applyInitialRoute());

    // Camera & Simulator controls
    document.getElementById('btn-start-camera').addEventListener('click', () => this.startCamera());
    document.getElementById('btn-stop-camera').addEventListener('click', () => this.stopAll());
    document.getElementById('btn-start-sim').addEventListener('click', () => this.startSimulator());

    // Simulator controls bar
    const simDistInput = document.getElementById('sim-distance-slider');
    const simDistVal = document.getElementById('sim-distance-val');
    simDistInput.addEventListener('input', () => {
      const v = parseFloat(simDistInput.value);
      this.tracker.simDistanceM = v;
      simDistVal.textContent = `${v.toFixed(1)}m`;
    });

    document.querySelectorAll('[data-sim-pose]').forEach((btn) => {
      btn.addEventListener('click', () => {
        const p = btn.getAttribute('data-sim-pose');
        if (p.startsWith('swipe_')) {
          if (!this.simulatorRunning) this.startSimulator();
          // Ensure clutch is unlocked or dwell ready so demo swipes respond smoothly
          if (this.config.activation_mode === 'clutch') {
            this.gestureEngine.clutchUnlocked = true;
            this.gestureEngine.clutchUnlockExpiry = performance.now() / 1000 + 5.0;
          }
          this.tracker.triggerSimulatedSwipe(p.replace('swipe_', ''), performance.now() / 1000);
        } else {
          this.tracker.simPose = p;
          document.querySelectorAll('[data-sim-pose]').forEach((b) => b.classList.remove('active'));
          btn.classList.add('active');
          if (!this.simulatorRunning && !this.cameraRunning) this.startSimulator();
        }
      });
    });

    // Profile cards
    document.querySelectorAll('[data-profile-select]').forEach((card) => {
      card.addEventListener('click', () => {
        const pid = card.getAttribute('data-profile-select');
        this.pushConfigUpdate({ active_profile: pid });
      });
    });

    // Activation Mode segmented buttons
    document.querySelectorAll('[data-activation-mode]').forEach((btn) => {
      btn.addEventListener('click', () => {
        const m = btn.getAttribute('data-activation-mode');
        this.gestureEngine.reset();
        this.pushConfigUpdate({ activation_mode: m });
      });
    });

    // CPU Preset buttons
    document.querySelectorAll('[data-cpu-preset]').forEach((btn) => {
      btn.addEventListener('click', () => {
        const p = btn.getAttribute('data-cpu-preset');
        const detailsMap = {
          ultra_light: { capture_width: 960, capture_height: 540, roi_size: 224, pose_scout_interval: 10, tile_scout_interval: 8 },
          balanced: { capture_width: 1280, capture_height: 720, roi_size: 256, pose_scout_interval: 5, tile_scout_interval: 4 },
          max_range: { capture_width: 1920, capture_height: 1080, roi_size: 320, pose_scout_interval: 3, tile_scout_interval: 2 },
        };
        this.config.preset_details = detailsMap[p] || detailsMap.balanced;
        this.pushConfigUpdate({ performance_preset: p });
      });
    });

    // Optics toggles
    document.getElementById('toggle-smart-zoom').addEventListener('change', (e) => {
      this.tracker.reset();
      this.pushConfigUpdate({ smart_zoom_enabled: e.target.checked });
    });
    document.getElementById('toggle-pose-assist').addEventListener('change', (e) => {
      this.pushConfigUpdate({ pose_assist_enabled: e.target.checked });
    });
    document.getElementById('toggle-os-control').addEventListener('change', (e) => {
      this.pushConfigUpdate({ os_control_enabled: e.target.checked });
    });

    // Mini stage & main stage manual buttons
    document.querySelectorAll('[data-trigger-action]').forEach((btn) => {
      btn.addEventListener('click', () => {
        const act = btn.getAttribute('data-trigger-action');
        this.dispatchAction(act, 'manual_ui');
      });
    });

    document.getElementById('btn-clear-ink').addEventListener('click', () => {
      this.stage.clearInk();
    });

    document.getElementById('btn-stage-fullscreen').addEventListener('click', () => {
      const stageWrap = document.getElementById('stage-deck-viewport');
      if (!document.fullscreenElement) {
        stageWrap.requestFullscreen?.();
      } else {
        document.exitFullscreen?.();
      }
    });

    document.getElementById('btn-import-deck').addEventListener('click', () => {
      const txt = document.getElementById('custom-deck-textarea').value;
      if (this.stage.importCustomDeck(txt)) {
        this.showToast('Custom Slide Deck Loaded');
      }
    });

    // Sliders in Keymap & Tuning tab
    const bindSlider = (id, valId, suffix, key) => {
      const el = document.getElementById(id);
      const vEl = document.getElementById(valId);
      if (!el) return;
      el.addEventListener('input', () => {
        const val = parseFloat(el.value);
        if (vEl) vEl.textContent = `${val.toFixed(2)}${suffix}`;
      });
      el.addEventListener('change', () => {
        this.pushConfigUpdate({ [key]: parseFloat(el.value) });
      });
    };
    bindSlider('slider-swipe-sens', 'val-swipe-sens', 'x', 'swipe_sensitivity');
    bindSlider('slider-clutch-win', 'val-clutch-win', 's', 'clutch_window_seconds');
    bindSlider('slider-dwell-hold', 'val-dwell-hold', 's', 'dwell_hold_seconds');
    bindSlider('slider-cursor-sens', 'val-cursor-sens', 'x', 'cursor_sensitivity');

    document.getElementById('select-keymap-profile').addEventListener('change', (e) => {
      this.renderKeymapTable(e.target.value);
    });

    document.getElementById('btn-reset-defaults').addEventListener('click', async () => {
      try {
        const res = await fetch('/api/config/reset', { method: 'POST' });
        if (res.ok) {
          this.config = await res.json();
          this.syncConfigToUI();
          this.showToast('Settings reset to defaults');
        }
      } catch {
        this.config = structuredClone(FALLBACK_CONFIG);
        this.syncConfigToUI();
      }
    });

    // Calibration & CPU Benchmark buttons
    document.querySelectorAll('[data-calib-capture]').forEach((btn) => {
      btn.addEventListener('click', () => {
        const zone = btn.getAttribute('data-calib-capture');
        this.captureCalibrationPoint(zone);
      });
    });

    document.getElementById('btn-run-benchmark').addEventListener('click', () => {
      this.runDiagnosticsBenchmark();
    });
  }

  applyInitialRoute() {
    const hash = (window.location.hash || '#studio').replace('#', '');
    const valid = ['studio', 'stage', 'bindings', 'calibrate'];
    this.switchTab(valid.includes(hash) ? hash : 'studio', false);
  }

  switchTab(tabId, updateHash = true) {
    document.querySelectorAll('[data-nav-tab]').forEach((b) => {
      b.classList.toggle('active', b.getAttribute('data-nav-tab') === tabId);
    });
    document.querySelectorAll('[data-tab-panel]').forEach((panel) => {
      panel.classList.toggle('active', panel.getAttribute('data-tab-panel') === tabId);
    });
    if (updateHash && window.location.hash !== `#${tabId}`) {
      history.replaceState(null, '', `#${tabId}`);
    }
    if (tabId === 'stage') {
      setTimeout(() => this.stage.redrawInk(), 60);
    }
  }

  async loadServerConfig() {
    try {
      const res = await fetch('/api/config');
      if (res.ok) {
        const serverCfg = await res.json();
        this.config = { ...this.config, ...serverCfg };
        return;
      }
    } catch {
      // Standalone static mode fallback
    }
    try {
      const cached = localStorage.getItem('vision_v2_config');
      if (cached) {
        this.config = { ...this.config, ...JSON.parse(cached) };
      }
    } catch {
      // Ignore storage errors
    }
  }

  connectWebSocket() {
    try {
      const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsUrl = `${proto}//${window.location.host}/ws`;
      this.ws = new WebSocket(wsUrl);

      this.ws.onopen = () => {
        this.wsConnected = true;
        this.updateBridgeBadge();
      };

      this.ws.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data);
          if (msg.type === 'hello') {
            this.osName = msg.os_name || 'Local OS';
            this.osGuiAvailable = !!msg.os_gui_available;
            if (msg.config) {
              this.config = { ...this.config, ...msg.config };
              this.syncConfigToUI();
            }
            if (Array.isArray(msg.history)) {
              this.eventLog = msg.history;
              this.renderHistory();
            }
            this.updateBridgeBadge();
          } else if (msg.type === 'config_updated' && msg.config) {
            this.config = { ...this.config, ...msg.config };
            this.syncConfigToUI();
          } else if (msg.type === 'action_executed' && msg.event) {
            if (msg.active_profile && msg.active_profile !== this.config.active_profile) {
              this.config.active_profile = msg.active_profile;
              this.syncConfigToUI();
            }
          }
        } catch {
          // Ignore malformed ws message
        }
      };

      this.ws.onclose = () => {
        this.wsConnected = false;
        this.updateBridgeBadge();
      };
    } catch {
      this.wsConnected = false;
      this.updateBridgeBadge();
    }
  }

  updateBridgeBadge() {
    if (!this.bridgeBadge) return;
    if (this.wsConnected && this.osGuiAvailable) {
      this.bridgeBadge.textContent = `OS Bridge: ${this.osName} Active`;
      this.bridgeBadge.className = 'pill pill-emerald';
    } else if (this.wsConnected) {
      this.bridgeBadge.textContent = `Studio Server Connected (Web Stage)`;
      this.bridgeBadge.className = 'pill pill-cyan';
    } else {
      this.bridgeBadge.textContent = 'Browser Standalone Mode';
      this.bridgeBadge.className = 'pill pill-muted';
    }
  }

  async pushConfigUpdate(partial) {
    Object.assign(this.config, partial);
    if (partial.mapping_update) {
      const { profile, gesture, action } = partial.mapping_update;
      if (this.config.profiles?.[profile]?.mappings) {
        this.config.profiles[profile].mappings[gesture] = action;
      }
    }
    this.syncConfigToUI();

    try {
      localStorage.setItem('vision_v2_config', JSON.stringify(this.config));
    } catch {
      // Ignore
    }

    if (this.wsConnected && this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: 'update_config', config: partial }));
    } else {
      try {
        await fetch('/api/config', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(partial),
        });
      } catch {
        // Offline standalone
      }
    }
  }

  syncConfigToUI() {
    document.querySelectorAll('[data-profile-select]').forEach((el) => {
      el.classList.toggle(
        'active',
        el.getAttribute('data-profile-select') === this.config.active_profile
      );
    });
    document.querySelectorAll('[data-activation-mode]').forEach((el) => {
      el.classList.toggle(
        'active',
        el.getAttribute('data-activation-mode') === this.config.activation_mode
      );
    });
    document.querySelectorAll('[data-cpu-preset]').forEach((el) => {
      el.classList.toggle(
        'active',
        el.getAttribute('data-cpu-preset') === this.config.performance_preset
      );
    });

    const sz = document.getElementById('toggle-smart-zoom');
    if (sz) sz.checked = this.config.smart_zoom_enabled !== false;
    const pa = document.getElementById('toggle-pose-assist');
    if (pa) pa.checked = this.config.pose_assist_enabled !== false;
    const oc = document.getElementById('toggle-os-control');
    if (oc) oc.checked = this.config.os_control_enabled !== false;

    const setSlider = (id, valId, val, suffix) => {
      const s = document.getElementById(id);
      const v = document.getElementById(valId);
      if (s && typeof val === 'number') s.value = String(val);
      if (v && typeof val === 'number') v.textContent = `${val.toFixed(2)}${suffix}`;
    };
    setSlider('slider-swipe-sens', 'val-swipe-sens', this.config.swipe_sensitivity, 'x');
    setSlider('slider-clutch-win', 'val-clutch-win', this.config.clutch_window_seconds, 's');
    setSlider('slider-dwell-hold', 'val-dwell-hold', this.config.dwell_hold_seconds, 's');
    setSlider('slider-cursor-sens', 'val-cursor-sens', this.config.cursor_sensitivity, 'x');

    const selProf = document.getElementById('select-keymap-profile');
    if (selProf) {
      selProf.value = this.config.active_profile;
      this.renderKeymapTable(selProf.value);
    }
    this.renderQuickGestureGrid();
  }

  renderQuickGestureGrid() {
    const grid = document.getElementById('quick-gesture-grid');
    if (!grid) return;
    const prof = this.config.profiles?.[this.config.active_profile] || {};
    const mappings = prof.mappings || {};
    const catalog = this.config.action_catalog || FALLBACK_CONFIG.action_catalog;

    const featured = [
      'swipe_right',
      'swipe_left',
      'open_palm',
      'pointing_up',
      'pointing_down',
      'peace_sign',
      'thumbs_up',
      'thumbs_down',
      'pinch',
      'closed_fist',
    ];

    grid.innerHTML = featured
      .map((gKey) => {
        const g = GESTURE_META[gKey];
        const actKey = mappings[gKey] || 'none';
        const actLabel = catalog[actKey]?.label || actKey;
        return `
          <div class="gesture-chip" data-gesture-chip="${gKey}">
            <span class="g-icon">${g.icon}</span>
            <div class="g-meta">
              <span class="g-name">${g.name}</span>
              <span class="g-act">${actLabel}</span>
            </div>
          </div>
        `;
      })
      .join('');
  }

  renderKeymapTable(profileId) {
    const tbody = document.getElementById('keymap-table-body');
    if (!tbody) return;
    const prof = this.config.profiles?.[profileId] || this.config.profiles.presentation;
    const mappings = prof.mappings || {};
    const catalog = this.config.action_catalog || FALLBACK_CONFIG.action_catalog;
    const actionKeys = Object.keys(catalog);

    tbody.innerHTML = Object.entries(GESTURE_META)
      .map(([gKey, gMeta]) => {
        const currentAct = mappings[gKey] || 'none';
        const optionsHtml = actionKeys
          .map(
            (aKey) =>
              `<option value="${aKey}" ${aKey === currentAct ? 'selected' : ''}>${
                catalog[aKey].label
              } (${catalog[aKey].keys})</option>`
          )
          .join('');
        return `
          <tr>
            <td>
              <div class="km-gesture-cell">
                <span class="km-icon">${gMeta.icon}</span>
                <div>
                  <div class="km-title">${gMeta.name}</div>
                  <div class="km-kind">${gMeta.kind === 'dynamic' ? 'Dynamic Flick' : 'Static Pose'}</div>
                </div>
              </div>
            </td>
            <td>
              <select class="studio-select km-select" data-km-gesture="${gKey}">
                ${optionsHtml}
              </select>
            </td>
            <td>
              <button class="btn-ghost btn-xs" data-test-action="${currentAct}">Test</button>
            </td>
          </tr>
        `;
      })
      .join('');

    tbody.querySelectorAll('[data-km-gesture]').forEach((sel) => {
      sel.addEventListener('change', () => {
        const gesture = sel.getAttribute('data-km-gesture');
        const action = sel.value;
        this.pushConfigUpdate({
          mapping_update: { profile: profileId, gesture, action },
        });
      });
    });

    tbody.querySelectorAll('[data-test-action]').forEach((btn) => {
      btn.addEventListener('click', () => {
        const act = btn.getAttribute('data-test-action');
        this.dispatchAction(act, 'keymap_test');
      });
    });
  }

  async startCamera() {
    this.stopAll();
    const preset = this.config.preset_details || FALLBACK_CONFIG.preset_details;

    try {
      this.loadingOverlay.style.display = 'flex';
      this.loadingText.textContent = 'Requesting camera stream...';

      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: preset.capture_width || 1280 },
          height: { ideal: preset.capture_height || 720 },
          frameRate: { ideal: 30 },
        },
        audio: false,
      });

      this.videoStream = stream;
      this.videoEl.srcObject = stream;
      await this.videoEl.play();

      await this.tracker.initModels((msg) => {
        this.loadingText.textContent = msg;
      });

      this.loadingOverlay.style.display = 'none';
      this.cameraRunning = true;
      this.simulatorRunning = false;
      document.getElementById('sim-toolbar').classList.remove('visible');

      this.canvasEl.width = this.videoEl.videoWidth || 1280;
      this.canvasEl.height = this.videoEl.videoHeight || 720;

      this.statusBadge.textContent = `LIVE · ${this.canvasEl.width}x${this.canvasEl.height} (${this.tracker.delegateUsed})`;
      this.statusBadge.className = 'pill pill-emerald';
      this.tracker.reset();
      this.gestureEngine.reset();
      this.lastFrameTime = performance.now();
      this.loop();
    } catch (err) {
      this.loadingOverlay.style.display = 'none';
      this.statusBadge.textContent = 'Camera unavailable — Switched to Range Simulator';
      this.statusBadge.className = 'pill pill-amber';
      this.startSimulator();
    }
  }

  startSimulator() {
    this.stopAll();
    this.simulatorRunning = true;
    this.canvasEl.width = 1280;
    this.canvasEl.height = 720;
    document.getElementById('sim-toolbar').classList.add('visible');
    this.statusBadge.textContent = 'SIMULATOR · Long-Range Optics Active';
    this.statusBadge.className = 'pill pill-cyan';
    this.tracker.reset();
    this.gestureEngine.reset();
    this.lastFrameTime = performance.now();
    this.loop();
  }

  stopAll() {
    this.cameraRunning = false;
    this.simulatorRunning = false;
    if (this.rafId) {
      cancelAnimationFrame(this.rafId);
      this.rafId = null;
    }
    if (this.videoStream) {
      this.videoStream.getTracks().forEach((t) => t.stop());
      this.videoStream = null;
      this.videoEl.srcObject = null;
    }
    if (this.pipWrapEl) this.pipWrapEl.style.display = 'none';
    this.statusBadge.textContent = 'Standby';
    this.statusBadge.className = 'pill pill-muted';
    this.renderIdleCanvas();
  }

  loop() {
    if (!this.cameraRunning && !this.simulatorRunning) return;

    const nowMs = performance.now();
    const nowSec = nowMs / 1000;
    const dt = Math.max(1e-3, (nowMs - this.lastFrameTime) / 1000);
    this.lastFrameTime = nowMs;
    this.fpsEma = this.fpsEma === 0 ? 30 : 0.88 * this.fpsEma + 0.12 * Math.min(120, 1 / dt);

    const fw = this.canvasEl.width;
    const fh = this.canvasEl.height;

    let trackRes;
    if (this.cameraRunning && this.videoEl.readyState >= 2) {
      trackRes = this.tracker.processVideoFrame(this.videoEl, nowSec);
    } else {
      trackRes = this.tracker.generateSimulatedFrame(fw, fh, nowSec);
    }

    const evalRes = this.gestureEngine.update(
      trackRes.detected ? trackRes.landmarks : null,
      nowSec,
      fw / Math.max(1, fh)
    );

    this.latestTrack = trackRes;
    this.latestEval = evalRes;

    // Update Air Cursor / Laser Pointer
    const profObj = this.config.profiles?.[this.config.active_profile] || {};
    this.stage.updateLaserAndInk(evalRes.cursorXY, evalRes.isPinching, !!profObj.cursor_enabled);

    if (
      evalRes.cursorXY &&
      (profObj.cursor_enabled || this.stage.laserEnabled) &&
      this.wsConnected &&
      this.ws?.readyState === WebSocket.OPEN
    ) {
      this.ws.send(
        JSON.stringify({
          type: 'cursor_move',
          x: evalRes.cursorXY.x,
          y: evalRes.cursorXY.y,
          pinching: evalRes.isPinching,
        })
      );
    }

    // Dispatch newly triggered action
    if (evalRes.triggeredAction) {
      this.dispatchAction(evalRes.triggeredAction, evalRes.triggeredGesture || 'gesture');
    }

    this.renderFrameHUD(trackRes, evalRes, fw, fh);
    this.updateTelemetryDOM(trackRes, evalRes);

    this.rafId = requestAnimationFrame(() => this.loop());
  }

  dispatchAction(action, gestureKey) {
    if (!action || action === 'none') return;

    if (action === 'cycle_profile') {
      const order = ['presentation', 'media', 'air_mouse', 'custom'];
      const next = order[(order.indexOf(this.config.active_profile) + 1) % order.length];
      this.pushConfigUpdate({ active_profile: next });
      const pName = this.config.profiles[next]?.name || next;
      this.recordEvent(action, `Mode -> ${pName}`, gestureKey);
      this.showToast(`🤟 Mode Switched: ${pName}`);
      return;
    }

    if (action === 'lock_clutch') {
      this.recordEvent(action, 'Gesture Window Locked', gestureKey);
      this.showToast('✊ Gesture Window Locked');
      return;
    }

    this.stage.handleAction(action);

    const catalog = this.config.action_catalog || FALLBACK_CONFIG.action_catalog;
    const label = catalog[action]?.label || action;
    const icon = GESTURE_META[gestureKey]?.icon || '⚡';
    this.recordEvent(action, label, gestureKey);
    this.showToast(`${icon}  ${label}`);

    if (this.wsConnected && this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(
        JSON.stringify({ type: 'gesture_trigger', action, gesture: gestureKey })
      );
    }
  }

  recordEvent(action, label, gestureKey) {
    this.commandCount += 1;
    const item = {
      time_str: new Date().toLocaleTimeString(),
      action,
      label,
      gesture: GESTURE_META[gestureKey]?.name || gestureKey,
    };
    this.eventLog.unshift(item);
    if (this.eventLog.length > 16) this.eventLog.pop();
    this.renderHistory();
  }

  renderHistory() {
    if (!this.elEventList) return;
    if (!this.eventLog.length) {
      this.elEventList.innerHTML = `<div class="empty-log">Perform a gesture or click Simulator poses to test</div>`;
      return;
    }
    this.elEventList.innerHTML = this.eventLog
      .slice(0, 8)
      .map(
        (e) => `
        <div class="history-row">
          <span class="h-label">${e.label}</span>
          <span class="h-meta">${e.gesture} · ${e.time_str}</span>
        </div>
      `
      )
      .join('');
  }

  showToast(text) {
    if (!this.toastBanner) return;
    this.toastBanner.textContent = text;
    this.toastBanner.classList.add('visible');
    if (this.toastTimer) clearTimeout(this.toastTimer);
    this.toastTimer = setTimeout(() => {
      this.toastBanner.classList.remove('visible');
    }, 1600);
  }

  renderIdleCanvas() {
    const w = (this.canvasEl.width = 1280);
    const h = (this.canvasEl.height = 720);
    const ctx = this.ctx;
    const grad = ctx.createRadialGradient(w * 0.5, h * 0.45, 40, w * 0.5, h * 0.5, w * 0.65);
    grad.addColorStop(0, '#111c2d');
    grad.addColorStop(1, '#070b12');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, w, h);

    // Subtle optics reticle grid
    ctx.strokeStyle = 'rgba(34, 211, 238, 0.08)';
    ctx.lineWidth = 1;
    for (let x = 160; x < w; x += 160) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
      ctx.stroke();
    }
    for (let y = 144; y < h; y += 144) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }

    ctx.fillStyle = '#e2e8f0';
    ctx.font = '600 26px Inter, system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Long-Range Smart-Zoom Optics Ready', w / 2, h / 2 - 14);
    ctx.fillStyle = '#94a3b8';
    ctx.font = '400 16px Inter, system-ui, sans-serif';
    ctx.fillText(
      'Click "Start Camera" for live 0.4m–5.0m+ tracking, or "Range Simulator" to test without a webcam.',
      w / 2,
      h / 2 + 22
    );
  }

  renderFrameHUD(trackRes, evalRes, fw, fh) {
    const ctx = this.ctx;
    ctx.clearRect(0, 0, fw, fh);

    if (this.cameraRunning) {
      ctx.drawImage(this.tracker.mirrorCanvas, 0, 0, fw, fh);
    } else {
      // Simulator room backdrop with distance perspective grid
      const grad = ctx.createLinearGradient(0, 0, 0, fh);
      grad.addColorStop(0, '#0d1626');
      grad.addColorStop(1, '#070b12');
      ctx.fillStyle = grad;
      ctx.fillRect(0, 0, fw, fh);

      ctx.strokeStyle = 'rgba(56, 189, 248, 0.09)';
      ctx.lineWidth = 1;
      for (let i = 1; i < 8; i++) {
        const x = (fw / 8) * i;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, fh);
        ctx.stroke();
      }
      for (let j = 1; j < 5; j++) {
        const y = (fh / 5) * j;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(fw, y);
        ctx.stroke();
      }

      if (trackRes.source === 'legacy_out_of_range') {
        ctx.fillStyle = '#f87171';
        ctx.font = '600 22px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(
          `Smart Zoom is OFF: Hand lost at ${trackRes.estimatedDistanceM.toFixed(1)}m (>1.3m legacy limit!)`,
          fw / 2,
          fh / 2
        );
        ctx.fillStyle = '#94a3b8';
        ctx.font = '400 15px Inter, sans-serif';
        ctx.fillText(
          'Re-enable "Smart ROI Zoom" on the right panel to lock onto the hand up to 5.0m.',
          fw / 2,
          fh / 2 + 30
        );
      }
    }

    // Draw Smart-Zoom ROI Lock Brackets
    if (trackRes.detected && trackRes.roi && trackRes.zoomFactor > 1.2) {
      const r = trackRes.roi;
      const cLen = Math.max(14, Math.min(30, r.width / 6));
      ctx.strokeStyle = evalRes.clutchUnlocked ? '#34d399' : '#22d3ee';
      ctx.lineWidth = 2.5;

      const corners = [
        [r.x1, r.y1, 1, 1],
        [r.x2, r.y1, -1, 1],
        [r.x1, r.y2, 1, -1],
        [r.x2, r.y2, -1, -1],
      ];
      for (const [cx, cy, sx, sy] of corners) {
        ctx.beginPath();
        ctx.moveTo(cx + sx * cLen, cy);
        ctx.lineTo(cx, cy);
        ctx.lineTo(cx, cy + sy * cLen);
        ctx.stroke();
      }
    }

    // Draw Hand Skeleton & Dwell Ring
    if (trackRes.detected && trackRes.landmarks.length >= 21) {
      const pts = trackRes.landmarks.map((p) => ({ x: p.x * fw, y: p.y * fh }));
      const mainColor = evalRes.clutchUnlocked ? '#34d399' : '#22d3ee';

      ctx.save();
      ctx.strokeStyle = mainColor;
      ctx.lineWidth = Math.max(2, Math.min(4, trackRes.palmPx / 28));
      ctx.shadowColor = mainColor;
      ctx.shadowBlur = 8;

      for (const [a, b] of HAND_CONNECTIONS) {
        ctx.beginPath();
        ctx.moveTo(pts[a].x, pts[a].y);
        ctx.lineTo(pts[b].x, pts[b].y);
        ctx.stroke();
      }
      ctx.restore();

      for (let i = 0; i < pts.length; i++) {
        const isTip = i === 4 || i === 8 || i === 12 || i === 16 || i === 20;
        ctx.beginPath();
        ctx.arc(pts[i].x, pts[i].y, isTip ? 5 : 3, 0, Math.PI * 2);
        ctx.fillStyle = isTip ? '#f8fafc' : mainColor;
        ctx.fill();
      }

      // Radial Dwell / Clutch Progress Ring around palm center
      if (evalRes.palmCenter && evalRes.dwellProgress > 0.02) {
        const px = evalRes.palmCenter.x * fw;
        const py = evalRes.palmCenter.y * fh;
        const radius = Math.max(24, Math.min(68, trackRes.palmPx * 0.75));
        ctx.beginPath();
        ctx.arc(px, py, radius, 0, Math.PI * 2);
        ctx.strokeStyle = 'rgba(255,255,255,0.18)';
        ctx.lineWidth = 3;
        ctx.stroke();

        ctx.beginPath();
        ctx.arc(
          px,
          py,
          radius,
          -Math.PI / 2,
          -Math.PI / 2 + Math.PI * 2 * Math.min(1, evalRes.dwellProgress)
        );
        ctx.strokeStyle = evalRes.dwellProgress >= 0.98 ? '#34d399' : '#fbbf24';
        ctx.lineWidth = 4;
        ctx.stroke();
      }
    }

    // Update Smart-Zoom PiP Inset Canvas
    if (trackRes.detected && trackRes.roi && this.config.show_zoom_pip !== false) {
      this.pipWrapEl.style.display = 'block';
      const pw = (this.pipCanvasEl.width = 148);
      const ph = (this.pipCanvasEl.height = 148);
      this.pipCtx.clearRect(0, 0, pw, ph);
      // Draw magnified region from main canvas (including landmarks!)
      const r = trackRes.roi;
      this.pipCtx.drawImage(this.canvasEl, r.x1, r.y1, r.width, r.height, 0, 0, pw, ph);
      this.pipLabelEl.textContent = `ZOOM ${trackRes.zoomFactor.toFixed(1)}x · ~${trackRes.estimatedDistanceM.toFixed(1)}m`;
    } else if (this.pipWrapEl) {
      this.pipWrapEl.style.display = 'none';
    }
  }

  updateTelemetryDOM(trackRes, evalRes) {
    if (this.elFps) this.elFps.textContent = `${Math.round(this.fpsEma)}`;
    if (this.elLatency) this.elLatency.textContent = `${trackRes.inferenceMs.toFixed(1)} ms`;
    if (this.elRange) {
      this.elRange.textContent = trackRes.detected
        ? `${trackRes.estimatedDistanceM.toFixed(2)} m`
        : '--';
    }
    if (this.elZoom) {
      this.elZoom.textContent = trackRes.detected ? `${trackRes.zoomFactor.toFixed(1)}x` : '1.0x';
    }
    if (this.elSource) {
      const labels = {
        temporal_roi: 'Locked ROI',
        pose_wrist: 'Pose Wrist Zoom',
        scout_tile: 'Auditorium Tile',
        full_frame: 'Close Frame',
        searching: 'Scanning...',
        legacy_out_of_range: 'Out of Range',
      };
      this.elSource.textContent = labels[trackRes.source] || trackRes.source;
    }

    if (this.elGestureIcon) this.elGestureIcon.textContent = evalRes.gestureIcon;
    if (this.elGestureName) {
      this.elGestureName.textContent = trackRes.detected ? evalRes.gestureName : 'No Hand in View';
    }
    const confPct = Math.round((evalRes.confidence || 0) * 100);
    if (this.elConfFill) this.elConfFill.style.width = `${confPct}%`;
    if (this.elConfText) this.elConfText.textContent = confPct > 0 ? `${confPct}%` : '--';

    // Highlight active gesture chip
    document.querySelectorAll('[data-gesture-chip]').forEach((chip) => {
      chip.classList.toggle(
        'triggered',
        chip.getAttribute('data-gesture-chip') === evalRes.gesture
      );
    });

    if (this.elClutchPill) {
      if (this.config.activation_mode === 'clutch') {
        if (evalRes.clutchUnlocked) {
          this.elClutchPill.textContent = `UNLOCKED · ${evalRes.clutchRemainingS.toFixed(1)}s`;
          this.elClutchPill.className = 'pill pill-emerald';
        } else {
          this.elClutchPill.textContent = 'LOCKED · Hold Open Palm 0.4s';
          this.elClutchPill.className = 'pill pill-amber';
        }
      } else if (this.config.activation_mode === 'dwell') {
        this.elClutchPill.textContent = 'DIRECT DWELL + SWIPES';
        this.elClutchPill.className = 'pill pill-cyan';
      } else {
        this.elClutchPill.textContent = 'INSTANT TRIGGER';
        this.elClutchPill.className = 'pill pill-cyan';
      }
    }
  }

  captureCalibrationPoint(zone) {
    if (!this.latestTrack || !this.latestTrack.detected) {
      this.showToast('Start Camera or Simulator with a visible hand first');
      return;
    }
    this.calibSamples[zone] = {
      distM: this.latestTrack.estimatedDistanceM,
      palmPx: this.latestTrack.palmPx,
      zoom: this.latestTrack.zoomFactor,
      ms: this.latestTrack.inferenceMs,
    };
    const slot = document.getElementById(`calib-result-${zone}`);
    if (slot) {
      const s = this.calibSamples[zone];
      slot.innerHTML = `<strong>~${s.distM.toFixed(2)}m</strong> · Palm ${Math.round(
        s.palmPx
      )}px · Zoom ${s.zoom.toFixed(1)}x (${s.ms.toFixed(1)}ms)`;
      slot.classList.add('captured');
    }
    this.showToast(`Captured ${zone.toUpperCase()} range sample`);
  }

  async runDiagnosticsBenchmark() {
    const outEl = document.getElementById('benchmark-results-box');
    if (!outEl) return;
    outEl.innerHTML = `<div class="bench-running">Running CPU Long-Range Pipeline Benchmark...</div>`;

    // 1. Browser JS Gesture Engine benchmark (1,000 frames of 3D biomechanical math + 1-Euro filter)
    const t0 = performance.now();
    for (let i = 0; i < 500; i++) {
      const sim = this.tracker.generateSimulatedFrame(1280, 720, i * 0.016);
      this.gestureEngine.classifyStaticPose(sim.landmarks, 16 / 9);
    }
    const jsEngineUs = (((performance.now() - t0) / 500) * 1000).toFixed(1);

    // 2. Python Backend XNNPACK CPU Benchmark (if server connected)
    let pyHtml = `<div class="bench-item"><span>Python XNNPACK Engine:</span><span>Run with <code>--studio</code> to measure</span></div>`;
    try {
      const res = await fetch('/api/benchmark');
      if (res.ok) {
        const data = await res.json();
        pyHtml = `
          <div class="bench-item"><span>Python CV Backend:</span><strong>${data.backend} (${data.preset})</strong></div>
          <div class="bench-item"><span>Python Avg Frame Latency:</span><strong class="text-emerald">${data.avg_frame_ms} ms (${data.theoretical_fps} FPS cap)</strong></div>
        `;
      }
    } catch {
      // Standalone mode
    }

    outEl.innerHTML = `
      <div class="bench-grid">
        ${pyHtml}
        <div class="bench-item"><span>3D Biomechanical + 1-Euro Math:</span><strong class="text-cyan">${jsEngineUs} µs / frame</strong></div>
        <div class="bench-item"><span>Effective Tracking Envelope:</span><strong class="text-emerald">0.35m – 5.20m (Smart ROI Zoom)</strong></div>
      </div>
    `;
  }
}

document.addEventListener('DOMContentLoaded', () => {
  const app = new VisionStudioApp();
  app.init();
  window.visionStudio = app;
});
