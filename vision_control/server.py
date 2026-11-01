"""
FastAPI + WebSocket Studio Server for Vision Control v2.

Provides:
  - Modern Web Control Studio & Interactive Presentation/Media Stage
  - Real-time WebSocket OS Bridge (/ws): browser or LAN phone camera gestures can
    drive desktop PowerPoint, Keynote, system volume, media keys, and Air Mouse!
  - Live profile & custom keybinding persistence (/api/config)
  - Built-in CPU inference benchmark endpoint (/api/benchmark)
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .actions import OSActionController
from .config import DEFAULT_PROFILES, SystemConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = REPO_ROOT / "web"
RESOURCES_DIR = REPO_ROOT / "resources"
CACHE_MODELS_DIR = REPO_ROOT / ".cache" / "models"
CACHE_VENDOR_DIR = REPO_ROOT / ".cache" / "vendor" / "mediapipe"


class StudioHub:
    """Manages shared state, OS action execution, and connected WebSocket clients."""

    def __init__(self, config: Optional[SystemConfig] = None) -> None:
        self.config = config or SystemConfig.load()
        self.actions = OSActionController(self.config)
        self.clients: Set[WebSocket] = set()
        self.started_at = time.time()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.clients.add(ws)
        await ws.send_json(
            {
                "type": "hello",
                "version": __version__,
                "os_name": self.actions.os_name,
                "os_gui_available": self.actions.gui_available,
                "config": self.config.to_dict(),
                "history": self.actions.recent_history(12),
            }
        )

    def disconnect(self, ws: WebSocket) -> None:
        self.clients.discard(ws)

    async def broadcast(self, payload: Dict[str, Any]) -> None:
        dead: List[WebSocket] = []
        for client in list(self.clients):
            try:
                await client.send_json(payload)
            except Exception:
                dead.append(client)
        for d in dead:
            self.clients.discard(d)


def create_app(config: Optional[SystemConfig] = None) -> FastAPI:
    hub = StudioHub(config=config)

    app = FastAPI(
        title="Vision Gesture Control Studio",
        version=__version__,
         description="Long-Range CPU-Optimized Hand Gesture Control for PowerPoint, Media & OS",
    )
    app.state.hub = hub

    # Permissive CORS & zero frame-ancestors block for seamless browser preview embedding
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if WEB_DIR.exists():
        app.mount("/web", StaticFiles(directory=str(WEB_DIR)), name="web")
    if RESOURCES_DIR.exists():
        app.mount("/resources", StaticFiles(directory=str(RESOURCES_DIR)), name="resources")
    if CACHE_VENDOR_DIR.exists():
        app.mount(
            "/vendor/mediapipe",
            StaticFiles(directory=str(CACHE_VENDOR_DIR)),
            name="vendor_mediapipe",
        )

    @app.get("/api/status")
    async def api_status() -> Dict[str, Any]:
        return {
            "ok": True,
            "version": __version__,
            "uptime_s": round(time.time() - hub.started_at, 1),
            "os_name": hub.actions.os_name,
            "os_gui_available": hub.actions.gui_available,
            "connected_clients": len(hub.clients),
            "local_models": {
                "hand_landmarker": (CACHE_MODELS_DIR / "hand_landmarker.task").exists(),
                "pose_landmarker_lite": (
                    CACHE_MODELS_DIR / "pose_landmarker_lite.task"
                ).exists(),
                "local_wasm": (CACHE_VENDOR_DIR / "vision_bundle.mjs").exists(),
            },
            "active_profile": hub.config.active_profile,
            "performance_preset": hub.config.performance_preset,
            "activation_mode": hub.config.activation_mode,
            "history": hub.actions.recent_history(12),
        }

    @app.get("/api/config")
    async def get_config() -> Dict[str, Any]:
        return hub.config.to_dict()

    @app.post("/api/config")
    async def update_config(request: Request) -> Dict[str, Any]:
        body = await request.json()
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="Expected JSON object")
        hub.config.update_from_dict(body)
        hub.config.save()
        cfg_dict = hub.config.to_dict()
        await hub.broadcast({"type": "config_updated", "config": cfg_dict})
        return cfg_dict

    @app.post("/api/config/reset")
    async def reset_config() -> Dict[str, Any]:
        import copy

        fresh = SystemConfig()
        fresh.profiles = copy.deepcopy(DEFAULT_PROFILES)
        hub.config = fresh
        hub.actions.config = fresh
        hub.config.save()
        cfg_dict = hub.config.to_dict()
        await hub.broadcast({"type": "config_updated", "config": cfg_dict})
        return cfg_dict

    @app.post("/api/action")
    async def trigger_action(request: Request) -> Dict[str, Any]:
        body = await request.json()
        action = str(body.get("action", ""))
        gesture = str(body.get("gesture", "manual"))
        evt = hub.actions.execute(action, gesture=gesture)
        if evt is None:
            return {"ok": False, "executed": None}
        await hub.broadcast(
            {
                "type": "action_executed",
                "event": evt,
                "active_profile": hub.config.active_profile,
            }
        )
        return {"ok": True, "event": evt, "active_profile": hub.config.active_profile}

    @app.get("/api/models/{model_name}")
    async def get_cached_model(model_name: str) -> FileResponse:
        allowed = {"hand_landmarker.task", "pose_landmarker_lite.task"}
        if model_name not in allowed:
            raise HTTPException(status_code=404, detail="Unknown model")
        target = CACHE_MODELS_DIR / model_name
        if not target.exists():
            raise HTTPException(status_code=404, detail="Model not cached locally")
        return FileResponse(
            str(target),
            media_type="application/octet-stream",
            headers={"Cache-Control": "public, max-age=86400"},
        )

    @app.get("/api/benchmark")
    async def run_cpu_benchmark() -> Dict[str, Any]:
        """Run a quick real CPU micro-benchmark on LongRangeHandTracker (256x256 ROI crop)."""

        def _bench() -> Dict[str, Any]:
            from .tracker import LongRangeHandTracker

            tracker = LongRangeHandTracker(config=hub.config)
            try:
                dummy_720p = np.zeros((720, 1280, 3), dtype=np.uint8)
                # Warmup
                for _ in range(3):
                    tracker.process_frame(dummy_720p)
                iters = 12
                t0 = time.perf_counter()
                for _ in range(iters):
                    tracker.process_frame(dummy_720p)
                avg_ms = (time.perf_counter() - t0) / iters * 1000.0
                return {
                    "ok": True,
                    "backend": tracker._backend_type,
                    "avg_frame_ms": round(avg_ms, 2),
                    "theoretical_fps": round(1000.0 / max(avg_ms, 0.1), 1),
                    "preset": hub.config.performance_preset,
                }
            finally:
                tracker.close()

        return await asyncio.to_thread(_bench)

    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket) -> None:
        await hub.connect(ws)
        try:
            while True:
                msg = await ws.receive_json()
                mtype = msg.get("type")

                if mtype == "gesture_trigger":
                    action = str(msg.get("action", ""))
                    gesture = str(msg.get("gesture", "browser"))
                    evt = hub.actions.execute(action, gesture=gesture)
                    if evt is not None:
                        await hub.broadcast(
                            {
                                "type": "action_executed",
                                "event": evt,
                                "active_profile": hub.config.active_profile,
                            }
                        )

                elif mtype == "cursor_move":
                    cx = msg.get("x")
                    cy = msg.get("y")
                    pinching = bool(msg.get("pinching", False))
                    if isinstance(cx, (int, float)) and isinstance(cy, (int, float)):
                        hub.actions.update_cursor((float(cx), float(cy)), pinching)

                elif mtype == "update_config":
                    updates = msg.get("config", {})
                    if isinstance(updates, dict):
                        hub.config.update_from_dict(updates)
                        hub.config.save()
                        await hub.broadcast(
                            {"type": "config_updated", "config": hub.config.to_dict()}
                        )

                elif mtype == "ping":
                    await ws.send_json({"type": "pong", "ts": msg.get("ts")})

        except WebSocketDisconnect:
            hub.disconnect(ws)
        except Exception:
            hub.disconnect(ws)

    # Serve HTML entry points cleanly
    @app.get("/")
    @app.get("/index.html")
    async def serve_index() -> FileResponse:
        return FileResponse(str(REPO_ROOT / "index.html"))

    @app.get("/{page_name}.html")
    async def serve_html_pages(page_name: str) -> FileResponse:
        allowed_pages = {"presentation", "training", "about"}
        if page_name in allowed_pages and (REPO_ROOT / f"{page_name}.html").exists():
            return FileResponse(str(REPO_ROOT / f"{page_name}.html"))
        return FileResponse(str(REPO_ROOT / "index.html"))

    @app.get("/styles.css")
    async def serve_root_css() -> FileResponse:
        return FileResponse(str(WEB_DIR / "css" / "studio.css"))

    return app
