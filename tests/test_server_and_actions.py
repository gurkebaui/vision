"""
Integration tests for FastAPI Studio Server, WebSocket OS Bridge, and OSActionController.
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from vision_control.config import SystemConfig
from vision_control.server import create_app


def test_rest_endpoints_and_profile_persistence(tmp_path: Path, monkeypatch):
    cfg_file = tmp_path / "settings.json"
    monkeypatch.setattr("vision_control.config.DEFAULT_CONFIG_PATH", cfg_file)

    cfg = SystemConfig(os_control_enabled=False)
    app = create_app(config=cfg)
    client = TestClient(app)

    # 1. Index HTML & Studio CSS
    r_idx = client.get("/")
    assert r_idx.status_code == 200
    assert "Vision Control" in r_idx.text

    r_css = client.get("/web/css/studio.css")
    assert r_css.status_code == 200

    # 2. Status & Config
    r_status = client.get("/api/status")
    assert r_status.status_code == 200
    assert r_status.json()["ok"] is True

    # 3. Update mapping & preset
    r_upd = client.post(
        "/api/config",
        json={
            "performance_preset": "ultra_light",
            "active_profile": "media",
            "mapping_update": {
                "profile": "custom",
                "gesture": "peace_sign",
                "action": "black_screen",
            },
        },
    )
    assert r_upd.status_code == 200
    body = r_upd.json()
    assert body["performance_preset"] == "ultra_light"
    assert body["active_profile"] == "media"
    assert body["profiles"]["custom"]["mappings"]["peace_sign"] == "black_screen"
    assert cfg_file.exists()

    # 4. Trigger action via REST
    r_act = client.post("/api/action", json={"action": "next_slide", "gesture": "swipe_right"})
    assert r_act.status_code == 200
    assert r_act.json()["event"]["action"] == "next_slide"

    # 5. CPU Benchmark endpoint
    r_bench = client.get("/api/benchmark")
    assert r_bench.status_code == 200
    assert r_bench.json()["ok"] is True
    assert r_bench.json()["avg_frame_ms"] > 0


def test_websocket_bidirectional_bridge(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "vision_control.config.DEFAULT_CONFIG_PATH", tmp_path / "ws_settings.json"
    )
    cfg = SystemConfig(os_control_enabled=False, active_profile="presentation")
    app = create_app(config=cfg)
    client = TestClient(app)

    with client.websocket_connect("/ws") as ws:
        hello = ws.receive_json()
        assert hello["type"] == "hello"
        assert hello["config"]["active_profile"] == "presentation"

        # Trigger 3-finger profile switch over WebSocket
        ws.send_json(
            {"type": "gesture_trigger", "action": "cycle_profile", "gesture": "three_fingers"}
        )
        evt_msg = ws.receive_json()
        assert evt_msg["type"] == "action_executed"
        assert evt_msg["active_profile"] == "media"
