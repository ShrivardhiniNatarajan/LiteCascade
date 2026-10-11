"""Unit tests for the LiteCascade FastAPI backend (Stage 16C)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api.main import app, state


@pytest.fixture(autouse=True)
def reset_engine_state():
    """Resets stream engine state before each test."""
    state.is_running = False
    state.speed = 10.0
    state.engine.reset()
    yield
    state.is_running = False
    state.engine.reset()


@pytest.fixture
def client():
    return TestClient(app)


def test_meta_endpoint_schema(client: TestClient):
    """Verify /api/meta returns expected schema, valid types, and provenance sources."""
    res = client.get("/api/meta")
    assert res.status_code == 200
    data = res.json()

    assert "data_source" in data
    assert "source" in data["data_source"]
    assert "is_synthetic" in data
    assert "model_format" in data
    assert "gate_loaded" in data
    assert "tau_b" in data
    assert "tau_m" in data
    assert "epsilon" in data
    assert "params" in data
    assert "macs" in data
    assert "available_families" in data

    # Verify model format is explicit string
    assert data["model_format"]["value"] in ["TFLite INT8", "PyTorch FP32"]
    assert isinstance(data["gate_loaded"]["value"], bool)


def test_stream_controls(client: TestClient):
    """Verify stream control actions: start, pause, step, reset."""
    # Step 1 window
    res_step = client.post("/api/stream/step")
    assert res_step.status_code == 200
    step_data = res_step.json()
    assert step_data["total_windows"] == 1
    assert step_data["is_running"] is False

    # Start stream
    res_start = client.post("/api/stream/start", json={"speed": 50})
    assert res_start.status_code == 200
    assert res_start.json()["is_running"] is True
    assert res_start.json()["speed"] == 50

    # Pause stream
    res_pause = client.post("/api/stream/pause")
    assert res_pause.status_code == 200
    assert res_pause.json()["is_running"] is False

    # Reset stream
    res_reset = client.post("/api/stream/reset")
    assert res_reset.status_code == 200
    assert res_reset.json()["total_windows"] == 0


def test_attack_injection_only_requested_family(client: TestClient):
    """Verify inject endpoint only returns requested family and handles errors cleanly."""
    # Invalid family -> 400 with descriptive error
    res_bad = client.post("/api/inject", json={"family": "NonExistentUnknownMalware", "duration_windows": 10})
    assert res_bad.status_code == 400
    assert "not available" in res_bad.json()["detail"]

    # Valid family injection
    res_ok = client.post("/api/inject", json={"family": "Mirai", "duration_windows": 10})
    assert res_ok.status_code == 200
    assert res_ok.json()["status"] == "scheduled"
    assert res_ok.json()["family"].lower() == "mirai"

    # Step through injected windows and verify family strictly matches
    injected_events = []
    for _ in range(10):
        client.post("/api/stream/step")
    recent = state.engine.get_recent_events(limit=10)
    for ev in recent:
        assert "mirai" in ev["family"].lower()
        assert ev["is_injected"] is True
        assert ev["true_label"] == 1


def test_stats_consistency_and_wilson_ci(client: TestClient):
    """Verify stats computation: exit_rate == mean(exit flags), Wilson CI when n>=50."""
    # Initial empty stats
    res_init = client.get("/api/stats")
    assert res_init.status_code == 200
    d_init = res_init.json()
    assert d_init["n"]["value"] == 0
    assert d_init["accuracy"]["value"] is None  # - until n >= 50

    # Step 60 windows
    for _ in range(60):
        client.post("/api/stream/step")

    res_stats = client.get("/api/stats")
    assert res_stats.status_code == 200
    d = res_stats.json()
    assert d["n"]["value"] == 60

    # Exit rate matches empirical count
    sentinel_count = sum(1 for e in state.engine.events if e["exit_used"] == "sentinel")
    expected_exit_rate = round(sentinel_count / 60.0, 4)
    assert abs(d["exit_rate"]["value"] - expected_exit_rate) < 1e-3

    # Accuracy has 95% Wilson CI populated
    assert d["accuracy"]["value"] is not None
    assert d["accuracy_ci_low"]["value"] is not None
    assert d["accuracy_ci_high"]["value"] is not None
    assert d["accuracy_ci_low"]["value"] <= d["accuracy"]["value"] <= d["accuracy_ci_high"]["value"]


def test_device_endpoint(client: TestClient):
    """Verify /api/device returns parsed JSON files or empty dict gracefully."""
    res = client.get("/api/device")
    assert res.status_code == 200
    data = res.json()
    assert isinstance(data, dict)
    if "rpi_benchmarks" in data:
        assert "measured_or_emulated" in data["rpi_benchmarks"]


def test_gate_loaded_false_is_surfaced(client: TestClient):
    """Verify that when no gate file exists, gate_loaded=False is surfaced visibly."""
    orig_loaded = state.engine.gate_loaded
    state.engine.gate_loaded = False
    try:
        res = client.get("/api/meta")
        assert res.json()["gate_loaded"]["value"] is False
    finally:
        state.engine.gate_loaded = orig_loaded
