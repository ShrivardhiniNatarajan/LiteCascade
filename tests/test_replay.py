"""Unit tests for the traffic replay engine (Stage 16)."""

from __future__ import annotations

import json
from pathlib import Path

from litecascade.deploy.replay import ReplayEngine


def test_replay_engine_determinism():
    """Verify that two runs with the same seed yield identical event sequences."""
    e1 = ReplayEngine(dataset_name="fixture", seed=42)
    e2 = ReplayEngine(dataset_name="fixture", seed=42)

    evs1 = [e1.step() for _ in range(10)]
    evs2 = [e2.step() for _ in range(10)]

    for ev1, ev2 in zip(evs1, evs2):
        assert ev1["prediction"] == ev2["prediction"]
        assert ev1["true_label"] == ev2["true_label"]
        assert ev1["exit_used"] == ev2["exit_used"]
        assert ev1["confidence"] == ev2["confidence"]


def test_replay_exit_rate_consistency():
    """Verify that summary exit_rate matches empirical event exit counts."""
    engine = ReplayEngine(dataset_name="fixture", seed=42)
    events = [engine.step() for _ in range(50)]

    summary = engine.get_summary()
    assert summary["total_windows"] == 50

    sentinel_count = sum(1 for e in events if e["exit_used"] == "sentinel")
    analyst_count = sum(1 for e in events if e["exit_used"] == "analyst")

    assert summary["sentinel_exits"] == sentinel_count
    assert summary["analyst_exits"] == analyst_count
    expected_exit_rate = sentinel_count / 50.0
    assert abs(summary["exit_rate"] - expected_exit_rate) < 1e-3


def test_attack_injection():
    """Verify that injected attack windows strictly reflect the specified family."""
    engine = ReplayEngine(dataset_name="fixture", seed=42)
    target_family = "Mirai_Injected_Test"
    engine.inject_attack(family=target_family, start_index=5, duration=5)

    events = [engine.step() for _ in range(15)]

    for i in range(5, 10):
        ev = events[i]
        assert ev["family"] == target_family
        assert ev["true_label"] == 1


def test_replay_summary_schema(tmp_path: Path):
    """Verify summary dictionary fields and types."""
    engine = ReplayEngine(dataset_name="fixture", seed=42)
    for _ in range(10):
        engine.step()

    summary = engine.get_summary()
    assert "total_windows" in summary
    assert "exit_rate" in summary
    assert "accuracy" in summary
    assert "avg_latency_ms" in summary
    assert "speedup_vs_analyst" in summary
    assert "is_synthetic" in summary

    out_file = tmp_path / "summary.json"
    out_file.write_text(json.dumps(summary), encoding="utf-8")
    assert out_file.exists()
