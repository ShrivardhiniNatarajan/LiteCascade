"""Unit tests for Stage 12: On-device benchmarking and MCU estimation scripts."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from scripts.device.bench_rpi import run_rpi_benchmark
from scripts.device.power_log import (
    compute_energy_metrics,
    parse_usb_power_meter_csv,
    run_mock_power_logging,
)
from scripts.estimate_mcu import estimate_mcu_resources

MODEL_PATH = Path("checkpoints/litecascade_int8.tflite")


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="checkpoints/litecascade_int8.tflite not found")
def test_estimate_mcu_schema_and_values():
    """Validates that MCU estimation outputs required keys with 'estimated' status."""
    report = estimate_mcu_resources(MODEL_PATH, target="esp32-s3")

    # Required Stage 12 fields
    assert report["measured_or_emulated"] == "estimated", "Must be strictly 'estimated' per Rule 6"
    assert "platform" in report
    assert "os" in report
    assert "cpu" in report
    assert "clock" in report
    assert report["n_runs"] == 0

    # Memory metrics
    assert report["flash_size_bytes"] > 0
    assert report["flash_size_kb"] > 0
    assert report["arena_size_bytes"] > 0
    assert report["arena_size_kb"] > 0

    # Timing estimates
    assert report["estimated_cycles"]["sentinel"] > 0
    assert report["estimated_cycles"]["cascade"] > 0
    assert report["estimated_latency_ms"]["sentinel_ms"] > 0


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="checkpoints/litecascade_int8.tflite not found")
def test_estimate_mcu_target_profiles():
    """Validates MCU estimation across different hardware targets."""
    for target in ["cortex-m4", "cortex-m7", "esp32-s3"]:
        rep = estimate_mcu_resources(MODEL_PATH, target=target)
        assert rep["target_profile"] == target
        assert rep["measured_or_emulated"] == "estimated"
        assert rep["flash_usage_pct"] < 100.0  # Fits in flash


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="checkpoints/litecascade_int8.tflite not found")
def test_bench_rpi_dry_run_schema():
    """Validates that bench_rpi in dry-run mode completes and returns valid metrics."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_json = Path(tmp_dir) / "rpi_test.json"
        report = run_rpi_benchmark(
            model_path=MODEL_PATH,
            n_runs=10,
            n_warmup=2,
            rho=0.88,
            dry_run=True,
            output_path=out_json,
        )

        assert report["measured_or_emulated"] == "emulated", "Dry run must mark 'emulated'"
        assert report["n_runs"] == 10
        assert "with_xnnpack" in report["results"]
        assert "without_xnnpack" in report["results"]

        for cfg in ["with_xnnpack", "without_xnnpack"]:
            data = report["results"][cfg]
            assert "sentinel_only" in data
            assert "analyst_only" in data
            assert "cascade" in data

            for path in ["sentinel_only", "analyst_only", "cascade"]:
                stats = data[path]
                assert stats["p50_ms"] > 0
                assert stats["p95_ms"] > 0
                assert stats["peak_rss_mb"] > 0
                assert "cpu_percent" in stats

        assert out_json.exists()


def test_compute_energy_metrics():
    """Validates physical formula E = (P_active - P_idle) * t."""
    # 4.2W active, 2.7W idle -> delta = 1.5W
    # t = 0.1 ms = 0.0001 s
    # E = 1.5 * 0.0001 = 0.00015 J = 0.15 mJ = 150 uJ
    metrics = compute_energy_metrics(p_idle=2.7, p_active=4.2, latency_ms=0.1)

    assert abs(metrics["delta_power_w"] - 1.5) < 1e-4
    assert abs(metrics["energy_per_inference_mj"] - 0.15) < 1e-4
    assert abs(metrics["energy_per_inference_uj"] - 150.0) < 1e-2


def test_power_log_mock_mode():
    """Validates mock power logging logic."""
    res = run_mock_power_logging(idle_sec=0.1, active_sec=0.1, latency_ms=0.08)
    assert res["idle_power_w"] > 0
    assert res["active_power_w"] > res["idle_power_w"]
    assert res["delta_power_w"] > 0
    assert res["energy_per_inference_uj"] > 0


def test_power_log_csv_parser():
    """Validates parser on a simulated USB power meter CSV."""
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        f.write("timestamp,voltage,current,phase\n")
        # 10 idle points (5V, 0.54A = 2.7W)
        for t in range(10):
            f.write(f"{t * 0.1},5.0,0.54,idle\n")
        # 10 active points (5V, 0.83A = 4.15W)
        for t in range(10, 20):
            f.write(f"{t * 0.1},5.0,0.83,active\n")
        tmp_csv = Path(f.name)

    try:
        p_idle, p_active, p_delta, duration = parse_usb_power_meter_csv(tmp_csv)
        assert abs(p_idle - 2.7) < 0.1
        assert abs(p_active - 4.15) < 0.1
        assert abs(p_delta - 1.45) < 0.1
        assert duration > 0
    finally:
        if tmp_csv.exists():
            tmp_csv.unlink()
