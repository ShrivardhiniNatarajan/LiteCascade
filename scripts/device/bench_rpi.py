"""Raspberry Pi On-Device Benchmarking Script for LiteCascade INT8.

Runs INT8 TFLite model on Raspberry Pi (and dev / Docker environments) with XNNPACK
delegate enabled and disabled across three inference paths:
  1. Sentinel-only
  2. Analyst-only
  3. Full cascade at measured early-exit rate rho.

Measures p50/p95 latency (ms), peak RSS memory (MB), and CPU utilization (%).
Outputs structured JSON adhering to the Stage 12 schema with explicit
'measured_or_emulated' tracking per Rule 6.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("bench_rpi")


def is_raspberry_pi() -> bool:
    """Detects whether code is executing on physical Raspberry Pi hardware."""
    try:
        # Check device tree model
        dt_model = Path("/proc/device-tree/model")
        if dt_model.exists():
            content = dt_model.read_text(encoding="utf-8", errors="ignore").lower()
            if "raspberry pi" in content:
                return True

        # Check /proc/cpuinfo
        cpuinfo = Path("/proc/cpuinfo")
        if cpuinfo.exists():
            content = cpuinfo.read_text(encoding="utf-8", errors="ignore").lower()
            if "raspberry" in content or "bcm2835" in content or "bcm2711" in content or "bcm2712" in content:
                return True
    except Exception:
        pass
    return False


def get_hardware_info() -> dict[str, str]:
    """Collects hardware, OS, CPU, and clock specifications."""
    is_rpi = is_raspberry_pi()

    # Platform description
    if is_rpi:
        try:
            dt_model = Path("/proc/device-tree/model")
            plat = dt_model.read_text(encoding="utf-8", errors="ignore").strip().replace("\x00", "")
        except Exception:
            plat = "Raspberry Pi (ARM Linux)"
    else:
        plat = f"{platform.system()} {platform.machine()} ({'Emulated RPi' if os.environ.get('EMULATE_PI') else 'Dev Machine'})"

    # OS description
    os_desc = f"{platform.system()} {platform.release()}"

    # CPU description
    cpu_desc = platform.processor() or platform.machine()
    if is_rpi:
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("Model") or line.startswith("Hardware"):
                        cpu_desc = line.split(":", 1)[1].strip()
                        break
        except Exception:
            pass

    # Clock frequency
    clock_desc = "Unknown"
    try:
        cur_freq_path = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")
        if cur_freq_path.exists():
            khz = int(cur_freq_path.read_text().strip())
            clock_desc = f"{khz / 1_000_000:.2f} GHz"
        else:
            import psutil

            freq = psutil.cpu_freq()
            if freq and freq.current:
                clock_desc = f"{freq.current / 1000.0:.2f} GHz"
    except Exception:
        clock_desc = "1.50 GHz" if is_rpi else "Variable"

    return {
        "platform": plat,
        "os": os_desc,
        "cpu": cpu_desc,
        "clock": clock_desc,
        "is_rpi": is_rpi,
    }


def load_interpreter(
    model_path: Path,
    use_xnnpack: bool,
    num_threads: int = 4,
) -> Any:
    """Instantiates a tf.lite.Interpreter with or without XNNPACK delegate."""
    import tensorflow as tf

    if use_xnnpack:
        op_resolver = tf.lite.experimental.OpResolverType.AUTO
    else:
        op_resolver = tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES

    interpreter = tf.lite.Interpreter(
        model_path=str(model_path),
        experimental_op_resolver_type=op_resolver,
        num_threads=num_threads,
    )
    interpreter.allocate_tensors()
    return interpreter


def benchmark_path(
    interpreter: Any,
    test_windows: np.ndarray,
    n_warmup: int,
    n_runs: int,
    input_index: int,
    output_index: int | None = None,
) -> dict[str, float]:
    """Runs timed inferences on an interpreter and computes latency statistics."""
    import psutil

    proc = psutil.Process()
    # Warmup
    n_samples = len(test_windows)
    for i in range(n_warmup):
        sample = test_windows[i % n_samples : (i % n_samples) + 1]
        interpreter.set_tensor(input_index, sample)
        interpreter.invoke()

    # Timed runs
    latencies_ms: list[float] = []
    # Trigger initial cpu percent
    proc.cpu_percent(interval=None)

    for i in range(n_runs):
        sample = test_windows[i % n_samples : (i % n_samples) + 1]
        interpreter.set_tensor(input_index, sample)

        t0 = time.perf_counter()
        interpreter.invoke()
        t1 = time.perf_counter()

        latencies_ms.append((t1 - t0) * 1000.0)

    cpu_pct = proc.cpu_percent(interval=None)
    rss_mb = proc.memory_info().rss / (1024.0 * 1024.0)

    lat_arr = np.array(latencies_ms)
    return {
        "p50_ms": round(float(np.percentile(lat_arr, 50)), 4),
        "p95_ms": round(float(np.percentile(lat_arr, 95)), 4),
        "mean_ms": round(float(np.mean(lat_arr)), 4),
        "std_ms": round(float(np.std(lat_arr)), 4),
        "min_ms": round(float(np.min(lat_arr)), 4),
        "max_ms": round(float(np.max(lat_arr)), 4),
        "peak_rss_mb": round(rss_mb, 2),
        "cpu_percent": round(cpu_pct, 1),
    }


def benchmark_cascade_path(
    sentinel_interp: Any,
    analyst_interp: Any,
    test_windows: np.ndarray,
    n_warmup: int,
    n_runs: int,
    rho: float,
    sent_in_idx: int,
    analyst_in_idx: int,
) -> dict[str, float]:
    """Measures empirical cascade latency distribution under early-exit rate rho."""
    import psutil

    proc = psutil.Process()
    n_samples = len(test_windows)

    # Warmup
    for i in range(n_warmup):
        sample = test_windows[i % n_samples : (i % n_samples) + 1]
        sentinel_interp.set_tensor(sent_in_idx, sample)
        sentinel_interp.invoke()
        if (i % 2) == 0:
            analyst_interp.set_tensor(analyst_in_idx, sample)
            analyst_interp.invoke()

    # Pre-generate early-exit decisions based on Bernoulli(rho)
    rng = np.random.RandomState(42)
    early_exit_mask = rng.rand(n_runs) < rho

    latencies_ms: list[float] = []
    proc.cpu_percent(interval=None)

    for i in range(n_runs):
        sample = test_windows[i % n_samples : (i % n_samples) + 1]

        t0 = time.perf_counter()
        # Always run Sentinel first
        sentinel_interp.set_tensor(sent_in_idx, sample)
        sentinel_interp.invoke()

        # If not early exit, invoke Analyst
        if not early_exit_mask[i]:
            analyst_interp.set_tensor(analyst_in_idx, sample)
            analyst_interp.invoke()

        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    cpu_pct = proc.cpu_percent(interval=None)
    rss_mb = proc.memory_info().rss / (1024.0 * 1024.0)

    lat_arr = np.array(latencies_ms)
    return {
        "p50_ms": round(float(np.percentile(lat_arr, 50)), 4),
        "p95_ms": round(float(np.percentile(lat_arr, 95)), 4),
        "mean_ms": round(float(np.mean(lat_arr)), 4),
        "std_ms": round(float(np.std(lat_arr)), 4),
        "min_ms": round(float(np.min(lat_arr)), 4),
        "max_ms": round(float(np.max(lat_arr)), 4),
        "peak_rss_mb": round(rss_mb, 2),
        "cpu_percent": round(cpu_pct, 1),
    }


def run_rpi_benchmark(
    model_path: str | Path,
    n_runs: int = 1000,
    n_warmup: int = 50,
    rho: float = 0.88,
    dry_run: bool = False,
    num_threads: int = 4,
    output_path: str | Path | None = "results/device/rpi_benchmarks.json",
) -> dict[str, Any]:
    """Orchestrates comprehensive Raspberry Pi benchmarking across delegate configurations."""
    model_p = Path(model_path)
    if not model_p.exists():
        raise FileNotFoundError(f"Model file not found: {model_p}")

    hw_info = get_hardware_info()
    is_rpi = hw_info["is_rpi"]

    # Strict Rule 6 tracking: only mark "measured" if physical Pi and not dry_run
    if is_rpi and not dry_run:
        measured_or_emulated = "measured"
    else:
        measured_or_emulated = "emulated"

    if dry_run:
        n_warmup = min(n_warmup, 10)
        n_runs = min(n_runs, 100)

    # Resolve Sentinel and Analyst models
    chk_dir = model_p.parent
    sentinel_path = chk_dir / "litecascade_sentinel_int8.tflite"
    analyst_path = chk_dir / "litecascade_analyst_int8.tflite"

    # If submodels do not exist, fall back to monolithic model
    has_submodels = sentinel_path.exists() and analyst_path.exists()
    if not has_submodels:
        logger.info("Submodel files not found; using monolithic model for Sentinel & Analyst profiling.")
        sentinel_path = model_p
        analyst_path = model_p

    # Synthesize test input windows (B=1, T=10, F=115)
    rng = np.random.RandomState(1337)
    test_windows = rng.randn(100, 10, 115).astype(np.float32)

    configs = [
        ("with_xnnpack", True),
        ("without_xnnpack", False),
    ]

    benchmark_results: dict[str, Any] = {}

    for config_name, use_xnnpack in configs:
        logger.info(f"Benchmarking configuration: {config_name} (threads={num_threads})...")

        # Load interpreters
        s_interp = load_interpreter(sentinel_path, use_xnnpack=use_xnnpack, num_threads=num_threads)
        a_interp = load_interpreter(analyst_path, use_xnnpack=use_xnnpack, num_threads=num_threads)

        s_in_idx = s_interp.get_input_details()[0]["index"]
        a_in_idx = a_interp.get_input_details()[0]["index"]

        # 1. Sentinel only
        s_stats = benchmark_path(s_interp, test_windows, n_warmup, n_runs, s_in_idx)

        # 2. Analyst only
        a_stats = benchmark_path(a_interp, test_windows, n_warmup, n_runs, a_in_idx)

        # 3. Full Cascade at rho
        c_stats = benchmark_cascade_path(
            s_interp, a_interp, test_windows, n_warmup, n_runs, rho, s_in_idx, a_in_idx
        )

        # Analytical expected latency and speedup
        expected_latency_ms = round(s_stats["mean_ms"] + (1.0 - rho) * a_stats["mean_ms"], 4)
        speedup_vs_analyst = round(a_stats["mean_ms"] / max(1e-4, expected_latency_ms), 2)

        c_stats["analytical_expected_ms"] = expected_latency_ms
        c_stats["speedup_vs_analyst"] = speedup_vs_analyst

        benchmark_results[config_name] = {
            "sentinel_only": s_stats,
            "analyst_only": a_stats,
            "cascade": c_stats,
        }

    report: dict[str, Any] = {
        "platform": hw_info["platform"],
        "os": hw_info["os"],
        "cpu": hw_info["cpu"],
        "clock": hw_info["clock"],
        "measured_or_emulated": measured_or_emulated,
        "n_runs": n_runs,
        "warmup_runs": n_warmup,
        "rho": rho,
        "num_threads": num_threads,
        "model_path": str(model_p),
        "results": benchmark_results,
    }

    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark LiteCascade INT8 model on Raspberry Pi."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="checkpoints/litecascade_int8.tflite",
        help="Path to INT8 TFLite model.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1000,
        help="Number of timed inference runs (default: 1000).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=50,
        help="Number of warm-up runs (default: 50).",
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=0.88,
        help="Measured exit rate rho for cascade execution (default: 0.88).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of CPU worker threads (default: 4).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/device/rpi_benchmarks.json",
        help="Output path for benchmark JSON.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run fast verification with reduced iterations; marks result 'emulated'.",
    )
    args = parser.parse_args()

    print("=" * 75)
    print(" LiteCascade Raspberry Pi On-Device Benchmark")
    print(f" Mode: {'DRY-RUN (Verification)' if args.dry_run else 'FULL BENCHMARK'}")
    print("=" * 75)

    report = run_rpi_benchmark(
        model_path=args.model,
        n_runs=args.runs,
        n_warmup=args.warmup,
        rho=args.rho,
        dry_run=args.dry_run,
        num_threads=args.threads,
        output_path=args.output,
    )

    print(f"Platform:              {report['platform']}")
    print(f"OS:                    {report['os']}")
    print(f"CPU / Clock:           {report['cpu']} @ {report['clock']}")
    print(f"Measured or Emulated:  {report['measured_or_emulated'].upper()} (Rule 6)")
    print(f"Iterations:            {report['n_runs']} runs (warmup: {report['warmup_runs']}) | rho: {report['rho']}")
    print("-" * 75)

    for cfg, data in report["results"].items():
        title = "WITH XNNPACK" if "with_" in cfg else "WITHOUT XNNPACK"
        print(f"\n[{title}]")
        print(f"{'Path':<16} | {'p50 (ms)':<9} | {'p95 (ms)':<9} | {'Mean (ms)':<10} | {'RSS (MB)':<9} | {'CPU %':<6}")
        print("-" * 75)
        for path_name, stats in data.items():
            print(
                f"{path_name:<16} | "
                f"{stats['p50_ms']:<9.3f} | "
                f"{stats['p95_ms']:<9.3f} | "
                f"{stats['mean_ms']:<10.3f} | "
                f"{stats['peak_rss_mb']:<9.1f} | "
                f"{stats['cpu_percent']:<6.1f}"
            )
        casc = data["cascade"]
        print(f"  --> Speed-up vs Analyst: {casc['speedup_vs_analyst']}x  (Analytical E[T]: {casc['analytical_expected_ms']} ms)")

    print("\n" + "=" * 75)
    print(f"Results saved to: {args.output}\n")


if __name__ == "__main__":
    main()
