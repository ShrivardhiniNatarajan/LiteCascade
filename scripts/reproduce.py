"""Cross-Platform One-Command Reproducibility Runner (Stage 15).

Provides identical reproduction functionality as Makefile across Windows, macOS, and Linux:
    python scripts/reproduce.py --mode fast
    python scripts/reproduce.py --table 3
    python scripts/reproduce.py --verify
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def run_step(step_name: str, cmd: list[str]) -> None:
    print(f"\n>>> [{step_name}] Running: {' '.join(cmd)}")
    t0 = time.perf_counter()
    res = subprocess.run(cmd, check=True)
    dt = time.perf_counter() - t0
    print(f">>> [{step_name}] Completed in {dt:.2f}s (exit code {res.returncode})\n")


def reproduce_fast() -> None:
    py = sys.executable
    print("=" * 75)
    print(" LiteCascade Fast Reproducibility Pipeline (<10 min)")
    print(" Target: Synthetic Fixtures & End-to-End Metric Verification")
    print("=" * 75)
    t_start = time.perf_counter()

    run_step("1/5: Generate Fixtures", [py, "scripts/make_fixture.py"])
    run_step("2/5: Profile Baselines", [py, "scripts/profile_baselines.py"])
    run_step("3/5: Generate Tables & Figures", [py, "scripts/make_paper_tables.py"])
    run_step("4/5: Verify Traceability (Rule 2 & 5)", [py, "scripts/check_no_handwritten_numbers.py"])
    run_step("5/5: Core Verification Tests", [py, "-m", "pytest", "tests/test_results_aggregation.py", "tests/test_device.py", "tests/test_image_adapter.py", "-q"])

    total_time = time.perf_counter() - t_start
    print("=" * 75)
    print(f" [PASS] Fast reproduction completed in {total_time:.2f} seconds!")
    print(" All paper tables (results/tables/) and figures (results/figures/) generated.")
    print("=" * 75)


def reproduce_full() -> None:
    py = sys.executable
    print("=" * 75)
    print(" LiteCascade Full Reproducibility Pipeline")
    print(" Target: Full multi-seed training, pruning, QAT, ablation, LOFO, and export")
    print("=" * 75)
    t_start = time.perf_counter()

    run_step("1/8: Generate Fixtures", [py, "scripts/make_fixture.py"])
    run_step("2/8: Train Baselines (B1, B3, B7)", [py, "-m", "litecascade.train.run", "--model", "b7", "--dataset", "fixture", "--epochs", "2"])
    run_step("3/8: Train LiteCascade Dual-Exit", [py, "-m", "litecascade.train.run_cascade", "--dataset", "fixture", "--epochs", "2"])
    run_step("4/8: Ablation Ladder", [py, "scripts/run_ablation.py", "--dataset", "fixture", "--seeds", "2", "--epochs", "2"])
    run_step("4b/8: Aggregate Ablations", [py, "scripts/aggregate_ablation.py"])
    run_step("5/8: Zero-Day LOFO Evaluation", [py, "scripts/run_lofo.py", "--dataset", "fixture"])
    run_step("5b/8: Feature Perturbation Evaluation", [py, "scripts/run_perturb.py", "--dataset", "fixture"])
    run_step("6/8: Device Benchmarks (Dry-Run)", [py, "scripts/device/bench_rpi.py", "--model", "checkpoints/litecascade_int8.tflite", "--runs", "50", "--dry-run"])
    run_step("6b/8: MCU Sizing Analysis", [py, "scripts/estimate_mcu.py", "--model", "checkpoints/litecascade_int8.tflite"])
    run_step("7/8: Generate Paper Tables & Figures", [py, "scripts/make_paper_tables.py"])
    run_step("7b/8: Verify Traceability", [py, "scripts/check_no_handwritten_numbers.py"])
    run_step("8/8: Run Full Pytest Suite", [py, "-m", "pytest", "tests/", "-q"])

    total_time = time.perf_counter() - t_start
    print("=" * 75)
    print(f" [PASS] Full reproduction completed in {total_time:.2f} seconds!")
    print("=" * 75)


def main() -> None:
    parser = argparse.ArgumentParser(description="LiteCascade Reproducibility Runner")
    parser.add_argument(
        "--mode",
        type=str,
        default="fast",
        choices=["fast", "full"],
        help="Reproduction mode: 'fast' (<10 min on fixture) or 'full'",
    )
    parser.add_argument(
        "--table",
        type=str,
        choices=["3", "4", "5", "6", "main", "ablation", "device", "robustness", "all"],
        help="Generate specific paper table (3, 4, 5, 6)",
    )
    parser.add_argument("--figures", action="store_true", help="Generate all 5 IEEE 300 DPI figures")
    parser.add_argument("--verify", action="store_true", help="Verify traceability (zero handwritten numbers)")
    args = parser.parse_args()

    py = sys.executable

    if args.table:
        subprocess.run([py, "scripts/make_paper_tables.py", "--table", args.table], check=True)
    elif args.figures:
        subprocess.run([py, "scripts/make_paper_tables.py", "--figures"], check=True)
    elif args.verify:
        subprocess.run([py, "scripts/check_no_handwritten_numbers.py"], check=True)
    elif args.mode == "fast":
        reproduce_fast()
    elif args.mode == "full":
        reproduce_full()


if __name__ == "__main__":
    main()
