"""
Stage 10: Ablation Ladder Aggregator.

Aggregates individual seed results from results/ablation/<name>/seed_<n>.json
and produces:
- results/tables/ablation.csv (with mean and std columns, dataset labeled synthetic)
- results/tables/ablation.tex (clean publication-ready LaTeX table)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ABLATION_ORDER = [
    "A0",
    "A1",
    "A2",
    "A3",
    "A4",
    "A5",
    "A6",
    "A7",
    "A8",
    "A9",
]

ABLATION_DESCRIPTIONS = {
    "A0": "Analyst only (no PIP init, no gate)",
    "A1": "+ PIP initialisation",
    "A2": "+ Sentinel with naive fixed-threshold gate (0.9)",
    "A3": "+ temperature scaling",
    "A4": "+ risk-controlled tau (Clopper-Pearson)",
    "A5": "+ distillation",
    "A6": "+ structured pruning",
    "A7": "+ INT8 QAT",
    "A8": "+ re-calibration after QAT",
    "A9": "Wrong order (QAT -> prune)",
}


def aggregate_ablation(
    ablation_dir: Path = Path("results/ablation"),
    tables_dir: Path = Path("results/tables"),
) -> list[dict[str, Any]]:
    """Aggregates all seed runs in ablation_dir and writes CSV/LaTeX tables."""
    tables_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []

    for abl_id in ABLATION_ORDER:
        folder = ablation_dir / abl_id
        if not folder.exists():
            logger.warning("Directory %s not found. Skipping %s.", folder, abl_id)
            continue

        seed_files = sorted(folder.glob("seed_*.json"))
        if not seed_files:
            logger.warning("No seed files found in %s.", folder)
            continue

        f1_list: list[float] = []
        rho_list: list[float] = []
        macs_list: list[float] = []
        size_list: list[float] = []
        variant_set: set[str] = set()
        dataset_set: set[str] = set()
        is_synth = False

        for sf in seed_files:
            with open(sf, encoding="utf-8") as f:
                data = json.load(f)
            f1_list.append(float(data.get("f1", 0.0)))
            rho_list.append(float(data.get("rho", 0.0)))
            macs_list.append(float(data.get("expected_macs", 0.0)))
            size_list.append(float(data.get("size_kb", 0.0)))
            variant_set.add(data.get("variant", "R"))
            dataset_set.add(data.get("dataset", "fixture"))
            if data.get("is_synthetic", False):
                is_synth = True

        n_seeds = len(seed_files)
        f1_arr = np.array(f1_list)
        rho_arr = np.array(rho_list)
        macs_arr = np.array(macs_list)
        size_arr = np.array(size_list)

        dataset_label = "/".join(sorted(dataset_set))
        if is_synth:
            dataset_label = f"{dataset_label} (synthetic)"

        row = {
            "ablation": abl_id,
            "description": ABLATION_DESCRIPTIONS.get(abl_id, ""),
            "variant": "/".join(sorted(variant_set)),
            "dataset": dataset_label,
            "is_synthetic": is_synth,
            "num_seeds": n_seeds,
            "f1_mean": float(np.mean(f1_arr)),
            "f1_std": float(np.std(f1_arr, ddof=1)) if n_seeds > 1 else 0.0,
            "rho_mean": float(np.mean(rho_arr)),
            "rho_std": float(np.std(rho_arr, ddof=1)) if n_seeds > 1 else 0.0,
            "expected_macs_mean": float(np.mean(macs_arr)),
            "expected_macs_std": float(np.std(macs_arr, ddof=1)) if n_seeds > 1 else 0.0,
            "size_kb_mean": float(np.mean(size_arr)),
            "size_kb_std": float(np.std(size_arr, ddof=1)) if n_seeds > 1 else 0.0,
        }
        summary_rows.append(row)

    csv_path = tables_dir / "ablation.csv"
    tex_path = tables_dir / "ablation.tex"

    # Write CSV
    fieldnames = [
        "ablation",
        "description",
        "variant",
        "dataset",
        "is_synthetic",
        "num_seeds",
        "f1_mean",
        "f1_std",
        "rho_mean",
        "rho_std",
        "expected_macs_mean",
        "expected_macs_std",
        "size_kb_mean",
        "size_kb_std",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in summary_rows:
            writer.writerow(r)

    # Write LaTeX
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("% LiteCascade Ablation Ladder Table (Table IV)\n")
        f.write("\\begin{table}[t]\n\\centering\n")
        f.write("\\caption{Ablation Ladder results (mean $\\pm$ std over seeds).}\n")
        f.write("\\label{tab:ablation}\n")
        f.write("\\begin{tabular}{llcccc}\n\\toprule\n")
        f.write("Ablation & Description & Macro F1 & Exit Rate $\\rho$ & Expected MACs & Size (KB) \\\\\n\\midrule\n")
        for r in summary_rows:
            f.write(
                f"{r['ablation']} & {r['description']} & "
                f"{r['f1_mean']:.4f} $\\pm$ {r['f1_std']:.4f} & "
                f"{r['rho_mean']:.3f} $\\pm$ {r['rho_std']:.3f} & "
                f"{r['expected_macs_mean']:.0f} $\\pm$ {r['expected_macs_std']:.0f} & "
                f"{r['size_kb_mean']:.1f} $\\pm$ {r['size_kb_std']:.1f} \\\\\n"
            )
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    logger.info("Successfully exported %s and %s (%d rows).", csv_path, tex_path, len(summary_rows))
    return summary_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate LiteCascade Ablation Ladder results")
    parser.add_argument("--ablation_dir", type=str, default="results/ablation")
    parser.add_argument("--tables_dir", type=str, default="results/tables")
    args = parser.parse_args()

    rows = aggregate_ablation(Path(args.ablation_dir), Path(args.tables_dir))

    # Print CSV content directly for immediate inspection
    csv_file = Path(args.tables_dir) / "ablation.csv"
    if csv_file.exists():
        print(f"\n--- Output of {csv_file} ({len(rows)} rows) ---")
        with open(csv_file, encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines[:15]:
                print(line.rstrip())


if __name__ == "__main__":
    main()
