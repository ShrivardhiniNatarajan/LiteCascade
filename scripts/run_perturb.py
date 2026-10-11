"""
Stage 11: Perturbation and Robustness Benchmarking (Table VI).

Evaluates LiteCascade under feature-level evasion emulations:
1. Clean (sigma = 0.0)
2. Gaussian Noise: sigma in {0.05, 0.10, 0.20} in scaled space
3. Inter-arrival time jitter: traffic delay injection emulation
4. Packet-size padding: dummy payload padding emulation

Produces:
- results/perturb/<dataset>/perturb_results.json
- results/tables/robustness.csv
- results/tables/robustness.tex
"""

# ruff: noqa: E402
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from litecascade.data.dataset import CascadeDataset, compute_class_weights
from litecascade.data.nbaiot import load_nbaiot
from litecascade.data.preprocess import Preprocessor
from litecascade.data.splits import create_group_splits
from litecascade.data.windows import create_windows
from litecascade.eval.gate import find_tau_b, fit_temperature
from litecascade.eval.perturb import (
    apply_gaussian_noise,
    apply_size_padding,
    apply_time_jitter,
    evaluate_under_perturbation,
    identify_feature_types,
)
from litecascade.models.litecascade import LiteCascade
from litecascade.train.loss import CascadeLoss
from litecascade.utils.seed import set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_clean_model(
    model: LiteCascade,
    train_loader: DataLoader,
    loss_fn: nn.Module,
    epochs: int = 5,
    lr: float = 1e-3,
) -> None:
    """Trains model on clean data."""
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    for _ in range(epochs):
        for xb, yb in train_loader:
            optimizer.zero_grad()
            l1, l2 = model(xb, mode="train")
            loss = loss_fn(l1, l2, yb)
            loss.backward()
            optimizer.step()


def run_perturb(
    dataset_name: str = "fixture",
    epochs: int = 5,
    seed: int = 42,
    variant: str = "R",
    epsilon: float = 0.005,
    output_dir: Path = Path("results/perturb"),
    tables_dir: Path = Path("results/tables"),
) -> list[dict[str, Any]]:
    """Runs perturbation evaluation on test windows and exports Table VI."""
    set_seed(seed)
    is_synthetic = dataset_name == "fixture"
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    df = load_nbaiot(use_synthetic=is_synthetic)
    splits = create_group_splits(df, dataset_name, seed=seed)

    feature_cols = [c for c in df.columns if c not in ["label_binary", "label_family", "group", "timestamp", "is_synthetic"]]
    input_dim = len(feature_cols)

    # Preprocessor fit on train only
    df_train = df.loc[splits["train"]]
    preproc = Preprocessor(n_components=16)
    preproc.fit(df_train, feature_cols)
    pca_weights = torch.tensor(preproc.pca.components_, dtype=torch.float32)

    # Windows
    X_train, y_train = create_windows(df_train, feature_cols, T=10)
    X_calib, y_calib = create_windows(df.loc[splits["calib"]], feature_cols, T=10)
    X_test, y_test = create_windows(df.loc[splits["test"]], feature_cols, T=10)

    train_ds = CascadeDataset(X_train, y_train)
    calib_ds = CascadeDataset(X_calib, y_calib)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    calib_loader = DataLoader(calib_ds, batch_size=32, shuffle=False)

    class_weights = compute_class_weights(y_train)

    # Train model
    logger.info("Training clean LiteCascade (%s) on %s...", variant, dataset_name)
    model = LiteCascade(F_in=input_dim, d=16, c=24, hidden=24, K=2, variant=variant)
    model.pip.linear.weight.data.copy_(pca_weights)
    loss_fn = CascadeLoss(gamma=0.0, delta=0.3, class_weights=class_weights)
    train_clean_model(model, train_loader, loss_fn, epochs=epochs)

    # Calibration on calib split
    model.eval()
    calib_l1, calib_y = [], []
    with torch.no_grad():
        for xb, yb in calib_loader:
            l1, _ = model(xb, mode="train")
            calib_l1.append(l1.numpy())
            calib_y.append(yb.numpy())

    T = fit_temperature(np.concatenate(calib_l1), np.concatenate(calib_y))
    tau_b, _, _ = find_tau_b(model, calib_loader, epsilon=epsilon, temperature=T)
    tau_m = 0.8
    logger.info("Calibrated gate: T=%.4f, tau_b=%.4f, tau_m=%.4f", T, tau_b, tau_m)

    # Identify features for perturbation
    timing_idx, size_idx, meta = identify_feature_types(feature_cols, total_features=input_dim)
    logger.info("Identified %d timing features (%s) and %d size features (%s)",
                len(timing_idx), meta["timing_columns"], len(size_idx), meta["size_columns"])

    # 1. Clean condition (sigma=0.0)
    clean_x = apply_gaussian_noise(X_test, sigma=0.0, seed=seed)
    clean_res = evaluate_under_perturbation(model, clean_x, y_test, tau_b=tau_b, tau_m=tau_m, temperature=T)
    clean_recall = clean_res["recall"]

    conditions: list[dict[str, Any]] = []

    def record_condition(
        name: str,
        category: str,
        param_desc: str,
        x_perturbed: np.ndarray,
        perturbed_cols: list[str] | list[int] | None = None,
    ) -> None:
        res = evaluate_under_perturbation(model, x_perturbed, y_test, tau_b=tau_b, tau_m=tau_m, temperature=T)
        rec = res["recall"]
        recall_loss = max(0.0, clean_recall - rec)

        rec_entry = {
            "condition": name,
            "category": category,
            "parameter": param_desc,
            "dataset": dataset_name,
            "is_synthetic": is_synthetic,
            "variant": variant,
            "f1_macro": res["f1_macro"],
            "recall": rec,
            "rho": res["rho"],
            "recall_loss": recall_loss,
            "within_epsilon": bool(recall_loss <= epsilon),
            "perturbed_features": [str(c) for c in (perturbed_cols or [])],
            "emulation_note": "Feature-level EMULATION of evasion attack",
        }
        conditions.append(rec_entry)
        logger.info(
            "[%s] F1=%.4f | Recall=%.4f | rho=%.3f | Recall Loss=%.4f (<=eps: %s)",
            name,
            res["f1_macro"],
            rec,
            res["rho"],
            recall_loss,
            rec_entry["within_epsilon"],
        )

    # Evaluate conditions
    record_condition("Clean (baseline)", "Clean", "sigma=0.0", clean_x)

    for sigma in [0.05, 0.10, 0.20]:
        x_noise = apply_gaussian_noise(X_test, sigma=sigma, seed=seed)
        record_condition(f"Gaussian (sigma={sigma:.2f})", "Gaussian Noise", f"sigma={sigma}", x_noise)

    x_jitter, _ = apply_time_jitter(X_test, feature_names=feature_cols, jitter_ratio=0.10, seed=seed)
    record_condition("Time Jitter (ratio=0.10)", "Evasion Emulation", "jitter=0.10", x_jitter, meta["timing_columns"])

    x_padding, _ = apply_size_padding(X_test, feature_names=feature_cols, pad_amount=0.20, seed=seed)
    record_condition("Packet Size Padding (+0.20)", "Evasion Emulation", "shift=+0.20", x_padding, meta["size_columns"])

    # Export JSON
    out_json = output_dir / f"perturb_{dataset_name}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"dataset": dataset_name, "conditions": conditions}, f, indent=2)

    # Export CSV
    csv_file = tables_dir / "robustness.csv"
    fieldnames = [
        "condition",
        "category",
        "parameter",
        "dataset",
        "is_synthetic",
        "variant",
        "f1_macro",
        "recall",
        "rho",
        "recall_loss",
        "within_epsilon",
    ]
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in conditions:
            row_dict = {k: r[k] for k in fieldnames}
            writer.writerow(row_dict)

    # Export LaTeX
    tex_file = tables_dir / "robustness.tex"
    with open(tex_file, "w", encoding="utf-8") as f:
        f.write("% LiteCascade Robustness & Perturbation Table (Table VI)\n")
        f.write("\\begin{table}[t]\n\\centering\n")
        f.write("\\caption{Robustness under feature-level evasion emulations.}\n")
        f.write("\\label{tab:robustness}\n")
        f.write("\\begin{tabular}{lccccc}\n\\toprule\n")
        f.write("Condition & Category & Macro F1 & Recall & Exit Rate $\\rho$ & Recall Loss \\\\\n\\midrule\n")
        for r in conditions:
            f.write(
                f"{r['condition']} & {r['category']} & {r['f1_macro']:.4f} & "
                f"{r['recall']:.4f} & {r['rho']:.3f} & {r['recall_loss']:.4f} \\\\\n"
            )
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    logger.info("Robustness benchmark complete! Exported %s and %s (%d conditions).",
                csv_file, tex_file, len(conditions))

    # Print CSV content directly for immediate inspection
    print(f"\n--- Output of {csv_file} ({len(conditions)} rows) ---")
    with open(csv_file, encoding="utf-8") as f:
        for line in f:
            print(line.rstrip())

    return conditions


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LiteCascade perturbation robustness benchmark")
    parser.add_argument("--dataset", type=str, default="fixture")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variant", type=str, default="R")
    parser.add_argument("--epsilon", type=float, default=0.005)
    args = parser.parse_args()

    run_perturb(
        dataset_name=args.dataset,
        epochs=args.epochs,
        seed=args.seed,
        variant=args.variant,
        epsilon=args.epsilon,
    )


if __name__ == "__main__":
    main()
