"""
Stage 11: Leave-One-Family-Out (LOFO) Zero-Day Evaluation.

For each malware family:
1. Train model strictly without it (train contains 0 samples of held-out family).
2. Test on held-out family (plus benign test samples).
3. Report per-family F1, recall, and mean across families for:
   - Teacher (CNN-BiLSTM)
   - Best lightweight baseline (B7 DSCNN)
   - LiteCascade Variant R
   - LiteCascade Variant T
4. Report exit-induced recall loss and verify if it stays within epsilon.
5. Export results/tables/lofo.csv and results/tables/lofo.tex.
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
from litecascade.data.windows import create_windows
from litecascade.eval.gate import find_tau_b, fit_temperature
from litecascade.eval.metrics import compute_detection_metrics
from litecascade.models.baselines import B7_DSCNN
from litecascade.models.litecascade import LiteCascade
from litecascade.models.teacher import Teacher_CNN_BiLSTM
from litecascade.train.loss import CascadeLoss
from litecascade.utils.seed import set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_classifier(
    model: nn.Module,
    loader: DataLoader,
    epochs: int = 5,
    lr: float = 1e-3,
    is_cascade: bool = False,
    loss_fn: nn.Module | None = None,
) -> None:
    """Trains standard or cascade model for given epochs."""
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    ce_loss = nn.CrossEntropyLoss()

    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            if is_cascade:
                l1, l2 = model(xb, mode="train")
                if loss_fn is not None:
                    loss = loss_fn(l1, l2, yb)
                else:
                    loss = ce_loss(l1, yb) + ce_loss(l2, yb)
            else:
                logits = model(xb)
                loss = ce_loss(logits, yb)

            loss.backward()
            optimizer.step()


def evaluate_standard_model(model: nn.Module, loader: DataLoader) -> tuple[float, float]:
    """Evaluates standard model (Teacher / Baseline). Returns (f1_macro, recall)."""
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb)
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(yb.numpy().tolist())

    metrics = compute_detection_metrics(np.array(all_labels), np.array(all_preds))
    rec = float(metrics.get("recall_malicious", metrics.get("recall_macro", 0.0)))
    return float(metrics["f1_macro"]), rec


def evaluate_cascade_lofo(
    model: LiteCascade,
    test_loader: DataLoader,
    calib_loader: DataLoader,
    epsilon: float = 0.005,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Calibrates LiteCascade and evaluates on LOFO test set."""
    model.eval()

    # 1. Temperature scaling
    calib_l1, calib_y = [], []
    with torch.no_grad():
        for xb, yb in calib_loader:
            l1, _ = model(xb, mode="train")
            calib_l1.append(l1.numpy())
            calib_y.append(yb.numpy())

    T = fit_temperature(np.concatenate(calib_l1), np.concatenate(calib_y))

    # 2. Clopper-Pearson tau_b
    tau_b, _, _ = find_tau_b(model, calib_loader, epsilon=epsilon, alpha=alpha, temperature=T)
    tau_m = 0.8

    # 3. Test evaluation
    all_preds: list[int] = []
    analyst_preds: list[int] = []
    all_labels: list[int] = []
    exited_count = 0
    total_count = 0

    with torch.no_grad():
        for xb, yb in test_loader:
            B = xb.size(0)
            l1, l2 = model(xb, mode="train")
            scaled_l1 = l1 / max(T, 1e-6)
            probs1 = torch.softmax(scaled_l1, dim=-1)
            p1_preds = torch.argmax(probs1, dim=-1)
            p2_preds = torch.argmax(l2, dim=-1)

            final_p = p2_preds.clone()
            for i in range(B):
                p1_max = probs1[i].max().item()
                pred1 = p1_preds[i].item()
                should_exit = False
                if pred1 == 0 and probs1[i, 0].item() >= tau_b:
                    should_exit = True
                elif pred1 != 0 and p1_max >= tau_m:
                    should_exit = True

                if should_exit:
                    final_p[i] = pred1
                    exited_count += 1

            total_count += B
            all_preds.extend(final_p.cpu().numpy().tolist())
            analyst_preds.extend(p2_preds.cpu().numpy().tolist())
            all_labels.extend(yb.numpy().tolist())

    y_arr = np.array(all_labels)
    pred_arr = np.array(all_preds)
    analyst_arr = np.array(analyst_preds)

    cascade_metrics = compute_detection_metrics(y_arr, pred_arr)
    analyst_metrics = compute_detection_metrics(y_arr, analyst_arr)

    rec_cascade = float(cascade_metrics.get("recall_malicious", 0.0))
    rec_analyst = float(analyst_metrics.get("recall_malicious", 0.0))
    recall_loss = max(0.0, rec_analyst - rec_cascade)
    rho = exited_count / total_count if total_count > 0 else 0.0

    return {
        "f1_macro": float(cascade_metrics["f1_macro"]),
        "recall": rec_cascade,
        "analyst_recall": rec_analyst,
        "rho": float(rho),
        "recall_loss": float(recall_loss),
        "within_epsilon": bool(recall_loss <= epsilon),
        "tau_b": float(tau_b),
        "temperature": float(T),
    }


def run_lofo(
    dataset_name: str = "fixture",
    epochs: int = 5,
    seed: int = 42,
    epsilon: float = 0.005,
    output_dir: Path = Path("results/lofo"),
    tables_dir: Path = Path("results/tables"),
) -> dict[str, Any]:
    """Executes full LOFO evaluation across all malware families."""
    set_seed(seed)
    is_synthetic = dataset_name == "fixture"
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load dataset
    df = load_nbaiot(use_synthetic=is_synthetic)
    feature_cols = [c for c in df.columns if c not in ["label_binary", "label_family", "group", "timestamp", "is_synthetic"]]
    input_dim = len(feature_cols)

    # Identify all malware families
    families = sorted([f for f in df["label_family"].unique() if f != "benign"])
    groups = sorted(df["group"].unique())
    logger.info("Running LOFO across %d malware families: %s with groups %s", len(families), families, groups)

    # Group splitting for leakage safety (Rule 7)
    if len(groups) >= 3:
        train_groups = groups[:-1]
        test_groups = [groups[-1]]
    else:
        train_groups = groups
        test_groups = groups

    results_per_family: dict[str, dict[str, Any]] = {}

    for fam in families:
        logger.info("=== LOFO: Held-out family '%s' ===", fam)

        # Build train, calib, test splits
        # TRAIN strictly excludes held-out family
        train_mask = df["group"].isin(train_groups) & (df["label_family"] != fam)
        test_mask = df["group"].isin(test_groups) & ((df["label_family"] == fam) | (df["label_family"] == "benign"))

        train_idx = df[train_mask].index.tolist()
        test_idx = df[test_mask].index.tolist()

        # Strict safety assertions (non-negotiable rules)
        assert fam not in df.loc[train_idx, "label_family"].values, f"CRITICAL: Held-out family {fam} leaked into train!"
        if len(groups) >= 3:
            assert set(df.loc[train_idx, "group"]).isdisjoint(set(df.loc[test_idx, "group"])), "Train and test groups overlap!"
        assert fam in df.loc[test_idx, "label_family"].values, f"Test set missing held-out family {fam}!"

        df_train = df.loc[train_idx]
        df_test = df.loc[test_idx]

        # Calibration slice (disjoint sample from train)
        calib_size = max(10, int(len(df_train) * 0.2))
        calib_idx = df_train.sample(n=calib_size, random_state=seed).index.tolist()
        train_sub_idx = [i for i in train_idx if i not in calib_idx]
        df_calib = df.loc[calib_idx]
        df_train_sub = df.loc[train_sub_idx]

        # Fit preprocessor strictly on train
        preproc = Preprocessor(n_components=16)
        preproc.fit(df_train_sub, feature_cols)
        pca_weights = torch.tensor(preproc.pca.components_, dtype=torch.float32)

        # Create windows
        X_train, y_train = create_windows(df_train_sub, feature_cols, T=10)
        X_calib, y_calib = create_windows(df_calib, feature_cols, T=10)
        X_test, y_test = create_windows(df_test, feature_cols, T=10)

        train_ds = CascadeDataset(X_train, y_train)
        calib_ds = CascadeDataset(X_calib, y_calib)
        test_ds = CascadeDataset(X_test, y_test)

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        calib_loader = DataLoader(calib_ds, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

        class_weights = compute_class_weights(y_train)

        # Model 1: Teacher (CNN-BiLSTM)
        logger.info("Training Teacher...")
        teacher = Teacher_CNN_BiLSTM(input_dim=input_dim, num_classes=2)
        train_classifier(teacher, train_loader, epochs=epochs, is_cascade=False)
        t_f1, t_rec = evaluate_standard_model(teacher, test_loader)

        # Model 2: Best Baseline (B7 DSCNN)
        logger.info("Training B7 DSCNN baseline...")
        b7 = B7_DSCNN(input_dim=input_dim, hidden_dim=16, num_classes=2)
        train_classifier(b7, train_loader, epochs=epochs, is_cascade=False)
        b7_f1, b7_rec = evaluate_standard_model(b7, test_loader)

        # Model 3: LiteCascade Variant R
        logger.info("Training LiteCascade (R)...")
        lc_r = LiteCascade(F_in=input_dim, d=16, c=24, hidden=24, K=2, variant="R")
        lc_r.pip.linear.weight.data.copy_(pca_weights)
        loss_r = CascadeLoss(gamma=0.0, delta=0.3, class_weights=class_weights)
        train_classifier(lc_r, train_loader, epochs=epochs, is_cascade=True, loss_fn=loss_r)
        lc_r_res = evaluate_cascade_lofo(lc_r, test_loader, calib_loader, epsilon=epsilon)

        # Model 4: LiteCascade Variant T
        logger.info("Training LiteCascade (T)...")
        lc_t = LiteCascade(F_in=input_dim, d=16, c=24, hidden=24, K=2, variant="T")
        lc_t.pip.linear.weight.data.copy_(pca_weights)
        loss_t = CascadeLoss(gamma=0.0, delta=0.3, class_weights=class_weights)
        train_classifier(lc_t, train_loader, epochs=epochs, is_cascade=True, loss_fn=loss_t)
        lc_t_res = evaluate_cascade_lofo(lc_t, test_loader, calib_loader, epsilon=epsilon)

        fam_record = {
            "family": fam,
            "teacher": {"f1_macro": t_f1, "recall": t_rec},
            "b7_dscnn": {"f1_macro": b7_f1, "recall": b7_rec},
            "litecascade_r": lc_r_res,
            "litecascade_t": lc_t_res,
        }
        results_per_family[fam] = fam_record
        logger.info(
            "[%s] Teacher F1=%.4f Rec=%.4f | B7 F1=%.4f Rec=%.4f | LC-R F1=%.4f Rec=%.4f Loss=%.4f (within_eps=%s) | LC-T F1=%.4f Rec=%.4f Loss=%.4f",
            fam,
            t_f1,
            t_rec,
            b7_f1,
            b7_rec,
            lc_r_res["f1_macro"],
            lc_r_res["recall"],
            lc_r_res["recall_loss"],
            lc_r_res["within_epsilon"],
            lc_t_res["f1_macro"],
            lc_t_res["recall"],
            lc_t_res["recall_loss"],
        )

    # Compute Means
    mean_teacher_f1 = float(np.mean([r["teacher"]["f1_macro"] for r in results_per_family.values()]))
    mean_teacher_rec = float(np.mean([r["teacher"]["recall"] for r in results_per_family.values()]))
    mean_b7_f1 = float(np.mean([r["b7_dscnn"]["f1_macro"] for r in results_per_family.values()]))
    mean_b7_rec = float(np.mean([r["b7_dscnn"]["recall"] for r in results_per_family.values()]))

    mean_lcr_f1 = float(np.mean([r["litecascade_r"]["f1_macro"] for r in results_per_family.values()]))
    mean_lcr_rec = float(np.mean([r["litecascade_r"]["recall"] for r in results_per_family.values()]))
    mean_lcr_loss = float(np.mean([r["litecascade_r"]["recall_loss"] for r in results_per_family.values()]))
    mean_lcr_rho = float(np.mean([r["litecascade_r"]["rho"] for r in results_per_family.values()]))

    mean_lct_f1 = float(np.mean([r["litecascade_t"]["f1_macro"] for r in results_per_family.values()]))
    mean_lct_rec = float(np.mean([r["litecascade_t"]["recall"] for r in results_per_family.values()]))
    mean_lct_loss = float(np.mean([r["litecascade_t"]["recall_loss"] for r in results_per_family.values()]))
    mean_lct_rho = float(np.mean([r["litecascade_t"]["rho"] for r in results_per_family.values()]))

    summary = {
        "dataset": dataset_name,
        "is_synthetic": is_synthetic,
        "per_family": results_per_family,
        "means": {
            "teacher": {"f1_macro": mean_teacher_f1, "recall": mean_teacher_rec},
            "b7_dscnn": {"f1_macro": mean_b7_f1, "recall": mean_b7_rec},
            "litecascade_r": {
                "f1_macro": mean_lcr_f1,
                "recall": mean_lcr_rec,
                "rho": mean_lcr_rho,
                "recall_loss": mean_lcr_loss,
                "within_epsilon": bool(mean_lcr_loss <= epsilon),
            },
            "litecascade_t": {
                "f1_macro": mean_lct_f1,
                "recall": mean_lct_rec,
                "rho": mean_lct_rho,
                "recall_loss": mean_lct_loss,
                "within_epsilon": bool(mean_lct_loss <= epsilon),
            },
        },
    }

    # Save JSON
    out_json = output_dir / f"lofo_{dataset_name}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Save CSV
    csv_file = tables_dir / "lofo.csv"
    fieldnames = [
        "family",
        "model",
        "dataset",
        "is_synthetic",
        "f1_macro",
        "recall",
        "rho",
        "recall_loss",
        "within_epsilon",
    ]
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for fam, rec in results_per_family.items():
            writer.writerow({
                "family": fam, "model": "Teacher", "dataset": dataset_name, "is_synthetic": is_synthetic,
                "f1_macro": rec["teacher"]["f1_macro"], "recall": rec["teacher"]["recall"], "rho": 0.0, "recall_loss": 0.0, "within_epsilon": True
            })
            writer.writerow({
                "family": fam, "model": "B7_DSCNN", "dataset": dataset_name, "is_synthetic": is_synthetic,
                "f1_macro": rec["b7_dscnn"]["f1_macro"], "recall": rec["b7_dscnn"]["recall"], "rho": 0.0, "recall_loss": 0.0, "within_epsilon": True
            })
            writer.writerow({
                "family": fam, "model": "LiteCascade_R", "dataset": dataset_name, "is_synthetic": is_synthetic,
                "f1_macro": rec["litecascade_r"]["f1_macro"], "recall": rec["litecascade_r"]["recall"],
                "rho": rec["litecascade_r"]["rho"], "recall_loss": rec["litecascade_r"]["recall_loss"],
                "within_epsilon": rec["litecascade_r"]["within_epsilon"]
            })
            writer.writerow({
                "family": fam, "model": "LiteCascade_T", "dataset": dataset_name, "is_synthetic": is_synthetic,
                "f1_macro": rec["litecascade_t"]["f1_macro"], "recall": rec["litecascade_t"]["recall"],
                "rho": rec["litecascade_t"]["rho"], "recall_loss": rec["litecascade_t"]["recall_loss"],
                "within_epsilon": rec["litecascade_t"]["within_epsilon"]
            })

        # Summary mean row
        writer.writerow({
            "family": "MEAN", "model": "LiteCascade_R", "dataset": dataset_name, "is_synthetic": is_synthetic,
            "f1_macro": mean_lcr_f1, "recall": mean_lcr_rec, "rho": mean_lcr_rho, "recall_loss": mean_lcr_loss,
            "within_epsilon": bool(mean_lcr_loss <= epsilon)
        })

    # Save LaTeX
    tex_file = tables_dir / "lofo.tex"
    with open(tex_file, "w", encoding="utf-8") as f:
        f.write("% Leave-One-Family-Out (LOFO) Zero-Day Evaluation\n")
        f.write("\\begin{table}[t]\n\\centering\n")
        f.write("\\caption{Zero-Day LOFO generalization results.}\n")
        f.write("\\label{tab:lofo}\n")
        f.write("\\begin{tabular}{lcccc}\n\\toprule\n")
        f.write("Held-Out Family & Model & F1 & Recall & Recall Loss \\\\\n\\midrule\n")
        for fam, rec in results_per_family.items():
            f.write(f"{fam} & Teacher & {rec['teacher']['f1_macro']:.4f} & {rec['teacher']['recall']:.4f} & -- \\\\\n")
            f.write(f"{fam} & B7 DSCNN & {rec['b7_dscnn']['f1_macro']:.4f} & {rec['b7_dscnn']['recall']:.4f} & -- \\\\\n")
            f.write(f"{fam} & LiteCascade (R) & {rec['litecascade_r']['f1_macro']:.4f} & {rec['litecascade_r']['recall']:.4f} & {rec['litecascade_r']['recall_loss']:.4f} \\\\\n")
            f.write(f"{fam} & LiteCascade (T) & {rec['litecascade_t']['f1_macro']:.4f} & {rec['litecascade_t']['recall']:.4f} & {rec['litecascade_t']['recall_loss']:.4f} \\\\\n")
        f.write("\\midrule\n")
        f.write(f"Mean & LiteCascade (R) & {mean_lcr_f1:.4f} & {mean_lcr_rec:.4f} & {mean_lcr_loss:.4f} \\\\\n")
        f.write(f"Mean & LiteCascade (T) & {mean_lct_f1:.4f} & {mean_lct_rec:.4f} & {mean_lct_loss:.4f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    logger.info("LOFO completed successfully! Exported %s and %s", csv_file, tex_file)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LOFO zero-day experiment")
    parser.add_argument("--dataset", type=str, default="fixture")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epsilon", type=float, default=0.005)
    args = parser.parse_args()

    run_lofo(
        dataset_name=args.dataset,
        epochs=args.epochs,
        seed=args.seed,
        epsilon=args.epsilon,
    )


if __name__ == "__main__":
    main()
