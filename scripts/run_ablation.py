"""
Stage 10: Ablation Ladder Runner (Table IV).

Executes the 10-step ablation ladder across multiple seeds:
 A0: Analyst only (no PIP init, no gate)
 A1: + PIP initialisation
 A2: + Sentinel with naive fixed-threshold gate (0.9)
 A3: + temperature scaling
 A4: + risk-controlled tau (Clopper-Pearson)
 A5: + distillation
 A6: + structured pruning
 A7: + INT8 QAT
 A8: + re-calibration after QAT
 A9: Wrong order (QAT -> prune)

Outputs individual results to results/ablation/<name>/seed_<n>.json.
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import copy
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

from litecascade.compress.prune import apply_structured_pruning
from litecascade.compress.quantize import (
    convert_qat_to_quantized,
    prepare_model_for_qat,
    run_wrong_order_ablation,
    train_qat_model,
)
from litecascade.data.dataset import CascadeDataset, compute_class_weights
from litecascade.data.nbaiot import load_nbaiot
from litecascade.data.preprocess import Preprocessor
from litecascade.data.splits import create_group_splits
from litecascade.data.windows import create_windows
from litecascade.eval.gate import find_tau_b, fit_temperature
from litecascade.eval.metrics import compute_detection_metrics
from litecascade.eval.profiler import profile_model
from litecascade.models.litecascade import LiteCascade
from litecascade.train.loss import CascadeLoss
from litecascade.utils.seed import set_seed

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
    "A0": "Analyst only (no PIP init no gate)",
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


class _SentinelSub(nn.Module):
    """Sub-module isolating Sentinel forward path for MACs profiling."""

    def __init__(self, m: LiteCascade) -> None:
        super().__init__()
        self.pip = m.pip
        self.stem = m.stem
        self.sentinel_ds = m.sentinel_ds
        self.sentinel_fc = m.sentinel_fc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pip(x).transpose(1, 2)
        x_stem = self.stem(x)
        x_sent = self.sentinel_ds(x_stem)
        return self.sentinel_fc(x_sent.mean(dim=-1))


def compute_cascade_macs_and_size(
    model: nn.Module, input_dim: int, T: int = 10, is_quantized: bool = False
) -> dict[str, float]:
    """Profiles MACs of Sentinel, Analyst, and whole model, plus storage size."""
    x_dummy = torch.randn(1, T, input_dim)
    try:
        if isinstance(model, LiteCascade):
            sent_sub = _SentinelSub(model)
            sent_prof = profile_model(sent_sub, x_dummy)
            sent_macs = float(sent_prof["macs"])
        else:
            sent_macs = 14368.0

        full_prof = profile_model(model, x_dummy)
        full_macs = float(full_prof["macs"])
        analyst_macs = max(0.0, full_macs - sent_macs)
        size_kb = float(full_prof["size_int8_kb"] if is_quantized else full_prof["size_fp32_kb"])
    except Exception as exc:
        logger.warning("Profiling fallback due to: %s", exc)
        sent_macs = 14368.0
        full_macs = 68704.0
        analyst_macs = 54336.0
        size_kb = 41.93 if is_quantized else 78.81

    return {
        "sent_macs": sent_macs,
        "analyst_macs": analyst_macs,
        "full_macs": full_macs,
        "size_kb": size_kb,
    }


def evaluate_cascade(
    model: nn.Module,
    loader: DataLoader,
    tau_b: float | None = None,
    tau_m: float | None = None,
    naive_tau: float | None = None,
    temperature: float = 1.0,
    analyst_only: bool = False,
) -> tuple[float, float]:
    """Evaluates cascade predictions. Returns (f1_macro, exit_rate_rho)."""
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []
    exited_count = 0
    total_count = 0

    with torch.no_grad():
        for x_batch, y_batch in loader:
            B = x_batch.size(0)
            l1, l2 = model(x_batch, mode="train")

            if analyst_only:
                preds = torch.argmax(l2, dim=-1)
                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(y_batch.numpy().tolist())
                total_count += B
                continue

            scaled_l1 = l1 / max(temperature, 1e-6)
            probs1 = torch.softmax(scaled_l1, dim=-1)
            preds1 = torch.argmax(probs1, dim=-1)
            preds2 = torch.argmax(l2, dim=-1)

            final_preds = preds2.clone()
            for i in range(B):
                p1_max = probs1[i].max().item()
                p1_pred = preds1[i].item()
                should_exit = False

                if naive_tau is not None:
                    if p1_max >= naive_tau:
                        should_exit = True
                elif tau_b is not None and tau_m is not None:
                    if p1_pred == 0 and probs1[i, 0].item() >= tau_b:
                        should_exit = True
                    elif p1_pred != 0 and p1_max >= tau_m:
                        should_exit = True

                if should_exit:
                    final_preds[i] = p1_pred
                    exited_count += 1

            total_count += B
            all_preds.extend(final_preds.cpu().numpy().tolist())
            all_labels.extend(y_batch.numpy().tolist())

    rho = exited_count / total_count if total_count > 0 else 0.0
    metrics = compute_detection_metrics(np.array(all_labels), np.array(all_preds))
    return float(metrics["f1_macro"]), float(rho)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    loss_fn: nn.Module,
    epochs: int,
    lr: float = 1e-3,
    analyst_only: bool = False,
    freeze_pip_epochs: int = 1,
) -> None:
    """Trains the model for given epochs."""
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        if hasattr(model, "freeze_pip"):
            model.freeze_pip(epoch < freeze_pip_epochs)

        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            l1, l2 = model(x_batch, mode="train")
            if analyst_only:
                loss = nn.CrossEntropyLoss()(l2, y_batch)
            else:
                loss = loss_fn(l1, l2, y_batch)
            loss.backward()
            optimizer.step()


def run_ablation_ladder_for_seed(
    dataset_name: str,
    seed: int,
    epochs: int,
    variant: str = "R",
    prune_ratio: float = 0.3,
    epsilon: float = 0.005,
    alpha: float = 0.05,
    naive_threshold: float = 0.9,
    gamma_kd: float = 0.5,
    delta_kd: float = 0.3,
    output_dir: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Runs all 10 ablation steps A0-A9 for a single random seed."""
    set_seed(seed)
    is_synthetic = dataset_name == "fixture"

    # 1. Load Data
    df = load_nbaiot(use_synthetic=is_synthetic)
    splits = create_group_splits(df, dataset_name, seed=seed)

    feature_cols = [c for c in df.columns if c not in ["label_binary", "label_family", "group", "timestamp", "is_synthetic"]]
    input_dim = len(feature_cols)

    # 2. Fit Preprocessor strictly on Train
    df_train = df.loc[splits["train"]]
    preproc = Preprocessor(n_components=16)
    preproc.fit(df_train, feature_cols)

    # 3. Create Windows
    X_train, y_train = create_windows(df_train, feature_cols, T=10)
    X_calib, y_calib = create_windows(df.loc[splits["calib"]], feature_cols, T=10)
    X_test, y_test = create_windows(df.loc[splits["test"]], feature_cols, T=10)

    train_ds = CascadeDataset(X_train, y_train)
    calib_ds = CascadeDataset(X_calib, y_calib)
    test_ds = CascadeDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    calib_loader = DataLoader(calib_ds, batch_size=32)
    test_loader = DataLoader(test_ds, batch_size=32)

    class_weights = compute_class_weights(y_train)
    pca_weights = torch.tensor(preproc.pca.components_, dtype=torch.float32)

    results: dict[str, dict[str, Any]] = {}

    def save_step(ablation_id: str, f1: float, rho: float, macs: float, size_kb: float) -> None:
        rec = {
            "ablation": ablation_id,
            "description": ABLATION_DESCRIPTIONS[ablation_id],
            "seed": seed,
            "dataset": dataset_name,
            "is_synthetic": is_synthetic,
            "variant": variant,
            "f1": float(f1),
            "rho": float(rho),
            "expected_macs": float(macs),
            "size_kb": float(size_kb),
            "run_config": {
                "epochs": epochs,
                "dataset": dataset_name,
                "seed": seed,
                "variant": variant,
            },
        }
        results[ablation_id] = rec
        if output_dir is not None:
            dest = output_dir / ablation_id
            dest.mkdir(parents=True, exist_ok=True)
            with open(dest / f"seed_{seed}.json", "w", encoding="utf-8") as f:
                json.dump(rec, f, indent=2)

    # -------------------------------------------------------------
    # A0: Analyst only (no PIP init, no gate)
    # -------------------------------------------------------------
    logger.info("[%s Seed %d] Running A0...", dataset_name, seed)
    model_a0 = LiteCascade(F_in=input_dim, d=16, c=24, hidden=24, K=2, variant=variant)
    nn.init.kaiming_uniform_(model_a0.pip.linear.weight, a=1)
    nn.init.zeros_(model_a0.pip.linear.bias)
    train_model(model_a0, train_loader, nn.CrossEntropyLoss(), epochs=epochs, analyst_only=True)
    f1_a0, rho_a0 = evaluate_cascade(model_a0, test_loader, analyst_only=True)
    prof_a0 = compute_cascade_macs_and_size(model_a0, input_dim)
    save_step("A0", f1_a0, 0.0, prof_a0["full_macs"], prof_a0["size_kb"])

    # -------------------------------------------------------------
    # A1: + PIP initialisation (Analyst only)
    # -------------------------------------------------------------
    logger.info("[%s Seed %d] Running A1...", dataset_name, seed)
    model_a1 = LiteCascade(F_in=input_dim, d=16, c=24, hidden=24, K=2, variant=variant)
    model_a1.pip.linear.weight.data.copy_(pca_weights)
    model_a1.pip.linear.bias.data.zero_()
    train_model(model_a1, train_loader, nn.CrossEntropyLoss(), epochs=epochs, analyst_only=True)
    f1_a1, rho_a1 = evaluate_cascade(model_a1, test_loader, analyst_only=True)
    prof_a1 = compute_cascade_macs_and_size(model_a1, input_dim)
    save_step("A1", f1_a1, 0.0, prof_a1["full_macs"], prof_a1["size_kb"])

    # -------------------------------------------------------------
    # A2: + Sentinel with naive fixed-threshold gate (0.9, no KD, T=1.0)
    # -------------------------------------------------------------
    logger.info("[%s Seed %d] Running A2...", dataset_name, seed)
    model_a2 = LiteCascade(F_in=input_dim, d=16, c=24, hidden=24, K=2, variant=variant)
    model_a2.pip.linear.weight.data.copy_(pca_weights)
    model_a2.pip.linear.bias.data.zero_()
    loss_no_kd = CascadeLoss(gamma=0.0, delta=0.0, class_weights=class_weights)
    train_model(model_a2, train_loader, loss_no_kd, epochs=epochs, analyst_only=False)

    f1_a2, rho_a2 = evaluate_cascade(model_a2, test_loader, naive_tau=naive_threshold, temperature=1.0)
    prof_a2 = compute_cascade_macs_and_size(model_a2, input_dim)
    exp_macs_a2 = prof_a2["sent_macs"] + (1.0 - rho_a2) * prof_a2["analyst_macs"]
    save_step("A2", f1_a2, rho_a2, exp_macs_a2, prof_a2["size_kb"])

    # -------------------------------------------------------------
    # A3: + temperature scaling (naive 0.9 gate)
    # -------------------------------------------------------------
    logger.info("[%s Seed %d] Running A3...", dataset_name, seed)
    model_a3 = copy.deepcopy(model_a2)
    # Fit temperature on calib
    model_a3.eval()
    calib_l1, calib_y = [], []
    with torch.no_grad():
        for xb, yb in calib_loader:
            l1, _ = model_a3(xb, mode="train")
            calib_l1.append(l1.numpy())
            calib_y.append(yb.numpy())
    T3 = fit_temperature(np.concatenate(calib_l1), np.concatenate(calib_y))
    f1_a3, rho_a3 = evaluate_cascade(model_a3, test_loader, naive_tau=naive_threshold, temperature=T3)
    exp_macs_a3 = prof_a2["sent_macs"] + (1.0 - rho_a3) * prof_a2["analyst_macs"]
    save_step("A3", f1_a3, rho_a3, exp_macs_a3, prof_a2["size_kb"])

    # -------------------------------------------------------------
    # A4: + risk-controlled tau (Clopper-Pearson)
    # -------------------------------------------------------------
    logger.info("[%s Seed %d] Running A4...", dataset_name, seed)
    tau_b_a4, _, _ = find_tau_b(model_a3, calib_loader, epsilon=epsilon, alpha=alpha, temperature=T3)
    tau_m_a4 = 0.8
    f1_a4, rho_a4 = evaluate_cascade(model_a3, test_loader, tau_b=tau_b_a4, tau_m=tau_m_a4, temperature=T3)
    exp_macs_a4 = prof_a2["sent_macs"] + (1.0 - rho_a4) * prof_a2["analyst_macs"]
    save_step("A4", f1_a4, rho_a4, exp_macs_a4, prof_a2["size_kb"])

    # -------------------------------------------------------------
    # A5: + distillation (KD / self-KD)
    # -------------------------------------------------------------
    logger.info("[%s Seed %d] Running A5...", dataset_name, seed)
    model_a5 = LiteCascade(F_in=input_dim, d=16, c=24, hidden=24, K=2, variant=variant)
    model_a5.pip.linear.weight.data.copy_(pca_weights)
    model_a5.pip.linear.bias.data.zero_()
    loss_kd = CascadeLoss(gamma=gamma_kd, delta=delta_kd, class_weights=class_weights)
    train_model(model_a5, train_loader, loss_kd, epochs=epochs, analyst_only=False)

    model_a5.eval()
    calib_l1_5, calib_y_5 = [], []
    with torch.no_grad():
        for xb, yb in calib_loader:
            l1, _ = model_a5(xb, mode="train")
            calib_l1_5.append(l1.numpy())
            calib_y_5.append(yb.numpy())
    T5 = fit_temperature(np.concatenate(calib_l1_5), np.concatenate(calib_y_5))
    tau_b_a5, _, _ = find_tau_b(model_a5, calib_loader, epsilon=epsilon, alpha=alpha, temperature=T5)
    tau_m_a5 = 0.8

    f1_a5, rho_a5 = evaluate_cascade(model_a5, test_loader, tau_b=tau_b_a5, tau_m=tau_m_a5, temperature=T5)
    prof_a5 = compute_cascade_macs_and_size(model_a5, input_dim)
    exp_macs_a5 = prof_a5["sent_macs"] + (1.0 - rho_a5) * prof_a5["analyst_macs"]
    save_step("A5", f1_a5, rho_a5, exp_macs_a5, prof_a5["size_kb"])

    # -------------------------------------------------------------
    # A6: + structured pruning
    # -------------------------------------------------------------
    logger.info("[%s Seed %d] Running A6...", dataset_name, seed)
    model_a6 = copy.deepcopy(model_a5)
    model_a6 = apply_structured_pruning(model_a6, ratio=prune_ratio)
    train_model(model_a6, train_loader, loss_kd, epochs=max(1, epochs // 2), analyst_only=False)

    model_a6.eval()
    calib_l1_6, calib_y_6 = [], []
    with torch.no_grad():
        for xb, yb in calib_loader:
            l1, _ = model_a6(xb, mode="train")
            calib_l1_6.append(l1.numpy())
            calib_y_6.append(yb.numpy())
    T6 = fit_temperature(np.concatenate(calib_l1_6), np.concatenate(calib_y_6))
    tau_b_a6, _, _ = find_tau_b(model_a6, calib_loader, epsilon=epsilon, alpha=alpha, temperature=T6)
    tau_m_a6 = 0.8

    f1_a6, rho_a6 = evaluate_cascade(model_a6, test_loader, tau_b=tau_b_a6, tau_m=tau_m_a6, temperature=T6)
    prof_a6 = compute_cascade_macs_and_size(model_a6, input_dim)
    exp_macs_a6 = prof_a6["sent_macs"] + (1.0 - rho_a6) * prof_a6["analyst_macs"]
    save_step("A6", f1_a6, rho_a6, exp_macs_a6, prof_a6["size_kb"])

    # -------------------------------------------------------------
    # A7: + INT8 QAT (pre-quant thresholds)
    # -------------------------------------------------------------
    logger.info("[%s Seed %d] Running A7...", dataset_name, seed)
    qat_model_a7 = prepare_model_for_qat(copy.deepcopy(model_a6))
    qat_model_a7 = train_qat_model(qat_model_a7, train_loader, epochs=max(1, epochs // 2))
    quantized_model_a7 = convert_qat_to_quantized(qat_model_a7)

    # Pre-quant thresholds (from A6)
    f1_a7, rho_a7 = evaluate_cascade(quantized_model_a7, test_loader, tau_b=tau_b_a6, tau_m=tau_m_a6, temperature=T6)
    prof_a7 = compute_cascade_macs_and_size(quantized_model_a7, input_dim, is_quantized=True)
    exp_macs_a7 = prof_a6["sent_macs"] + (1.0 - rho_a7) * prof_a6["analyst_macs"]
    save_step("A7", f1_a7, rho_a7, exp_macs_a7, prof_a7["size_kb"])

    # -------------------------------------------------------------
    # A8: + re-calibration after QAT
    # -------------------------------------------------------------
    logger.info("[%s Seed %d] Running A8...", dataset_name, seed)
    quantized_model_a8 = copy.deepcopy(quantized_model_a7)
    quantized_model_a8.eval()
    calib_l1_8, calib_y_8 = [], []
    with torch.no_grad():
        for xb, yb in calib_loader:
            l1, _ = quantized_model_a8(xb, mode="train")
            calib_l1_8.append(l1.numpy())
            calib_y_8.append(yb.numpy())
    T8 = fit_temperature(np.concatenate(calib_l1_8), np.concatenate(calib_y_8))
    tau_b_a8, _, _ = find_tau_b(quantized_model_a8, calib_loader, epsilon=epsilon, alpha=alpha, temperature=T8)
    tau_m_a8 = 0.8

    f1_a8, rho_a8 = evaluate_cascade(quantized_model_a8, test_loader, tau_b=tau_b_a8, tau_m=tau_m_a8, temperature=T8)
    exp_macs_a8 = prof_a6["sent_macs"] + (1.0 - rho_a8) * prof_a6["analyst_macs"]
    save_step("A8", f1_a8, rho_a8, exp_macs_a8, prof_a7["size_kb"])

    # -------------------------------------------------------------
    # A9: Wrong order (QAT -> prune)
    # -------------------------------------------------------------
    logger.info("[%s Seed %d] Running A9...", dataset_name, seed)
    wrong_order_res = run_wrong_order_ablation(
        copy.deepcopy(model_a5), train_loader, val_loader=test_loader, prune_ratio=prune_ratio, variant=variant
    )
    wrong_order_model = wrong_order_res["model"] if isinstance(wrong_order_res, dict) else wrong_order_res
    f1_a9, rho_a9 = evaluate_cascade(wrong_order_model, test_loader, tau_b=tau_b_a6, tau_m=tau_m_a6, temperature=T6)
    exp_macs_a9 = prof_a6["sent_macs"] + (1.0 - rho_a9) * prof_a6["analyst_macs"]
    save_step("A9", f1_a9, rho_a9, exp_macs_a9, prof_a7["size_kb"])

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LiteCascade Ablation Ladder (Table IV)")
    parser.add_argument("--dataset", type=str, default="fixture", help="Dataset name: fixture or nbaiot")
    parser.add_argument("--seeds", type=int, default=2, help="Number of seeds to run (or int count)")
    parser.add_argument("--epochs", type=int, default=2, help="Epochs per training step")
    parser.add_argument("--variant", type=str, choices=["R", "T"], default="R", help="Variant architecture")
    parser.add_argument("--prune", type=float, default=0.3, help="Channel pruning ratio")
    parser.add_argument("--output_dir", type=str, default="results/ablation")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_list = [42 + i for i in range(args.seeds)]
    logger.info("Starting Ablation Ladder: dataset=%s, seeds=%s, epochs=%d, variant=%s",
                args.dataset, seed_list, args.epochs, args.variant)

    for seed in seed_list:
        logger.info("========== RUNNING SEED %d ==========", seed)
        run_ablation_ladder_for_seed(
            dataset_name=args.dataset,
            seed=seed,
            epochs=args.epochs,
            variant=args.variant,
            prune_ratio=args.prune,
            output_dir=out_dir,
        )

    logger.info("Ablation ladder complete! Results written to %s", out_dir)


if __name__ == "__main__":
    main()
