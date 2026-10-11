"""
Feature-level perturbation engine for evasion emulation and robustness benchmarking (Table VI).

Implements:
1. Gaussian noise: sigma in {0.0, 0.05, 0.10, 0.20} in scaled feature space.
2. Inter-arrival time jitter: perturbs timing/rate features (emulating delay injection).
3. Packet-size padding: shifts size/byte features positively (emulating payload padding).

These perturbations are documented as feature-level EMULATIONS of evasion attacks.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from litecascade.data.dataset import CascadeDataset
from litecascade.eval.metrics import compute_detection_metrics

logger = logging.getLogger(__name__)

# Documented fallback feature column indices when feature names lack semantic labels
TIMING_FALLBACK_INDICES = [0, 1, 2, 3]
SIZE_FALLBACK_INDICES = [4, 5, 6, 7]

TIMING_KEYWORDS = ["time", "interval", "jitter", "dur", "iat", "rate", "weight"]
SIZE_KEYWORDS = ["size", "byte", "bytes", "length", "len", "pkt", "count", "mean"]


def identify_feature_types(
    feature_names: list[str] | None = None,
    total_features: int = 16,
) -> tuple[list[int], list[int], dict[str, Any]]:
    """
    Identifies timing and size feature indices from column names or documented fallbacks.

    Returns:
        (timing_indices, size_indices, documentation_metadata)
    """
    timing_indices: list[int] = []
    size_indices: list[int] = []
    timing_cols: list[str] = []
    size_cols: list[str] = []

    if feature_names:
        for i, name in enumerate(feature_names):
            name_lower = name.lower()
            if any(kw in name_lower for kw in TIMING_KEYWORDS):
                timing_indices.append(i)
                timing_cols.append(name)
            elif any(kw in name_lower for kw in SIZE_KEYWORDS):
                size_indices.append(i)
                size_cols.append(name)

    # Use documented fallback if no columns matched semantically
    if not timing_indices:
        timing_indices = [i for i in TIMING_FALLBACK_INDICES if i < total_features]
        timing_cols = [f"index_{i}" for i in timing_indices]

    if not size_indices:
        size_indices = [i for i in SIZE_FALLBACK_INDICES if i < total_features]
        size_cols = [f"index_{i}" for i in size_indices]

    metadata = {
        "timing_columns": timing_cols,
        "timing_indices": timing_indices,
        "size_columns": size_cols,
        "size_indices": size_indices,
        "emulation_note": "Feature-level EMULATION of network evasion attacks.",
    }
    return timing_indices, size_indices, metadata


def apply_gaussian_noise(
    x: np.ndarray,
    sigma: float = 0.05,
    seed: int = 42,
) -> np.ndarray:
    """
    Applies Gaussian noise N(0, sigma^2) in scaled space.
    When sigma == 0.0, returns exact identical data.
    """
    if sigma == 0.0:
        return x.copy()

    rng = np.random.RandomState(seed)
    noise = rng.normal(loc=0.0, scale=sigma, size=x.shape).astype(x.dtype)
    return x + noise


def apply_time_jitter(
    x: np.ndarray,
    feature_names: list[str] | None = None,
    jitter_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[np.ndarray, list[int]]:
    """
    Emulates inter-arrival time jitter by perturbing designated timing features.
    Leaves non-timing features completely untouched.
    """
    total_features = x.shape[-1]
    timing_indices, _, _ = identify_feature_types(feature_names, total_features)

    x_pert = x.copy()
    if not timing_indices or jitter_ratio == 0.0:
        return x_pert, timing_indices

    rng = np.random.RandomState(seed)
    for col in timing_indices:
        std = float(np.std(x[:, :, col]))
        scale = max(std * jitter_ratio, 1e-4)
        noise = rng.normal(loc=0.0, scale=scale, size=x[:, :, col].shape).astype(x.dtype)
        x_pert[:, :, col] += noise

    return x_pert, timing_indices


def apply_size_padding(
    x: np.ndarray,
    feature_names: list[str] | None = None,
    pad_amount: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, list[int]]:
    """
    Emulates packet-size padding by shifting designated size features positively.
    Leaves non-size features completely untouched.
    """
    total_features = x.shape[-1]
    _, size_indices, _ = identify_feature_types(feature_names, total_features)

    x_pert = x.copy()
    if not size_indices or pad_amount == 0.0:
        return x_pert, size_indices

    for col in size_indices:
        x_pert[:, :, col] += pad_amount

    return x_pert, size_indices


def evaluate_under_perturbation(
    model: nn.Module,
    x_test: np.ndarray,
    y_test: np.ndarray,
    tau_b: float = 0.5,
    tau_m: float = 0.8,
    temperature: float = 1.0,
    batch_size: int = 32,
) -> dict[str, float]:
    """
    Evaluates cascade model on perturbed input windows.
    Returns dictionary with f1_macro, recall, fpr, accuracy, rho, and analyst recall.
    """
    model.eval()
    test_ds = CascadeDataset(x_test, y_test)
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    all_preds: list[int] = []
    analyst_preds: list[int] = []
    all_labels: list[int] = []
    exited_count = 0
    total_count = 0

    with torch.no_grad():
        for xb, yb in loader:
            B = xb.size(0)
            if hasattr(model, "pip") and hasattr(model, "sentinel_fc"):
                # LiteCascade model
                l1, l2 = model(xb, mode="train")
                scaled_l1 = l1 / max(temperature, 1e-6)
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

                all_preds.extend(final_p.cpu().numpy().tolist())
                analyst_preds.extend(p2_preds.cpu().numpy().tolist())
            else:
                # Standard model (Teacher or Baseline)
                logits = model(xb)
                preds = torch.argmax(logits, dim=-1)
                all_preds.extend(preds.cpu().numpy().tolist())
                analyst_preds.extend(preds.cpu().numpy().tolist())

            total_count += B
            all_labels.extend(yb.numpy().tolist())

    y_arr = np.array(all_labels)
    pred_arr = np.array(all_preds)
    analyst_arr = np.array(analyst_preds)

    metrics = compute_detection_metrics(y_arr, pred_arr)
    analyst_metrics = compute_detection_metrics(y_arr, analyst_arr)

    rho = exited_count / total_count if total_count > 0 else 0.0
    rec_val = float(metrics.get("recall_malicious", metrics.get("recall_macro", 0.0)))
    analyst_rec_val = float(analyst_metrics.get("recall_malicious", analyst_metrics.get("recall_macro", 0.0)))

    return {
        "f1_macro": float(metrics["f1_macro"]),
        "recall": rec_val,
        "recall_malicious": rec_val,
        "fpr": float(metrics["fpr"]),
        "accuracy": float(metrics["accuracy"]),
        "rho": float(rho),
        "analyst_recall": analyst_rec_val,
    }
