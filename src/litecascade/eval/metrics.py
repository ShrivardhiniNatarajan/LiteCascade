import datetime
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)


def compute_detection_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute standard detection metrics for binary classification."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec_malicious = recall_score(y_true, y_pred, zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall_malicious": float(rec_malicious),
        "recall_macro": float(rec_macro),
        "f1_macro": float(f1_macro),
        "fpr": float(fpr),
        "mcc": float(mcc),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def bootstrap_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_bootstraps: int = 100) -> dict[str, tuple[float, float]]:
    """Compute 95% CI for macro-F1 and malicious recall via bootstrapping."""
    f1_scores = []
    rec_scores = []

    n = len(y_true)
    for _ in range(n_bootstraps):
        idx = np.random.randint(0, n, n)
        y_true_boot = y_true[idx]
        y_pred_boot = y_pred[idx]

        f1_scores.append(f1_score(y_true_boot, y_pred_boot, average="macro", zero_division=0))
        rec_scores.append(recall_score(y_true_boot, y_pred_boot, zero_division=0))

    return {
        "f1_macro_95ci": (
            float(np.percentile(f1_scores, 2.5)),
            float(np.percentile(f1_scores, 97.5)),
        ),
        "recall_malicious_95ci": (
            float(np.percentile(rec_scores, 2.5)),
            float(np.percentile(rec_scores, 97.5)),
        ),
    }


def get_git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
    except Exception:
        return "unknown"


def save_metrics(run_id: str, metrics: dict[str, Any], seed: int, config_hash: str, is_emulated: bool = False):
    """Save metrics to results/<run_id>/metrics.json."""
    out_dir = Path("results") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "run_id": run_id,
        "date": datetime.datetime.now().isoformat(),
        "git_hash": get_git_hash(),
        "seed": seed,
        "config_hash": config_hash,
        "measured_or_emulated": "emulated" if is_emulated else "measured",
        "metrics": metrics,
    }

    out_path = out_dir / "metrics.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=4)

    return out_path
