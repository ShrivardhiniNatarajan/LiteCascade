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
    """Detection metrics. Class 0 is benign; any class > 0 is malicious.

    Binary detection metrics (precision, malicious recall, FPR, MCC, confusion counts) are computed on
    (label > 0). ``f1_macro`` / ``recall_macro`` / ``accuracy`` use the full label set, so for K > 2 they
    are multi-class scores; for K = 2 everything coincides with the usual binary definitions.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    if len(y_true) == 0:
        return {k: 0.0 for k in ("accuracy", "precision", "recall_malicious", "recall_macro", "f1_macro", "fpr", "mcc")} | {
            "tn": 0, "fp": 0, "fn": 0, "tp": 0
        }  # fmt: skip
    yt_bin = (y_true > 0).astype(int)
    yp_bin = (y_pred > 0).astype(int)
    cm = confusion_matrix(yt_bin, yp_bin, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(yt_bin, yp_bin, zero_division=0)
    rec_malicious = recall_score(yt_bin, yp_bin, zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    mcc = matthews_corrcoef(yt_bin, yp_bin) if len(np.unique(yt_bin)) > 1 or len(np.unique(yp_bin)) > 1 else 0.0

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


def bootstrap_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, n_bootstraps: int = 200, seed: int = 0
) -> dict[str, tuple[float, float]]:
    """Compute 95% CI for macro-F1 and malicious recall via bootstrapping."""
    f1_scores = []
    rec_scores = []

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    rng = np.random.default_rng(seed)
    for _ in range(n_bootstraps):
        idx = rng.integers(0, n, n)
        y_true_boot = y_true[idx]
        y_pred_boot = y_pred[idx]

        f1_scores.append(f1_score(y_true_boot, y_pred_boot, average="macro", zero_division=0))
        rec_scores.append(recall_score(y_true_boot > 0, y_pred_boot > 0, zero_division=0))

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


def save_metrics(
    run_id: str, metrics: dict[str, Any], seed: int, config_hash: str, is_emulated: bool = False, out_root: str = "results"
):
    """Save metrics to <out_root>/<run_id>/metrics.json.

    ``is_emulated`` marks runs on the SYNTHETIC fixture ("synthetic_fixture") as opposed to a real dataset.
    """
    out_dir = Path(out_root) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "run_id": run_id,
        "date": datetime.datetime.now().isoformat(),
        "git_hash": get_git_hash(),
        "seed": seed,
        "config_hash": config_hash,
        "data": "synthetic_fixture" if is_emulated else "real_dataset",
        "metrics": metrics,
    }

    out_path = out_dir / "metrics.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=4, default=_json_default)

    return out_path


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
