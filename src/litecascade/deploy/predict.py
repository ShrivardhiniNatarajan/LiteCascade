"""Batch inference on a traffic CSV with a saved bundle (used by the CLI and the web dashboard).

Input CSV: one row per record (flow / packet-window statistics) with the bundle's feature columns.
Optional columns: ``segment`` or ``group`` (separate streams; windows never mix them), ``timestamp``
(ordering), and a label column (``label_binary``, ``label`` or ``Label``; strings starting with "benign"
or the value 0 mean benign) which enables metrics. A prediction is produced for every record that closes a
full window of T records; the first T-1 records of each stream have no prediction.

    python -m litecascade.deploy.predict --bundle artifacts/fixture_R_int8 --csv examples/fixture_sample.csv --out preds.csv
"""

import argparse
import json

import numpy as np
import pandas as pd

from litecascade.data.windows import create_windows
from litecascade.deploy.bundle import Bundle
from litecascade.eval.metrics import compute_detection_metrics

LABEL_CANDIDATES = ["label_binary", "label", "Label", "class"]


def _binary_labels(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return (series.fillna(0) > 0).astype(int)
    s = series.astype(str).str.strip().str.lower()
    return (~(s.str.startswith("benign") | (s == "0") | (s == "normal"))).astype(int)


def prepare_frame(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, dict]:
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV is missing {len(missing)} of the {len(feature_cols)} feature columns this model needs, "
            f"e.g. {missing[:5]}. Upload a CSV with the same feature columns the model was trained on."
        )
    feats = df[feature_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float32")
    n_invalid = int(feats.isna().any(axis=1).sum())
    seg_col = "segment" if "segment" in df.columns else ("group" if "group" in df.columns else None)
    extra = {
        "segment": df[seg_col].astype(str) if seg_col else pd.Series("upload", index=df.index),
        "row_order": pd.Series(np.arange(len(df)), index=df.index),
    }
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        if ts.notna().all():
            extra["timestamp"] = ts
    label_col = next((c for c in LABEL_CANDIDATES if c in df.columns), None)
    if label_col:
        extra["label_binary"] = _binary_labels(df[label_col])
    out = pd.concat([feats, pd.DataFrame(extra, index=df.index)], axis=1)
    return out, {"invalid_rows_imputed": n_invalid, "label_column": label_col, "segment_column": seg_col}


def predict_dataframe(bundle: Bundle, df: pd.DataFrame, batch_size: int = 512) -> tuple[pd.DataFrame, dict]:
    frame, info = prepare_frame(df, bundle.feature_cols)
    T = bundle.window
    has_labels = "label_binary" in frame.columns
    X, y, meta = create_windows(
        frame, bundle.feature_cols, T=T, label_col="label_binary" if has_labels else "_none_", return_meta=True
    )
    if len(X) == 0:
        raise ValueError(f"No complete window: every stream needs at least T={T} records.")
    X = np.nan_to_num(bundle.preprocessor.scale_array(X), nan=0.0)  # NaN -> 0 = train median after RobustScaler

    out = bundle.predict_windows(X, batch_size=batch_size)
    preds, probs, elapsed = out["preds"], out["probs"], out["seconds"]
    cascade = bundle.kind == "cascade"
    exits = out["exit_mask"] if cascade else np.zeros(len(preds), bool)
    p1s = out["sentinel_probs"] if cascade else probs
    n_analyst = out["n_analyst"] if cascade else 0

    names = bundle.class_names
    res = pd.DataFrame(
        {
            "row": meta["end_row"].to_numpy(),
            "segment": meta["segment"].to_numpy(),
            "prediction": np.where(preds > 0, "malicious", "benign"),
            "predicted_class": [names[i] for i in preds],
            "malicious_score": 1.0 - probs[:, 0],
            "exit_stage": np.where(exits, "sentinel", "analyst") if cascade else "single",
            "sentinel_benign_prob": p1s[:, 0],
        }
    )
    if has_labels:
        res["true_label"] = np.where(y > 0, "malicious", "benign")
        res["correct"] = (preds > 0).astype(int) == y

    n = len(res)
    summary = {
        "model": bundle.name,
        "is_synthetic_model": bool(bundle.meta.get("is_synthetic")),
        "dataset": bundle.meta.get("dataset"),
        "quantization": bundle.meta.get("quantization"),
        "records": int(len(df)),
        "windows": int(n),
        "window_length": T,
        "segments": int(res["segment"].nunique()),
        "predicted_malicious": int((preds > 0).sum()),
        "predicted_benign": int((preds == 0).sum()),
        "model_kind": bundle.kind,
        "display_name": bundle.display_name,
        "exit_rate": float(exits.mean()),
        "sentinel_exits": int(exits.sum()),
        "analyst_invocations": int(n_analyst),
        "inference_ms_total": elapsed * 1000,
        "inference_ms_per_window": elapsed * 1000 / n,
        "gate": {k: bundle.gate.get(k) for k in ("tau_b", "tau_m", "T1", "T2", "epsilon", "alpha", "feasible", "verify_alerts")}
        if cascade
        else None,
        **info,
    }
    if has_labels:
        summary["metrics"] = compute_detection_metrics(y, (preds > 0).astype(int))
    return res, summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="predictions.csv")
    args = ap.parse_args()
    res, summary = predict_dataframe(Bundle(args.bundle), pd.read_csv(args.csv))
    res.to_csv(args.out, index=False)
    print(json.dumps(summary, indent=2, default=float))
    print(f"Wrote {len(res)} window predictions to {args.out}")


if __name__ == "__main__":
    main()
