"""Self-contained model bundles: weights + architecture + train-fitted preprocessor + metadata.

    artifacts/<name>/
        bundle.json         kind (cascade | baseline), architecture, quantization, gate, features, classes, metrics
        model.pt            state_dict (float or INT8)
        preprocessor.joblib train-fitted RobustScaler (+ PCA)

Every bundle exposes the same ``predict_windows`` call, so the CLI, the web API and the live replay can run
LiteCascade and the baseline deep-learning models side by side.
Bundles are pickled sklearn objects / torch state dicts: only load bundles you produced yourself.
"""

import datetime
import json
import time
from pathlib import Path

import joblib
import numpy as np
import torch

from litecascade.compress.quantize import build_int8_skeleton, quantize_dynamic_int8
from litecascade.models.litecascade import LiteCascade

FORMAT_VERSION = 2

# Human-readable names / families for the model zoo (used by the dashboard and reports).
MODEL_INFO = {
    "teacher": ("Teacher CNN-BiLSTM", "hybrid", "2x Conv1d + 2-layer BiLSTM(128) + attention pooling; distillation source"),
    "b1": ("B1 L1-MLP (SNIPE)", "mlp", "MLP on the flattened window with L1 regularisation"),
    "b3": ("B3 LSTM", "recurrent", "single-layer LSTM, last time step"),
    "b4": ("B4 BiLSTM", "recurrent", "bidirectional LSTM, last time step"),
    "b5": ("B5 PCA-BiLSTM INT8", "recurrent", "BiLSTM on 32 PCA components, dynamic INT8 (Wang et al.)"),
    "b6": ("B6 CNN-BiLSTM", "hybrid", "Conv1d + max-pool + BiLSTM (Jouhari & Guizani)"),
    "b7": ("B7 DS-CNN", "convolutional", "depthwise-separable Conv1d + global average pooling"),
    "b8": ("B8 Tiny Transformer", "attention", "2-layer, 2-head Transformer encoder, mean pooling"),
    "b9": ("B9 BiGRU", "recurrent", "bidirectional GRU, mean pooling"),
    "b10": ("B10 TCN", "convolutional", "3 residual dilated causal conv blocks (dilation 1, 2, 4)"),
    "b11": ("B11 ResNet-1D", "convolutional", "2 residual Conv1d stages + global average pooling"),
}


def _base_meta(data, metrics, extra):
    return {
        "format_version": FORMAT_VERSION,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "feature_cols": data.feature_cols,
        "window": data.T,
        "class_names": data.class_names,
        "task": data.task,
        "dataset": data.dataset,
        "is_synthetic": data.is_synthetic,
        "metrics": metrics or {},
        **(extra or {}),
    }


def save_bundle(
    out_dir, model, data, gate: dict, quantization: str = "none", metrics: dict | None = None, extra: dict | None = None
):
    """LiteCascade bundle (cascade with calibrated gate)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")
    joblib.dump(data.preprocessor, out / "preprocessor.joblib")
    meta = {"kind": "cascade", "model_config": model.config(), "quantization": quantization, "gate": gate}
    meta.update(_base_meta(data, metrics, extra))
    (out / "bundle.json").write_text(json.dumps(meta, indent=2, default=float))
    return out


def save_baseline_bundle(out_dir, arch: str, model, data, metrics=None, input_kind="scaled", quantization="none", extra=None):
    """Single-exit deep-learning baseline bundle. ``input_kind="pca"`` feeds PCA scores (B5)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")
    joblib.dump(data.preprocessor, out / "preprocessor.joblib")
    input_dim = data.preprocessor.pca.n_components_ if input_kind == "pca" else data.n_features
    if arch == "b1":
        input_dim *= data.T
    meta = {
        "kind": "baseline",
        "arch": arch,
        "input_dim": int(input_dim),
        "input_kind": input_kind,
        "quantization": quantization,
        "gate": None,
    }
    meta.update(_base_meta(data, metrics, extra))
    (out / "bundle.json").write_text(json.dumps(meta, indent=2, default=float))
    return out


class Bundle:
    def __init__(self, path):
        self.path = Path(path)
        self.meta = json.loads((self.path / "bundle.json").read_text())
        self.kind = self.meta.get("kind", "cascade")
        q = self.meta.get("quantization", "none")
        state = torch.load(self.path / "model.pt", map_location="cpu", weights_only=False)
        if self.kind == "cascade":
            cfg = self.meta["model_config"]
            model = LiteCascade(**cfg) if q == "none" else build_int8_skeleton(cfg, q)
            model.load_state_dict(state)
            model.verify_alerts = bool(self.meta["gate"].get("verify_alerts", False))
        else:
            from litecascade.train.run import get_model

            model = get_model(self.meta["arch"], self.meta["input_dim"], len(self.meta["class_names"]))
            model.load_state_dict(state)
            model.eval()
            if q == "dynamic":
                model = quantize_dynamic_int8(model)
        model.eval()
        self.model = model
        self.preprocessor = joblib.load(self.path / "preprocessor.joblib")

    # ------------------------------------------------------------------ metadata
    @property
    def name(self) -> str:
        return self.path.name

    @property
    def gate(self) -> dict | None:
        return self.meta.get("gate")

    @property
    def feature_cols(self) -> list[str]:
        return self.meta["feature_cols"]

    @property
    def window(self) -> int:
        return int(self.meta["window"])

    @property
    def class_names(self) -> list[str]:
        return self.meta["class_names"]

    @property
    def arch(self) -> str:
        if self.kind == "cascade":
            return "litecascade_" + self.meta["model_config"]["variant"]
        return self.meta["arch"]

    @property
    def display_name(self) -> str:
        if self.kind == "cascade":
            q = self.meta.get("quantization", "none")
            return f"LiteCascade-{self.meta['model_config']['variant']}" + (" INT8" if q != "none" else " FP32")
        return MODEL_INFO.get(self.arch, (self.arch,))[0]

    @property
    def family(self) -> str:
        return "cascade" if self.kind == "cascade" else MODEL_INFO.get(self.arch, ("", "other"))[1]

    @property
    def description(self) -> str:
        if self.kind == "cascade":
            v = self.meta["model_config"]["variant"]
            return (
                "PCA-initialised projection + depthwise-separable stem; Sentinel exit with a Clopper-Pearson gate; "
                f"Analyst = 2 residual DS-TCN blocks{' + BiGRU' if v == 'R' else ''} + attention pooling"
            )
        return MODEL_INFO.get(self.arch, ("", "", ""))[2]

    def n_params(self) -> int | None:
        prof = self.meta.get("metrics", {}).get("profile", {})
        if isinstance(prof.get("params"), dict):
            return prof["params"].get("total")
        return self.meta.get("metrics", {}).get("params")

    # ------------------------------------------------------------------ inference
    def prepare_input(self, X_scaled: np.ndarray) -> np.ndarray:
        if self.kind == "baseline" and self.meta.get("input_kind") == "pca":
            return self.preprocessor.pca_array(X_scaled)
        return X_scaled

    @torch.no_grad()
    def predict_windows(self, X_scaled: np.ndarray, batch_size: int = 512) -> dict:
        """Run the model on scaled windows (N, T, F). For LiteCascade this is genuine cascade inference."""
        X = self.prepare_input(np.asarray(X_scaled, dtype=np.float32))
        preds, probs, p1s, exits, n_analyst = [], [], [], [], 0
        t0 = time.perf_counter()
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(np.ascontiguousarray(X[i : i + batch_size]))
            if self.kind == "cascade":
                out = self.model.cascade_predict(xb, self.gate["tau_b"], self.gate["tau_m"])
                preds.append(out["preds"].numpy())
                probs.append(out["probs"].numpy())
                p1s.append(out["sentinel_probs"].numpy())
                exits.append(out["exit_mask"].numpy())
                n_analyst += out["n_analyst"]
            else:
                p = torch.softmax(self.model(xb).float(), dim=-1)
                preds.append(p.argmax(-1).numpy())
                probs.append(p.numpy())
        elapsed = time.perf_counter() - t0
        out = {
            "preds": np.concatenate(preds) if preds else np.zeros(0, int),
            "probs": np.concatenate(probs) if probs else np.zeros((0, len(self.class_names))),
            "seconds": elapsed,
        }
        if self.kind == "cascade":
            out["sentinel_probs"] = np.concatenate(p1s) if p1s else out["probs"]
            out["exit_mask"] = np.concatenate(exits) if exits else np.zeros(0, bool)
            out["n_analyst"] = n_analyst
        return out


def list_bundles(root="artifacts") -> list[Path]:
    root = Path(root)
    if not root.exists():
        return []
    return sorted(p.parent for p in root.glob("*/bundle.json"))
