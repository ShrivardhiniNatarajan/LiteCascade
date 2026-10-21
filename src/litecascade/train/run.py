"""Train / evaluate / profile one baseline (B1-B11) or the teacher on the shared leakage-safe data pipeline.

    python -m litecascade.train.run --model b3 --dataset fixture --epochs 5
    python -m litecascade.train.run --model dl --dataset fixture --epochs 8 --bundles artifacts

``--model dl`` trains every deep-learning model (all but the B2 tree ensemble) and ``--bundles`` saves each one
as a loadable bundle for the dashboard. B5 follows its source design: PCA-reduced inputs (fit on train) +
BiLSTM, evaluated after torch dynamic INT8 quantization. B2 is a Random Forest (``--tree xgb`` for XGBoost)
on flattened windows.
"""

import argparse
import io
import logging
import pickle
import time
import uuid
from pathlib import Path

import numpy as np
import torch

from litecascade.data.pipeline import DATASETS, prepare_data
from litecascade.eval.metrics import bootstrap_metrics, compute_detection_metrics, save_metrics
from litecascade.eval.profiler import profile_model
from litecascade.models.baselines import (
    B1_MLP,
    B3_LSTM,
    B7_DSCNN,
    B10_TCN,
    B2_Trees,
    B4_BiLSTM,
    B5_WangBiLSTM,
    B6_CNN_BiLSTM,
    B8_Transformer,
    B9_BiGRU,
    B11_ResNet1D,
)
from litecascade.train.cascade_trainer import make_loader
from litecascade.train.trainer import UnifiedTrainer
from litecascade.utils.seed import set_seed

logger = logging.getLogger(__name__)

ALL_MODELS = ["teacher", "b1", "b2", "b3", "b4", "b5", "b6", "b7", "b8", "b9", "b10", "b11"]
DL_MODELS = [m for m in ALL_MODELS if m != "b2"]


def get_model(model_name: str, input_dim: int, num_classes: int = 2, tree: str = "rf", seed: int = 42):
    name = model_name.lower()
    if name == "b1":
        return B1_MLP(input_dim, num_classes=num_classes)
    if name == "b2":
        return B2_Trees(model_type=tree, random_state=seed)
    if name == "b3":
        return B3_LSTM(input_dim, num_classes=num_classes)
    if name == "b4":
        return B4_BiLSTM(input_dim, num_classes=num_classes)
    if name == "b5":
        return B5_WangBiLSTM(input_dim, num_classes=num_classes)
    if name == "b6":
        return B6_CNN_BiLSTM(input_dim, num_classes=num_classes)
    if name == "b7":
        return B7_DSCNN(input_dim, num_classes=num_classes)
    if name == "b8":
        return B8_Transformer(input_dim, num_classes=num_classes)
    if name == "b9":
        return B9_BiGRU(input_dim, num_classes=num_classes)
    if name == "b10":
        return B10_TCN(input_dim, num_classes=num_classes)
    if name == "b11":
        return B11_ResNet1D(input_dim, num_classes=num_classes)
    if name == "teacher":
        from litecascade.models.teacher import TeacherCNNBiLSTM

        return TeacherCNNBiLSTM(input_dim, num_classes=num_classes)
    raise ValueError(f"Unknown baseline: {model_name}")


def _profile_tree(model: B2_Trees, x_flat: np.ndarray, runs: int = 200) -> dict:
    est = model.model
    if hasattr(est, "estimators_"):
        nodes = int(sum(t.tree_.node_count for t in est.estimators_))
    else:
        nodes = int(est.get_booster().trees_to_dataframe().shape[0])
    size_kb = len(pickle.dumps(est)) / 1024
    for _ in range(5):
        model.predict_proba(x_flat)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model.predict_proba(x_flat)
        times.append(time.perf_counter() - t0)
    t = np.array(times) * 1000
    return {
        "params": nodes,
        "params_note": "tree nodes",
        "macs": None,
        "size_fp32_kb": size_kb,
        "size_int8_kb": None,
        "latency_p50_ms": float(np.percentile(t, 50)),
        "latency_p95_ms": float(np.percentile(t, 95)),
        "latency_device": "host CPU (not target IoT hardware)",
    }


def run_one(
    model_name: str,
    data,
    epochs: int,
    seed: int,
    tree: str,
    results: str,
    latency_runs: int,
    checkpoints: str,
    artifacts: str | None = None,
):
    set_seed(seed)
    name = model_name.lower()
    X = {s: data.X[s] for s in ("train", "val", "test")}
    if name == "b5":  # PCA-reduced inputs (fit on train records only, inside prepare_data)
        X = {s: data.preprocessor.pca_array(v) for s, v in X.items()}
    T, F_in = X["train"].shape[1], X["train"].shape[2]
    input_dim = T * F_in if name == "b1" else F_in
    model = get_model(name, input_dim, data.num_classes, tree=tree, seed=seed)

    run_id = f"{name}_{data.dataset}_{seed}_{uuid.uuid4().hex[:4]}"
    run_dir = Path(results) / run_id
    loaders = {s: make_loader(X[s], data.y[s], 128, shuffle=(s == "train"), seed=seed) for s in X}
    trainer = UnifiedTrainer(name, model, device="cpu", run_dir=str(run_dir), seed=seed)
    from litecascade.data.dataset import compute_class_weights

    trainer.fit(
        loaders["train"], loaders["val"], epochs=epochs, class_weights=compute_class_weights(data.y["train"], data.num_classes)
    )
    metrics = trainer.evaluate(loaders["test"], bootstrap=True)

    if isinstance(model, B2_Trees):
        metrics.update(_profile_tree(model, X["test"][:1].reshape(1, -1), runs=min(latency_runs, 200)))
    else:
        metrics.update(profile_model(model, torch.as_tensor(X["test"][:1]), latency_runs=latency_runs))
        if name == "b5":  # evaluate the dynamically quantized model, as in the source design
            from litecascade.compress.quantize import quantize_dynamic_int8

            q = quantize_dynamic_int8(model)
            with torch.no_grad():
                preds = torch.cat([q(xb).argmax(1) for xb, _ in loaders["test"]]).numpy()
            metrics["fp32"] = {k: metrics[k] for k in ("f1_macro", "recall_malicious", "fpr")}
            metrics.update(compute_detection_metrics(data.y["test"], preds))
            metrics.update(bootstrap_metrics(data.y["test"], preds))
            buf = io.BytesIO()
            torch.save(q.state_dict(), buf)
            metrics["size_int8_kb"] = buf.tell() / 1024
            metrics["evaluated_model"] = "dynamic INT8"

    if name == "teacher":
        path = Path(checkpoints) / f"teacher_{data.dataset}_{data.task}_T{data.T}_s{seed}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "input_dim": F_in, "num_classes": data.num_classes, "val_f1": None}, path)
        logger.info(f"Teacher checkpoint: {path}")

    metrics["model"] = name
    metrics["dataset"] = data.dataset
    if artifacts and not isinstance(model, B2_Trees):
        from litecascade.deploy.bundle import save_baseline_bundle

        bdir = Path(artifacts) / f"{data.dataset}_{name}"
        save_baseline_bundle(
            bdir, name, model, data, metrics={**metrics, "run_id": run_id},
            input_kind="pca" if name == "b5" else "scaled", quantization="dynamic" if name == "b5" else "none",
        )  # fmt: skip
        logger.info(f"Bundle: {bdir}")
    save_metrics(run_id, metrics, seed, config_hash="N/A", is_emulated=data.is_synthetic, out_root=results)
    logger.info(f"Finished {run_id}. Test F1: {metrics['f1_macro']:.4f}")
    return run_id, metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=str, required=True, help="b1..b11, teacher, all, or dl (all deep-learning models)")
    parser.add_argument("--dataset", type=str, default="fixture", choices=DATASETS)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--max-rows-per-file", type=int, default=None)
    parser.add_argument("--max-train-windows", type=int, default=None)
    parser.add_argument("--task", default="binary", choices=["binary", "family"])
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--split-mode", default="auto", choices=["auto", "group", "time"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--tree", default="rf", choices=["rf", "xgb"])
    parser.add_argument("--latency-runs", type=int, default=300)
    parser.add_argument("--results", default="results")
    parser.add_argument("--checkpoints", default="checkpoints")
    parser.add_argument("--bundles", default=None, help="also save each DL model as a bundle under this folder")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    set_seed(args.seed)

    data = prepare_data(
        args.dataset, root=args.data_root, T=args.window, split_mode=args.split_mode, seed=args.seed, task=args.task,
        max_rows_per_file=args.max_rows_per_file, max_train_windows=args.max_train_windows,
    )  # fmt: skip
    choice = args.model.lower()
    models = ALL_MODELS if choice == "all" else DL_MODELS if choice == "dl" else [choice]
    out = {}
    for m in models:
        out[m] = run_one(
            m, data, args.epochs, args.seed, args.tree, args.results, args.latency_runs, args.checkpoints, args.bundles
        )
    return out


if __name__ == "__main__":
    main()
