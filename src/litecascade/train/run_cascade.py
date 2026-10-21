"""Train, calibrate, compress, evaluate and package LiteCascade.

Chain (paper Sec. IV.F-G): teacher -> joint training with distillation (eq. 5) -> temperature scaling +
risk-controlled gate (eqs. 2-4) -> structured pruning + fine-tuning -> INT8 QAT -> re-calibration on the
quantized model. Each stage is evaluated on the held-out test split and profiled on the host CPU.

    python -m litecascade.train.run_cascade --dataset fixture --variant R --epochs 15 --prune 0.3 --quantize qat

Ablation switches (Table IV): --no-kd --no-self-kd --no-pip-init --no-temp-scaling --naive-gate
--verify-alerts --conservative-gate.
"""

import argparse
import json
import logging
import uuid
from pathlib import Path

import numpy as np
import torch

from litecascade.compress.prune import prune_litecascade
from litecascade.compress.quantize import convert_qat, prepare_qat, quantize_dynamic_int8
from litecascade.data.pipeline import DATASETS, prepare_data
from litecascade.deploy.bundle import save_bundle
from litecascade.eval.metrics import _json_default, get_git_hash
from litecascade.eval.profiler import profile_cascade, serialized_size_kb
from litecascade.models.teacher import TeacherCNNBiLSTM
from litecascade.train.cascade_trainer import (
    build_cascade,
    calibrate_model,
    evaluate_model,
    naive_gate,
    predict_logits,
    train_cascade,
    train_teacher,
)
from litecascade.utils.seed import set_seed

logger = logging.getLogger(__name__)


def get_teacher(data, args):
    """Load checkpoints/teacher_<dataset>_<task>_T<T>.pt if compatible, else train and save it."""
    path = Path(args.checkpoints) / f"teacher_{data.dataset}_{data.task}_T{data.T}_s{args.seed}.pt"
    if path.exists():
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if ck.get("input_dim") == data.n_features and ck.get("num_classes") == data.num_classes:
            model = TeacherCNNBiLSTM(data.n_features, data.num_classes)
            model.load_state_dict(ck["state_dict"])
            logger.info(f"Loaded teacher from {path}")
            return model, ck.get("val_f1")
    model, val_f1 = train_teacher(data, epochs=args.teacher_epochs, seed=args.seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "input_dim": data.n_features, "num_classes": data.num_classes, "val_f1": val_f1}, path
    )
    return model, val_f1


def stage_record(name, model, data, gate, profile_model_float=None, quantization="none"):
    rep = evaluate_model(model, data, gate)
    prof_src = profile_model_float or model
    prof = profile_cascade(prof_src, data.T, data.n_features, rho=rep["rho_raw"], latency_runs=200)
    if quantization != "none":  # latency of the actual (quantized) model
        q = profile_cascade(
            model, data.T, data.n_features, rho=rep["rho_raw"], latency_runs=200, X_eval=data.X["test"], gate=gate
        )
        prof.update({k: q[k] for k in q if k.startswith(("latency", "measured"))})
    else:
        q = profile_cascade(model, data.T, data.n_features, latency_runs=200, X_eval=data.X["test"], gate=gate)
        prof.update({k: q[k] for k in q if k.startswith("measured")})
    prof["size_kb_serialized"] = serialized_size_kb(model)
    rec = {"stage": name, "quantization": quantization, "gate": gate, "test": rep, "profile": prof}
    c = rep["cascade"]
    logger.info(
        f"[{name}] test cascade F1={c['f1_macro']:.4f} recall={c['recall_malicious']:.4f} rho={rep['rho_raw']:.3f} "
        f"exit-miss={rep['exit_induced_miss_rate']:.4f} size={prof['size_kb_serialized']:.1f}KB"
    )
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="fixture", choices=DATASETS)
    ap.add_argument("--data-root", default=None, help="root folder of the raw dataset (defaults to data/raw/<dataset>)")
    ap.add_argument("--max-rows-per-file", type=int, default=None)
    ap.add_argument("--max-train-windows", type=int, default=None)
    ap.add_argument("--task", default="binary", choices=["binary", "family"])
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--split-mode", default="auto", choices=["auto", "group", "time"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--variant", choices=["R", "T"], default="R")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--teacher-epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--gamma", type=float, default=0.5, help="weight of teacher->Analyst KD")
    ap.add_argument("--delta", type=float, default=0.3, help="weight of Analyst->Sentinel self-KD")
    ap.add_argument("--tau-kd", type=float, default=3.0)
    ap.add_argument("--epsilon", type=float, default=0.005)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--target-fpr", type=float, default=0.01)
    ap.add_argument("--prune", type=float, default=0.0, help="structured pruning ratio (0 = off)")
    ap.add_argument("--finetune-epochs", type=int, default=5)
    ap.add_argument("--quantize", default="none", choices=["none", "dynamic", "qat"])
    ap.add_argument("--qat-epochs", type=int, default=3)
    ap.add_argument("--no-kd", action="store_true")
    ap.add_argument("--no-self-kd", action="store_true")
    ap.add_argument("--no-pip-init", action="store_true")
    ap.add_argument("--no-temp-scaling", action="store_true")
    ap.add_argument("--naive-gate", action="store_true", help="fixed tau_b=tau_m=0.9, no temperature (ablation)")
    ap.add_argument("--verify-alerts", action="store_true", help="tau_m = never: every alert goes to the Analyst")
    ap.add_argument("--conservative-gate", action="store_true", help="count every malicious benign-exit as a miss")
    ap.add_argument(
        "--tag", default="main", help="free-form label stored in the report ('ablation' runs are excluded from the main table)"
    )
    ap.add_argument("--results", default="results")
    ap.add_argument("--checkpoints", default="checkpoints")
    ap.add_argument("--bundle", default=None, help="write the final model bundle to <artifacts>/<name>")
    ap.add_argument("--artifacts", default="artifacts")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    set_seed(args.seed)

    data = prepare_data(
        args.dataset, root=args.data_root, T=args.window, stride=args.stride, split_mode=args.split_mode, seed=args.seed,
        task=args.task, max_rows_per_file=args.max_rows_per_file, max_train_windows=args.max_train_windows,
    )  # fmt: skip
    logger.info(f"Data: {json.dumps(data.summary())} split_mode={data.info['split_mode']}")

    teacher_logits, teacher_info = None, None
    if not args.no_kd:
        teacher, t_val = get_teacher(data, args)
        teacher_logits = predict_logits(teacher, data.X["train"])  # aligned 1:1 with the training windows
        from litecascade.eval.metrics import compute_detection_metrics

        teacher_info = {
            "val_f1": t_val,
            "test": compute_detection_metrics(data.y["test"], predict_logits(teacher, data.X["test"]).argmax(1)),
        }

    gate_kwargs = dict(
        epsilon=args.epsilon, alpha=args.alpha, target_fpr=args.target_fpr, verify_alerts=args.verify_alerts,
        temperature=not args.no_temp_scaling, conservative=args.conservative_gate,
    )  # fmt: skip
    train_kwargs = dict(gamma=args.gamma, delta=0.0 if args.no_self_kd else args.delta, tau_kd=args.tau_kd, seed=args.seed)

    def calibrate(m):
        if args.naive_gate:
            return naive_gate()
        return calibrate_model(m, data, **gate_kwargs)

    # 1) joint training
    model = build_cascade(data, variant=args.variant, pip_init=not args.no_pip_init)
    model, history = train_cascade(
        model, data, teacher_logits, epochs=args.epochs, lr=args.lr,
        pip_warmup_epochs=0 if args.no_pip_init else 3, tag="joint", **train_kwargs,
    )  # fmt: skip
    stages = [stage_record("joint_kd", model, data, calibrate(model))]
    float_model = model

    # 2) structured pruning + fine-tuning
    if args.prune > 0:
        float_model = prune_litecascade(model, args.prune)
        float_model, _ = train_cascade(
            float_model, data, teacher_logits, epochs=args.finetune_epochs, lr=args.lr / 2, pip_warmup_epochs=0, tag="prune-ft",
            **train_kwargs,
        )  # fmt: skip
        stages.append(stage_record(f"pruned_{args.prune:g}", float_model, data, calibrate(float_model)))

    final_model, final_gate, quant = float_model, stages[-1]["gate"], "none"
    # 3) INT8
    if args.quantize != "none":
        float_gate = stages[-1]["gate"]
        if args.quantize == "qat":
            qat = prepare_qat(float_model)
            qat, _ = train_cascade(
                qat, data, teacher_logits, epochs=args.qat_epochs, lr=args.lr / 4, pip_warmup_epochs=0, tag="qat", **train_kwargs
            )
            int8 = convert_qat(qat)
        else:
            int8 = quantize_dynamic_int8(float_model)
        int8.T1.fill_(float_gate["T1"])
        int8.T2.fill_(float_gate["T2"])
        # before re-calibration: thresholds/temperatures from the float model (shows why step iv matters)
        stages.append(stage_record(f"int8_{args.quantize}_float_gate", int8, data, float_gate, float_model, args.quantize))
        new_gate = calibrate(int8)
        stages.append(stage_record(f"int8_{args.quantize}_recalibrated", int8, data, new_gate, float_model, args.quantize))
        final_model, final_gate, quant = int8, new_gate, args.quantize

    run_id = f"cascade_{args.variant}_{args.dataset}_{args.seed}_{uuid.uuid4().hex[:4]}"
    run_dir = Path(args.results) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "run_id": run_id,
        "data_kind": "SYNTHETIC fixture - not evidence of real-world performance" if data.is_synthetic else "real dataset",
        "git_hash": get_git_hash(),
        "config": vars(args),
        "data": data.info,
        "teacher": teacher_info,
        "history": history,
        "stages": stages,
    }
    with open(run_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2, default=_json_default)
    torch.save(final_model.state_dict(), run_dir / "final_model.pt")
    if args.bundle:
        final = stages[-1]
        save_bundle(
            Path(args.artifacts) / args.bundle, final_model, data, final_gate, quantization=quant,
            metrics={"test": final["test"], "profile": final["profile"], "run_id": run_id},
        )  # fmt: skip
        logger.info(f"Bundle written to {Path(args.artifacts) / args.bundle}")
    logger.info(f"Report: {run_dir / 'report.json'}")
    return report


if __name__ == "__main__":
    np.set_printoptions(precision=4)
    main()
