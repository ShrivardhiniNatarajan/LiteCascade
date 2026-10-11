"""
Compression pipeline CLI:
Structured Pruning -> QAT -> ONNX/TFLite INT8 Export -> Post-Quant Re-calibration.
"""

import argparse
import json
import logging
import os
import sys
import uuid
from pathlib import Path

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import torch
from torch.utils.data import DataLoader

from litecascade.compress.export import (
    TFLiteInterpreterWrapper,
    check_tflite_parity,
    export_onnx,
    export_tflite,
)
from litecascade.compress.prune import (
    apply_structured_pruning,
    count_zero_channels,
    fine_tune_pruned_model,
)
from litecascade.compress.quantize import (
    convert_qat_to_quantized,
    prepare_model_for_qat,
    run_wrong_order_ablation,
    train_qat_model,
)
from litecascade.data.dataset import CascadeDataset, compute_class_weights
from litecascade.data.nbaiot import load_nbaiot
from litecascade.data.splits import create_group_splits
from litecascade.data.windows import create_windows
from litecascade.eval.gate import calibrate_and_report
from litecascade.eval.metrics import compute_detection_metrics
from litecascade.models.litecascade import LiteCascade
from litecascade.utils.seed import set_seed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_f1(model: torch.nn.Module, data_loader: DataLoader) -> float:
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            _, l2 = model(x_batch, mode="train")
            all_preds.extend(torch.argmax(l2, dim=-1).cpu().numpy())
            all_labels.extend(y_batch.numpy())

    if len(all_labels) == 0:
        return 0.0
    metrics = compute_detection_metrics(np.array(all_labels), np.array(all_preds))
    return float(metrics["f1_macro"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, default="", help="Run ID to compress")
    parser.add_argument("--prune", type=float, default=0.3, help="Pruning ratio {0.2, 0.3, 0.5}")
    parser.add_argument("--dataset", type=str, default="fixture")
    parser.add_argument("--variant", type=str, choices=["T", "R"], default="T")
    parser.add_argument("--epochs", type=int, default=5, help="Fine-tuning and QAT epochs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    # 1. Load Data
    df = load_nbaiot(use_synthetic=(args.dataset == "fixture"))
    splits = create_group_splits(df, args.dataset, seed=args.seed)

    feature_cols = [c for c in df.columns if c not in ["label_binary", "label_family", "group", "timestamp", "is_synthetic"]]
    input_dim = len(feature_cols)

    X_train, y_train = create_windows(df.loc[splits["train"]], feature_cols, T=10)
    X_calib, y_calib = create_windows(df.loc[splits["calib"]], feature_cols, T=10)
    X_val, y_val = create_windows(df.loc[splits["val"]], feature_cols, T=10)
    X_test, y_test = create_windows(df.loc[splits["test"]], feature_cols, T=10)

    train_ds = CascadeDataset(X_train, y_train)
    calib_ds = CascadeDataset(X_calib, y_calib)
    val_ds = CascadeDataset(X_val, y_val)
    test_ds = CascadeDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    calib_loader = DataLoader(calib_ds, batch_size=32)
    val_loader = DataLoader(val_ds, batch_size=32)
    test_loader = DataLoader(test_ds, batch_size=32)

    class_weights = compute_class_weights(y_train)

    # 2. Load or instantiate base model
    run_id = args.run if args.run else f"compress_{args.variant}_{uuid.uuid4().hex[:4]}"
    run_dir = Path(f"results/{run_id}")
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = Path("checkpoints")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    model = LiteCascade(F_in=input_dim, d=16, c=24, hidden=24, K=2, variant=args.variant)
    ckpt_path = Path(f"results/{args.run}/best_model.pt") if args.run else None
    if ckpt_path and ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=False))
        logger.info(f"Loaded existing checkpoint from {ckpt_path}")
    else:
        logger.info("Initializing baseline LiteCascade model...")

    # Record baseline stats
    total_params_orig = sum(p.numel() for p in model.parameters())
    f1_baseline = evaluate_f1(model, test_loader)
    logger.info(f"Baseline FP32 Model | Params: {total_params_orig} | Test F1: {f1_baseline:.4f}")

    # 3. Structured Pruning
    logger.info(f"Applying structured channel pruning at ratio {args.prune}...")
    pruned_model = apply_structured_pruning(model, ratio=args.prune)
    prune_stats = count_zero_channels(pruned_model)
    logger.info(
        f"Channel pruning rate: {prune_stats['channel_prune_rate'] * 100:.1f}%, Weight prune rate: {prune_stats['weight_prune_rate'] * 100:.1f}%"
    )

    fine_tune_pruned_model(
        pruned_model,
        train_loader,
        val_loader,
        epochs=args.epochs,
        class_weights=class_weights,
    )
    f1_pruned = evaluate_f1(pruned_model, test_loader)
    logger.info(f"After Pruning + Fine-tuning | Test F1: {f1_pruned:.4f}")

    # 4. Quantization-Aware Training (QAT)
    logger.info("Starting Quantization-Aware Training (QAT)...")
    prepared_model = prepare_model_for_qat(pruned_model, variant=args.variant)
    qat_trained = train_qat_model(prepared_model, train_loader, epochs=args.epochs, class_weights=class_weights)
    quantized_model = convert_qat_to_quantized(qat_trained, variant=args.variant)

    f1_quantized = evaluate_f1(quantized_model, test_loader)
    f1_drop_pp = (f1_pruned - f1_quantized) * 100.0
    h4_hypothesis_holds = bool(f1_drop_pp <= 0.5)
    logger.info(f"After QAT | Test F1: {f1_quantized:.4f} | F1 drop: {f1_drop_pp:.2f} pp (H4 <= 0.5 pp: {h4_hypothesis_holds})")

    # 5. Export to ONNX and TFLite
    dummy_input = torch.randn(1, 10, input_dim)
    onnx_path = checkpoints_dir / f"litecascade_{args.variant}.onnx"
    export_onnx(pruned_model, dummy_input, onnx_path)

    # Save compressed weights to run directory
    torch.save(pruned_model.state_dict(), run_dir / "best_model.pt")

    tflite_fp32_path = checkpoints_dir / f"litecascade_{args.variant}_fp32.tflite"
    tflite_int8_path = checkpoints_dir / f"litecascade_{args.variant}_int8.tflite"
    export_tflite(pruned_model, tflite_fp32_path, representative_dataset=X_train, int8=False)
    export_tflite(pruned_model, tflite_int8_path, representative_dataset=X_train, int8=True)

    # Standard path expected by downstream stages
    standard_int8_path = checkpoints_dir / "litecascade_int8.tflite"
    if tflite_int8_path.exists():
        import shutil

        shutil.copyfile(tflite_int8_path, standard_int8_path)

    size_fp32_kb = os.path.getsize(tflite_fp32_path) / 1024.0
    size_int8_kb = os.path.getsize(tflite_int8_path) / 1024.0
    logger.info(
        f"TFLite Models | FP32: {size_fp32_kb:.2f} KB | INT8: {size_int8_kb:.2f} KB (Compression: {size_fp32_kb / size_int8_kb:.1f}x)"
    )

    # 6. Parity Check
    tflite_wrapper = TFLiteInterpreterWrapper(tflite_int8_path)
    parity_report = check_tflite_parity(quantized_model, tflite_wrapper, X_test, max_samples=min(1000, len(X_test)))
    logger.info(
        f"TFLite Parity Agreement: {parity_report['overall_agreement_pct']:.2f}% (Argmax pass >= 99%: {parity_report['parity_pass']})"
    )

    # 7. Post-Quantization Re-calibration (rerun Stage 8 logic on quantized model)
    logger.info("Re-calibrating temperature and risk-controlled thresholds on the quantized model...")
    gate_report = calibrate_and_report(quantized_model, calib_loader, test_loader, run_id, epsilon=0.005)
    exit_rate_recalibrated = float(gate_report.get("rho_raw", gate_report.get("test_metrics", {}).get("exit_rate_raw", 0.0)))
    logger.info(f"Post-Quant Recalibration | Exit Rate: {exit_rate_recalibrated * 100:.2f}%")

    # 8. Run Wrong-Order Ablation (QAT -> Prune)
    ablation_a9 = run_wrong_order_ablation(model, train_loader, val_loader, prune_ratio=args.prune, variant=args.variant)
    f1_wrong_order = evaluate_f1(ablation_a9["model"], test_loader)
    logger.info(f"Ablation A9 (QAT -> Prune) Test F1: {f1_wrong_order:.4f} vs Normal (Prune -> QAT): {f1_quantized:.4f}")

    # 9. Record full compression report
    compression_report = {
        "run_id": run_id,
        "variant": args.variant,
        "prune_ratio": args.prune,
        "params_original": total_params_orig,
        "params_active_conv": prune_stats["total_weights"] - prune_stats["zero_weights"],
        "weight_prune_rate": prune_stats["weight_prune_rate"],
        "channel_prune_rate": prune_stats["channel_prune_rate"],
        "f1_baseline_fp32": f1_baseline,
        "f1_pruned": f1_pruned,
        "f1_quantized_int8": f1_quantized,
        "f1_drop_pp": f1_drop_pp,
        "h4_hypothesis_holds": h4_hypothesis_holds,
        "size_fp32_kb": size_fp32_kb,
        "size_int8_kb": size_int8_kb,
        "compression_ratio": float(size_fp32_kb / max(1e-4, size_int8_kb)),
        "parity_report": parity_report,
        "post_quant_exit_rate": exit_rate_recalibrated,
        "f1_wrong_order_ablation": f1_wrong_order,
    }

    report_path = run_dir / "compression_report.json"
    with open(report_path, "w") as f:
        json.dump(compression_report, f, indent=2)

    logger.info(f"Compression report saved to {report_path}")
    print("\n=== COMPRESSION CHAIN SUMMARY ===")
    print(f"FP32 Size: {size_fp32_kb:.2f} KB | INT8 Size: {size_int8_kb:.2f} KB")
    print(f"Argmax Agreement: {parity_report['overall_agreement_pct']:.2f}%")
    print(f"F1 Drop after QAT: {f1_drop_pp:.2f} pp (Hypothesis H4 <= 0.5 pp: {h4_hypothesis_holds})")
    print(f"Exit Rate after Re-calibration: {exit_rate_recalibrated * 100:.2f}%")


if __name__ == "__main__":
    main()
