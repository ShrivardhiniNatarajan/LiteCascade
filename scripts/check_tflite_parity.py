"""
Parity check CLI: Compares PyTorch vs TFLite INT8 outputs on test windows.
"""

import argparse
import json
import logging
from pathlib import Path

import torch

from litecascade.compress.export import (
    TFLiteInterpreterWrapper,
    check_tflite_parity,
)
from litecascade.data.nbaiot import load_nbaiot
from litecascade.data.splits import create_group_splits
from litecascade.data.windows import create_windows
from litecascade.models.litecascade import LiteCascade

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, default="", help="Run ID")
    parser.add_argument("--dataset", type=str, default="fixture")
    parser.add_argument("--variant", type=str, choices=["T", "R"], default="T")
    parser.add_argument("--samples", type=int, default=1000)
    args = parser.parse_args()

    # 1. Load Data
    df = load_nbaiot(use_synthetic=(args.dataset == "fixture"))
    splits = create_group_splits(df, args.dataset, seed=42)

    feature_cols = [c for c in df.columns if c not in ["label_binary", "label_family", "group", "timestamp", "is_synthetic"]]
    input_dim = len(feature_cols)

    X_test, _ = create_windows(df.loc[splits["test"]], feature_cols, T=10)

    variant = args.variant
    if args.run:
        report_path = Path(f"results/{args.run}/compression_report.json")
        if report_path.exists():
            with open(report_path) as f:
                rep = json.load(f)
                variant = rep.get("variant", variant)

    # 2. Load PyTorch model
    model = LiteCascade(F_in=input_dim, d=16, c=24, hidden=24, K=2, variant=variant)
    if args.run:
        ckpt_path = Path(f"results/{args.run}/best_model.pt")
        if ckpt_path.exists():
            model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=False))
            logger.info(f"Loaded PyTorch weights from {ckpt_path}")
    model.eval()

    # 3. Locate TFLite model
    tflite_candidates = [
        Path(f"checkpoints/litecascade_{args.variant}_int8.tflite"),
        Path("checkpoints/litecascade_int8.tflite"),
    ]
    tflite_path = None
    for p in tflite_candidates:
        if p.exists():
            tflite_path = p
            break

    if not tflite_path:
        raise FileNotFoundError("Could not find INT8 TFLite model in checkpoints/. Run scripts/compress.py first.")

    logger.info(f"Checking parity against {tflite_path} on {min(len(X_test), args.samples)} samples...")
    tflite_wrapper = TFLiteInterpreterWrapper(tflite_path)

    report = check_tflite_parity(model, tflite_wrapper, X_test, max_samples=args.samples)
    print("\n=== TFLITE PARITY REPORT ===")
    print(json.dumps(report, indent=2))

    if report["parity_pass"]:
        print(f"\n[PASS] Argmax agreement: {report['overall_agreement_pct']:.2f}% >= 99.0%")
    else:
        print(f"\n[FAIL] Argmax agreement: {report['overall_agreement_pct']:.2f}% < 99.0%")


if __name__ == "__main__":
    main()
