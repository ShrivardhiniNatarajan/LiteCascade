"""Calibrate gate thresholds for a trained LiteCascade run."""
import argparse
import json
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from litecascade.data.dataset import CascadeDataset
from litecascade.data.nbaiot import load_nbaiot
from litecascade.data.splits import create_group_splits
from litecascade.data.windows import create_windows
from litecascade.eval.gate import calibrate_and_report
from litecascade.models.litecascade import LiteCascade
from litecascade.utils.seed import set_seed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, required=True, help="Run ID")
    parser.add_argument("--epsilon", type=float, default=0.005)
    parser.add_argument("--dataset", type=str, default="fixture")
    parser.add_argument("--variant", type=str, default="R")
    args = parser.parse_args()

    set_seed(42)

    df = load_nbaiot(use_synthetic=(args.dataset == "fixture"))
    splits = create_group_splits(df, args.dataset, seed=42)

    feature_cols = [c for c in df.columns if c not in ["label_binary", "label_family", "group", "timestamp", "is_synthetic"]]
    input_dim = len(feature_cols)

    X_calib, y_calib = create_windows(df.loc[splits["calib"]], feature_cols, T=10)
    X_test, y_test = create_windows(df.loc[splits["test"]], feature_cols, T=10)

    calib_ds = CascadeDataset(X_calib, y_calib)
    test_ds = CascadeDataset(X_test, y_test)

    calib_loader = DataLoader(calib_ds, batch_size=32)
    test_loader = DataLoader(test_ds, batch_size=32)

    # Load model
    run_dir = Path(f"results/{args.run}")
    model = LiteCascade(F_in=input_dim, d=16, c=24, hidden=24, K=2, variant=args.variant)
    
    ckpt_path = run_dir / "best_model.pt"
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state)
        logger.info(f"Loaded checkpoint from {ckpt_path}")
    else:
        logger.warning(f"No checkpoint found at {ckpt_path}, using random weights")

    report = calibrate_and_report(model, calib_loader, test_loader, args.run, epsilon=args.epsilon)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
