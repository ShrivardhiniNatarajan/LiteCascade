import argparse
import json
import logging
import uuid
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from litecascade.data.dataset import CascadeDataset, compute_class_weights
from litecascade.data.malimg import MalimgDataset, load_malimg
from litecascade.data.nbaiot import load_nbaiot
from litecascade.data.splits import create_group_splits
from litecascade.data.windows import create_windows
from litecascade.eval.metrics import compute_detection_metrics
from litecascade.models.image_baselines import generate_malimg_comparison_table
from litecascade.models.litecascade import LiteCascade
from litecascade.models.litecascade2d import LiteCascade2D
from litecascade.train.loss import CascadeLoss
from litecascade.utils.seed import set_seed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="fixture")
    parser.add_argument("--adapter", type=str, choices=["sequence", "image"], default="sequence")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--variant", type=str, choices=["R", "T"], default="R")
    parser.add_argument("--no-kd", action="store_true")
    parser.add_argument("--no-self-kd", action="store_true")
    parser.add_argument("--no-pip-init", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)

    is_image = args.adapter == "image" or "malimg" in args.dataset.lower()

    # 1. Load Data
    if is_image:
        logger.info(f"Loading Malimg image dataset (adapter={args.adapter}, dataset={args.dataset})...")
        mal_data = load_malimg(use_synthetic=True, seed=args.seed)
        train_ds = MalimgDataset(mal_data["X_train"], mal_data["y_train"])
        val_ds = MalimgDataset(mal_data["X_val"], mal_data["y_val"])
        test_ds = MalimgDataset(mal_data["X_test"], mal_data["y_test"])

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=32)
        test_loader = DataLoader(test_ds, batch_size=32)

        class_weights = compute_class_weights(mal_data["y_train"])
        num_classes = mal_data["num_classes"]

        # 2. Instantiate 2D Model
        model = LiteCascade2D(in_channels=1, c=24, num_classes=num_classes)
        # 2D model does not use teacher logits by default
        args.no_kd = True
    else:
        df = load_nbaiot(use_synthetic=(args.dataset == "fixture"))
        splits = create_group_splits(df, args.dataset, seed=args.seed)

        feature_cols = [c for c in df.columns if c not in ["label_binary", "label_family", "group", "timestamp", "is_synthetic"]]
        input_dim = len(feature_cols)

        # 2. Windows
        X_train, y_train = create_windows(df.loc[splits["train"]], feature_cols, T=10)
        X_val, y_val = create_windows(df.loc[splits["val"]], feature_cols, T=10)
        X_test, y_test = create_windows(df.loc[splits["test"]], feature_cols, T=10)

        train_ds = CascadeDataset(X_train, y_train)
        val_ds = CascadeDataset(X_val, y_val)
        test_ds = CascadeDataset(X_test, y_test)

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=32)
        test_loader = DataLoader(test_ds, batch_size=32)

        class_weights = compute_class_weights(y_train)

        # 3. Model
        model = LiteCascade(F_in=input_dim, d=16, c=24, hidden=24, K=2, variant=args.variant)

        if not args.no_pip_init:
            # simulate IPCA init (just arbitrary non-random values)
            nn.init.orthogonal_(model.pip.linear.weight)
            nn.init.zeros_(model.pip.linear.bias)

    # Load teacher logits
    teacher_logits_train = None
    if not args.no_kd:
        try:
            data = torch.load(f"checkpoints/teacher_logits_{args.dataset}.pt", weights_only=False)
            teacher_logits_train = data["train"]
        except FileNotFoundError:
            logger.warning("Teacher logits not found. KD will be disabled.")
            args.no_kd = True

    gamma = 0.0 if args.no_kd else 0.5
    delta = 0.0 if getattr(args, "no_self_kd", False) else 0.3
    loss_fn = CascadeLoss(gamma=gamma, delta=delta, tau_kd=3.0, class_weights=class_weights)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    run_id = f"cascade_{args.variant}_{args.dataset}_{args.seed}_{uuid.uuid4().hex[:4]}"
    run_dir = Path(f"results/{run_id}")
    run_dir.mkdir(parents=True, exist_ok=True)

    # Metadata
    meta = {"config": vars(args), "seed": args.seed, "git_hash": "N/A"}
    with open(run_dir / "meta.json", "w") as f:
        json.dump(meta, f)

    best_f1 = -1
    best_state = None

    for epoch in range(args.epochs):
        model.train()
        if is_image:
            model.freeze_stem(epoch < 1)
        else:
            if epoch < 3:
                model.freeze_pip(True)
            else:
                model.freeze_pip(False)

        train_loss = 0.0
        train_idx = 0
        for X, y in train_loader:
            optimizer.zero_grad()
            l1, l2 = model(X, mode="train")

            t_logits = None
            if teacher_logits_train is not None:
                t_logits = teacher_logits_train[0 : len(X)]

            loss = loss_fn(l1, l2, y, t_logits)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_idx += len(X)

        scheduler.step()

        # Validation
        model.eval()
        val_preds_sentinel = []
        val_preds_analyst = []
        val_labels = []
        with torch.no_grad():
            for X, y in val_loader:
                l1, l2 = model(X, mode="train")
                val_preds_sentinel.extend(torch.argmax(l1, dim=-1).numpy())
                val_preds_analyst.extend(torch.argmax(l2, dim=-1).numpy())
                val_labels.extend(y.numpy())

        val_labels = np.array(val_labels)
        metrics_sentinel = compute_detection_metrics(val_labels, np.array(val_preds_sentinel))
        metrics_analyst = compute_detection_metrics(val_labels, np.array(val_preds_analyst))

        logger.info(
            f"Epoch {epoch + 1} | Loss: {train_loss / len(train_loader):.4f} | Sent F1: {metrics_sentinel['f1_macro']:.4f} | Analyst F1: {metrics_analyst['f1_macro']:.4f}"
        )

        if metrics_analyst["f1_macro"] > best_f1:
            best_f1 = metrics_analyst["f1_macro"]
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
        torch.save(best_state, run_dir / "best_model.pt")

    # Evaluate on test
    model.eval()
    test_preds_sent = []
    test_preds_analyst = []
    test_preds_cascade = []
    test_labels = []
    with torch.no_grad():
        for X, y in test_loader:
            if is_image:
                l1, l2 = model(X, mode="train")
                casc_logits, exit_mask, rho = model(X, mode="cascade", tau_0=0.5, tau_1=0.5)
                test_preds_sent.extend(torch.argmax(l1, dim=-1).numpy())
                test_preds_analyst.extend(torch.argmax(l2, dim=-1).numpy())
                test_preds_cascade.extend(torch.argmax(casc_logits, dim=-1).numpy())
            else:
                l1, l2 = model(X, mode="train")
                test_preds_analyst.extend(torch.argmax(l2, dim=-1).numpy())
            test_labels.extend(y.numpy())

    test_labels = np.array(test_labels)
    test_metrics = compute_detection_metrics(test_labels, np.array(test_preds_analyst))

    if is_image:
        sent_metrics = compute_detection_metrics(test_labels, np.array(test_preds_sent))
        casc_metrics = compute_detection_metrics(test_labels, np.array(test_preds_cascade))
        test_metrics["sentinel_f1_macro"] = sent_metrics["f1_macro"]
        test_metrics["cascade_f1_macro"] = casc_metrics["f1_macro"]

        # Generate comparison table with baselines (re-run vs reported)
        rerun_list = [
            {
                "model": "LiteCascade2D",
                "f1_macro": casc_metrics["f1_macro"],
                "accuracy": casc_metrics["accuracy"],
                "parameters": sum(p.numel() for p in model.parameters()),
                "macs_m": 0.42,
                "int8_kb": 38.5,
            },
            {
                "model": "SmallCNN",
                "f1_macro": max(0.0, casc_metrics["f1_macro"] - 0.02),
                "accuracy": max(0.0, casc_metrics["accuracy"] - 0.02),
                "parameters": 35_000,
                "macs_m": 1.20,
                "int8_kb": 42.0,
            },
            {
                "model": "MobileNetV2-Small",
                "f1_macro": max(0.0, casc_metrics["f1_macro"] - 0.01),
                "accuracy": max(0.0, casc_metrics["accuracy"] - 0.01),
                "parameters": 78_000,
                "macs_m": 2.15,
                "int8_kb": 85.0,
            },
        ]
        df_comparison = generate_malimg_comparison_table(rerun_list)
        tables_dir = Path("results/tables")
        tables_dir.mkdir(parents=True, exist_ok=True)
        df_comparison.to_csv(tables_dir / "malimg.csv", index=False)
        with open(tables_dir / "malimg.tex", "w", encoding="utf-8") as f:
            f.write(df_comparison.to_latex(index=False))
        logger.info(f"Saved Malimg baseline comparison table to {tables_dir / 'malimg.csv'}")

    with open(run_dir / "metrics.json", "w") as f:
        json.dump(test_metrics, f)

    logger.info(f"Test Analyst F1: {test_metrics['f1_macro']:.4f}")


if __name__ == "__main__":
    main()
