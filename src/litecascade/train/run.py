import argparse
import logging
import uuid

import torch
from torch.utils.data import DataLoader

from litecascade.data.dataset import CascadeDataset, compute_class_weights
from litecascade.data.nbaiot import load_nbaiot
from litecascade.data.splits import create_group_splits
from litecascade.data.windows import create_windows
from litecascade.eval.metrics import save_metrics
from litecascade.eval.profiler import profile_model
from litecascade.models.baselines import (
    B1_MLP,
    B3_LSTM,
    B7_DSCNN,
    B2_Trees,
    B4_BiLSTM,
    B5_WangBiLSTM,
    B6_CNN_BiLSTM,
    B8_Transformer,
)
from litecascade.train.trainer import UnifiedTrainer
from litecascade.utils.seed import set_seed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_model(model_name: str, input_dim: int):
    name = model_name.lower()
    if name == "b1":
        return B1_MLP(input_dim)
    if name == "b2":
        return B2_Trees()
    if name == "b3":
        return B3_LSTM(input_dim)
    if name == "b4":
        return B4_BiLSTM(input_dim)
    if name == "b5":
        return B5_WangBiLSTM(input_dim)
    if name == "b6":
        return B6_CNN_BiLSTM(input_dim)
    if name == "b7":
        return B7_DSCNN(input_dim)
    if name == "b8":
        return B8_Transformer(input_dim)
    if name == "teacher":
        from litecascade.models.teacher import TeacherCNNBiLSTM
        return TeacherCNNBiLSTM(input_dim, num_classes=2)
    raise ValueError(f"Unknown baseline: {model_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Baseline B1-B8")
    parser.add_argument("--dataset", type=str, default="fixture")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    set_seed(args.seed)

    # 1. Load data
    df = load_nbaiot(use_synthetic=(args.dataset == "fixture"))
    splits = create_group_splits(df, args.dataset, seed=args.seed)

    feature_cols = [c for c in df.columns if c not in ["label_binary", "label_family", "group", "timestamp", "is_synthetic"]]
    input_dim = len(feature_cols)

    if args.model.lower() == "b1":
        # B1 MLP expects flattened input (T=10 default, so T*F)
        input_dim = 10 * input_dim

    # 2. Windowing
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

    # 3. Model setup
    model = get_model(args.model, input_dim)

    run_id = f"{args.model}_{args.dataset}_{args.seed}_{uuid.uuid4().hex[:4]}"
    run_dir = f"results/{run_id}"

    # 4. Train
    trainer = UnifiedTrainer(args.model, model, device="cpu", run_dir=run_dir, seed=args.seed)
    trainer.fit(train_loader, val_loader, epochs=args.epochs, class_weights=class_weights)

    # 5. Evaluate on Test
    test_metrics = trainer.evaluate(test_loader, bootstrap=True)

    # 6. Profile
    if not isinstance(model, B2_Trees):
        dummy_input = torch.randn(1, 10, len(feature_cols))
        prof = profile_model(model, dummy_input)
        test_metrics.update(prof)

    if args.model.lower() == "teacher":
        # export logits
        model.eval()
        with torch.no_grad():
            from pathlib import Path
            Path("checkpoints").mkdir(exist_ok=True)
            
            # Use non-shuffled train loader
            train_loader_seq = DataLoader(train_ds, batch_size=32, shuffle=False)
            train_logits = []
            for X, y in train_loader_seq:
                train_logits.append(model(X.to("cpu")).cpu())
            train_logits = torch.cat(train_logits, dim=0)
            
            calib_logits = []
            for X, y in val_loader:
                calib_logits.append(model(X.to("cpu")).cpu())
            calib_logits = torch.cat(calib_logits, dim=0)
            
            torch.save({
                "train": train_logits,
                "calib": calib_logits
            }, f"checkpoints/teacher_logits_{args.dataset}.pt")

    save_metrics(run_id, test_metrics, args.seed, config_hash="N/A", is_emulated=(args.dataset == "fixture"))
    logger.info(f"Finished {run_id}. Test F1: {test_metrics['f1_macro']:.4f}")


if __name__ == "__main__":
    main()
