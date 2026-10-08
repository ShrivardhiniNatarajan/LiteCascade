import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Setup basic logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def run_eda(dataset_name: str, df: pd.DataFrame, out_dir: Path):
    logger.info(f"Running EDA for {dataset_name}...")

    # 1. Feature counts & shapes
    row_count, col_count = df.shape
    feature_cols = [
        c
        for c in df.columns
        if c not in ["label_binary", "label_family", "group", "timestamp", "is_synthetic"]
    ]

    # 2. Missing, Inf, Duplicates
    num_df = df.select_dtypes(include=[np.number])
    inf_count = int(np.isinf(num_df).values.sum())
    nan_count = int(df.isna().sum().sum())
    duplicate_count = int(df.duplicated().sum())
    duplicate_rate = duplicate_count / row_count if row_count > 0 else 0

    # 3. Distributions
    binary_dist = df["label_binary"].value_counts().to_dict()
    family_dist = df["label_family"].value_counts().to_dict()
    group_sizes = df["group"].value_counts().to_dict()

    stats = {
        "dataset": dataset_name,
        "row_count": int(row_count),
        "total_columns": int(col_count),
        "feature_count": len(feature_cols),
        "missing_nan_count": nan_count,
        "infinity_count": inf_count,
        "duplicate_count": duplicate_count,
        "duplicate_rate": float(duplicate_rate),
        "class_distribution": {str(k): int(v) for k, v in binary_dist.items()},
        "family_distribution": {str(k): int(v) for k, v in family_dist.items()},
        "group_sizes": {str(k): int(v) for k, v in group_sizes.items()},
    }

    # Save JSON
    json_path = out_dir / f"{dataset_name}.json"
    with open(json_path, "w") as f:
        json.dump(stats, f, indent=4)

    logger.info(f"Saved EDA stats to {json_path}")

    # Save Plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Binary class
    pd.Series(binary_dist).plot(kind="bar", ax=axes[0], color=["blue", "red"])
    axes[0].set_title("Binary Class Distribution")
    axes[0].set_ylabel("Count")

    # Plot 2: Family class
    pd.Series(family_dist).plot(kind="bar", ax=axes[1], color="purple")
    axes[1].set_title("Family Distribution")
    axes[1].set_ylabel("Count")

    # Plot 3: Group sizes
    pd.Series(group_sizes).plot(kind="bar", ax=axes[2], color="green")
    axes[2].set_title("Per-Group Sizes")
    axes[2].set_ylabel("Count")

    plt.tight_layout()
    plot_path = out_dir / f"{dataset_name}_distributions.png"
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Saved EDA plots to {plot_path}")


def main():
    out_dir = Path("results/eda")
    out_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path("data/processed")
    datasets = ["nbaiot", "ciciot2023", "iot23"]

    for ds in datasets:
        file_path = data_dir / f"synthetic_{ds}.csv"
        if not file_path.exists():
            logger.warning(f"File {file_path} not found. Skipping EDA for {ds}.")
            continue

        df = pd.read_csv(file_path)
        run_eda(ds, df, out_dir)


if __name__ == "__main__":
    main()
