from pathlib import Path

import numpy as np
import pandas as pd

from litecascade.utils.seed import set_seed


def generate_synthetic_data(dataset_name: str, num_rows: int = 2000) -> pd.DataFrame:
    """Generate a small synthetic dataset for testing."""
    set_seed(42)

    # 4 families, 3 groups
    families = ["benign", "mirai", "gafgyt", "tsunami"]
    groups = ["device_A", "device_B", "device_C"]

    # Generate 10 continuous features
    features = {f"feature_{i}": np.random.randn(num_rows) for i in range(10)}

    # Add random duplicates, inf, NaN to test cleaning
    # (Just a few to trigger the cleaning logic in EDA/Loaders)
    features["feature_0"][10:15] = np.inf
    features["feature_1"][20:25] = np.nan

    df = pd.DataFrame(features)

    # Add constant column
    df["constant_feature"] = 1.0

    # Labels and metadata
    df["label_family"] = np.random.choice(families, size=num_rows)
    df["label_binary"] = (df["label_family"] != "benign").astype(int)
    df["group"] = np.random.choice(groups, size=num_rows)

    # Random timestamps over a 1 day period
    base_time = pd.Timestamp("2026-01-01")
    df["timestamp"] = base_time + pd.to_timedelta(np.random.randint(0, 86400, size=num_rows), unit="s")

    df["is_synthetic"] = True

    # Add a few explicit duplicates
    df = pd.concat([df, df.iloc[:10]], ignore_index=True)

    return df


def main():
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    for dataset in ["nbaiot", "ciciot2023", "iot23"]:
        df = generate_synthetic_data(dataset)
        out_path = output_dir / f"synthetic_{dataset}.csv"
        df.to_csv(out_path, index=False)
        print(f"Generated {out_path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
