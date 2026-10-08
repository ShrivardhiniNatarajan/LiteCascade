import pandas as pd
import numpy as np

def _clean(df):
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    df = df.drop_duplicates()
    cols_to_drop = [c for c in df.columns if df[c].nunique() <= 1 and c != "is_synthetic"]
    return df.drop(columns=cols_to_drop)

def load_nbaiot(use_synthetic=False):
    if use_synthetic:
        from litecascade.utils.seed import set_seed
        set_seed(42)
        families = ["benign", "mirai", "gafgyt", "tsunami"]
        groups = ["device_A", "device_B", "device_C"]
        features = {f"feature_{i}": np.random.randn(2000) for i in range(115)}
        features["feature_0"][10:15] = np.inf
        features["feature_1"][20:25] = np.nan
        df = pd.DataFrame(features)
        df["constant_feature"] = 1.0
        df["label_family"] = np.random.choice(families, size=2000)
        df["label_binary"] = (df["label_family"] != "benign").astype(int)
        df["group"] = np.random.choice(groups, size=2000)
        df["timestamp"] = pd.Timestamp("2026-01-01") + pd.to_timedelta(np.random.randint(0, 86400, size=2000), unit="s")
        df["is_synthetic"] = True
        df = pd.concat([df, df.iloc[:10]], ignore_index=True)
        return _clean(df)
    return pd.DataFrame()
