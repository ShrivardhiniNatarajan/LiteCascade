import numpy as np
import pandas as pd


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    df = df.drop_duplicates()
    cols_to_drop = [c for c in df.columns if df[c].nunique() <= 1 and c != "is_synthetic"]
    return df.drop(columns=cols_to_drop)


def load_nbaiot(use_synthetic: bool = False) -> pd.DataFrame:
    if use_synthetic:
        from litecascade.utils.seed import set_seed

        set_seed(42)
        groups = ["device_A", "device_B", "device_C"]

        # Generate realistic IoT device traffic with temporal sessions per device
        records = []
        for grp in groups:
            # 5 sessions per family alternating with clean benign baseline traffic
            for _ in range(5):
                for fam in ["benign", "mirai", "benign", "tsunami", "benign", "gafgyt"]:
                    sub_n = 25
                    sub_feats = {f"feature_{i}": np.random.randn(sub_n) for i in range(115)}
                    sub = pd.DataFrame(sub_feats)
                    sub["group"] = grp
                    sub["label_family"] = fam
                    sub["label_binary"] = 0 if fam == "benign" else 1

                    # Distinct flow signatures for IoT malware families
                    if fam == "mirai":
                        for ch in [0, 2, 4, 6]:
                            sub[f"feature_{ch}"] += 3.8
                        for ch in range(8, 14):
                            sub[f"feature_{ch}"] += 2.5
                    elif fam == "gafgyt":
                        for ch in [1, 3, 5, 7]:
                            sub[f"feature_{ch}"] += 3.5
                        for ch in range(8, 14):
                            sub[f"feature_{ch}"] += 2.5
                    elif fam == "tsunami":
                        for ch in range(16, 24):
                            sub[f"feature_{ch}"] += 3.8
                        for ch in range(8, 14):
                            sub[f"feature_{ch}"] += 2.5

                    records.append(sub)

        df = pd.concat(records, ignore_index=True)

        # Invalidate a few elements to verify data cleaning in tests
        df.loc[10:14, "feature_0"] = np.inf
        df.loc[20:24, "feature_1"] = np.nan
        df["constant_feature"] = 1.0

        # Monotonically increasing timestamps
        base_time = pd.Timestamp("2026-01-01")
        df["timestamp"] = base_time + pd.to_timedelta(range(len(df)), unit="s")
        df["is_synthetic"] = True
        df = pd.concat([df, df.iloc[:10]], ignore_index=True)
        return _clean(df)

    return pd.DataFrame()
