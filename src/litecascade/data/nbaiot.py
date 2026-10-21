"""N-BaIoT loader (Meidan et al., 2018) — 9 devices, 115 statistical features, Mirai + BASHLITE(gafgyt).

Two on-disk layouts are supported (see docs/DATASETS.md):

1. UCI layout (after extracting the .rar archives)::

       <root>/Danmini_Doorbell/benign_traffic.csv
       <root>/Danmini_Doorbell/gafgyt_attacks/{combo,junk,scan,tcp,udp}.csv
       <root>/Danmini_Doorbell/mirai_attacks/{ack,scan,syn,udp,udpplain}.csv

2. Kaggle flat layout::

       <root>/1.benign.csv, <root>/1.gafgyt.combo.csv, <root>/1.mirai.ack.csv, ...

The CSVs carry no timestamps; records are kept in file order, each file is its own segment
(windows never span two files) and the device is the split group.
"""

from pathlib import Path

import pandas as pd

from .clean import add_order_columns, clean_df

DEFAULT_ROOT = Path("data/raw/nbaiot")


def _family_from_uci(device_dir: Path, csv_path: Path) -> str:
    rel = csv_path.relative_to(device_dir)
    if rel.stem.startswith("benign"):
        return "benign"
    botnet = rel.parts[0].replace("_attacks", "") if len(rel.parts) > 1 else "unknown"
    return f"{botnet}_{rel.stem}"


def _discover(root: Path) -> list[tuple[str, str, Path]]:
    """Return (device, family, path) for every CSV under root."""
    found = []
    # Kaggle flat layout: "<device>.<botnet>.<attack>.csv" or "<device>.benign.csv"
    for p in sorted(root.glob("*.csv")):
        parts = p.stem.split(".")
        if len(parts) >= 2 and parts[1] == "benign":
            found.append((f"device_{parts[0]}", "benign", p))
        elif len(parts) >= 3:
            found.append((f"device_{parts[0]}", f"{parts[1]}_{parts[2]}", p))
    # UCI layout: one folder per device
    for device_dir in sorted(d for d in root.iterdir() if d.is_dir()):
        for p in sorted(device_dir.rglob("*.csv")):
            found.append((device_dir.name, _family_from_uci(device_dir, p), p))
    return found


def load_nbaiot(
    use_synthetic: bool = False,
    root: str | Path | None = None,
    max_rows_per_file: int | None = None,
    devices: list[str] | None = None,
) -> pd.DataFrame:
    """Load N-BaIoT into the common schema (see ``litecascade.data.clean``).

    ``use_synthetic=True`` loads the small synthetic fixture written in the same layout.
    ``max_rows_per_file`` reads only the first N rows of each CSV (keeps temporal contiguity).
    """
    if use_synthetic:
        from .synthetic import ensure_fixture

        root = ensure_fixture() / "nbaiot"
    root = Path(root) if root is not None else DEFAULT_ROOT
    if not root.exists():
        raise FileNotFoundError(f"N-BaIoT not found at '{root}'. See docs/DATASETS.md for download/layout instructions.")

    frames = []
    for device, family, path in _discover(root):
        if devices and device not in devices:
            continue
        df = pd.read_csv(path, nrows=max_rows_per_file)
        df = df.apply(pd.to_numeric, errors="coerce").astype("float32")
        df["label_family"] = family
        df["label_binary"] = int(family != "benign")
        df["group"] = device
        add_order_columns(df, segment=f"{device}/{path.stem}")
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No N-BaIoT CSV files found under '{root}'.")

    df = pd.concat(frames, ignore_index=True)
    df["is_synthetic"] = bool(use_synthetic)
    return clean_df(df)
