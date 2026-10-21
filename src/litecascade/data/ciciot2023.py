"""CICIoT2023 loader (Neto et al., 2023) — 46 flow features + ``label`` (33 attacks + benign).

Expected layout: any number of CSVs (``part-*.csv`` or ``Merged*.csv``) under ``<root>`` (searched recursively).
Each file is one split group and one segment. The published CSVs have no timestamps; rows are kept in
file order. NOTE: the public CSV rows are not guaranteed to be temporally ordered, so windows with T > 1
may not carry real temporal structure on this dataset; ``--window 1`` is a sensible choice here.
"""

from pathlib import Path

import pandas as pd

from .clean import add_order_columns, clean_df

DEFAULT_ROOT = Path("data/raw/ciciot2023")

# 34 raw labels -> 7 attack categories + benign (category scheme from the CICIoT2023 paper).
_CATEGORY_PREFIXES = [
    ("benign", "benign"),
    ("ddos", "ddos"),
    ("dos", "dos"),
    ("mirai", "mirai"),
    ("recon", "recon"),
    ("vulnerabilityscan", "recon"),
    ("mitm", "spoofing"),
    ("dns_spoofing", "spoofing"),
    ("dictionarybruteforce", "bruteforce"),
]
_WEB = {"sqlinjection", "commandinjection", "backdoor_malware", "uploading_attack", "xss", "browserhijacking"}


def label_to_category(label: str) -> str:
    key = str(label).strip().lower()
    if key in _WEB:
        return "web"
    for prefix, cat in _CATEGORY_PREFIXES:
        if key.startswith(prefix):
            return cat
    return "other"


def load_ciciot2023(
    use_synthetic: bool = False,
    root: str | Path | None = None,
    max_rows_per_file: int | None = None,
    max_files: int | None = None,
    family_level: str = "category",
) -> pd.DataFrame:
    """Load CICIoT2023 into the common schema. ``family_level`` is "category" (8 classes) or "raw" (34)."""
    if use_synthetic:
        from .synthetic import ensure_fixture

        root = ensure_fixture() / "ciciot2023"
    root = Path(root) if root is not None else DEFAULT_ROOT
    if not root.exists():
        raise FileNotFoundError(f"CICIoT2023 not found at '{root}'. See docs/DATASETS.md.")

    files = sorted(root.rglob("*.csv"))[:max_files] if max_files else sorted(root.rglob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CICIoT2023 CSV files found under '{root}'.")

    frames = []
    for path in files:
        df = pd.read_csv(path, nrows=max_rows_per_file)
        label_col = next((c for c in df.columns if c.strip().lower() == "label"), None)
        if label_col is None:
            raise ValueError(f"{path} has no 'label' column; is this a CICIoT2023 CSV?")
        labels = df.pop(label_col).astype(str)
        df = df.apply(pd.to_numeric, errors="coerce").astype("float32")
        df["label_family"] = labels.map(label_to_category) if family_level == "category" else labels.str.lower()
        df["label_binary"] = (~labels.str.lower().str.startswith("benign")).astype(int)
        df["group"] = path.stem
        add_order_columns(df, segment=path.stem)
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)
    df["is_synthetic"] = bool(use_synthetic)
    return clean_df(df)
