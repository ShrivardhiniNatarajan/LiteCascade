import pandas as pd
import pytest

from litecascade.data.ciciot2023 import load_ciciot2023
from litecascade.data.iot23 import load_iot23
from litecascade.data.nbaiot import load_nbaiot


@pytest.mark.parametrize("loader_func", [load_nbaiot, load_ciciot2023, load_iot23])
def test_loader_schema(loader_func):
    """Test that loaders return a DataFrame matching the required schema."""
    # This will load the synthetic fixture due to use_synthetic=True by default
    df = loader_func(use_synthetic=True)

    assert isinstance(df, pd.DataFrame)

    # Check required columns
    required_cols = {"label_binary", "label_family", "group", "timestamp"}
    assert required_cols.issubset(df.columns), f"Missing columns: {required_cols - set(df.columns)}"

    # Check data types
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])

    # Feature columns should exist
    feature_cols = [c for c in df.columns if c not in required_cols and c != "is_synthetic"]
    assert len(feature_cols) > 0, "No feature columns found"

    # Check for NaN/Inf/Duplicates (should be cleaned by the loader)
    import numpy as np

    num_df = df.select_dtypes(include=[np.number])
    assert np.isinf(num_df).values.sum() == 0, "Found INF values after cleaning"
    assert df.isna().sum().sum() == 0, "Found NaN values after cleaning"
    assert df.duplicated().sum() == 0, "Found duplicate rows after cleaning"

    # Check for constant columns (should be removed by the loader)
    for col in df.columns:
        assert df[col].nunique() > 1 or col == "is_synthetic", f"Column {col} is constant but wasn't removed"
