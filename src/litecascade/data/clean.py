import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the dataframe by handling inf/NaN, constant cols, and duplicates.

    Args:
        df: Input pandas DataFrame.

    Returns:
        Cleaned pandas DataFrame.
    """
    initial_rows, initial_cols = df.shape

    # 1. Handle Inf/NaN
    inf_count = np.isinf(df.select_dtypes(include=[np.number])).values.sum()
    if inf_count > 0:
        logger.info(f"Replacing {inf_count} INF values with NaN")
        df = df.replace([np.inf, -np.inf], np.nan)

    nan_counts = df.isna().sum().sum()
    if nan_counts > 0:
        logger.info(f"Dropping rows with NaN values. Found {nan_counts} NaNs.")
        df = df.dropna()

    # 2. Handle duplicates
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        logger.info(f"Dropping {dup_count} duplicate rows.")
        df = df.drop_duplicates()

    # 3. Handle constant columns
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    if constant_cols:
        logger.info(f"Dropping {len(constant_cols)} constant columns: {constant_cols}")
        df = df.drop(columns=constant_cols)

    final_rows, final_cols = df.shape
    logger.info(f"Cleaned dataset: {initial_rows}x{initial_cols} -> {final_rows}x{final_cols}")

    return df
