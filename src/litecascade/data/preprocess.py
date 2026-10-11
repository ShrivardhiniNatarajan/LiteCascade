"""Preprocessor: RobustScaler + optional PCA for LiteCascade data pipeline."""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler


class Preprocessor:
    """Fit on train, transform any split."""

    def __init__(self, n_components: int | None = None):
        self.n_components = n_components
        self.scaler = RobustScaler()
        self.pca = PCA(n_components=n_components) if n_components else None
        self._feature_cols: list[str] = []

    def fit(self, df_train: pd.DataFrame, feature_cols: list[str]) -> "Preprocessor":
        self._feature_cols = list(feature_cols)
        X = df_train[feature_cols].values
        self.scaler.fit(X)
        if self.pca is not None:
            self.pca.fit(self.scaler.transform(X))
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[self._feature_cols].values
        X_scaled = self.scaler.transform(X)

        non_feature = [c for c in df.columns if c not in self._feature_cols]
        result = df[non_feature].copy()

        if self.pca is not None:
            X_pca = self.pca.transform(X_scaled)
            for i in range(X_pca.shape[1]):
                result[f"pca_{i}"] = X_pca[:, i]
        else:
            for i, col in enumerate(self._feature_cols):
                result[col] = X_scaled[:, i]

        return result
