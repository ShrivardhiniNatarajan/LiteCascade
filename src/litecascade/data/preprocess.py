"""Preprocessor: RobustScaler + optional (Incremental)PCA. Always fit on the TRAIN split only."""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.preprocessing import RobustScaler


class Preprocessor:
    """Fit on train, transform any split.

    With ``n_components`` set, ``transform`` replaces the features by PCA scores (used by baseline B5).
    LiteCascade instead consumes the scaled features and uses ``pca_init_params`` to initialise its
    learnable PIP layer (paper eq. 1: z = W_p (x - mu)).
    """

    def __init__(self, n_components: int | None = None, incremental: bool = False, batch_size: int = 4096):
        self.n_components = n_components
        self.incremental = incremental
        self.scaler = RobustScaler()
        if n_components:
            self.pca = IncrementalPCA(n_components=n_components, batch_size=batch_size) if incremental else PCA(n_components)
        else:
            self.pca = None
        self._feature_cols: list[str] = []

    @property
    def feature_cols(self) -> list[str]:
        return list(self._feature_cols)

    def fit(self, df_train: pd.DataFrame, feature_cols: list[str]) -> "Preprocessor":
        self._feature_cols = list(feature_cols)
        X = df_train[feature_cols].to_numpy(dtype=np.float64)
        self.scaler.fit(X)
        # RobustScaler leaves scale=1 for zero-IQR features; that is fine (no division by zero).
        if self.pca is not None:
            self.pca.fit(self.scaler.transform(X))
        return self

    def scale_array(self, X: np.ndarray) -> np.ndarray:
        """Scale a (..., F) array of raw features with the train-fitted scaler."""
        shape = X.shape
        out = self.scaler.transform(X.reshape(-1, shape[-1]).astype(np.float64))
        return np.clip(out, -1e4, 1e4).astype(np.float32).reshape(shape)

    def pca_array(self, X: np.ndarray) -> np.ndarray:
        """Project a scaled (..., F) array onto the PCA components."""
        shape = X.shape
        out = self.pca.transform(X.reshape(-1, shape[-1]))
        return out.astype(np.float32).reshape(*shape[:-1], out.shape[-1])

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        X_scaled = self.scale_array(df[self._feature_cols].to_numpy())
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

    def pca_init_params(self) -> tuple[np.ndarray, np.ndarray]:
        """(W_p, mu) with W_p of shape (d, F): top-d principal axes of the scaled train features."""
        if self.pca is None:
            raise ValueError("Preprocessor was built without n_components")
        return self.pca.components_.astype(np.float32), self.pca.mean_.astype(np.float32)
