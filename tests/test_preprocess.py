import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler

from litecascade.data.preprocess import Preprocessor


def test_preprocessor_fit_on_train_only():
    # Create train and test sets
    np.random.seed(42)
    df_train = pd.DataFrame(
        {
            "feature_1": np.random.randn(100) * 10 + 5,
            "feature_2": np.random.randn(100) * 2 - 3,
            "group": ["A"] * 100,
        }
    )

    df_test = pd.DataFrame(
        {
            "feature_1": np.random.randn(50) * 100 + 50,  # very different dist
            "feature_2": np.random.randn(50) * 20 - 30,
            "group": ["B"] * 50,
        }
    )

    feature_cols = ["feature_1", "feature_2"]

    preproc = Preprocessor(n_components=1)
    preproc.fit(df_train, feature_cols)

    # Extract equivalent from raw sklearn to compare
    scaler_true = RobustScaler()
    scaler_true.fit(df_train[feature_cols].values)

    pca_true = PCA(n_components=1)
    pca_true.fit(scaler_true.transform(df_train[feature_cols].values))

    # 1. Assert scaler statistics equal those computed on train only
    assert np.allclose(preproc.scaler.center_, scaler_true.center_)
    assert np.allclose(preproc.scaler.scale_, scaler_true.scale_)

    # 2. Check PCA values match train fit (signs might flip, so check abs or mean)
    assert np.allclose(preproc.pca.mean_, pca_true.mean_)

    # 3. Transform test and verify the components are appended correctly
    df_test_transformed = preproc.transform(df_test)
    assert "pca_0" in df_test_transformed.columns
    assert "group" in df_test_transformed.columns
    assert "feature_1" not in df_test_transformed.columns  # original features replaced by PCA
