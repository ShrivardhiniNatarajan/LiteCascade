import numpy as np

from litecascade.eval.perturb import (
    apply_gaussian_noise,
    apply_size_padding,
    apply_time_jitter,
    evaluate_under_perturbation,
    identify_feature_types,
)
from litecascade.models.litecascade import LiteCascade


def test_gaussian_noise_zero_sigma():
    """Assert sigma=0 returns identical data."""
    x = np.random.randn(20, 10, 16).astype(np.float32)
    x_pert = apply_gaussian_noise(x, sigma=0.0)
    np.testing.assert_array_equal(x, x_pert)


def test_perturb_reproducible_seeds():
    """Assert seeds are reproducible and distinct seeds produce distinct noise."""
    x = np.random.randn(20, 10, 16).astype(np.float32)

    pert_a1 = apply_gaussian_noise(x, sigma=0.1, seed=42)
    pert_a2 = apply_gaussian_noise(x, sigma=0.1, seed=42)
    pert_b = apply_gaussian_noise(x, sigma=0.1, seed=99)

    np.testing.assert_array_equal(pert_a1, pert_a2)
    assert not np.array_equal(pert_a1, pert_b)


def test_time_jitter_columns():
    """Assert time jitter only modifies designated timing columns."""
    feature_names = [f"feat_{i}" for i in range(12)]
    feature_names[0] = "packet_time_diff"
    feature_names[1] = "flow_duration"

    x = np.ones((10, 10, 12), dtype=np.float32)
    x_pert, timing_cols = apply_time_jitter(x, feature_names=feature_names, jitter_ratio=0.2, seed=42)

    assert 0 in timing_cols
    assert 1 in timing_cols

    # Timing columns must be modified
    assert not np.array_equal(x[:, :, 0], x_pert[:, :, 0])
    assert not np.array_equal(x[:, :, 1], x_pert[:, :, 1])

    # Non-timing columns must remain strictly identical
    for col in range(2, 12):
        np.testing.assert_array_equal(x[:, :, col], x_pert[:, :, col])


def test_size_padding_columns():
    """Assert packet size padding shifts size columns positively and leaves others untouched."""
    feature_names = [f"feat_{i}" for i in range(12)]
    feature_names[4] = "total_bytes"
    feature_names[5] = "packet_length"

    x = np.zeros((10, 10, 12), dtype=np.float32)
    x_pert, size_cols = apply_size_padding(x, feature_names=feature_names, pad_amount=0.5, seed=42)

    assert 4 in size_cols
    assert 5 in size_cols

    # Size columns must have positive shift
    np.testing.assert_allclose(x_pert[:, :, 4], 0.5)
    np.testing.assert_allclose(x_pert[:, :, 5], 0.5)

    # Non-size columns must remain strictly identical
    for col in [0, 1, 2, 3, 6, 7, 8, 9, 10, 11]:
        np.testing.assert_array_equal(x[:, :, col], x_pert[:, :, col])


def test_evaluate_under_perturbation():
    """Assert evaluation function correctly computes metrics and rho on clean data."""
    model = LiteCascade(F_in=8, d=8, c=16, hidden=16, K=2, variant="R")
    x_test = np.random.randn(30, 10, 8).astype(np.float32)
    y_test = np.random.randint(0, 2, size=(30,))

    res = evaluate_under_perturbation(model, x_test, y_test, tau_b=0.5, tau_m=0.8, temperature=1.0)

    assert "f1_macro" in res
    assert "recall" in res
    assert "rho" in res
    assert "analyst_recall" in res
    assert 0.0 <= res["rho"] <= 1.0
    assert 0.0 <= res["f1_macro"] <= 1.0


def test_identify_feature_types_fallback():
    """Assert identify_feature_types falls back gracefully when names are generic."""
    timing_idx, size_idx, meta = identify_feature_types(None, total_features=16)
    assert len(timing_idx) > 0
    assert len(size_idx) > 0
    assert "EMULATION" in meta["emulation_note"]
