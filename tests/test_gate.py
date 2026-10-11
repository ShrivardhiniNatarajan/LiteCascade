import numpy as np
import torch
from litecascade.eval.gate import clopper_pearson_upper, fit_temperature

def test_cp_upper_bound_m0():
    """m=0, N=1000, alpha=0.05 -> upper bound = 1-0.05^(1/1000) ~ 0.00299"""
    ub = clopper_pearson_upper(0, 1000, alpha=0.05)
    expected = 1 - 0.05 ** (1.0 / 1000)
    assert abs(ub - expected) < 1e-4, f"Expected ~{expected:.6f}, got {ub:.6f}"

def test_cp_monotonicity():
    """U_CP increases with m."""
    N = 100
    prev = -1
    for m in range(0, 20):
        ub = clopper_pearson_upper(m, N, alpha=0.05)
        assert ub > prev, f"Non-monotonic at m={m}: {ub} <= {prev}"
        prev = ub

def test_temperature_scaling_preserves_argmax():
    """Temperature scaling does not change argmax."""
    logits = np.random.randn(50, 5).astype(np.float32)
    labels = np.random.randint(0, 5, 50)
    
    argmax_before = logits.argmax(axis=1)
    T = fit_temperature(logits, labels)
    
    scaled = logits / T
    argmax_after = scaled.argmax(axis=1)
    assert np.array_equal(argmax_before, argmax_after)
