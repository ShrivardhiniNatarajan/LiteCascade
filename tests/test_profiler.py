import pytest
import torch
import torch.nn as nn

from litecascade.eval.profiler import expected_cost, profile_model


def test_expected_cost():
    # E[C] = 0.06 + (1 - 0.8) * 0.12 = 0.06 + 0.2 * 0.12 = 0.06 + 0.024 = 0.084
    cost = expected_cost(0.06, 0.12, 0.8)
    assert pytest.approx(cost, 1e-4) == 0.084


def test_profile_model():
    model = nn.Linear(10, 5)
    x = torch.randn(1, 10)

    profile = profile_model(model, x)

    # params: 10 * 5 + 5 = 55
    assert profile["params"] == 55
    # MACs: 10 * 5 = 50 for a single batched sample of size 1 (1x10 @ 10x5 -> 1x5 output, 50 multiplications)
    assert profile["macs"] == 50

    assert "size_fp32_kb" in profile
    assert "size_int8_kb" in profile
    assert "latency_p50_ms" in profile
    assert "process_rss_mb" in profile
