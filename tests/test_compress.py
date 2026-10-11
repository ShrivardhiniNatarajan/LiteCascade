import os
import tempfile

import numpy as np
import pytest
import torch

from litecascade.compress.export import (
    TFLiteInterpreterWrapper,
    check_tflite_parity,
    export_onnx,
    export_tflite,
)
from litecascade.compress.prune import (
    apply_structured_pruning,
    count_zero_channels,
)
from litecascade.compress.quantize import (
    convert_qat_to_quantized,
    prepare_model_for_qat,
)
from litecascade.models.litecascade import LiteCascade


@pytest.fixture
def dummy_model_r():
    return LiteCascade(F_in=16, d=16, c=24, hidden=24, K=2, variant="R")


@pytest.fixture
def dummy_model_t():
    return LiteCascade(F_in=16, d=16, c=24, hidden=24, K=2, variant="T")


@pytest.mark.parametrize("ratio", [0.2, 0.3, 0.5])
def test_structured_pruning_ratios(dummy_model_r, ratio):
    stats_before = count_zero_channels(dummy_model_r)
    pruned = apply_structured_pruning(dummy_model_r, ratio=ratio)
    stats_after = count_zero_channels(pruned)

    assert stats_after["zero_weights"] > stats_before["zero_weights"]
    assert stats_after["weight_prune_rate"] > 0.0
    assert stats_after["zero_channels"] > 0

    # Ensure forward pass works after pruning
    x = torch.randn(2, 10, 16)
    l1, l2 = pruned(x, mode="train")
    assert l1.shape == (2, 2)
    assert l2.shape == (2, 2)


def test_qat_preparation_and_conversion_variant_t(dummy_model_t):
    # Variant T: Full QAT
    prepared = prepare_model_for_qat(dummy_model_t, variant="T")
    x = torch.randn(2, 10, 16)
    _ = prepared(x, mode="train")

    quantized = convert_qat_to_quantized(prepared, variant="T")
    quantized.eval()
    with torch.no_grad():
        out1, out2 = quantized(x, mode="train")
    assert out1.shape == (2, 2)
    assert out2.shape == (2, 2)


def test_qat_preparation_and_conversion_variant_r(dummy_model_r):
    # Variant R: QAT on Conv/Linear, Dynamic Quant on GRU
    prepared = prepare_model_for_qat(dummy_model_r, variant="R")
    x = torch.randn(2, 10, 16)
    _ = prepared(x, mode="train")

    quantized = convert_qat_to_quantized(prepared, variant="R")
    quantized.eval()
    with torch.no_grad():
        out1, out2 = quantized(x, mode="train")
    assert out1.shape == (2, 2)
    assert out2.shape == (2, 2)


def test_export_onnx(dummy_model_r):
    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_file = os.path.join(tmpdir, "model.onnx")
        dummy_in = torch.randn(1, 10, 16)
        out_path = export_onnx(dummy_model_r, dummy_in, onnx_file)
        assert os.path.exists(out_path)
        assert os.path.getsize(out_path) > 1000


def test_export_tflite_and_interpreter(dummy_model_r):
    with tempfile.TemporaryDirectory() as tmpdir:
        fp32_file = os.path.join(tmpdir, "model_fp32.tflite")
        int8_file = os.path.join(tmpdir, "model_int8.tflite")
        rep_data = np.random.randn(20, 10, 16).astype(np.float32)

        export_tflite(dummy_model_r, fp32_file, representative_dataset=rep_data, int8=False)
        export_tflite(dummy_model_r, int8_file, representative_dataset=rep_data, int8=True)

        assert os.path.exists(fp32_file)
        assert os.path.exists(int8_file)

        fp32_size = os.path.getsize(fp32_file)
        int8_size = os.path.getsize(int8_file)
        # INT8 should be smaller than FP32
        assert int8_size < fp32_size, f"Expected INT8 ({int8_size}) < FP32 ({fp32_size})"

        # Test inference via wrapper
        wrapper = TFLiteInterpreterWrapper(int8_file)
        test_x = np.random.randn(4, 10, 16).astype(np.float32)
        out0, out1 = wrapper.predict(test_x)
        assert out0.shape == (4, 2)
        assert out1.shape == (4, 2)


def test_parity_check(dummy_model_r):
    with tempfile.TemporaryDirectory() as tmpdir:
        int8_file = os.path.join(tmpdir, "model_int8.tflite")
        rep_data = np.random.randn(20, 10, 16).astype(np.float32)
        export_tflite(dummy_model_r, int8_file, representative_dataset=rep_data, int8=True)

        wrapper = TFLiteInterpreterWrapper(int8_file)
        test_windows = np.random.randn(20, 10, 16).astype(np.float32)
        report = check_tflite_parity(dummy_model_r, wrapper, test_windows, max_samples=20)

        assert "sentinel_agreement_pct" in report
        assert "analyst_agreement_pct" in report
        assert "overall_agreement_pct" in report
        assert "max_abs_logit_diff" in report
        assert report["overall_agreement_pct"] >= 0.0
