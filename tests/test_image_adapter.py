"""Unit tests for Stage 13: 2D Malimg Image Adapter and Baselines."""

from __future__ import annotations

import torch

from litecascade.data.malimg import (
    MALIMG_FAMILIES,
    create_malimg_lofo,
    load_malimg,
)
from litecascade.models.image_baselines import (
    MobileNetV2_Small,
    SmallCNN,
    generate_malimg_comparison_table,
)
from litecascade.models.litecascade2d import LiteCascade2D


def test_litecascade2d_shapes_train_mode():
    """Validates tensor output shapes in train mode on 64x64 grayscale inputs."""
    batch_size = 4
    channels = 1
    h, w = 64, 64
    num_classes = 25

    model = LiteCascade2D(in_channels=channels, c=24, num_classes=num_classes)
    x = torch.randn(batch_size, channels, h, w)

    l1, l2 = model(x, mode="train")
    assert l1.shape == (batch_size, num_classes), f"Sentinel output shape mismatch: {l1.shape}"
    assert l2.shape == (batch_size, num_classes), f"Analyst output shape mismatch: {l2.shape}"


def test_litecascade2d_shapes_cascade_mode():
    """Validates tensor output shapes and exit rate in cascade mode."""
    batch_size = 8
    model = LiteCascade2D(in_channels=1, c=24, num_classes=25)
    x = torch.randn(batch_size, 1, 64, 64)

    final_logits, exit_mask, rho = model(x, mode="cascade", tau_0=0.2, tau_1=0.8)
    assert final_logits.shape == (batch_size, 25)
    assert exit_mask.shape == (batch_size,)
    assert 0.0 <= rho <= 1.0


def test_litecascade2d_gate_logic():
    """Validates risk-controlled gate decision behavior."""
    model = LiteCascade2D(in_channels=1, c=24, num_classes=25)
    model.eval()

    # Extreme thresholds: tau_0=0.0, tau_1=0.0 => all samples should early exit
    x = torch.randn(6, 1, 64, 64)
    _, exit_mask, rho = model(x, mode="cascade", tau_0=0.0, tau_1=0.0)
    assert torch.all(exit_mask), "All samples must exit when tau_1=0.0"
    assert rho == 1.0

    # Extreme thresholds: tau_0=-1.0, tau_1=2.0 => no sample should early exit
    _, exit_mask_none, rho_none = model(x, mode="cascade", tau_0=-1.0, tau_1=2.0)
    assert not torch.any(exit_mask_none), "No sample should exit when tau_1=2.0"
    assert rho_none == 0.0


def test_image_baselines_forward():
    """Validates SmallCNN and MobileNetV2-Small forward passes."""
    x = torch.randn(3, 1, 64, 64)

    small_cnn = SmallCNN(in_channels=1, num_classes=25)
    out_cnn = small_cnn(x)
    assert out_cnn.shape == (3, 25)

    mobilenet = MobileNetV2_Small(in_channels=1, num_classes=25)
    out_mb = mobilenet(x)
    assert out_mb.shape == (3, 25)


def test_malimg_loader_and_lofo():
    """Validates Malimg synthetic loader and LOFO zero-leakage guarantee."""
    data = load_malimg(use_synthetic=True, num_samples=200, seed=42)
    assert data["X_train"].shape[1:] == (1, 64, 64)
    assert len(data["families"]) == 25
    assert len(data["X_train"]) + len(data["X_val"]) + len(data["X_test"]) == 200

    # Test LOFO
    held_out = "Allaple.A"
    lofo = create_malimg_lofo(held_out, num_samples=200, seed=42)
    held_idx = MALIMG_FAMILIES.index(held_out)

    assert not (lofo["y_train"] == held_idx).any(), "Training set contains held-out family!"
    assert (lofo["y_test"] == held_idx).all(), "Test set must strictly contain held-out family!"


def test_comparison_table_status_separation():
    """Validates that table strictly separates 're-run' vs 'reported, not re-run' baselines."""
    mock_rerun = [
        {"model": "LiteCascade2D", "f1_macro": 0.985, "parameters": 25000, "macs_m": 0.42, "int8_kb": 38.5},
        {"model": "SmallCNN", "f1_macro": 0.965, "parameters": 35000, "macs_m": 1.20, "int8_kb": 42.0},
        {"model": "MobileNetV2-Small", "f1_macro": 0.975, "parameters": 78000, "macs_m": 2.15, "int8_kb": 85.0},
    ]

    df = generate_malimg_comparison_table(mock_rerun)

    assert "Status" in df.columns, "Must contain 'Status' column per Stage 13 requirements"
    rerun_rows = df[df["Status"] == "re-run"]
    reported_rows = df[df["Status"] == "reported, not re-run"]

    assert len(rerun_rows) == 3
    assert len(reported_rows) == 1
    assert reported_rows.iloc[0]["Model"] == "MALITE"
    assert "MALITE" not in rerun_rows["Model"].values
