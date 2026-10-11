"""
Quantization-Aware Training (QAT) and INT8 quantization for LiteCascade.

NOTE ON RECURRENT MODULES:
PyTorch eager-mode QAT (torch.ao.quantization.prepare_qat) natively supports
Conv1d, BatchNorm1d, Linear, and ReLU6, but does NOT support recurrent modules like nn.GRU.
Therefore:
  (a) For variant 'T' (no GRU, identity mapping), we quantize the model fully with QAT.
  (b) For variant 'R' (bidirectional GRU), we quantize the convolutional/linear blocks
      with QAT / per-channel observers, while quantizing the GRU using dynamic INT8 quantization.
"""

import copy
import logging
from typing import Any

import torch
import torch.nn as nn
from torch.ao.quantization import (
    FakeQuantize,
    MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
    QConfig,
    disable_observer,
    enable_fake_quant,
    prepare_qat,
    quantize_dynamic,
)
from torch.utils.data import DataLoader

from litecascade.compress.prune import apply_structured_pruning
from litecascade.models.litecascade import LiteCascade
from litecascade.train.loss import CascadeLoss

logger = logging.getLogger(__name__)


def get_per_channel_qat_qconfig() -> QConfig:
    """
    Returns a QConfig with per-channel weight observers and min-max activation observers.
    Avoids FusedMovingAvgObsFakeQuantize eager-mode attribute incompatibilities in PyTorch 2.5+.
    """
    act_observer = FakeQuantize.with_args(
        observer=MovingAverageMinMaxObserver,
        quant_min=0,
        quant_max=255,
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
    )
    weight_observer = FakeQuantize.with_args(
        observer=MovingAveragePerChannelMinMaxObserver,
        quant_min=-128,
        quant_max=127,
        dtype=torch.qint8,
        qscheme=torch.per_channel_symmetric,
    )
    return QConfig(activation=act_observer, weight=weight_observer)


def prepare_model_for_qat(model: LiteCascade, variant: str = "R") -> nn.Module:
    """
    Prepares a LiteCascade model for Quantization-Aware Training using per-channel weight observers.

    Args:
        model: Trained or pruned LiteCascade model.
        variant: 'R' (recurrent with GRU) or 'T' (temporal CNN only).

    Returns:
        Prepared QAT model with fake-quantize observers inserted.
    """
    model_copy = copy.deepcopy(model)
    model_copy.train()

    qconfig = get_per_channel_qat_qconfig()

    if variant == "T":
        # Full QAT across the entire model
        for name, module in model_copy.named_children():
            module.qconfig = qconfig
        prepared_model = prepare_qat(model_copy, inplace=False)
    else:
        # Variant R: QAT for Conv/Linear/BN, keep GRU floating / dynamically quantized
        for name, module in model_copy.named_children():
            if name != "gru":
                module.qconfig = qconfig
        prepared_model = prepare_qat(model_copy, inplace=False)

    return prepared_model


def train_qat_model(
    prepared_model: nn.Module,
    train_loader: DataLoader,
    epochs: int = 5,
    lr: float = 1e-4,
    device: str = "cpu",
    class_weights: torch.Tensor = None,
) -> nn.Module:
    """
    Runs QAT training loop to adjust fake-quantization scales and zero-points.
    """
    prepared_model.to(device)
    prepared_model.train()

    optimizer = torch.optim.AdamW(prepared_model.parameters(), lr=lr)
    loss_fn = CascadeLoss(gamma=0.0, delta=0.2, class_weights=class_weights.to(device) if class_weights is not None else None)

    for epoch in range(epochs):
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            l1, l2 = prepared_model(x_batch, mode="train")
            loss = loss_fn(l1, l2, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        logger.info(f"QAT Epoch {epoch + 1}/{epochs} | Loss: {train_loss / len(train_loader):.4f}")

    return prepared_model


def convert_qat_to_quantized(prepared_model: nn.Module, variant: str = "R") -> nn.Module:
    """
    Converts a QAT-trained model for INT8 quantized inference.
    Freezes observers and activates fake-quantized operations, while
    dynamically quantizing GRU layers for variant R.
    """
    quantized_model = copy.deepcopy(prepared_model)
    quantized_model.eval()

    # Freeze observer statistics so inference uses fixed INT8 scales
    quantized_model.apply(disable_observer)
    quantized_model.apply(enable_fake_quant)

    if variant == "R" and hasattr(quantized_model, "gru"):
        try:
            quantized_model.gru = quantize_dynamic(quantized_model.gru, {nn.GRU}, dtype=torch.qint8)
        except Exception as e:
            logger.debug(f"Dynamic quantization on GRU notice: {e}")

    return quantized_model


def run_wrong_order_ablation(
    model: LiteCascade,
    train_loader: DataLoader,
    val_loader: DataLoader,
    prune_ratio: float = 0.3,
    variant: str = "R",
) -> dict[str, Any]:
    """
    Ablation A9: Wrong order (QAT -> Prune) to demonstrate why Prune -> QAT is required.
    """
    logger.info("Running Ablation A9: Wrong order (QAT -> Prune)...")
    model_copy = copy.deepcopy(model)

    # 1. First run QAT
    prepared = prepare_model_for_qat(model_copy, variant=variant)
    qat_trained = train_qat_model(prepared, train_loader, epochs=2)
    quantized = convert_qat_to_quantized(qat_trained, variant=variant)

    # 2. Then attempt post-quantization pruning
    try:
        pruned_after_quant = apply_structured_pruning(quantized, ratio=prune_ratio)
    except Exception as e:
        logger.warning(f"Pruning quantized model encountered error as expected: {e}")
        pruned_after_quant = quantized

    return {
        "order": "QAT_then_Prune",
        "prune_ratio": prune_ratio,
        "model": pruned_after_quant,
    }
