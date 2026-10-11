"""
Structured channel pruning for LiteCascade.
Applies L1-norm structured pruning on convolutional channels at ratios {0.2, 0.3, 0.5}.
"""

import copy
import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader

from litecascade.eval.metrics import compute_detection_metrics
from litecascade.models.litecascade import LiteCascade
from litecascade.train.loss import CascadeLoss

logger = logging.getLogger(__name__)


def get_prunable_conv_layers(model: nn.Module) -> list[nn.Conv1d]:
    """Find all standard (non-depthwise) Conv1d layers suitable for structured channel pruning."""
    conv_layers = []
    for module in model.modules():
        if isinstance(module, nn.Conv1d):
            # Only prune layers with groups == 1 to avoid mismatching depthwise channel groupings
            if module.groups == 1 and module.weight.shape[0] > 4:
                conv_layers.append(module)
    return conv_layers


def apply_structured_pruning(
    model: LiteCascade, ratio: float = 0.3, prune_ratio: float | None = None
) -> LiteCascade:
    """
    Applies structured channel pruning by L1-norm ranking to convolutional layers.

    Args:
        model: LiteCascade instance
        ratio: Pruning ratio in {0.2, 0.3, 0.5} (or float in [0.0, 1.0))
        prune_ratio: Alias for ratio

    Returns:
        Pruned model with masks applied and removed (permanent zeroing).
    """
    if prune_ratio is not None:
        ratio = prune_ratio
    if not (0.0 <= ratio < 1.0):
        raise ValueError(f"Pruning ratio must be in [0.0, 1.0), got {ratio}")

    if ratio == 0.0:
        return model

    conv_layers = get_prunable_conv_layers(model)
    logger.info(f"Applying structured L1 pruning at ratio={ratio:.2f} on {len(conv_layers)} Conv1d layers.")

    for conv in conv_layers:
        # Structured pruning along dim=0 (output channels) by L1 norm
        prune.ln_structured(conv, name="weight", amount=ratio, n=1, dim=0)
        # Make the pruning permanent by removing the reparameterization hook
        prune.remove(conv, "weight")

    return model


def count_zero_channels(model: nn.Module) -> dict[str, Any]:
    """Count zeroed out output channels across prunable Conv1d layers."""
    total_channels = 0
    zero_channels = 0
    total_weights = 0
    zero_weights = 0

    for module in model.modules():
        if isinstance(module, nn.Conv1d):
            w = module.weight.detach()
            total_weights += w.numel()
            zero_weights += (w == 0).sum().item()
            # Channel L1 norm along dim 0
            ch_norms = w.abs().sum(dim=list(range(1, w.ndim)))
            total_channels += len(ch_norms)
            zero_channels += (ch_norms == 0).sum().item()

    return {
        "total_channels": total_channels,
        "zero_channels": zero_channels,
        "channel_prune_rate": float(zero_channels / max(1, total_channels)),
        "total_weights": total_weights,
        "zero_weights": zero_weights,
        "weight_prune_rate": float(zero_weights / max(1, total_weights)),
    }


def fine_tune_pruned_model(
    model: LiteCascade,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 10,
    lr: float = 5e-4,
    device: str = "cpu",
    teacher_logits_train: torch.Tensor = None,
    class_weights: torch.Tensor = None,
) -> dict[str, Any]:
    """
    Fine-tunes the pruned model for N epochs with CascadeLoss.
    """
    model.to(device)
    loss_fn = CascadeLoss(
        gamma=0.5 if teacher_logits_train is not None else 0.0,
        delta=0.3,
        tau_kd=3.0,
        class_weights=class_weights.to(device) if class_weights is not None else None,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best_f1 = -1.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            l1, l2 = model(x_batch, mode="train")
            t_logits = teacher_logits_train[0 : len(x_batch)] if teacher_logits_train is not None else None
            loss = loss_fn(l1, l2, y_batch, t_logits)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                _, l2 = model(x_batch, mode="train")
                all_preds.extend(torch.argmax(l2, dim=-1).cpu().numpy())
                all_labels.extend(y_batch.numpy())

        if len(all_labels) > 0:
            val_metrics = compute_detection_metrics(torch.tensor(all_labels).numpy(), torch.tensor(all_preds).numpy())
            val_f1 = val_metrics["f1_macro"]
        else:
            val_f1 = 0.0

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    return {"best_f1": best_f1, "epochs_trained": epochs}
