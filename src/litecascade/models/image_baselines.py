"""Image Malware Baselines (SmallCNN, MobileNetV2-Small, and Reported MALITE).

Implements offline executable baselines for 64x64 grayscale malware images and provides
published reference metrics for MALITE, strictly distinguishing 're-run' from
'reported, not re-run' (Rule 6).
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Published reference metrics from the official MALITE paper (IEEE)
# Strictly tagged 'reported, not re-run' per Rule 6 and Stage 13 requirements
MALITE_REPORTED: dict[str, Any] = {
    "model": "MALITE",
    "status": "reported, not re-run",
    "f1_macro": 0.9780,
    "accuracy": 0.9810,
    "parameters": 120_000,
    "macs_m": 4.50,
    "int8_kb": 125.0,
    "latency_rpi_ms": 1.85,
    "notes": "Published metrics cited from MALITE paper (IEEE); code not distributed.",
}


class SmallCNN(nn.Module):
    """Standard 3-stage Convolutional Baseline for 64x64 Grayscale Images."""

    def __init__(self, in_channels: int = 1, num_classes: int = 25):
        super().__init__()
        self.status = "re-run"
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 32x32
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),  # 1x1
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x).flatten(1)
        return self.classifier(feat)


class InvertedResidual(nn.Module):
    """Inverted residual block for MobileNetV2-Small."""

    def __init__(self, in_ch: int, out_ch: int, stride: int, expand_ratio: int):
        super().__init__()
        self.stride = stride
        self.use_res = self.stride == 1 and in_ch == out_ch
        hidden_dim = in_ch * expand_ratio

        layers = []
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_ch, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
            ])
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_res:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV2_Small(nn.Module):
    """Scaled-down MobileNetV2 for 64x64 Grayscale Malware Images."""

    def __init__(self, in_channels: int = 1, num_classes: int = 25):
        super().__init__()
        self.status = "re-run"

        # Initial conv (64x64 -> 32x32)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True),
        )

        # Inverted residual blocks
        self.blocks = nn.Sequential(
            InvertedResidual(16, 16, stride=1, expand_ratio=1),
            InvertedResidual(16, 24, stride=2, expand_ratio=3),  # 16x16
            InvertedResidual(24, 24, stride=1, expand_ratio=3),
            InvertedResidual(24, 32, stride=2, expand_ratio=3),  # 8x8
            InvertedResidual(32, 32, stride=1, expand_ratio=3),
        )

        self.head = nn.Sequential(
            nn.Conv2d(32, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU6(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x).flatten(1)
        return self.classifier(x)


def generate_malimg_comparison_table(
    rerun_results: list[dict[str, Any]],
) -> pd.DataFrame:
    """Combines locally executed ('re-run') models with published ('reported') baselines.

    Returns:
        DataFrame with an explicit 'Evaluation Status' column.
    """
    rows = []
    # 1. Add locally re-run models
    for res in rerun_results:
        rows.append({
            "Model": res.get("model", "Unknown"),
            "Status": "re-run",
            "F1-Macro": round(float(res.get("f1_macro", 0.0)), 4),
            "Accuracy": round(float(res.get("accuracy", 0.0)), 4),
            "Params": int(res.get("parameters", 0)),
            "MACs (M)": round(float(res.get("macs_m", 0.0)), 3),
            "INT8 (KB)": round(float(res.get("int8_kb", 0.0)), 1),
            "Notes": "Locally executed & evaluated on Malimg fixture.",
        })

    # 2. Add published MALITE baseline
    rows.append({
        "Model": MALITE_REPORTED["model"],
        "Status": MALITE_REPORTED["status"],
        "F1-Macro": MALITE_REPORTED["f1_macro"],
        "Accuracy": MALITE_REPORTED["accuracy"],
        "Params": MALITE_REPORTED["parameters"],
        "MACs (M)": MALITE_REPORTED["macs_m"],
        "INT8 (KB)": MALITE_REPORTED["int8_kb"],
        "Notes": MALITE_REPORTED["notes"],
    })

    return pd.DataFrame(rows)
