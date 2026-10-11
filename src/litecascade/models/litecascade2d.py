"""2D LiteCascade Image Adapter Model for 64x64 Malware Texture Classification.

Replaces the 1D PIP stem with a 2D Depthwise-Separable stem while preserving
the dual-exit Sentinel/Analyst architecture, risk-controlled gate, and multi-task loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DSBlock2D(nn.Module):
    """2D Depthwise-Separable Convolution Block with BatchNorm and ReLU6."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.pointwise(self.depthwise(x))))


class ResidualInvertedBottleneck2D(nn.Module):
    """2D Inverted Bottleneck Block with expansion factor 2."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion: int = 2,
        stride: int = 1,
        kernel_size: int = 3,
        padding: int = 1,
    ):
        super().__init__()
        self.stride = stride
        self.use_res_connect = self.stride == 1 and in_channels == out_channels
        hidden_dim = in_channels * expansion

        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
        )
        self.depthwise = nn.Sequential(
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=hidden_dim,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
        )
        self.project = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.project(self.depthwise(self.expand(x)))
        if self.use_res_connect:
            return x + out
        return out


class LiteCascade2D(nn.Module):
    """LiteCascade 2D Image Adapter Architecture for 64x64 Malware Images.

    Stem: 2D Depthwise-Separable blocks (replaces 1D PIP stem)
    Sentinel Exit: DSBlock2D + GAP + Linear (fast early-exit)
    Analyst Exit: 2D Inverted Bottlenecks + Attention/GAP + Linear (deep analyst)
    """

    def __init__(
        self,
        in_channels: int = 1,
        c: int = 24,
        num_classes: int = 25,
        hidden: int = 24,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.c = c
        self.num_classes = num_classes

        # 1. 2D Shared Stem (64x64 -> 32x32 -> 16x16)
        self.stem1 = DSBlock2D(in_channels, c, kernel_size=3, stride=2, padding=1)
        self.stem2 = DSBlock2D(c, c, kernel_size=3, stride=2, padding=1)

        # 2. Sentinel Branch (16x16 -> 8x8 -> 1x1)
        self.sentinel_ds = DSBlock2D(c, c, kernel_size=3, stride=2, padding=1)
        self.sentinel_gap = nn.AdaptiveAvgPool2d((1, 1))
        self.sentinel_fc = nn.Linear(c, num_classes)
        self.T1 = nn.Parameter(torch.ones(1))

        # 3. Analyst Branch (16x16 -> 16x16 -> 8x8)
        self.analyst_blk1 = ResidualInvertedBottleneck2D(c, c, expansion=2, stride=1)
        self.analyst_blk2 = ResidualInvertedBottleneck2D(c, c, expansion=2, stride=2)
        self.analyst_gap = nn.AdaptiveAvgPool2d((1, 1))
        self.analyst_fc = nn.Linear(c, num_classes)
        self.T2 = nn.Parameter(torch.ones(1))

    def forward(
        self,
        x: torch.Tensor,
        mode: str = "train",
        tau_0: float = 0.5,
        tau_1: float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, float]:
        """Forward pass supporting train mode and cascade mode with early exit."""
        # 1. Stem
        x_stem = self.stem2(self.stem1(x))

        # 2. Sentinel Exit
        x_sent = self.sentinel_ds(x_stem)
        x_sent_pool = self.sentinel_gap(x_sent).flatten(1)
        logits_sent = self.sentinel_fc(x_sent_pool)

        # 3. Analyst Exit
        x_an = self.analyst_blk2(self.analyst_blk1(x_stem))
        x_an_pool = self.analyst_gap(x_an).flatten(1)
        logits_analyst = self.analyst_fc(x_an_pool)

        if mode == "train":
            return logits_sent, logits_analyst

        # Cascade mode with calibrated temperature and thresholds
        t1 = torch.clamp(self.T1, min=1e-4)
        probs_sent = F.softmax(logits_sent / t1, dim=-1)
        conf_sent = probs_sent.max(dim=-1).values

        # Risk-controlled exit decision
        early_exit = (conf_sent >= tau_1) | (conf_sent <= tau_0)
        final_logits = torch.where(early_exit.unsqueeze(-1), logits_sent, logits_analyst)
        exit_rate = float(early_exit.float().mean().item())

        return final_logits, early_exit, exit_rate

    def freeze_stem(self, freeze: bool = True) -> None:
        """Freezes or unfreezes the shared stem parameters."""
        for p in self.stem1.parameters():
            p.requires_grad = not freeze
        for p in self.stem2.parameters():
            p.requires_grad = not freeze
