"""LiteCascade: PIP layer -> shared DS stem -> Sentinel exit (+ gate) -> Analyst exit (paper Sec. IV.B-E).

* PIP (eq. 1):  z_t = W_p (x_t - mu), initialised from the train-split PCA and fine-tuned.
* Stem:         one depthwise-separable Conv1d over time (k=3, d -> c).
* Sentinel:     one more DS block, global average pooling, linear; p1 = softmax(W1 GAP(H) / T1) (eq. 2).
                T1 is NOT trained with the network; it is fitted afterwards on the calibration split.
* Analyst:      two residual inverted-bottleneck DS-TCN blocks (dilation 1, 2; expansion 2),
                BiGRU(hidden) for variant "R" (omitted in variant "T"), attention pooling, linear.
* Gate (eq. 3): exit iff (argmax p1 = 0 and p1[0] >= tau_b) or (argmax p1 != 0 and max p1 >= tau_m).
                In ``cascade_predict`` the Analyst is executed ONLY for windows that did not exit.

QuantStub/DeQuantStub/FloatFunctional are identities in float mode; they mark the INT8 region
(PIP + stem + conv blocks) for eager-mode quantization-aware training (see litecascade.compress.quantize).
"""

import numpy as np
import torch
import torch.nn as nn
from torch.ao.nn.quantized import FloatFunctional
from torch.ao.quantization import DeQuantStub, QuantStub


class PIPLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    @torch.no_grad()
    def init_from_pca(self, components: np.ndarray, mean: np.ndarray):
        """W = components (d x F), b = -W mu, so that z = W (x - mu) exactly as in eq. 1."""
        W = torch.as_tensor(components, dtype=torch.float32)
        mu = torch.as_tensor(mean, dtype=torch.float32)
        if W.shape != self.linear.weight.shape:
            raise ValueError(f"PCA components {tuple(W.shape)} != PIP weight {tuple(self.linear.weight.shape)}")
        self.linear.weight.copy_(W)
        self.linear.bias.copy_(-(W @ mu))

    def forward(self, x):
        return self.linear(x)


class DSBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1):
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.pointwise(self.depthwise(x))))

    def fuse_spec(self, prefix):
        return [[f"{prefix}.pointwise", f"{prefix}.bn"]]


class ResidualInvertedBottleneckTCN(nn.Module):
    def __init__(self, in_channels, out_channels, expansion=2, kernel_size=3, padding=1, dilation=1, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or in_channels * expansion
        self.use_res_connect = in_channels == out_channels
        self.expand = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=1, bias=False), nn.BatchNorm1d(hidden_dim), nn.ReLU6(inplace=True)
        )
        self.depthwise = nn.Sequential(
            nn.Conv1d(
                hidden_dim, hidden_dim, kernel_size=kernel_size, padding=padding, dilation=dilation, groups=hidden_dim, bias=False
            ),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU6(inplace=True),
        )
        self.project = nn.Sequential(nn.Conv1d(hidden_dim, out_channels, kernel_size=1, bias=False), nn.BatchNorm1d(out_channels))
        self.skip_add = FloatFunctional()

    def forward(self, x):
        out = self.project(self.depthwise(self.expand(x)))
        if self.use_res_connect:
            return self.skip_add.add(x, out)
        return out

    def fuse_spec(self, prefix):
        return [[f"{prefix}.{m}.0", f"{prefix}.{m}.1"] for m in ("expand", "depthwise", "project")]


class LiteCascade(nn.Module):
    def __init__(
        self, F_in, d=32, c=24, hidden=24, T_len=10, K=2, variant="R", verify_alerts=False, analyst_hidden=None, sentinel_c=None
    ):
        """``analyst_hidden`` (h1, h2) and ``sentinel_c`` override channel widths after structured pruning."""
        super().__init__()
        if variant not in ("R", "T"):
            raise ValueError("variant must be 'R' or 'T'")
        self.variant, self.K, self.F_in, self.d, self.c, self.hidden, self.T_len = variant, K, F_in, d, c, hidden, T_len
        self.verify_alerts = verify_alerts
        h1, h2 = analyst_hidden or (c * 2, c * 2)
        self.analyst_hidden = (int(h1), int(h2))
        self.sentinel_c = int(sentinel_c or c)

        self.quant = QuantStub()
        self.pip = PIPLayer(F_in, d)
        self.stem = DSBlock(d, c, kernel_size=3, padding=1)

        # Sentinel exit
        self.sentinel_ds = DSBlock(c, self.sentinel_c, kernel_size=3, padding=1)
        self.dequant_s = DeQuantStub()
        self.sentinel_fc = nn.Linear(self.sentinel_c, K)

        # Analyst exit
        self.analyst_blk1 = ResidualInvertedBottleneckTCN(c, c, dilation=1, padding=1, hidden_dim=h1)
        self.analyst_blk2 = ResidualInvertedBottleneckTCN(c, c, dilation=2, padding=2, hidden_dim=h2)
        self.dequant_a = DeQuantStub()
        if variant == "R":
            self.gru = nn.GRU(input_size=c, hidden_size=hidden, batch_first=True, bidirectional=True)
            attn_in = hidden * 2
        else:
            self.gru = nn.Identity()
            attn_in = c
        self.attn_query = nn.Parameter(torch.randn(1, 1, attn_in) * attn_in**-0.5)
        self.analyst_fc = nn.Linear(attn_in, K)

        # Calibrated temperatures (fitted post-hoc on the calibration split, never trained).
        self.register_buffer("T1", torch.ones(1))
        self.register_buffer("T2", torch.ones(1))

    # ------------------------------------------------------------------ building blocks
    def config(self) -> dict:
        return {
            "F_in": self.F_in, "d": self.d, "c": self.c, "hidden": self.hidden, "T_len": self.T_len,
            "K": self.K, "variant": self.variant, "analyst_hidden": list(self.analyst_hidden), "sentinel_c": self.sentinel_c,
        }  # fmt: skip

    def freeze_pip(self, freeze=True):
        for p in self.pip.parameters():
            p.requires_grad = not freeze

    def init_pip_from_pca(self, components, mean):
        self.pip.init_from_pca(components, mean)

    def stem_forward(self, x):
        """(B, T, F) -> shared stem features (B, c, T)."""
        z = self.pip(self.quant(x))
        return self.stem(z.transpose(1, 2).contiguous())

    def sentinel_logits(self, h):
        s = self.dequant_s(self.sentinel_ds(h))
        return self.sentinel_fc(s.mean(dim=-1))

    def analyst_logits(self, h):
        a = self.analyst_blk2(self.analyst_blk1(h))
        a = self.dequant_a(a).transpose(1, 2)  # (B, T, c)
        if self.variant == "R":
            a, _ = self.gru(a)
        weights = torch.softmax(torch.matmul(a, self.attn_query.transpose(1, 2)), dim=1)  # (B, T, 1)
        return self.analyst_fc(torch.sum(weights * a, dim=1))

    # backwards-compatible name
    def _analyst_forward(self, h):
        return self.analyst_logits(h)

    # ------------------------------------------------------------------ gate
    @staticmethod
    def exit_mask(probs1: torch.Tensor, tau_b: float, tau_m: float, verify_alerts: bool = False) -> torch.Tensor:
        """Eq. 3 on calibrated Sentinel probabilities (vectorised)."""
        conf, pred = probs1.max(dim=-1)
        benign_exit = (pred == 0) & (probs1[:, 0] >= tau_b)
        if verify_alerts:
            return benign_exit
        return benign_exit | ((pred != 0) & (conf >= tau_m))

    @torch.no_grad()
    def cascade_predict(self, x, tau_b: float, tau_m: float):
        """Genuine cascade inference: the Analyst runs only on windows the gate did not release."""
        h = self.stem_forward(x)
        p1 = torch.softmax(self.sentinel_logits(h) / self.T1, dim=-1)
        exit_mask = self.exit_mask(p1, tau_b, tau_m, self.verify_alerts)
        probs = p1.clone()
        idx = torch.nonzero(~exit_mask, as_tuple=True)[0]
        if len(idx) > 0:
            l2 = self.analyst_logits(h.index_select(0, idx))
            probs[idx] = torch.softmax(l2 / self.T2, dim=-1)
        return {
            "preds": probs.argmax(dim=-1),
            "probs": probs,
            "sentinel_probs": p1,
            "exit_mask": exit_mask,
            "n_analyst": int(len(idx)),
        }

    def forward(self, x, mode="train", tau_b=0.0, tau_m=0.0):
        """mode="train": raw (uncalibrated) logits of both exits, (B, K) each.
        mode="infer": (preds, exit_mask, info) via the gate, skipping the Analyst for exited windows."""
        if mode == "train":
            h = self.stem_forward(x)
            return self.sentinel_logits(h), self.analyst_logits(h)
        if mode == "infer":
            out = self.cascade_predict(x, tau_b, tau_m)
            return out["preds"], out["exit_mask"], {"probs": out["probs"], "n_analyst": out["n_analyst"]}
        raise ValueError(f"unknown mode {mode}")

    # ------------------------------------------------------------------ reporting / quantization helpers
    def param_breakdown(self) -> dict:
        def n(*mods):
            return sum(p.numel() for m in mods for p in m.parameters())

        shared = n(self.pip, self.stem)
        sentinel_exit = n(self.sentinel_ds, self.sentinel_fc)
        analyst_exit = n(self.analyst_blk1, self.analyst_blk2, self.gru, self.analyst_fc) + self.attn_query.numel()
        return {
            "pip": n(self.pip),
            "stem": n(self.stem),
            "sentinel_exit_only": sentinel_exit,
            "analyst_exit_only": analyst_exit,
            "sentinel_path_total": shared + sentinel_exit,
            "total": shared + sentinel_exit + analyst_exit,
        }

    def fuse_spec(self):
        spec = self.stem.fuse_spec("stem") + self.sentinel_ds.fuse_spec("sentinel_ds")
        spec += self.analyst_blk1.fuse_spec("analyst_blk1") + self.analyst_blk2.fuse_spec("analyst_blk2")
        return spec

    def float_modules(self):
        """Modules kept outside the static-INT8 region (dynamic INT8 is applied to them instead)."""
        return [self.sentinel_fc, self.gru, self.analyst_fc]


class SentinelPath(nn.Module):
    """PIP + stem + Sentinel exit, for profiling the early-exit path cost C1."""

    def __init__(self, model: LiteCascade):
        super().__init__()
        self.m = model

    def forward(self, x):
        return self.m.sentinel_logits(self.m.stem_forward(x))


class AnalystPath(nn.Module):
    """Analyst exit on precomputed stem features, for profiling the incremental cost C2."""

    def __init__(self, model: LiteCascade):
        super().__init__()
        self.m = model

    def forward(self, h):
        return self.m.analyst_logits(h)


class FullPath(nn.Module):
    """Both exits (what an always-full model pays: C1 + C2)."""

    def __init__(self, model: LiteCascade):
        super().__init__()
        self.m = model

    def forward(self, x):
        h = self.m.stem_forward(x)
        return self.m.sentinel_logits(h), self.m.analyst_logits(h)
