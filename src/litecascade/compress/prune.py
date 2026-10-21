"""Structured (channel) L1-norm pruning for LiteCascade (paper Sec. IV.G step i).

Whole channels are removed and the affected layers are rebuilt with smaller weight tensors, so parameter
count, MACs and file size really shrink (unlike unstructured masks). Pruned channel groups:

* hidden (expanded) channels of both Analyst inverted-bottleneck blocks: expand-conv filters ranked by
  L1 norm; the matching BN entries, depthwise filters and project-conv input columns are removed;
* output channels of the Sentinel DS block: pointwise filters ranked by L1 norm; matching BN entries and
  Sentinel classifier input columns are removed.

The shared stem / residual stream width ``c`` is left intact (pruning it would couple every block).
Fine-tune after pruning (see ``litecascade.train.cascade_trainer``).
"""

import copy

import torch
import torch.nn as nn

from litecascade.models.litecascade import LiteCascade


def _keep_idx(weight: torch.Tensor, ratio: float) -> torch.Tensor:
    """Indices of output channels to keep (largest L1 norms), sorted."""
    n = weight.shape[0]
    n_keep = max(1, int(round(n * (1 - ratio))))
    scores = weight.detach().abs().reshape(n, -1).sum(dim=1)
    return torch.sort(torch.topk(scores, n_keep).indices).values


def _slice_bn(bn: nn.BatchNorm1d, idx) -> nn.BatchNorm1d:
    new = nn.BatchNorm1d(len(idx), eps=bn.eps, momentum=bn.momentum)
    with torch.no_grad():
        new.weight.copy_(bn.weight[idx])
        new.bias.copy_(bn.bias[idx])
        new.running_mean.copy_(bn.running_mean[idx])
        new.running_var.copy_(bn.running_var[idx])
        new.num_batches_tracked.copy_(bn.num_batches_tracked)
    return new


def _copy_conv(old: nn.Conv1d, in_ch: int, out_ch: int, weight: torch.Tensor, groups: int = 1) -> nn.Conv1d:
    new = nn.Conv1d(
        in_ch,
        out_ch,
        old.kernel_size[0],
        stride=old.stride,
        padding=old.padding,
        dilation=old.dilation,
        groups=groups,
        bias=False,
    )
    with torch.no_grad():
        new.weight.copy_(weight)
    return new


def _prune_bottleneck(blk, ratio: float) -> int:
    idx = _keep_idx(blk.expand[0].weight, ratio)
    h = len(idx)
    exp, dw, proj = blk.expand[0], blk.depthwise[0], blk.project[0]
    blk.expand[0] = _copy_conv(exp, exp.in_channels, h, exp.weight[idx])
    blk.expand[1] = _slice_bn(blk.expand[1], idx)
    blk.depthwise[0] = _copy_conv(dw, h, h, dw.weight[idx], groups=h)
    blk.depthwise[1] = _slice_bn(blk.depthwise[1], idx)
    blk.project[0] = _copy_conv(proj, h, proj.out_channels, proj.weight[:, idx])
    return h


def _prune_sentinel(model: LiteCascade, ratio: float) -> int:
    ds = model.sentinel_ds
    idx = _keep_idx(ds.pointwise.weight, ratio)
    ds.pointwise = _copy_conv(ds.pointwise, ds.pointwise.in_channels, len(idx), ds.pointwise.weight[idx])
    ds.bn = _slice_bn(ds.bn, idx)
    fc = nn.Linear(len(idx), model.K)
    with torch.no_grad():
        fc.weight.copy_(model.sentinel_fc.weight[:, idx])
        fc.bias.copy_(model.sentinel_fc.bias)
    model.sentinel_fc = fc
    return len(idx)


def prune_litecascade(model: LiteCascade, ratio: float = 0.3, prune_sentinel: bool = True) -> LiteCascade:
    """Return a physically smaller copy of ``model`` with ``ratio`` of the prunable channels removed."""
    if not 0.0 <= ratio < 1.0:
        raise ValueError("ratio must be in [0, 1)")
    m = copy.deepcopy(model).cpu()
    if ratio == 0:
        return m
    h1 = _prune_bottleneck(m.analyst_blk1, ratio)
    h2 = _prune_bottleneck(m.analyst_blk2, ratio)
    m.analyst_hidden = (h1, h2)
    if prune_sentinel:
        m.sentinel_c = _prune_sentinel(m, ratio)
    return m
