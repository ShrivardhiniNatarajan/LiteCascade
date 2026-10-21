"""Efficiency profiling on the HOST CPU: params, MACs, serialized size, latency, RSS.

All latencies are measured on the machine running the code (not on a Raspberry Pi / MCU) and are labelled
as such in reports. MACs are counted with fvcore plus custom handles for LSTM/GRU and scaled-dot-product
attention, which fvcore otherwise silently counts as zero.
"""

import io
import logging
import time
import warnings
from typing import Any

import numpy as np
import psutil
import torch
import torch.nn as nn
from fvcore.nn import ActivationCountAnalysis, FlopCountAnalysis, parameter_count
from fvcore.nn.jit_handles import get_shape

logger = logging.getLogger(__name__)


def expected_cost(c1: float, c2: float, rho: float) -> float:
    """E[C] = C1 + (1 - rho) * C2 (paper eq. 6)."""
    return c1 + (1 - rho) * c2


def speed_up(c1: float, c2: float, rho: float) -> float:
    """S = (C1 + C2) / E[C] (paper eq. 6)."""
    e_c = expected_cost(c1, c2, rho)
    if e_c == 0:
        return 0.0
    return (c1 + c2) / e_c


# --------------------------------------------------------------------------------------------- MAC handles
def _const(v):
    try:
        return v.toIValue()
    except Exception:
        return None


def _rnn_handle(gates: int):
    """aten::lstm/gru(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first)."""

    def handle(inputs, outputs):
        x_shape = get_shape(inputs[0])
        out_shape = get_shape(outputs[0])
        num_layers = _const(inputs[4]) or 1
        bidir = bool(_const(inputs[7]))
        batch_first = bool(_const(inputs[8]))
        dirs = 2 if bidir else 1
        hidden = out_shape[-1] // dirs
        if batch_first:
            b, t, i = x_shape
        else:
            t, b, i = x_shape
        macs, inp = 0, i
        for _ in range(num_layers):
            macs += dirs * b * t * gates * hidden * (inp + hidden)
            inp = hidden * dirs
        return macs

    return handle


def _sdpa_handle(inputs, outputs):
    q = get_shape(inputs[0])
    k = get_shape(inputs[1])
    *batch, length, e = q
    s = k[-2]
    return int(np.prod(batch)) * length * s * e * 2  # QK^T and AV


CUSTOM_HANDLES = {
    "aten::lstm": _rnn_handle(4),
    "aten::gru": _rnn_handle(3),
    "aten::scaled_dot_product_attention": _sdpa_handle,
}


def count_macs(model: nn.Module, inputs) -> int:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        flops = FlopCountAnalysis(model, inputs).set_op_handle(**CUSTOM_HANDLES)
        flops.unsupported_ops_warnings(False)
        flops.uncalled_modules_warnings(False)
        flops.tracer_warnings("none")
        return int(flops.total())


def serialized_size_kb(model: nn.Module) -> float:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / 1024.0


def measure_latency(fn, *args, warmup: int = 20, runs: int = 200) -> dict:
    with torch.no_grad():
        for _ in range(warmup):
            fn(*args)
        times = []
        for _ in range(runs):
            start = time.perf_counter()
            fn(*args)
            times.append(time.perf_counter() - start)
    t = np.array(times) * 1000.0
    return {"p50_ms": float(np.percentile(t, 50)), "p95_ms": float(np.percentile(t, 95)), "mean_ms": float(t.mean())}


def profile_model(model: nn.Module, input_tensor: torch.Tensor, latency_runs: int = 1000) -> dict[str, Any]:
    """Profile a single-input model for params, MACs, size, latency (host CPU), memory."""
    model.eval()
    params = parameter_count(model)
    total_params = params.get("", sum(params.values())) if isinstance(params, dict) else params
    total_macs = count_macs(model, input_tensor)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        acts = ActivationCountAnalysis(model, input_tensor)
        acts.unsupported_ops_warnings(False)
        acts.uncalled_modules_warnings(False)
        acts.tracer_warnings("none")
        total_acts = acts.total()

    size_fp32_kb = serialized_size_kb(model)
    # Dynamic INT8 (weights of Linear/LSTM/GRU only; convolutions stay FP32): a real, runnable model.
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            quantized = torch.ao.quantization.quantize_dynamic(model, {nn.Linear, nn.LSTM, nn.GRU}, dtype=torch.qint8)
        size_int8_kb = serialized_size_kb(quantized)
    except Exception as e:  # pragma: no cover - depends on torch build
        logger.warning(f"dynamic quantization failed: {e}")
        size_int8_kb = float("nan")

    lat = measure_latency(model, input_tensor, warmup=min(50, latency_runs), runs=latency_runs)
    rss_mb = psutil.Process().memory_info().rss / (1024 * 1024)
    return {
        "params": int(total_params),
        "macs": int(total_macs),
        "activation_memory": int(total_acts),
        "size_fp32_kb": float(size_fp32_kb),
        "size_int8_kb": float(size_int8_kb),
        "size_int8_note": "torch dynamic INT8 (Linear/RNN weights only), serialized state_dict",
        "latency_p50_ms": lat["p50_ms"],
        "latency_p95_ms": lat["p95_ms"],
        "latency_device": "host CPU (not target IoT hardware)",
        "process_rss_mb": float(rss_mb),
    }


def profile_cascade(model, T: int, F: int, rho: float | None = None, latency_runs: int = 300, X_eval=None, gate=None) -> dict:
    """Per-path cost of a LiteCascade model.

    * analytic: C1 = MACs(PIP + stem + Sentinel), C2 = MACs(Analyst on stem features); E[C], S from eq. 6.
    * measured (host CPU, batch 1): Sentinel-path and full-path latency; and, if ``X_eval``/``gate`` are
      given, the mean per-window latency of real cascade inference over those windows vs. always-full.
    """
    from litecascade.models.litecascade import FullPath, SentinelPath

    model.eval()
    x = torch.randn(1, T, F)
    # MACs are counted on the float graph (fvcore does not count quantized kernels); for an INT8 model pass
    # its float counterpart if MACs are needed - the MAC count does not change with precision.
    try:
        c1 = count_macs(SentinelPath(model), x)
        full = count_macs(FullPath(model), x)
        c2 = full - c1
    except Exception as e:  # quantized kernels cannot be traced by fvcore
        logger.info(f"MAC counting skipped: {e}")
        c1 = full = c2 = None
    out = {
        "params": model.param_breakdown(),
        "macs_sentinel_path": c1,
        "macs_analyst_incremental": c2,
        "macs_full": full,
        "latency_sentinel_path": measure_latency(SentinelPath(model), x, runs=latency_runs),
        "latency_full_path": measure_latency(FullPath(model), x, runs=latency_runs),
        "latency_device": "host CPU, batch 1 (not target IoT hardware)",
    }
    if rho is not None and c1 is not None:
        out["rho"] = float(rho)
        out["expected_macs"] = expected_cost(c1, c2, rho)
        out["analytic_speedup_macs"] = speed_up(c1, c2, rho)
    if X_eval is not None and gate is not None and len(X_eval):
        xs = torch.as_tensor(X_eval[:latency_runs], dtype=torch.float32)
        tb, tm = gate["tau_b"], gate["tau_m"]
        casc_t, full_t = [], []
        with torch.no_grad():
            for i in range(min(10, len(xs))):  # warm-up
                model.cascade_predict(xs[i : i + 1], tb, tm)
                FullPath(model)(xs[i : i + 1])
            for i in range(len(xs)):
                xi = xs[i : i + 1]
                t0 = time.perf_counter()
                model.cascade_predict(xi, tb, tm)
                casc_t.append(time.perf_counter() - t0)
                t0 = time.perf_counter()
                FullPath(model)(xi)
                full_t.append(time.perf_counter() - t0)
        out["measured_cascade_mean_ms"] = float(np.mean(casc_t) * 1000)
        out["measured_full_mean_ms"] = float(np.mean(full_t) * 1000)
        out["measured_speedup"] = float(np.mean(full_t) / np.mean(casc_t))
        out["measured_on_windows"] = int(len(xs))
    return out
