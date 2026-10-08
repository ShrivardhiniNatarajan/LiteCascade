import os
import tempfile
import time
from typing import Any

import numpy as np
import psutil
import torch
import torch.nn as nn
from fvcore.nn import ActivationCountAnalysis, FlopCountAnalysis, parameter_count


def expected_cost(c1: float, c2: float, rho: float) -> float:
    """Compute expected cost E[C] = C1 + (1-rho)*C2.

    Args:
        c1: Cost of early exit (Sentinel).
        c2: Cost of full model (Analyst).
        rho: Exit rate at Sentinel.

    Returns:
        Expected cost.
    """
    return c1 + (1 - rho) * c2


def speed_up(c1: float, c2: float, rho: float) -> float:
    """Compute speed-up S = (C1 + C2) / E[C]."""
    e_c = expected_cost(c1, c2, rho)
    if e_c == 0:
        return 0.0
    return (c1 + c2) / e_c


def profile_model(model: nn.Module, input_tensor: torch.Tensor) -> dict[str, Any]:
    """Profile the model for params, MACs, size, latency, memory."""
    model.eval()

    # 1. Parameter count
    params = parameter_count(model)
    if isinstance(params, dict):
        # fvcore returns a dict where "" is the total
        total_params = params.get("", sum(params.values()))
    else:
        total_params = params

    # 2. MACs
    flops = FlopCountAnalysis(model, input_tensor)
    flops.unsupported_ops_warnings(False)
    flops.uncalled_modules_warnings(False)
    flops.tracer_warnings("none")
    total_macs = flops.total()

    # 3. Activation Memory Estimate
    acts = ActivationCountAnalysis(model, input_tensor)
    total_acts = acts.total()

    # 4. FP32 and INT8 file size
    import os

    fd, path = tempfile.mkstemp(suffix=".pt")
    os.close(fd)
    torch.save(model.state_dict(), path)
    size_fp32_kb = os.path.getsize(path) / 1024.0
    os.remove(path)

    # Dynamic quantize linear layers as an estimate for INT8 size
    try:
        quantized = torch.ao.quantization.quantize_dynamic(model, {nn.Linear, nn.LSTM, nn.GRU}, dtype=torch.qint8)
        fd, path = tempfile.mkstemp(suffix=".pt")
        os.close(fd)
        torch.save(quantized.state_dict(), path)
        size_int8_kb = os.path.getsize(path) / 1024.0
        os.remove(path)
    except Exception:
        size_int8_kb = size_fp32_kb / 4.0  # rough estimate if QAT fails

    # 5. CPU Latency
    with torch.no_grad():
        # Warmup
        for _ in range(50):
            _ = model(input_tensor)

        # Timed runs
        times = []
        for _ in range(1000):
            start = time.perf_counter()
            _ = model(input_tensor)
            times.append(time.perf_counter() - start)

    p50_latency = float(np.percentile(times, 50)) * 1000.0  # ms
    p95_latency = float(np.percentile(times, 95)) * 1000.0  # ms

    # 6. RSS
    process = psutil.Process()
    rss_mb = process.memory_info().rss / (1024 * 1024)

    return {
        "params": int(total_params),
        "macs": int(total_macs),
        "activation_memory": int(total_acts),
        "size_fp32_kb": float(size_fp32_kb),
        "size_int8_kb": float(size_int8_kb),
        "latency_p50_ms": p50_latency,
        "latency_p95_ms": p95_latency,
        "process_rss_mb": float(rss_mb),
    }
