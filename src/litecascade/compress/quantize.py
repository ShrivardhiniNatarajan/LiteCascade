"""INT8 quantization for LiteCascade using what this PyTorch build actually supports on CPU.

* ``dynamic``: torch dynamic INT8 for Linear + GRU weights (activations quantized on the fly);
  convolutions stay FP32. Works for every model, including baselines.
* ``qat``: eager-mode quantization-aware training (Jacob et al., 2018) of the static INT8 region
  (PIP + stem + all DS/TCN conv blocks, with Conv+BN folding and per-channel weight scales where the
  backend's default QAT qconfig provides them), then conversion; the remaining Linear/GRU layers
  (Sentinel/Analyst classifiers, BiGRU, attention pooling) get dynamic INT8.

Export to TFLite / TFLite-Micro is NOT implemented here (TensorFlow is not a dependency of the runnable
pipeline); the INT8 models are PyTorch quantized modules executed on the host CPU.
After quantization the gate MUST be re-calibrated (``litecascade.eval.gate.calibrate_arrays``).
"""

import copy
import warnings

import torch
import torch.ao.quantization as tq
import torch.nn as nn

from litecascade.models.litecascade import LiteCascade

_PREFERRED_ENGINES = ["x86", "fbgemm", "onednn", "qnnpack"]


def select_engine() -> str:
    supported = torch.backends.quantized.supported_engines
    for e in _PREFERRED_ENGINES:
        if e in supported:
            torch.backends.quantized.engine = e
            return e
    raise RuntimeError(f"No usable quantized engine in {supported}")


def quantize_dynamic_int8(model: nn.Module) -> nn.Module:
    select_engine()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return tq.quantize_dynamic(copy.deepcopy(model).cpu().eval(), {nn.Linear, nn.GRU, nn.LSTM}, dtype=torch.qint8)


def prepare_qat(model: LiteCascade) -> LiteCascade:
    """Fuse Conv+BN, attach fake-quant observers to the static INT8 region. Returns a new model in train mode."""
    engine = select_engine()
    m = copy.deepcopy(model).cpu()
    m.train()
    m.qconfig = tq.get_default_qat_qconfig(engine)
    for mod in m.float_modules():
        mod.qconfig = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tq.fuse_modules_qat(m, m.fuse_spec(), inplace=True)
        tq.prepare_qat(m, inplace=True)
    return m


def convert_qat(qat_model: LiteCascade) -> LiteCascade:
    """Convert a QAT-trained model to a real INT8 model (static region) + dynamic INT8 for Linear/GRU."""
    select_engine()
    m = copy.deepcopy(qat_model).cpu().eval()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = tq.convert(m, inplace=False)
        m = tq.quantize_dynamic(m, {nn.Linear, nn.GRU}, dtype=torch.qint8)
    m.eval()
    return m


def build_int8_skeleton(config: dict, mode: str) -> nn.Module:
    """Rebuild an (untrained) INT8 model with the right structure so a saved INT8 state_dict can be loaded."""
    float_model = LiteCascade(**config)
    if mode == "dynamic":
        return quantize_dynamic_int8(float_model)
    if mode == "qat":
        return convert_qat(prepare_qat(float_model))
    raise ValueError(mode)
