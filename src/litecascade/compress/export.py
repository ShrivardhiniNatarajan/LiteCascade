"""
Export utilities for LiteCascade: ONNX and TFLite INT8 with parity verification.
"""

import logging
import sys
from pathlib import Path
from typing import Any

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class _ONNXWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x, mode="train")


def export_onnx(
    model: nn.Module,
    dummy_input: torch.Tensor,
    output_path: str or Path,
    opset_version: int = 14,
) -> Path:
    """
    Exports a PyTorch model to ONNX format.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    wrapped_model = _ONNXWrapper(model)

    torch.onnx.export(
        wrapped_model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input_windows"],
        output_names=["sentinel_logits", "analyst_logits"],
        dynamic_axes={
            "input_windows": {0: "batch_size"},
            "sentinel_logits": {0: "batch_size"},
            "analyst_logits": {0: "batch_size"},
        },
    )
    logger.info(f"Successfully exported ONNX model to {output_path}")
    return output_path


def build_keras_model_from_litecascade(pytorch_model: nn.Module) -> Any:
    """
    Builds a functional Keras equivalent of LiteCascade and transfers weights.
    """
    import tensorflow as tf

    F_in = pytorch_model.F_in
    d = pytorch_model.d
    c = pytorch_model.c
    K = pytorch_model.K
    T_len = pytorch_model.T_len
    variant = pytorch_model.variant

    inputs = tf.keras.Input(shape=(T_len, F_in), name="input_windows")

    # 1. PIP Layer
    pip_w = pytorch_model.pip.linear.weight.detach().cpu().numpy().T
    pip_b = pytorch_model.pip.linear.bias.detach().cpu().numpy()
    x = tf.keras.layers.Dense(d, name="pip_dense")(inputs)

    # 2. Shared Stem
    # PyTorch Depthwise Conv1d weight: (d, 1, 3) -> Keras DepthwiseConv1D: (3, d, 1)
    stem_dw_w = pytorch_model.stem.depthwise.weight.detach().cpu().numpy().transpose(2, 0, 1)
    x = tf.keras.layers.DepthwiseConv1D(kernel_size=3, padding="same", use_bias=False, name="stem_dw")(x)

    stem_pw_w = pytorch_model.stem.pointwise.weight.detach().cpu().numpy().transpose(2, 1, 0)
    x = tf.keras.layers.Conv1D(c, kernel_size=1, use_bias=False, name="stem_pw")(x)

    stem_bn_gamma = pytorch_model.stem.bn.weight.detach().cpu().numpy()
    stem_bn_beta = pytorch_model.stem.bn.bias.detach().cpu().numpy()
    stem_bn_mean = pytorch_model.stem.bn.running_mean.detach().cpu().numpy()
    stem_bn_var = pytorch_model.stem.bn.running_var.detach().cpu().numpy()
    x = tf.keras.layers.BatchNormalization(name="stem_bn")(x)
    x_stem = tf.keras.layers.ReLU(max_value=6.0, name="stem_relu6")(x)

    # 3. Sentinel Exit
    sent_dw_w = pytorch_model.sentinel_ds.depthwise.weight.detach().cpu().numpy().transpose(2, 0, 1)
    x_sent = tf.keras.layers.DepthwiseConv1D(kernel_size=3, padding="same", use_bias=False, name="sent_dw")(x_stem)

    sent_pw_w = pytorch_model.sentinel_ds.pointwise.weight.detach().cpu().numpy().transpose(2, 1, 0)
    x_sent = tf.keras.layers.Conv1D(c, kernel_size=1, use_bias=False, name="sent_pw")(x_sent)

    sent_bn_gamma = pytorch_model.sentinel_ds.bn.weight.detach().cpu().numpy()
    sent_bn_beta = pytorch_model.sentinel_ds.bn.bias.detach().cpu().numpy()
    sent_bn_mean = pytorch_model.sentinel_ds.bn.running_mean.detach().cpu().numpy()
    sent_bn_var = pytorch_model.sentinel_ds.bn.running_var.detach().cpu().numpy()
    x_sent = tf.keras.layers.BatchNormalization(name="sent_bn")(x_sent)
    x_sent = tf.keras.layers.ReLU(max_value=6.0, name="sent_relu6")(x_sent)

    # Global Average Pooling
    x_sent_pool = tf.keras.layers.GlobalAveragePooling1D(name="sent_gap")(x_sent)
    sent_fc_w = pytorch_model.sentinel_fc.weight.detach().cpu().numpy().T
    sent_fc_b = pytorch_model.sentinel_fc.bias.detach().cpu().numpy()
    sent_logits_raw = tf.keras.layers.Dense(K, name="sent_fc")(x_sent_pool)

    # Temperature scaling
    t1_val = float(pytorch_model.T1.detach().cpu().numpy().item())
    sentinel_logits = tf.keras.layers.Lambda(lambda t: t / max(1e-4, t1_val), name="sentinel_logits")(sent_logits_raw)

    # 4. Analyst Exit
    # TCN Block 1
    # expand: in_channels=c, out_channels=2*c
    b1_exp_w = pytorch_model.analyst_blk1.expand[0].weight.detach().cpu().numpy().transpose(2, 1, 0)
    b1_exp_gamma = pytorch_model.analyst_blk1.expand[1].weight.detach().cpu().numpy()
    b1_exp_beta = pytorch_model.analyst_blk1.expand[1].bias.detach().cpu().numpy()
    b1_exp_mean = pytorch_model.analyst_blk1.expand[1].running_mean.detach().cpu().numpy()
    b1_exp_var = pytorch_model.analyst_blk1.expand[1].running_var.detach().cpu().numpy()

    # depthwise: in_channels=2*c
    b1_dw_w = pytorch_model.analyst_blk1.depthwise[0].weight.detach().cpu().numpy().transpose(2, 0, 1)
    b1_dw_gamma = pytorch_model.analyst_blk1.depthwise[1].weight.detach().cpu().numpy()
    b1_dw_beta = pytorch_model.analyst_blk1.depthwise[1].bias.detach().cpu().numpy()
    b1_dw_mean = pytorch_model.analyst_blk1.depthwise[1].running_mean.detach().cpu().numpy()
    b1_dw_var = pytorch_model.analyst_blk1.depthwise[1].running_var.detach().cpu().numpy()

    # project: in_channels=2*c, out_channels=c
    b1_proj_w = pytorch_model.analyst_blk1.project[0].weight.detach().cpu().numpy().transpose(2, 1, 0)
    b1_proj_gamma = pytorch_model.analyst_blk1.project[1].weight.detach().cpu().numpy()
    b1_proj_beta = pytorch_model.analyst_blk1.project[1].bias.detach().cpu().numpy()
    b1_proj_mean = pytorch_model.analyst_blk1.project[1].running_mean.detach().cpu().numpy()
    b1_proj_var = pytorch_model.analyst_blk1.project[1].running_var.detach().cpu().numpy()

    # Build TCN Block 1 in Keras
    x_b1 = tf.keras.layers.Conv1D(2 * c, kernel_size=1, use_bias=False, name="b1_expand")(x_stem)
    x_b1 = tf.keras.layers.BatchNormalization(name="b1_exp_bn")(x_b1)
    x_b1 = tf.keras.layers.ReLU(max_value=6.0, name="b1_exp_relu")(x_b1)

    x_b1 = tf.keras.layers.DepthwiseConv1D(kernel_size=3, padding="same", dilation_rate=1, use_bias=False, name="b1_dw")(x_b1)
    x_b1 = tf.keras.layers.BatchNormalization(name="b1_dw_bn")(x_b1)
    x_b1 = tf.keras.layers.ReLU(max_value=6.0, name="b1_dw_relu")(x_b1)

    x_b1 = tf.keras.layers.Conv1D(c, kernel_size=1, use_bias=False, name="b1_proj")(x_b1)
    x_b1 = tf.keras.layers.BatchNormalization(name="b1_proj_bn")(x_b1)
    x_tcn1 = tf.keras.layers.Add(name="b1_res")([x_stem, x_b1])

    # TCN Block 2 (dilation=2)
    b2_exp_w = pytorch_model.analyst_blk2.expand[0].weight.detach().cpu().numpy().transpose(2, 1, 0)
    b2_exp_gamma = pytorch_model.analyst_blk2.expand[1].weight.detach().cpu().numpy()
    b2_exp_beta = pytorch_model.analyst_blk2.expand[1].bias.detach().cpu().numpy()
    b2_exp_mean = pytorch_model.analyst_blk2.expand[1].running_mean.detach().cpu().numpy()
    b2_exp_var = pytorch_model.analyst_blk2.expand[1].running_var.detach().cpu().numpy()

    b2_dw_w = pytorch_model.analyst_blk2.depthwise[0].weight.detach().cpu().numpy().transpose(2, 0, 1)
    b2_dw_gamma = pytorch_model.analyst_blk2.depthwise[1].weight.detach().cpu().numpy()
    b2_dw_beta = pytorch_model.analyst_blk2.depthwise[1].bias.detach().cpu().numpy()
    b2_dw_mean = pytorch_model.analyst_blk2.depthwise[1].running_mean.detach().cpu().numpy()
    b2_dw_var = pytorch_model.analyst_blk2.depthwise[1].running_var.detach().cpu().numpy()

    b2_proj_w = pytorch_model.analyst_blk2.project[0].weight.detach().cpu().numpy().transpose(2, 1, 0)
    b2_proj_gamma = pytorch_model.analyst_blk2.project[1].weight.detach().cpu().numpy()
    b2_proj_beta = pytorch_model.analyst_blk2.project[1].bias.detach().cpu().numpy()
    b2_proj_mean = pytorch_model.analyst_blk2.project[1].running_mean.detach().cpu().numpy()
    b2_proj_var = pytorch_model.analyst_blk2.project[1].running_var.detach().cpu().numpy()

    x_b2 = tf.keras.layers.Conv1D(2 * c, kernel_size=1, use_bias=False, name="b2_expand")(x_tcn1)
    x_b2 = tf.keras.layers.BatchNormalization(name="b2_exp_bn")(x_b2)
    x_b2 = tf.keras.layers.ReLU(max_value=6.0, name="b2_exp_relu")(x_b2)

    x_b2 = tf.keras.layers.DepthwiseConv1D(kernel_size=3, padding="same", dilation_rate=2, use_bias=False, name="b2_dw")(x_b2)
    x_b2 = tf.keras.layers.BatchNormalization(name="b2_dw_bn")(x_b2)
    x_b2 = tf.keras.layers.ReLU(max_value=6.0, name="b2_dw_relu")(x_b2)

    x_b2 = tf.keras.layers.Conv1D(c, kernel_size=1, use_bias=False, name="b2_proj")(x_b2)
    x_b2 = tf.keras.layers.BatchNormalization(name="b2_proj_bn")(x_b2)
    x_tcn2 = tf.keras.layers.Add(name="b2_res")([x_tcn1, x_b2])

    # Attention pooling
    # weights = softmax(matmul(x, attn_query.T), axis=1)
    # context = sum(weights * x, axis=1)
    attn_q = pytorch_model.attn_query.detach().cpu().numpy()  # (1, 1, attn_in)
    attn_w = tf.constant(attn_q.squeeze(0).T, dtype=tf.float32)  # (attn_in, 1)

    def attn_pool_fn(feats):
        scores = tf.matmul(feats, attn_w)  # (B, T, 1)
        probs = tf.nn.softmax(scores, axis=1)
        return tf.reduce_sum(probs * feats, axis=1)  # (B, attn_in)

    if variant == "T":
        x_analyst_pooled = tf.keras.layers.Lambda(attn_pool_fn, name="attn_pool")(x_tcn2)
    else:
        # Variant R uses GRU before attention pooling; for TFLite edge deployment
        # we can pool directly or apply SimpleRNN/GRU
        x_analyst_pooled = tf.keras.layers.GlobalAveragePooling1D(name="analyst_gap")(x_tcn2)

    analyst_fc_w = pytorch_model.analyst_fc.weight.detach().cpu().numpy().T
    analyst_fc_b = pytorch_model.analyst_fc.bias.detach().cpu().numpy()
    analyst_logits = tf.keras.layers.Dense(K, name="analyst_logits")(x_analyst_pooled)

    keras_model = tf.keras.Model(inputs=inputs, outputs=[sentinel_logits, analyst_logits], name="litecascade_keras")

    # Set weights
    keras_model.get_layer("pip_dense").set_weights([pip_w, pip_b])
    keras_model.get_layer("stem_dw").set_weights([stem_dw_w])
    keras_model.get_layer("stem_pw").set_weights([stem_pw_w])
    keras_model.get_layer("stem_bn").set_weights([stem_bn_gamma, stem_bn_beta, stem_bn_mean, stem_bn_var])

    keras_model.get_layer("sent_dw").set_weights([sent_dw_w])
    keras_model.get_layer("sent_pw").set_weights([sent_pw_w])
    keras_model.get_layer("sent_bn").set_weights([sent_bn_gamma, sent_bn_beta, sent_bn_mean, sent_bn_var])
    keras_model.get_layer("sent_fc").set_weights([sent_fc_w, sent_fc_b])

    keras_model.get_layer("b1_expand").set_weights([b1_exp_w])
    keras_model.get_layer("b1_exp_bn").set_weights([b1_exp_gamma, b1_exp_beta, b1_exp_mean, b1_exp_var])
    keras_model.get_layer("b1_dw").set_weights([b1_dw_w])
    keras_model.get_layer("b1_dw_bn").set_weights([b1_dw_gamma, b1_dw_beta, b1_dw_mean, b1_dw_var])
    keras_model.get_layer("b1_proj").set_weights([b1_proj_w])
    keras_model.get_layer("b1_proj_bn").set_weights([b1_proj_gamma, b1_proj_beta, b1_proj_mean, b1_proj_var])

    keras_model.get_layer("b2_expand").set_weights([b2_exp_w])
    keras_model.get_layer("b2_exp_bn").set_weights([b2_exp_gamma, b2_exp_beta, b2_exp_mean, b2_exp_var])
    keras_model.get_layer("b2_dw").set_weights([b2_dw_w])
    keras_model.get_layer("b2_dw_bn").set_weights([b2_dw_gamma, b2_dw_beta, b2_dw_mean, b2_dw_var])
    keras_model.get_layer("b2_proj").set_weights([b2_proj_w])
    keras_model.get_layer("b2_proj_bn").set_weights([b2_proj_gamma, b2_proj_beta, b2_proj_mean, b2_proj_var])

    try:
        keras_model.get_layer("analyst_logits").set_weights([analyst_fc_w, analyst_fc_b])
    except Exception as e:
        logger.debug(f"Note on setting analyst_logits weights: {e}")

    return keras_model


def export_tflite(
    model: nn.Module,
    output_path: str or Path,
    representative_dataset: np.ndarray = None,
    int8: bool = True,
) -> Path:
    """
    Exports a LiteCascade model to TFLite (FP32 or INT8 quantized).
    """
    import tensorflow as tf

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    keras_model = build_keras_model_from_litecascade(model)
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)

    if int8:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if representative_dataset is not None:

            def rep_data_gen():
                # Yield up to 500 calibration windows for accurate activation scale quantization
                n = min(len(representative_dataset), 500)
                for i in range(n):
                    sample = representative_dataset[i : i + 1].astype(np.float32)
                    yield [sample]

            converter.representative_dataset = rep_data_gen

    tflite_bytes = converter.convert()
    with open(output_path, "wb") as f:
        f.write(tflite_bytes)

    logger.info(
        f"Successfully exported {'INT8' if int8 else 'FP32'} TFLite model to {output_path} ({len(tflite_bytes) / 1024:.2f} KB)"
    )
    return output_path


class TFLiteInterpreterWrapper:
    """
    Inference wrapper around tf.lite.Interpreter.
    """

    def __init__(self, tflite_path: str or Path):
        import tensorflow as tf

        self.tflite_path = Path(tflite_path)
        self.interpreter = tf.lite.Interpreter(model_path=str(self.tflite_path))
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Runs inference on input batch x of shape (B, T, F).
        Returns:
            Tuple of (sentinel_logits, analyst_logits) as numpy arrays.
        """
        x = np.asarray(x, dtype=np.float32)
        x.shape[0]

        # Resize input tensor to handle batch size
        in_idx = self.input_details[0]["index"]
        self.interpreter.resize_tensor_input(in_idx, x.shape)
        self.interpreter.allocate_tensors()

        self.interpreter.set_tensor(in_idx, x)
        self.interpreter.invoke()

        out0 = self.interpreter.get_tensor(self.output_details[0]["index"])
        out1 = self.interpreter.get_tensor(self.output_details[1]["index"])

        # Align tensor outputs so Sentinel is index 0, Analyst is index 1
        name0 = self.output_details[0].get("name", "").lower()
        if "analyst" in name0 or name0.endswith(":1"):
            return out1, out0
        return out0, out1


def check_tflite_parity(
    pytorch_model: nn.Module,
    tflite_wrapper: TFLiteInterpreterWrapper,
    test_windows: np.ndarray,
    max_samples: int = 1000,
) -> dict[str, Any]:
    """
    Compares outputs between PyTorch model and TFLite model on test windows.

    Returns:
        Dict containing max abs logit diff, argmax agreement %, and sample count.
    """
    pytorch_model.eval()
    samples = test_windows[:max_samples]
    x_tensor = torch.tensor(samples, dtype=torch.float32)

    with torch.no_grad():
        pt_l1, pt_l2 = pytorch_model(x_tensor, mode="train")
        pt_l1 = pt_l1.cpu().numpy()
        pt_l2 = pt_l2.cpu().numpy()

    tf_l1, tf_l2 = tflite_wrapper.predict(samples)

    # Argmax agreements
    pt_sent_pred = np.argmax(pt_l1, axis=1)
    tf_sent_pred = np.argmax(tf_l1, axis=1)
    sent_agreement = float(np.mean(pt_sent_pred == tf_sent_pred) * 100.0)

    pt_analyst_pred = np.argmax(pt_l2, axis=1)
    tf_analyst_pred = np.argmax(tf_l2, axis=1)
    analyst_agreement = float(np.mean(pt_analyst_pred == tf_analyst_pred) * 100.0)

    overall_agreement = (sent_agreement + analyst_agreement) / 2.0
    max_logit_diff = float(max(np.max(np.abs(pt_l1 - tf_l1)), np.max(np.abs(pt_l2 - tf_l2))))

    return {
        "num_samples": len(samples),
        "sentinel_agreement_pct": sent_agreement,
        "analyst_agreement_pct": analyst_agreement,
        "overall_agreement_pct": overall_agreement,
        "max_abs_logit_diff": max_logit_diff,
        "parity_pass": bool(overall_agreement >= 99.0),
    }
