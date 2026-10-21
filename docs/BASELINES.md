# Baseline Implementations

This document tracks where our re-implementations of the baselines in `src/litecascade/models/baselines.py` deviate from the original papers for practical or compatibility reasons.

* **B1 (L1-regularised MLP - SNIPE style):** We flatten the `(B, T, F)` temporal window into `(B, T*F)` before passing to the FC layers. The original SNIPE operates on statistical aggregates, but we use the flattened raw window features to ensure apples-to-apples feature parity.
* **B2 (RandomForest/XGBoost):** We similarly flatten `(B, T, F)` into a 2D array. The trainer automatically wraps the sklearn/xgboost API.
* **B3 (LSTM) & B4 (BiLSTM):** We take the hidden state of the final timestep `T` as the representation for classification.
* **B5 (BiLSTM + IPCA + Quantization - Wang et al.):** Same architecture as B4, trained on PCA-reduced inputs (d = 32 components fitted on the train split only; `IncrementalPCA` is used automatically above 200k train records) and **evaluated after torch dynamic INT8 quantization** (`litecascade.train.run`). The FP32 scores are kept under `metrics.fp32`. Wang et al.'s DNN front-end is not reproduced.
* **B6 (CNN-BiLSTM - Jouhari & Guizani):** We use a `Conv1d` over the time dimension. A `MaxPool1d(2)` is applied if `T >= 2`. The features are then passed to a BiLSTM, which takes the last timestep.
* **B7 (Depthwise-Separable CNN):** Uses a `groups=in_channels` Conv1d layer, followed by a pointwise `1x1` Conv1d, ReLU, and Global Average Pooling.
* **B8 (Tiny Transformer):** A 2-layer, 2-head encoder block. Mean pooling over the sequence dimension is used for the final classification layer.

## Common protocol (all baselines and LiteCascade)

* Same data path: `litecascade.data.pipeline.prepare_data` (group/time split with disjoint calibration split,
  RobustScaler fitted on train records only, windows of T = 10 records labelled by their last record).
* Class-balanced cross-entropy, AdamW + cosine schedule, model selection on validation macro-F1.
* MACs are counted with fvcore plus custom handles for `aten::lstm`, `aten::gru` and
  `aten::scaled_dot_product_attention` (fvcore alone reports **0** MACs for recurrent layers).
* "Size INT8" for B1/B3-B8 = serialized torch dynamic-INT8 model (Linear/RNN weights only; convolutions remain FP32).
* Latency = host-CPU wall clock, batch 1, PyTorch eager. Not a Raspberry Pi / MCU measurement.
* B2 "params" is the number of tree nodes; no MAC count is reported for trees.
