# Baseline Implementations

This document tracks where our re-implementations of the baselines in `src/litecascade/models/baselines.py` deviate from the original papers for practical or compatibility reasons.

* **B1 (L1-regularised MLP - SNIPE style):** We flatten the `(B, T, F)` temporal window into `(B, T*F)` before passing to the FC layers. The original SNIPE operates on statistical aggregates, but we use the flattened raw window features to ensure apples-to-apples feature parity.
* **B2 (RandomForest/XGBoost):** We similarly flatten `(B, T, F)` into a 2D array. The trainer automatically wraps the sklearn/xgboost API.
* **B3 (LSTM) & B4 (BiLSTM):** We take the hidden state of the final timestep `T` as the representation for classification.
* **B5 (BiLSTM + IPCA + Quantization - Wang et al.):** Shares the exact PyTorch architecture as B4. The IPCA transform and dynamic quantization occur as part of the preprocessing pipeline and evaluation routines rather than embedded inside the `nn.Module`.
* **B6 (CNN-BiLSTM - Jouhari & Guizani):** We use a `Conv1d` over the time dimension. A `MaxPool1d(2)` is applied if `T >= 2`. The features are then passed to a BiLSTM, which takes the last timestep.
* **B7 (Depthwise-Separable CNN):** Uses a `groups=in_channels` Conv1d layer, followed by a pointwise `1x1` Conv1d, ReLU, and Global Average Pooling.
* **B8 (Tiny Transformer):** A 2-layer, 2-head encoder block. Mean pooling over the sequence dimension is used for the final classification layer.
