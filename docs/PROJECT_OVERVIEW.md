# LiteCascade: what was built

LiteCascade is a deep-learning intrusion and malware detector for IoT devices that spends compute only where it is
needed. A tiny **Sentinel** network classifies every window of traffic; a statistically calibrated **gate** lets
confident windows leave immediately, and only uncertain windows run the larger **Analyst** network. This document
explains every part of the project, what was verified, and what was not.

> **Data used for every number in this project:** a deterministic **synthetic** fixture written in the real N-BaIoT
> file layout (10 synthetic devices, 115 features, 16,000 records). The real N-BaIoT, CICIoT2023 and IoT-23 data are
> supported by the loaders but were not available here, so no real-dataset results exist. Latencies are measured on
> the development PC, not on a Raspberry Pi or microcontroller.

## 1. Problem

A device sees a stream of flow-statistics records x_t ∈ ℝ^F (F = 115 for N-BaIoT). The model classifies windows of
T = 10 consecutive records, X_t = [x_{t−9}, …, x_t], as benign or malicious. The label of a window is the label of its
last record, so a decision can be made as soon as that record arrives. Goal: detection quality close to a large
deep model at a fraction of its compute, with a provable limit on the extra misses caused by early exits.

## 2. Pipeline at a glance

1. **Load** the dataset in its native layout (N-BaIoT UCI/Kaggle CSVs, CICIoT2023 CSVs, IoT-23 Zeek logs) into one schema.
2. **Clean**: drop ±inf/NaN rows, exact duplicates, constant columns.
3. **Split without leakage**: whole devices go to train / validation / calibration / test (or chronological blocks
   with a purge gap). The calibration split is disjoint from all others.
4. **Fit preprocessing on training records only**: RobustScaler, plus PCA used to initialise the PIP layer.
5. **Window** each split separately (windows never cross a device file).
6. **Train** the CNN-BiLSTM teacher, then LiteCascade jointly with distillation (eq. 5).
7. **Calibrate** temperatures and the risk-controlled gate on the calibration split (eqs. 2–4).
8. **Compress**: structured pruning + fine-tuning, INT8 quantization-aware training, re-calibration.
9. **Evaluate and profile** on the held-out test devices; package each model as a bundle; serve the dashboard.

## 3. Deep-learning models

All models use the same data, splits, class-balanced loss, AdamW + cosine schedule and validation-based model selection.

| Model | Family | Idea | Params | MACs / window |
|---|---|---|---|---|
| **LiteCascade-R** | cascade | PIP + DS stem → Sentinel (gate) → Analyst with DS-TCN + BiGRU + attention | 16,356 (pruned) | 51.1K Sentinel path / 159.2K full |
| **LiteCascade-T** | cascade | same without the BiGRU (microcontroller-friendly) | 9,084 (pruned) | 89.7K full |
| Teacher CNN-BiLSTM | hybrid | 2×Conv1d + 2-layer BiLSTM(128) + attention; distillation source | 739,843 | 7.35M |
| B1 L1-MLP (SNIPE) | MLP | MLP on the flattened window, L1 penalty | 75,810 | 75.7K |
| B3 LSTM | recurrent | single LSTM, last step | 46,466 | 458K |
| B4 BiLSTM | recurrent | bidirectional LSTM | 92,930 | 917K |
| B5 PCA-BiLSTM INT8 | recurrent | BiLSTM on 32 PCA components, dynamic INT8 (Wang et al.) | 50,434 | 492K |
| B6 CNN-BiLSTM | hybrid | Conv1d + max-pool + BiLSTM (Jouhari & Guizani) | 88,962 | 549K |
| B7 DS-CNN | convolutional | depthwise-separable Conv1d + global pooling | 8,014 | 77.2K |
| B8 Tiny Transformer | attention | 2-layer, 2-head encoder | 20,866 | 220K |
| **B9 BiGRU** (new) | recurrent | bidirectional GRU, mean pooling | 47,714 | 470K |
| **B10 TCN** (new) | convolutional | 3 residual dilated causal conv blocks (d = 1, 2, 4) | 22,402 | 221K |
| **B11 ResNet-1D** (new) | convolutional | 2 residual conv stages + global pooling | 38,786 | 387K |

B2 (Random Forest / XGBoost) is kept as a non-deep reference in the reports but is not part of the dashboard zoo.
MACs are counted with fvcore plus custom handlers for LSTM/GRU and attention (fvcore alone reports 0 for recurrent layers).

### 3.1 LiteCascade in detail

- **PIP layer (eq. 1)** z_t = W_p(x_t − μ): a linear layer initialised with the top-32 PCA axes of the training data
  (W = components, b = −Wμ), frozen for 3 warm-up epochs, then fine-tuned.
- **Shared stem**: depthwise (k = 3) + pointwise convolution 32 → 24 channels, BatchNorm, ReLU6.
- **Sentinel exit (eq. 2)**: one more DS block, global average pooling, linear. p₁ = softmax(z₁/T₁).
- **Analyst exit**: two residual inverted-bottleneck DS-TCN blocks (dilation 1 and 2, expansion 2), a BiGRU with
  24 hidden units (variant R only), attention pooling, linear.
- **Gate (eq. 3)**: exit if (ŷ₁ = 0 and p₁,₀ ≥ τ_b) or (ŷ₁ ≠ 0 and max p₁ ≥ τ_m). In deployment the Analyst code is
  executed **only** for windows that did not exit (verified with a forward hook in the tests).
- Unpruned with F = 115: Sentinel exit 746 params (paper target < 2K), Analyst exit 12,722 (≈ 14K), total 18,092
  (≈ 20K); 0.053M / 0.176M MACs (paper estimate ≈ 0.06M / 0.18M).

### 3.2 Training objective (eq. 5)

L = CE(p₁, y) + CE(p₂, y) + γ·τ²·KL(σ(t/τ) ‖ σ(s₂/τ)) + δ·KL(p₂ ‖ p₁), with γ = 0.5, δ = 0.3, τ = 3.
Teacher logits are computed for exactly the training windows and travel with them through the shuffled data loader.
The self-distillation target p₂ is detached, so it trains the Sentinel only.

### 3.3 Risk-controlled gate (eq. 4)

- T₁ and T₂ are fitted by minimising NLL on the calibration split (log-parametrised, so T > 0 and predictions do not change).
- An exit-induced miss is a malicious calibration window that exits as benign while the Analyst would have flagged it.
  τ_b = min{τ : U_CP(m(τ), N_m; α) ≤ ε}, with U_CP the one-sided Clopper–Pearson upper bound. Because m(τ) can only fall
  as τ rises, this equals a fixed-sequence test, so the bound holds with probability ≥ 1 − α under exchangeability.
- Thresholds are placed at float32 values just above a calibration score so ties cannot add exits. If even zero misses
  cannot satisfy the bound (N_m < ln α / ln(1 − ε) = 598 for ε = 0.5 %, α = 0.05), benign exits are disabled.
- τ_m targets a false-alarm rate on benign calibration windows (empirical, no guarantee), or `--verify-alerts`
  sends every alert to the Analyst.

### 3.4 Compression chain

1. Joint training with distillation.
2. Structured L1-norm pruning (30 %) of Analyst expansion channels and Sentinel channels; layers are rebuilt smaller; 5 fine-tuning epochs.
3. INT8 quantization-aware training of PIP + stem + all conv blocks (Conv+BN fused), then dynamic INT8 for the Linear/GRU layers.
4. Gate re-calibration on the INT8 model, because quantization changes confidences.

## 4. Results on the synthetic fixture (test split: 2 unseen devices, 3,110 windows)

These numbers show the pipeline works. They are not evidence of performance on real traffic.

| Model | macro-F1 [95 % CI] | Recall | FPR | Size KB |
|---|---|---|---|---|
| Teacher CNN-BiLSTM | 0.936 [0.928, 0.943] | 0.891 | 0.020 | 2,898 |
| B1 L1-MLP | 0.948 [0.940, 0.955] | 0.914 | 0.018 | 299 |
| B10 TCN | 0.941 [0.933, 0.949] | 0.899 | 0.017 | 93 |
| B7 DS-CNN | 0.939 [0.932, 0.946] | 0.897 | 0.020 | 34 |
| B5 PCA-BiLSTM INT8 | 0.935 [0.926, 0.944] | 0.894 | 0.025 | 57 |
| B6 CNN-BiLSTM | 0.935 [0.925, 0.943] | 0.891 | 0.023 | 352 |
| B9 BiGRU | 0.934 [0.925, 0.941] | 0.891 | 0.024 | 190 |
| B11 ResNet-1D | 0.933 [0.926, 0.940] | 0.896 | 0.032 | 162 |
| B4 BiLSTM | 0.930 [0.920, 0.937] | 0.888 | 0.030 | 366 |
| B3 LSTM | 0.925 [0.916, 0.933] | 0.875 | 0.025 | 184 |
| B8 Tiny Transformer | 0.907 [0.897, 0.915] | 0.875 | 0.063 | 91 |
| **LiteCascade-R INT8** | **0.937** | 0.897 | 0.025 | **50.3** |
| LiteCascade-T INT8 | 0.924 | 0.872 | 0.027 | 39.3 |

LiteCascade-R through the compression chain:

| Stage | Cascade F1 | Exit rate ρ | Exit-induced miss rate | Size KB | Measured speed-up vs. both exits |
|---|---|---|---|---|---|
| Joint training + KD | 0.938 | 88.2 % | 0.20 % | 93.9 | 2.36× |
| Pruned 30 % + fine-tune | 0.939 | 86.4 % | 0.00 % | 86.5 | 2.25× |
| INT8 QAT, old (FP32) gate | 0.937 | 87.6 % | 0.00 % | 50.3 | 2.66× |
| INT8 QAT, re-calibrated gate | 0.937 | 86.4 % | 0.00 % | 50.3 | 2.76× |

Gate of the deployed INT8 model: ε = 0.5 %, α = 0.05, N_m = 1,528 malicious calibration windows (≥ 598 needed),
feasible. Every stage keeps the measured exit-induced miss rate below ε. Honest observations:

- On this fixture several small baselines (B1 MLP, B10 TCN, B7 DS-CNN) match or beat the teacher and LiteCascade on F1.
  The synthetic signal is easy enough that the Sentinel alone is almost as accurate as the Analyst.
- In the ablations, removing temperature scaling changes nothing for this binary task. That is expected: with two
  classes, temperature scaling is monotone in p₀, so the quantile-based τ_b selects the same exits.
- Removing PCA initialisation of the PIP layer lowered cascade F1 from 0.938 to 0.920 and recall from 0.895 to 0.841.
- INT8 kernels in PyTorch eager mode are slower than FP32 on this PC; size, not speed, is what INT8 buys here.

## 5. Dashboard

`python -m litecascade.web.app` → http://127.0.0.1:8000

| Page | What it shows |
|---|---|
| Live cascade | Held-out windows replayed in real time; animated packets travel PIP → stem → Sentinel → gate and exit early or run the Analyst; live exit rate, compute saved, alerts, F1, latency; score stream; event feed; attack-burst injection |
| Model arena | All 14 models race on the same windows: live verdict LEDs, running F1 rings, latency sparklines, confusion counts, animated leaderboard |
| Benchmarks | Stored test metrics; animated Pareto chart (F1 vs. MACs / params / size / latency); sortable table |
| Gate lab | ε and α slider: τ_b, exit rate, exit-induced misses and F1 recomputed from the real calibration outputs |
| Architecture | Parameter budget per block, MAC split, the six equations, the model zoo |
| Analyze CSV | Upload traffic, compare any set of models, download all predictions |
| Documentation | This document and the generated evaluation report |

Every prediction shown is produced by the model. Ground-truth labels of the replayed windows are used only to score
the predictions, never to produce them.

## 6. How to reproduce

```
pip install -r requirements.txt
pip install -e . --no-deps
python scripts/run_demo.py --ablations                                      # fixture, cascade + baselines + report
python -m litecascade.train.run --model dl --dataset fixture --epochs 8 --bundles artifacts   # all DL models as bundles
cd dashboard && npm ci && npm run build && cd ..                            # build the React dashboard
python -m litecascade.web.app                                               # open http://127.0.0.1:8000
python -m pytest tests -q                                                   # 69 tests
```

## 7. Verified vs. not done

**Verified by tests or measurement:** loader parsing of all three native formats; zero group overlap between splits;
scaler fitted on training records only; PIP implements eq. 1 exactly; the self-KD gradient reaches only the Sentinel;
KD terms match their formulas; Clopper–Pearson selection (including a Monte-Carlo coverage test of the guarantee);
tie safety; the Analyst is skipped for exited windows; live inference agrees with offline evaluation; pruning
shrinks parameters and MACs; INT8 models reload exactly; fvcore RNN MAC counts; the web API end to end.

**Not done:** real-dataset experiments; Raspberry Pi / MCU latency, RAM and energy; TFLite / TFLite-Micro export;
the Malimg image adapter; leave-one-family-out and perturbation robustness tables; the "wrong order" (QAT → prune) ablation.
