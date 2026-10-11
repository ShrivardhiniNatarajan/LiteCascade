# LiteCascade: Lightweight Early-Exit Cascade for IoT Malware and Intrusion Detection

[![Release](https://img.shields.io/badge/Release-v1.0.0-blue.svg)](https://github.com/ShrivardhiniNatarajan/LiteCascade/releases/tag/v1.0)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-brightgreen.svg)](https://www.python.org/)
[![PyTorch 2.5+](https://img.shields.io/badge/PyTorch-2.5%2B-ee4c2c.svg)](https://pytorch.org/)
[![TFLite INT8](https://img.shields.io/badge/Deployment-TFLite%20INT8%20%2F%20TFLM-orange.svg)](https://www.tensorflow.org/lite)
[![Tests](https://img.shields.io/badge/Tests-67%20Passed-success.svg)](tests/)

**LiteCascade** is a lightweight, early-exit neural cascade architecture engineered for edge IoT network intrusion detection and malware classification. By coupling a sub-10K parameter INT8 **Sentinel** classifier with a full-capacity sequence **Analyst** model through risk-controlled, temperature-scaled confidence gating, LiteCascade achieves a **2.64x empirical inference speedup** on Raspberry Pi 4 and fits comfortably within microcontroller memory constraints (**41.93 KB** Flash, **20.0 KB** SRAM Tensor Arena) while matching full-capacity teacher detection accuracy (**97.4% Macro-F1**, bounded recall loss $\Delta R \le 0.005$).

---

## 📑 Table of Contents

1. [Architecture Overview](#-architecture-overview)
2. [Key Empirical Results](#-key-empirical-results)
3. [Environment & Installation](#-environment--installation)
4. [Dataset Setup](#-dataset-setup)
5. [One-Command Reproduction](#-one-command-reproduction)
6. [On-Device Hardware Benchmarks](#-on-device-hardware-benchmarks)
7. [Limitations & Threats to Validity](#-limitations--threats-to-validity)
8. [Project Roadmap & Completed Milestones](#-project-roadmap--completed-milestones)
9. [Citation & License](#-citation--license)

---

## 🧠 Architecture Overview

Standard edge intrusion detection faces a harsh trade-off: ultra-compact models sacrifice detection sensitivity on novel zero-day attacks, whereas deep sequence models (BiLSTM, Transformers) exceed the memory, cycle, and thermal limits of low-power IoT gateways and microcontrollers.

LiteCascade resolves this with an asymmetrical dual-exit cascade:

```
                      Input Window x in R^{T x F}
                                 │
                   ┌─────────────┴─────────────┐
                   │    PIPLayer (1D Conv)     │  Patch-based Input Projection
                   └─────────────┬─────────────┘
                                 │
                   ┌─────────────┴─────────────┐
                   │   Shared DSBlock (Stem)   │  Depthwise-Separable Conv
                   └─────────────┬─────────────┘
                                 │
                   ┌─────────────┴─────────────┐
                   │       SENTINEL EXIT       │  1D CNN + Global Pooling
                   │     (Sub-10K Params)      │  Temperature Scaled (T1)
                   └─────────────┬─────────────┘
                                 │
                   ┌─────────────┴─────────────┐
                   │   Risk-Controlled Gate    │  Is Conf(x) >= tau?
                   │    (Clopper-Pearson)      │  (Controlled Risk eps <= 1%)
                   └──────┬─────────────┬──────┘
             YES (88.0%)  │             │  NO (12.0%)
                          ▼             ▼
                 [EARLY EXIT]    ┌───────────────────────────┐
                Sentinel Result  │       ANALYST EXIT        │  TCN + BiGRU
                                 │  (Full Sequence Context)  │  Temperature Scaled (T2)
                                 └─────────────┬─────────────┘
                                               ▼
                                         [FINAL EXIT]
                                         Analyst Result
```

### Key Innovations:
1. **Patch-based Input Projection (`PIPLayer`)**: Converts raw feature windows into dense temporal patches without heavy MLP expansion.
2. **Temperature-Scaled Calibration**: Calibrates Sentinel logits via NLL minimization ($T_1 = 1.15$, $T_2 = 1.08$), eliminating overconfidence.
3. **Risk-Controlled Confidence Gating**: Employs finite-sample Clopper-Pearson binomial confidence bounds on calibration splits to derive dual thresholds $(\tau_0, \tau_1)$ guaranteeing exit-induced recall degradation $\Delta R \le \epsilon$ with probability $\ge 1 - \alpha$.
4. **End-to-End Compression Chain**: Applies 30% structured L1-norm channel pruning followed by INT8 Quantization-Aware Training (QAT) and post-quantization gate re-calibration, producing a 41.93 KB INT8 TFLite flatbuffer.

---

## 📊 Key Empirical Results

All metrics below are 100% traceable to experimental runs in `results/` (enforced via `scripts/check_no_handwritten_numbers.py`):

### Table III: Comparative Evaluation vs. Baselines (Paper Table III)

| Model | Macro-F1 | Recall | Params | MACs (M) | INT8 Size (KB) | Latency p50 (ms) | Speed-up vs Analyst |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B1 (MLP SNIPE)** | 0.942 | 0.938 | 20,290 | 0.076 | 79.3 | 0.082 | - |
| **B2 (Tree / RF)** | 0.938 | 0.932 | - | - | 120.5 | 0.045 | - |
| **B3 (LSTM)** | 0.955 | 0.952 | 13,010 | 0.0001 | 50.9 | 0.210 | - |
| **B4 (BiLSTM)** | 0.961 | 0.958 | 25,378 | 0.0003 | 99.2 | 0.420 | - |
| **B5 (Wang BiLSTM)** | 0.959 | 0.955 | 25,378 | 0.0003 | 99.2 | 0.415 | - |
| **B6 (CNN-BiLSTM)** | 0.968 | 0.964 | 56,850 | 0.221 | 222.1 | 0.490 | - |
| **B7 (DSCNN)** | 0.958 | 0.954 | 8,858 | 0.077 | 34.6 | 0.095 | - |
| **B8 (Transformer)** | 0.965 | 0.961 | 15,634 | 0.207 | 61.2 | 0.380 | - |
| **Teacher (CNN-BiLSTM)**| **0.976** | **0.973** | 226,434 | 0.796 | 884.5 | 0.650 | 1.00x (ref) |
| **LiteCascade (Sentinel)** | 0.951 | 0.948 | **2,410** | **0.008** | **9.4** | **0.052** | - |
| **LiteCascade (Analyst)** | 0.973 | 0.970 | 8,411 | 0.221 | 32.5 | 0.384 | 1.00x (ref) |
| **LiteCascade (Cascade $\rho=0.88$)** | **0.974** | **0.971** | **10,821** | **0.038** | **41.93** | **0.091** | **2.64x** |

### Robustness & Zero-Day Generalization (Paper Table VI)
- **Leave-One-Family-Out (LOFO):** Tested on zero-day malware families held out completely during training (Gafgyt, Mirai, Tsunami). LiteCascade sustains $\mathbf{\Delta R \le 0.005}$, strictly satisfying risk tolerance $\epsilon = 0.010$.
- **Adversarial Perturbations:** Evaluated against Gaussian noise ($\sigma \le 0.20$), inter-arrival timing jitter, and packet padding; Macro-F1 degrades by $< 2.8\%$.

---

## 💻 Environment & Installation

### Requirements
- Python 3.10, 3.11, or 3.12
- Operating System: Linux, macOS, or Windows 10/11
- Minimum RAM: 4 GB (8 GB recommended)

### 1. Clone & Setup Virtual Environment
```bash
git clone https://github.com/ShrivardhiniNatarajan/LiteCascade.git
cd LiteCascade

# Create and activate virtual environment
python -m venv .venv

# On Linux / macOS:
source .venv/bin/activate

# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies
```bash
# Editable install with development and profiling tools
pip install -e .[dev]

# Alternatively, restore locked exact dependencies:
pip install -r requirements-lock.txt
```

### 3. Verify Installation
```bash
# Run pytest verification suite (all 67 unit tests)
python -m pytest -q

# Run code style & lint checks
python -m ruff check src scripts tests
```

---

## 📦 Dataset Setup

LiteCascade is evaluated across four major IoT intrusion detection and malware benchmarks:

1. **N-BaIoT**: Device-partitioned network traffic from 9 commercial IoT devices under Mirai and BASHLITE.
2. **CICIoT2023**: 33 attack classifications across 105 real smart-home and industrial IoT devices.
3. **IoT-23**: 23 long-running captures of IoT malware network flows alongside benign traffic.
4. **Malimg**: 25-class grayscale malware texture images ($64 \times 64$ byte matrices) for the 2D adapter.

### Quick Start: Synthetic Fixtures (Zero Download Required)
To verify pipelines instantly without downloading gigabytes of raw PCAPs/CSVs, LiteCascade provides an automated fixture generator that builds schema-identical synthetic data:
```bash
python scripts/make_fixture.py
```

### Real Benchmark Setup
To reproduce experiments on the raw datasets, follow the step-by-step download and directory layout guide in [docs/DATASETS.md](docs/DATASETS.md).

---

## 🔁 One-Command Reproduction

LiteCascade supports full one-command reproduction via both `make` (Linux/macOS) and the cross-platform Python runner `scripts/reproduce.py` (Windows/Linux/macOS).

### Fast Reproduction (<10 Minutes)
Generates synthetic fixtures, profiles all baseline models, trains/evaluates the cascade, validates zero handwritten numbers, and runs core unit tests:
```bash
# Using Python runner (any platform):
python scripts/reproduce.py --mode fast

# Using Makefile (Linux / macOS):
make reproduce-fast
```

### Full Reproduction Pipeline
Runs the complete multi-seed training, pruning, INT8 QAT, ablation ladder (A0–A9), zero-day LOFO cross-validation, on-device profiling, and LaTeX table export:
```bash
# Using Python runner:
python scripts/reproduce.py --mode full

# Using Makefile:
make reproduce-full
```

### Individual Paper Table Reproduction
Each table and figure in the IEEE manuscript can be individually regenerated from ground-truth results:

| Target Table | Python Runner Command | Makefile Command | Output Artifact |
| :--- | :--- | :--- | :--- |
| **Table III (Main Comparison)** | `python scripts/reproduce.py --table 3` | `make table3` | `results/tables/main.{csv,tex}` |
| **Table IV (Ablation Ladder)** | `python scripts/reproduce.py --table 4` | `make table4` | `results/tables/ablation.{csv,tex}` |
| **Table V (Device Benchmarks)** | `python scripts/reproduce.py --table 5` | `make table5` | `results/tables/device.{csv,tex}` |
| **Table VI (Robustness / LOFO)**| `python scripts/reproduce.py --table 6` | `make table6` | `results/tables/robustness.{csv,tex}` |
| **All 5 IEEE Figures (300 DPI)**| `python scripts/reproduce.py --figures` | `make figures` | `results/figures/*.png` |
| **Traceability Verification** | `python scripts/reproduce.py --verify` | `make verify-numbers` | Console Pass Report |

---

## 📱 On-Device Hardware Benchmarks

LiteCascade models are validated on physical hardware targets and low-power microcontroller profiles:

### 1. Raspberry Pi 4 Model B
- **SoC:** Broadcom BCM2711, Quad-core Cortex-A72 @ 1.5 GHz, 4 GB LPDDR4.
- **Runtime:** TensorFlow Lite INT8 with XNNPACK multi-threading (4 threads).
- **Latency:** Sentinel p50 = **0.052 ms** | Analyst p50 = **0.384 ms** | Cascade p50 = **0.091 ms** (**2.64x speedup**).
- **Energy:** Active inference consumption = **117.0 $\mu\text{J}$** (measured via synchronized INA219 current sensor).
- **Benchmark Script:**
  ```bash
  python scripts/device/bench_rpi.py --model checkpoints/litecascade_int8.tflite --runs 1000
  ```

### 2. Microcontroller Deployment (ESP32-S3 & ARM Cortex-M)
- **TFLite-Micro Flatbuffer Footprint:** **41.93 KB** Flash.
- **Static Tensor Arena Requirement:** **20.0 KB** SRAM.
- **Cycle Estimates:**
  - ESP32-S3 (Xtensa Dual-Core LX7 @ 240 MHz): Sentinel = 35,040 cycles (0.146 ms) | Cascade = 48,480 cycles (0.202 ms).
  - ARM Cortex-M4 (@ 168 MHz with CMSIS-NN): Sentinel = 24,528 cycles (0.146 ms).
- **Estimation & Guide:** See [scripts/device/bench_tflm.md](scripts/device/bench_tflm.md) and run:
  ```bash
  python scripts/estimate_mcu.py --model checkpoints/litecascade_int8.tflite
  ```

---

## ⚠️ Limitations & Threats to Validity

In accordance with Section VII of the IEEE manuscript:

1. **Feature Extraction Overhead:** Sliding-window feature extraction and robust scaling operate outside the neural network execution graph. On ultra-low-power microcontrollers, streaming packet parsing cycles must be budgeted alongside neural inference.
2. **Feature-Space Evasion Emulation:** Adversarial evaluations (Gaussian noise, packet-size padding, timing jitter) are performed in the scaled feature space and emulate evasion behavior; they do not represent physical network-layer packet reinjection.
3. **Quantization on Recurrent Architectures:** PyTorch eager-mode QAT cannot quantize recurrent GRU kernels statically due to framework limitations; LiteCascade quantizes the CNN stem statically and applies dynamic INT8 quantization to GRU, while the exported TFLite model is fully quantized to INT8.
4. **Finite Sample Size Bound in Hypothesis Testing:** With $N=5$ seeds, the mathematical lower bound for a two-sided Wilcoxon signed-rank test is $p = 2 \times (0.5)^5 = 0.0625$. While statistically significant at $\alpha = 0.10$, claiming $p < 0.05$ requires $N \ge 6$ seeds.

---

## 🗺️ Project Roadmap & Completed Milestones

| Stage | Milestone Description | Milestone Tag | Status |
| :---: | :--- | :---: | :---: |
| **0** | Scaffold, modular packaging, strict `.gitignore`, seeds | `m0-scaffold` | ✅ Completed |
| **1** | EDA, schema loaders (N-BaIoT, CICIoT, IoT-23), synthetic fixture | `m1-eda` | ✅ Completed |
| **2** | Group-aware leakage-safe preprocessing, windowing, RobustScaler/IPCA | `m2-prep` | ✅ Completed |
| **3** | Detection metrics, bootstrap 95% CIs, `fvcore` hardware profiler | `m3-eval` | ✅ Completed |
| **4** | SOTA Baselines B1–B8, UnifiedTrainer, AMP, class weighting | `m4-baselines` | ✅ Completed |
| **5** | CNN-BiLSTM Teacher model, logits export | `m5-teacher` | ✅ Completed |
| **6** | LiteCascade architecture (PIPLayer, Stem, Sentinel, Analyst) | `m6-model` | ✅ Completed |
| **7** | Joint training, multi-task cascade loss, self-KD & teacher-KD | `m7-train` | ✅ Completed |
| **8** | Temperature scaling calibration & Clopper-Pearson risk gate | `m8-gate` | ✅ Completed |
| **9** | Structured L1 pruning, INT8 QAT, TFLite export, gate recalibration | `m9-compress` | ✅ Completed |
| **10** | 10-step ablation ladder across seeds (A0–A9), LaTeX aggregation | `m10-ablation` | ✅ Completed |
| **11** | Zero-day LOFO evaluation & feature perturbation robustness engine | `m11-robust` | ✅ Completed |
| **12** | On-device Raspberry Pi benchmarking, power logging, MCU sizing | `m12-device` | ✅ Completed |
| **13** | 2D image adapter on Malimg, multiclass macro metrics, 2D baselines | `m13-image` | ✅ Completed |
| **14** | Paper tables (III–VI), 5 IEEE 300 DPI figures, Wilcoxon hypothesis tests | `m14-results` | ✅ Completed |
| **15** | Reproducibility release (v1.0), MODEL_CARD, CITATION.cff, lockfile | `v1.0` | ✅ Completed |

---

## 📖 Citation & License

### Citation
If you use LiteCascade models, code, or benchmarks in your research, please cite our paper:

```bibtex
@article{natarajan2026litecascade,
  title     = {LiteCascade: A Lightweight Early-Exit Cascade for IoT Malware and Intrusion Detection},
  author    = {Natarajan, Shrivardhini},
  journal   = {IEEE Internet of Things Journal},
  year      = {2026},
  note      = {Under review}
}
```

### License
This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.
