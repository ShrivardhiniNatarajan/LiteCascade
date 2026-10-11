# Model Card: LiteCascade

## 1. Model Details

- **Model Name:** LiteCascade (Lightweight Early-Exit Cascade for IoT Malware and Intrusion Detection)
- **Model Version:** v1.0.0
- **Release Date:** October 2026
- **License:** MIT License
- **Frameworks:** PyTorch 2.5+, TensorFlow Lite 2.18+ / LiteRT, TFLite-Micro (C++ / CMSIS-NN)
- **Model Type:** Dual-exit neural cascade classifier with risk-controlled confidence gating.
- **Components:**
  - **STEM:** Patch-based Input Projection (`PIPLayer`) with shared Depthwise-Separable convolutional block (`DSBlock`).
  - **Sentinel Exit:** Lightweight classifier with global pooling, temperature-scaled logit scaling ($T_1 = 1.15$), and dual risk-controlled thresholds $(\tau_0, \tau_1)$.
  - **Analyst Exit:** Full-capacity sequence model (Temporal Convolutional Network + Bidirectional GRU with self-attention) with temperature scaling ($T_2 = 1.08$).
  - **Compression:** 30% structured L1-norm channel pruning, INT8 Quantization-Aware Training (QAT), exported to TFLite INT8 flatbuffer.

---

## 2. Intended Use

### Intended Use Cases
- **Edge Gateway Intrusion Detection:** Inline classification of sliding temporal feature windows ($T=10$ steps, 12 features) on microcomputers such as Raspberry Pi 4 Model B.
- **Microcontroller-Class Deployment:** Standalone execution on resource-constrained microcontrollers (ESP32-S3, ARM Cortex-M4/M7) using TFLite-Micro with static Tensor Arena allocation ($\le 20\text{ KB}$).
- **Energy-Constrained IoT Nodes:** Battery-powered IoT devices benefiting from a 2.64x empirical speedup and reduced active CPU utilization via early exit ($\rho \approx 88.0\%$).

### Out-of-Scope and Misuse Cases
- **Carrier-Grade Backbone Routing:** Not intended as a drop-in replacement for 100 Gbps ASIC/FPGA hardware pipelines without specialized hardware offload.
- **Uncalibrated Threshold Deployment:** Deploying arbitrary, unverified thresholds $\tau$ without Clopper-Pearson binomial calibration risks violating the safety guarantee ($\Delta R \le \epsilon$).
- **Sole Source of Safety-Critical Defense:** Must not serve as the exclusive defense layer in life-critical medical devices or autonomous avionics without secondary mechanical/fail-safe isolation.

---

## 3. Training & Evaluation Data

- **Datasets:**
  - **N-BaIoT:** Real IoT device traffic under Mirai and BASHLITE/Gafgyt botnet attacks (115 temporal statistics).
  - **CICIoT2023:** Extensive multi-class IoT attack suite across 33 attack variants on 105 IoT devices.
  - **IoT-23:** Network traffic captures of benign IoT devices and malicious captures from infected Raspberry Pis.
  - **Malimg (Image Adapter):** 25-class malware bytecode textures ($64 \times 64$ grayscale bytes).
- **Data Partitioning Protocol:**
  - **Group-Aware Splitting:** 60% Train, 20% Validation/Calibration, 20% Test. Device IDs and capture sessions are strictly grouped to guarantee zero cross-device temporal leakage.
  - **Leave-One-Family-Out (LOFO):** Zero-day evaluation protocol where an entire malware family (e.g., Mirai, Gafgyt, Tsunami) is completely held out from training and seen exclusively at test time.

---

## 4. Quantitative Performance Summary

The metrics below trace directly to verifiable experiment artifacts in `results/`:

| Model / Configuration | Macro-F1 | Malicious Recall | Params | MACs (M) | INT8 Size (KB) | Speed-up vs Analyst |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **B1 (MLP SNIPE)** | 0.942 | 0.938 | 20,290 | 0.076 | 79.3 | - |
| **B3 (LSTM)** | 0.955 | 0.952 | 13,010 | 0.0001 | 50.9 | - |
| **B4 (BiLSTM)** | 0.961 | 0.958 | 25,378 | 0.0003 | 99.2 | - |
| **B7 (DSCNN)** | 0.958 | 0.954 | 8,858 | 0.077 | 34.6 | - |
| **Teacher (CNN-BiLSTM)** | **0.976** | **0.973** | 226,434 | 0.796 | 884.5 | 1.00x (ref) |
| **LiteCascade (Sentinel)** | 0.951 | 0.948 | **2,410** | **0.008** | **9.4** | - |
| **LiteCascade (Analyst)** | 0.973 | 0.970 | 8,411 | 0.221 | 32.5 | 1.00x (ref) |
| **LiteCascade (Cascade $\rho=0.88$)**| **0.974** | **0.971** | **10,821** | **0.038** | **41.93** | **2.64x** |

### Robustness & Zero-Day Generalization
- **LOFO Zero-Day Recall Loss:** On held-out malware families, LiteCascade experiences $\Delta R \le 0.005$, fully within the target bound $\epsilon = 0.010$.
- **Adversarial Jitter & Padding:** Under Gaussian feature noise ($\sigma \le 0.20$), inter-arrival timing jitter, and packet padding emulations, Macro-F1 degrades gracefully by $< 2.8\%$.

---

## 5. Technical Specifications & Hardware Profiles

- **Model Dimensions:**
  - Stem + Sentinel Parameters: 2,410
  - Analyst Parameters: 8,411
  - Total Empirical Architecture Parameters: 10,821
- **Flatbuffer Memory:**
  - TFLite INT8 Quantized File: **41.93 KB** (Flash storage)
  - TFLite-Micro Tensor Arena: **20.0 KB** (SRAM requirement)
- **Benchmarked Runtimes:**
  - **Raspberry Pi 4 Model B (Broadcom BCM2711, 1.5 GHz):**
    - Sentinel p50 Latency: 0.052 ms
    - Analyst p50 Latency: 0.384 ms
    - Cascade p50 Latency (XNNPACK 4T): **0.091 ms**
    - Energy per inference: **117.0 $\mu\text{J}$**
  - **ESP32-S3 (Xtensa Dual-Core LX7, 240 MHz):**
    - Estimated Sentinel Latency: 0.146 ms
    - Estimated Analyst Latency: 0.612 ms
    - Estimated Cascade Latency: 0.202 ms

---

## 6. Environmental and Ethical Considerations

- **Carbon & Compute Footprint:** Training LiteCascade requires $< 1$ GPU hour on an NVIDIA RTX 4090 or $< 4$ CPU hours on modern commodity workstations. Edge inference draws under $120\ \mu\text{J}$ per sample, dramatically reducing IoT carbon footprint compared to cloud-offloaded deep networks.
- **Dual-Use & Adversarial Risks:** Attackers may analyze early-exit thresholds to craft perturbation attacks that force samples to the slower Analyst exit (Denial of Service / battery exhaustion). LiteCascade mitigates this via risk-controlled threshold selection and bounded worst-case latency.
- **Data Privacy:** LiteCascade processes summary flow features or spatial byte textures entirely on-device, eliminating the privacy hazard of transmitting raw network packet payloads off-premise.

---

## 7. Limitations & Operational Caveats

1. **Feature Preprocessing Overhead:** Scaler transforms and window buffering operate outside the neural network runtime. On ultra-low-power microcontrollers, feature extraction latency must be budgeted alongside neural inference.
2. **Evasion Emulation:** Robustness evaluations represent feature-space emulations of evasion perturbations, not packet-level traffic morphing via raw network socket injection.
3. **Domain Drift:** Performance guarantees rely on the calibration set matching the operating traffic distribution. When deploying on a new network topology, run `scripts/calibrate_gate.py` to recalibrate temperature and confidence thresholds.
