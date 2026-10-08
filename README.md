# LiteCascade: Lightweight Early-Exit Cascade for IoT Malware/Intrusion Detection

LiteCascade is an early-exit neural network cascade designed to provide highly accurate, resource-efficient, and fast IoT malware/intrusion detection. This repository tracks the model's implementation from data preprocessing and baseline evaluation to joint cascade training, quantization (TFLite INT8), and on-device (Raspberry Pi/MCU) benchmarking.

## 🚀 Current Progress

This project is being developed in strict sequential stages. The following components have been fully implemented, tested, and verified:

### Stage 0: Project Scaffolding
* Established a modular Python package structure (`src/litecascade/`) with strict linting (`ruff`), testing (`pytest`), and standard configuration patterns (`Makefile`, `pyproject.toml`).
* Configured reproducibility utilities ensuring fixed seeds across PyTorch, NumPy, and Python standard libraries.

### Stage 1: Data Loaders and EDA
* Implemented strict-schema pandas dataloaders for standard IoT security datasets (N-BaIoT, CICIoT2023, IoT-23).
* Developed a synthetic data fixture generator for safe, fast continuous integration without distributing heavy raw data files.
* Built Exploratory Data Analysis (`eda.py`) scripts for dataset sanity checks and distribution visualization.

### Stage 2: Leakage-Safe Preprocessing
* **Group-Aware Splitting:** Enforces strict boundary rules guaranteeing zero overlap of devices/captures across train, validation, calibration, and test splits to prevent data leakage.
* **Zero-Day Splits:** Generates Leave-One-Family-Out (LOFO) splits to test model robustness on unseen malware.
* **Windowing:** Creates sliding temporal windows without crossing group boundaries.
* **Feature Scaling & IPCA:** Safely fits `RobustScaler` and `IncrementalPCA` exclusively on training data, applying deterministic transforms to downstream sets. Wrapped in native PyTorch `DataLoader`s with auto-calculated class weights.

### Stage 3: Evaluation Metrics and Hardware Profiling
* **Metrics Engine:** Calculates robust detection metrics (Macro-F1, Precision, Malicious Recall, MCC, FPR) with 95% Confidence Intervals determined via non-parametric bootstrapping.
* **Efficiency Profiler:** Leverages `fvcore` and `psutil` to extract physical deployment costs, including Parameter Count, Multiply-Accumulate Operations (MACs), estimated Activation Memory, static file sizes (FP32 vs INT8), and simulated CPU latency (p50/p95 ms).
* Implemented theoretical `expected_cost` and speed-up calculation logic for the final cascade evaluation.

### Stage 4: State-of-the-Art Baselines
* Implemented 8 distinct baseline architectures reflecting prior literature and standard approaches:
  * **B1:** L1-Regularized MLP (SNIPE-style)
  * **B2:** Random Forest / XGBoost Wrapper
  * **B3 & B4:** LSTM and BiLSTM
  * **B5:** BiLSTM with IPCA
  * **B6:** CNN-BiLSTM (Jouhari & Guizani style)
  * **B7:** Depthwise-Separable CNN (DSCNN)
  * **B8:** Tiny Transformer Encoder (2 layers, 2 heads)
* **Unified Trainer:** Developed a centralized trainer managing AdamW optimization, Cosine Annealing, AMP (Automatic Mixed Precision), early stopping, and seamless abstraction over both PyTorch neural networks and Sklearn/XGBoost tree structures.

## 💻 Quickstart (Developer Guide)

**1. Setup Environment**
```bash
make setup
```
*(On Windows, you can activate the environment and install requirements manually via `.venv\Scripts\activate` and `pip install -r requirements.txt`).*

**2. Run the Test Suite**
```bash
make test
```

**3. Train a Baseline Model**
You can run any implemented baseline (b1-b8) on the synthetic data fixture:
```bash
python -m litecascade.train.run --model b3 --dataset fixture --epochs 2
```
This will save the model weights and a detailed `metrics.json` containing the detection performance and hardware profile inside the `results/` folder.
