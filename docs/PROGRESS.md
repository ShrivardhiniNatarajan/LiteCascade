# LiteCascade Progress Log

## Stage 0: Environment Setup & Initialization
**Date:** 2026-10-08
* **What was done:** 
  * Acknowledged non-negotiable rules.
  * Git was already initialized with remote origin by the user.
  * Created basic directory scaffolding (`src/`, `tests/`, `configs/`, `data/`, `checkpoints/`, `results/`, `docs/`).
  * Created strict `.gitignore` to prevent committing large files and datasets.
  * Created this `PROGRESS.md` tracking log.
* **Verification output summary:** Directory creation successful. Git status reflects untracked files properly ignoring `data/`, `checkpoints/`, and `results/`.
* **Known issues:** None so far. Waiting for explicit requirements for dependency setup (e.g. PyTorch, TFLite).

### Stage 0: Project Scaffold (completed)
* **What was done:** 
  * Updated directory layout (added __init__.py files, `configs/`, `scripts/`, etc.).
  * Added `pyproject.toml`, `requirements.txt`, `Makefile`, `README.md`.
  * Created `utils/seed.py` and `utils/config.py` with tests.
  * Initialized venv and installed all dependencies (`torch`, `tensorflow`, `ruff`, etc.).
  * Fixed lint errors and ran test suite successfully.
* **Verification output summary:** `make lint` passes with 0 errors. `pytest tests/` passes 5 tests in ~3.84s.
* **Known issues:** None.

### Stage 1: Dataset Loaders and EDA (completed)
* **What was done:** 
  * Wrote `docs/DATASETS.md` explaining how to manually download and organize the raw datasets.
  * Created `scripts/make_fixture.py` which generates 2k-row synthetic data files per dataset schema to unblock tests.
  * Created `src/litecascade/data/clean.py` utility to consistently drop NaN, drop inf, drop constant columns, and drop duplicates across all loaders.
  * Wrote dataset loaders for N-BaIoT, CICIoT2023, and IoT-23 in `src/litecascade/data/` that default to synthetic data for CI.
  * Wrote `scripts/eda.py` which analyzes dataset distributions (missing, rows, inf, classes, groups) and exports the statistics to JSON and bar charts to PNG.
  * Added test cases validating the schema loaded by each data loader, ensuring data integrity post-cleaning.
* **Verification output summary:** Both scripts ran flawlessly (`scripts/make_fixture.py` and `scripts/eda.py`). The dataset loader tests pass, verifying the schema correctness. `ruff check` passes completely.
* **Known issues:** Real data logic in loaders throws NotImplementedError if `use_synthetic=False`, as we cannot access real raw datasets yet.

### Stage 2: Preprocessing Pipeline (completed)
* **What was done:**
  * Implemented `splits.py` for group-aware temporal splitting (`create_group_splits`) and leave-one-family-out cross-validation (`create_lofo_split`). Verified zero group overlaps. Splits metadata mapped and saved to JSON files.
  * Implemented `windows.py` to create sliding windows of $T$ records while strictly preventing overlap across group boundaries. Label is assigned to the last record's label.
  * Implemented `preprocess.py` featuring a `Preprocessor` class using `RobustScaler` and `IncrementalPCA` (d=32). Strictly fits on training data and applies transform consistently. Exposes `pca.components_` and `pca.mean_` for future initialization.
  * Implemented PyTorch Dataset/DataLoader wrappers (`dataset.py`) including a utility to compute balanced class weights.
  * Wrote tests for overlap validation, scaler statistics, window constraints, and LOFO exclusion logic.
* **Verification output summary:** `ruff` completes with 0 errors. `pytest` executes 12 tests successfully with zero failures.
* **Known issues:** Warning regarding `np.random.shuffle(unique_groups)` on Pandas string arrays (minor typing warning, safely operates on the underlying object list).

### Stage 3: Metrics and Profiler (completed)
* **What was done:**
  * Implemented `metrics.py` covering all classification metrics (precision, recall, macro-f1, fpr, mcc, acc, confusion matrix stats).
  * Added `bootstrap_metrics` for computing 95% Confidence Intervals via non-parametric bootstrapping.
  * Implemented `profiler.py` leveraging `fvcore` for MACs/activation estimates, `psutil` for memory, `time.perf_counter` for hardware p50/p95 latency testing, and INT8/FP32 model size estimation via tmp file saves on disk.
  * Added `expected_cost` helper that factors Sentinel-Analyst cost via $E[C] = c_1 + (1-\rho) \cdot c_2$, returning relative `speed_up`.
  * Covered implementations comprehensively with unit tests over deterministic toy distributions and tiny manual models (e.g. tracking `nn.Linear` 55-param properties).
* **Verification output summary:** `ruff` completes format verification, and `pytest tests/test_metrics.py tests/test_profiler.py` succeeds fully (all assertions validated on metrics definitions, zero failures). The `check_leakage` fix for Stage 2 was also included.
* **Known issues:** `fvcore` and Torch JIT throw a minor warning regarding trace modes during profile runs, and Torch AO triggers deprecation logs; safely ignored since measurements compute fine.

### Stage 4: Baselines B1–B8 (completed)
* **What was done:**
  * Created `src/litecascade/models/baselines.py` housing B1 (MLP), B2 (Tree-wrapper for RF/XGBoost), B3 (LSTM), B4 (BiLSTM), B5 (BiLSTM+IPCA pipeline marker), B6 (CNN-BiLSTM), B7 (DSCNN), and B8 (Transformer).
  * Devised `src/litecascade/train/trainer.py` to wrap PyTorch backprop + Tree `.fit()` behind a single clean `UnifiedTrainer` module, with class weights, L1-regularization for B1, early stopping and validation loops.
  * Designed CLI runner `python -m litecascade.train.run` taking `--model`, `--dataset`, `--epochs`, generating robust deterministic results.
  * Explicitly documented any model modifications/deviations in `docs/BASELINES.md`.
* **Verification output summary:** PyTorch Baselines correctly instantiate and emit sensible properties (parameters/MACs > 0). The `run` CLI successfully trains B3 for 2 epochs on the fixture.
* **Known issues:** Extremely small synthetic datasets may have a 0-item evaluation set due to strict group splitting; added a fallback guard in `trainer.py`. PyTorch 2.x `torch.ao` module gives deprecation warnings during profiling, but correctly builds quantized graphs.
