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
