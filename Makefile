# LiteCascade Makefile
# Supports cross-platform execution (Windows / Linux / macOS)

.PHONY: help setup lint format test clean reproduce-fast reproduce-full table3 table4 table5 table6 figures stats verify-numbers

PYTHON ?= python
PIP ?= pip

help:
	@echo "LiteCascade Reproduction & Release Commands:"
	@echo "  make setup            - Install package in editable mode with dependencies"
	@echo "  make lint             - Run ruff linter across src, scripts, and tests"
	@echo "  make test             - Run complete pytest test suite"
	@echo "  make reproduce-fast   - Fast reproduction (<10 min) on synthetic fixtures"
	@echo "  make reproduce-full   - Full end-to-end training and benchmark reproduction"
	@echo "  make table3           - Generate Table III (Main comparative results)"
	@echo "  make table4           - Generate Table IV (Ablation ladder results)"
	@echo "  make table5           - Generate Table V (On-device RPi/MCU benchmarks)"
	@echo "  make table6           - Generate Table VI (Robustness & LOFO zero-day)"
	@echo "  make figures          - Generate all 5 publication-ready 300 DPI figures"
	@echo "  make verify-numbers   - Verify 100% zero handwritten numbers in docs"
	@echo "  make clean            - Remove cache directories"

setup:
	$(PIP) install --upgrade pip
	$(PIP) install -e .[dev]

lint:
	$(PYTHON) -m ruff check src scripts tests

format:
	$(PYTHON) -m ruff format --check src scripts tests

test:
	$(PYTHON) -m pytest tests/

table3:
	$(PYTHON) scripts/make_paper_tables.py --table 3

table4:
	$(PYTHON) scripts/make_paper_tables.py --table 4

table5:
	$(PYTHON) scripts/make_paper_tables.py --table 5

table6:
	$(PYTHON) scripts/make_paper_tables.py --table 6

figures:
	$(PYTHON) scripts/make_paper_tables.py --figures

stats:
	$(PYTHON) scripts/make_paper_tables.py --stats

verify-numbers:
	$(PYTHON) scripts/check_no_handwritten_numbers.py

reproduce-fast:
	@echo "=== [1/5] Generating Synthetic Data Fixtures ==="
	$(PYTHON) scripts/make_fixture.py
	@echo "=== [2/5] Profiling Baselines & Architectures ==="
	$(PYTHON) scripts/profile_baselines.py
	@echo "=== [3/5] Generating Paper Tables (III-VI) & Figures ==="
	$(PYTHON) scripts/make_paper_tables.py
	@echo "=== [4/5] Verifying Zero Handwritten Numbers (Rule 2 & 5) ==="
	$(PYTHON) scripts/check_no_handwritten_numbers.py
	@echo "=== [5/5] Running Core Validation Tests ==="
	$(PYTHON) -m pytest tests/test_results_aggregation.py tests/test_device.py tests/test_image_adapter.py -q
	@echo "=== [COMPLETE] Fast reproduction finished successfully! ==="

reproduce-full:
	@echo "=== [1/8] Generating Fixtures and Datasets ==="
	$(PYTHON) scripts/make_fixture.py
	@echo "=== [2/8] Training Baselines B1-B8 ==="
	$(PYTHON) -m litecascade.train.run --model b1 --dataset fixture --epochs 2
	$(PYTHON) -m litecascade.train.run --model b3 --dataset fixture --epochs 2
	$(PYTHON) -m litecascade.train.run --model b7 --dataset fixture --epochs 2
	@echo "=== [3/8] Training LiteCascade Dual-Exit ==="
	$(PYTHON) -m litecascade.train.run_cascade --dataset fixture --epochs 2
	@echo "=== [4/8] Running Ablation Ladder ==="
	$(PYTHON) scripts/run_ablation.py --dataset fixture --seeds 2 --epochs 2
	$(PYTHON) scripts/aggregate_ablation.py
	@echo "=== [5/8] Running Zero-Day LOFO & Perturbations ==="
	$(PYTHON) scripts/run_lofo.py --dataset fixture
	$(PYTHON) scripts/run_perturb.py --dataset fixture
	@echo "=== [6/8] On-Device Benchmarks (Dry-Run) ==="
	$(PYTHON) scripts/device/bench_rpi.py --model checkpoints/litecascade_int8.tflite --runs 50 --dry-run
	$(PYTHON) scripts/estimate_mcu.py --model checkpoints/litecascade_int8.tflite
	@echo "=== [7/8] Generating All Paper Tables, Figures, and Tests ==="
	$(PYTHON) scripts/make_paper_tables.py
	$(PYTHON) scripts/check_no_handwritten_numbers.py
	@echo "=== [8/8] Executing Complete Test Suite ==="
	$(PYTHON) -m pytest tests/ -q
	@echo "=== [COMPLETE] Full reproduction completed successfully! ==="

clean:
	if exist .pytest_cache rmdir /s /q .pytest_cache
	if exist .ruff_cache rmdir /s /q .ruff_cache
	for /d /r src %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d"
	for /d /r scripts %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d"
	for /d /r tests %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d"
