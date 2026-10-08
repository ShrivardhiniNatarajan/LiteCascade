.PHONY: setup lint test clean

VENV = .venv
PYTHON = $(VENV)/Scripts/python
PIP = $(VENV)/Scripts/pip

setup:
	python -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e .[dev]

lint:
	$(PYTHON) -m ruff check src tests
	$(PYTHON) -m ruff format --check src tests

test:
	$(PYTHON) -m pytest tests/

clean:
	if exist $(VENV) rmdir /s /q $(VENV)
	if exist .pytest_cache rmdir /s /q .pytest_cache
	if exist .ruff_cache rmdir /s /q .ruff_cache
	for /d /r src %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d"
	for /d /r tests %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d"
