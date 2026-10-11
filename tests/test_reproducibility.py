"""Unit tests for Stage 15: Reproducibility and release package (v1.0)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml


def test_license_exists():
    lic_path = Path("LICENSE")
    assert lic_path.exists(), "LICENSE file is missing."
    text = lic_path.read_text(encoding="utf-8")
    assert "MIT License" in text
    assert "LiteCascade Authors" in text


def test_citation_cff_schema():
    cff_path = Path("CITATION.cff")
    assert cff_path.exists(), "CITATION.cff is missing."
    with open(cff_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert data["cff-version"] == "1.2.0"
    assert "LiteCascade" in data["title"]
    assert data["version"] == "1.0.0"
    assert data["license"] == "MIT"
    assert len(data["authors"]) > 0
    assert "preferred-citation" in data


def test_model_card_completeness():
    mc_path = Path("docs/MODEL_CARD.md")
    assert mc_path.exists(), "docs/MODEL_CARD.md is missing."
    text = mc_path.read_text(encoding="utf-8")
    assert "Model Details" in text
    assert "Intended Use" in text
    assert "Training & Evaluation Data" in text
    assert "Quantitative Performance Summary" in text
    assert "Technical Specifications" in text
    assert "Ethical Considerations" in text
    assert "Limitations & Operational Caveats" in text


def test_requirements_lock():
    lock_path = Path("requirements-lock.txt")
    assert lock_path.exists(), "requirements-lock.txt is missing."
    text = lock_path.read_text(encoding="utf-8")
    assert len(text.strip().splitlines()) >= 20, "Lockfile seems empty or corrupted."
    assert "torch" in text.lower()


def test_makefile_reproducibility_targets():
    make_path = Path("Makefile")
    assert make_path.exists(), "Makefile is missing."
    text = make_path.read_text(encoding="utf-8")
    for target in ["reproduce-fast", "reproduce-full", "table3", "table4", "table5", "table6"]:
        assert target in text, f"Target {target} not defined in Makefile."


def test_reproduce_script_help():
    res = subprocess.run([sys.executable, "scripts/reproduce.py", "--help"], capture_output=True, text=True)
    assert res.returncode == 0, f"reproduce.py --help failed: {res.stderr}"
    assert "LiteCascade Reproducibility Runner" in res.stdout
