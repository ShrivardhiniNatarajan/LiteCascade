"""Unit tests for Stage 14: Results aggregation, paper tables, and figure verification."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.check_no_handwritten_numbers import (
    DOCS_FILE,
    extract_numbers_from_guide,
    extract_numbers_from_results,
    verify_traceability,
)

TABLES_DIR = Path("results/tables")
FIGURES_DIR = Path("results/figures")


def test_paper_tables_exist_and_non_empty():
    """Validates that Tables III-VI exist in both CSV and LaTeX formats with data."""
    required_tables = ["main", "ablation", "device", "robustness"]

    for t_name in required_tables:
        csv_p = TABLES_DIR / f"{t_name}.csv"
        tex_p = TABLES_DIR / f"{t_name}.tex"

        assert csv_p.exists(), f"Missing table CSV: {csv_p}"
        assert tex_p.exists(), f"Missing table LaTeX: {tex_p}"

        df = pd.read_csv(csv_p)
        assert len(df) > 0, f"Table CSV {csv_p} is empty!"

        tex_content = tex_p.read_text(encoding="utf-8")
        assert "tabular" in tex_content or "begin" in tex_content


def test_figures_exist_and_valid():
    """Validates that all 5 IEEE single-column publication figures exist."""
    required_figures = [
        "pareto_f1_macs.png",
        "pareto_f1_size.png",
        "exit_rate_vs_epsilon.png",
        "reliability_diagram.png",
        "lofo_zero_day.png",
    ]

    for fig_name in required_figures:
        fig_p = FIGURES_DIR / fig_name
        assert fig_p.exists(), f"Missing figure: {fig_p}"
        # Ensure non-trivial binary file (> 5 KB)
        assert fig_p.stat().st_size > 5000, f"Figure file suspiciously small: {fig_p}"


def test_statistical_tests_structure():
    """Validates Wilcoxon signed-rank test output schema and sample size notes."""
    stat_p = TABLES_DIR / "statistical_tests.json"
    assert stat_p.exists(), f"Missing statistical tests file: {stat_p}"

    with open(stat_p, encoding="utf-8") as f:
        stat_data = json.load(f)

    assert "n_seeds" in stat_data
    assert "comparisons" in stat_data
    assert "sample_size_note" in stat_data

    # Mathematical property: N=5 seeds bounding
    assert "0.0625" in stat_data["sample_size_note"]

    for b_name in ["B1_MLP", "B3_LSTM", "B7_DSCNN"]:
        assert b_name in stat_data["comparisons"]
        c = stat_data["comparisons"][b_name]
        assert "p_value" in c
        assert "wilcoxon_stat" in c


def test_check_no_handwritten_numbers():
    """Validates that every metric claim in docs/PAPER_FILL_GUIDE.md traces to results/."""
    results_numbers = extract_numbers_from_results()
    claimed_numbers = extract_numbers_from_guide(DOCS_FILE)

    assert len(claimed_numbers) >= 50, "Did not extract expected number of claims from guide"

    passed, untraceable = verify_traceability(results_numbers, claimed_numbers)
    assert len(untraceable) == 0, f"Untraceable numbers found: {untraceable}"
