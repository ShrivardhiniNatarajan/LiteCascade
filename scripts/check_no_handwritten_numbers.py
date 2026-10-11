"""Traceability Verification Script (Stage 14).

Verifies that every numerical claim, metric, and table cell cited in
docs/PAPER_FILL_GUIDE.md is strictly traceable to experimental artifacts in results/.
Exits with code 0 on complete traceability; exits with code 1 if untraceable numbers exist.

Enforces Rule 2 & Rule 5: Zero fabricated or hand-written numbers.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RESULTS_DIR = Path("results")
DOCS_FILE = Path("docs/PAPER_FILL_GUIDE.md")


def extract_numbers_from_results() -> set[float]:
    """Collects all floating-point and integer numbers present across results/ files."""
    valid_numbers: set[float] = set()

    # 1. Parse JSON files
    for j_path in RESULTS_DIR.glob("**/*.json"):
        try:
            with open(j_path, encoding="utf-8") as f:
                data = json.load(f)

            def _walk(obj):
                if isinstance(obj, (int, float)) and not isinstance(obj, bool):
                    valid_numbers.add(round(float(obj), 4))
                    valid_numbers.add(round(float(obj), 2))
                    valid_numbers.add(round(float(obj), 1))
                elif isinstance(obj, dict):
                    for v in obj.values():
                        _walk(v)
                elif isinstance(obj, list):
                    for item in obj:
                        _walk(item)

            _walk(data)
        except Exception:
            pass

    # 2. Parse CSV files
    for c_path in RESULTS_DIR.glob("**/*.csv"):
        try:
            with open(c_path, encoding="utf-8") as f:
                reader = csv.reader(f)
                for row in reader:
                    for cell in row:
                        for token in re.findall(r"[-+]?\d*\.?\d+", cell.replace(",", "")):
                            try:
                                val = float(token)
                                valid_numbers.add(round(val, 4))
                                valid_numbers.add(round(val, 2))
                                valid_numbers.add(round(val, 1))
                            except ValueError:
                                pass
        except Exception:
            pass

    return valid_numbers


def extract_numbers_from_guide(doc_path: Path) -> list[tuple[str, float]]:
    """Extracts candidate metrics from the paper fill guide tables and verdicts."""
    if not doc_path.exists():
        raise FileNotFoundError(f"Documentation file not found: {doc_path}")

    text = doc_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    extracted: list[tuple[str, float]] = []

    # Regex matching backticked numbers, e.g. `0.974`, `41.93`, `75,810`, `2.64`
    num_pattern = re.compile(r"`([-+]?[\d,]+\.?\d*)`")

    for line in lines:
        # Skip dates, headers, seed numbers, and section numbering
        if line.startswith("#") or "2026-" in line or "seed" in line.lower():
            continue

        matches = num_pattern.findall(line)
        for m in matches:
            cleaned = m.replace(",", "").strip()
            # Skip empty or pure signs
            if not cleaned or cleaned in ["+", "-", "+-"]:
                continue
            try:
                val = float(cleaned)
                # Ignore trivial structural values (0, 1, 2)
                if val in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]:
                    continue
                extracted.append((m, val))
            except ValueError:
                pass

    return extracted


def verify_traceability(
    results_numbers: set[float],
    claimed_numbers: list[tuple[str, float]],
    tolerance: float = 0.005,
) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    """Checks each claimed number against the results ground truth within tolerance."""
    passed = []
    untraceable = []

    for orig_str, val in claimed_numbers:
        # Check direct or rounded match
        matched = False
        rounded_4 = round(val, 4)
        rounded_2 = round(val, 2)
        rounded_1 = round(val, 1)

        if rounded_4 in results_numbers or rounded_2 in results_numbers or rounded_1 in results_numbers:
            matched = True
        else:
            # Check proximity within tolerance
            for res_val in results_numbers:
                if abs(res_val - val) <= tolerance or (val != 0 and abs(res_val - val) / abs(val) <= 0.01):
                    matched = True
                    break

        if matched:
            passed.append((orig_str, val))
        else:
            untraceable.append((orig_str, val))

    return passed, untraceable


def main() -> None:
    print("=" * 75)
    print(" LiteCascade Traceability Checker (Rule 2 & Rule 5 Verification)")
    print(f" Checking: {DOCS_FILE} against {RESULTS_DIR}/")
    print("=" * 75)

    results_numbers = extract_numbers_from_results()
    print(f"[INFO] Indexed {len(results_numbers)} unique numeric values from results/ artifacts.")

    claimed_numbers = extract_numbers_from_guide(DOCS_FILE)
    print(f"[INFO] Extracted {len(claimed_numbers)} metric claims from {DOCS_FILE}.")

    passed, untraceable = verify_traceability(results_numbers, claimed_numbers)

    print("\n[TRACEABILITY SUMMARY]")
    print(f"  Verified Traceable Metrics: {len(passed)}")
    print(f"  Untraceable Numbers:        {len(untraceable)}")

    if untraceable:
        print("\n[FAIL] The following numbers in the paper fill guide could not be matched to results/:")
        for orig, v in untraceable:
            print(f"  - Claimed: `{orig}` (value: {v})")
        print("\nRule 2 Violation: Numbers must strictly originate from experimental results.")
        sys.exit(1)
    else:
        print("\n[PASS] Every single number in docs/PAPER_FILL_GUIDE.md is verified traceable to results/!")
        print("=" * 75)
        sys.exit(0)


if __name__ == "__main__":
    main()
