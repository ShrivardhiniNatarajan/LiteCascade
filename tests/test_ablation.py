import csv
import json
from pathlib import Path

from scripts.aggregate_ablation import aggregate_ablation
from scripts.run_ablation import ABLATION_ORDER, run_ablation_ladder_for_seed


def test_ablation_ladder_execution(tmp_path: Path):
    """Verifies that running the ablation ladder produces all 10 steps A0-A9."""
    results = run_ablation_ladder_for_seed(
        dataset_name="fixture",
        seed=42,
        epochs=1,
        variant="R",
        output_dir=tmp_path,
    )

    assert len(results) == 10
    for abl_id in ABLATION_ORDER:
        assert abl_id in results
        entry = results[abl_id]
        assert "f1" in entry
        assert "rho" in entry
        assert "expected_macs" in entry
        assert "size_kb" in entry
        assert entry["is_synthetic"] is True
        assert entry["dataset"] == "fixture"
        assert (tmp_path / abl_id / "seed_42.json").exists()

    # A0 and A1 must be analyst-only with zero exit rate
    assert results["A0"]["rho"] == 0.0
    assert results["A1"]["rho"] == 0.0


def test_aggregate_ablation_csv_and_tex(tmp_path: Path):
    """Verifies that aggregate_ablation produces valid CSV and LaTeX with 10 rows and std."""
    ablation_dir = tmp_path / "ablation"
    tables_dir = tmp_path / "tables"

    # Populate 10 ablation folders with 2 seeds each having different numbers
    for i, abl_id in enumerate(ABLATION_ORDER):
        folder = ablation_dir / abl_id
        folder.mkdir(parents=True, exist_ok=True)
        for seed_idx, seed in enumerate([42, 43]):
            data = {
                "ablation": abl_id,
                "description": f"Test {abl_id}",
                "seed": seed,
                "dataset": "fixture",
                "is_synthetic": True,
                "variant": "R",
                "f1": 0.50 + 0.05 * seed_idx + 0.01 * i,
                "rho": 0.10 * seed_idx,
                "expected_macs": 50000.0 + 1000.0 * seed_idx,
                "size_kb": 60.0 - 5.0 * seed_idx,
            }
            with open(folder / f"seed_{seed}.json", "w", encoding="utf-8") as f:
                json.dump(data, f)

    rows = aggregate_ablation(ablation_dir, tables_dir)
    assert len(rows) == 10

    csv_path = tables_dir / "ablation.csv"
    tex_path = tables_dir / "ablation.tex"

    assert csv_path.exists()
    assert tex_path.exists()

    # Read CSV
    with open(csv_path, encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    assert len(reader) == 10
    for row in reader:
        assert float(row["f1_std"]) > 0.0  # std column populated when changing seed changes numbers
        assert "synthetic" in row["dataset"]

    # Read LaTeX
    tex_content = tex_path.read_text(encoding="utf-8")
    assert "\\begin{table}" in tex_content
    for abl_id in ABLATION_ORDER:
        assert abl_id in tex_content


def test_variant_t_ablation(tmp_path: Path):
    """Verifies that the ablation ladder supports Variant T."""
    results = run_ablation_ladder_for_seed(
        dataset_name="fixture",
        seed=42,
        epochs=1,
        variant="T",
        output_dir=tmp_path,
    )
    assert results["A0"]["variant"] == "T"
    assert results["A7"]["variant"] == "T"
