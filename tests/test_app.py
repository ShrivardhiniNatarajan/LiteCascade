"""Unit tests for the Streamlit demo dashboard (Stage 16)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_app_headless_flag():
    """Verify that --headless-test executes end-to-end and exits cleanly with 0."""
    res = subprocess.run([sys.executable, "app_streamlit_legacy/main.py", "--headless-test"], capture_output=True, text=True)
    assert res.returncode == 0, f"Headless run failed: {res.stderr}"
    assert "Headless sanity check passed successfully!" in res.stdout


def test_streamlit_apptest_run():
    """Verify that Streamlit AppTest initializes and mounts components without error."""
    app_path = (Path(__file__).parent.parent / "app_streamlit_legacy" / "main.py").resolve()
    at = AppTest.from_file(str(app_path), default_timeout=30)
    at.run()
    assert not at.exception, f"Streamlit AppTest raised exception: {at.exception}"


def test_missing_files_graceful_handling():
    """Verify that missing result files return None and do not crash data loaders."""
    from app_streamlit_legacy.main import load_csv_safely, load_json_safely

    missing_path = Path("results/tables/non_existent_file_test.csv")
    res_csv = load_csv_safely(missing_path)
    assert res_csv is None

    missing_json = Path("results/device/non_existent_device_test.json")
    res_json = load_json_safely(missing_json)
    assert res_json is None
