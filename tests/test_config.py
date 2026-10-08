from pathlib import Path

import pytest

from litecascade.utils.config import load_config, override_config


def test_load_config(tmp_path: Path):
    """Test loading a basic YAML config."""
    config_content = """
    model:
      name: dummy
      layers: 3
    epochs: 10
    """
    config_file = tmp_path / "test.yaml"
    config_file.write_text(config_content)

    config = load_config(config_file)
    assert config["epochs"] == 10
    assert config["model"]["name"] == "dummy"
    assert config["model"]["layers"] == 3


def test_load_config_not_found():
    """Test loading a non-existent config raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_config("non_existent_file.yaml")


def test_override_config():
    """Test that nested config override works correctly."""
    base_config = {"model": {"name": "dummy", "layers": 3}, "epochs": 10, "lr": 0.01}

    overrides = {"model": {"layers": 5}, "epochs": 20, "new_key": "added"}

    result = override_config(base_config, overrides)

    assert result["model"]["name"] == "dummy"  # Should be untouched
    assert result["model"]["layers"] == 5  # Should be updated
    assert result["epochs"] == 20  # Should be updated
    assert result["lr"] == 0.01  # Should be untouched
    assert result["new_key"] == "added"  # Should be added
