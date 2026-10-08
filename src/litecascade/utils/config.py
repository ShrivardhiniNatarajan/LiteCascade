from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        config_path (str | Path): Path to the YAML file.

    Returns:
        Dict[str, Any]: Parsed configuration dictionary.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config or {}


def override_config(base_config: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Recursively override a base configuration with new values.

    Args:
        base_config (Dict[str, Any]): The original configuration.
        overrides (Dict[str, Any]): The new configuration values to apply.

    Returns:
        Dict[str, Any]: The overridden configuration.
    """
    result = base_config.copy()

    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = override_config(result[key], value)
        else:
            result[key] = value

    return result
