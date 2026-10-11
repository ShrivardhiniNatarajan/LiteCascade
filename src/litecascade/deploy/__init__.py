"""Deployment and live replay engine modules for LiteCascade."""

def __getattr__(name: str):
    if name == "ReplayEngine":
        from .replay import ReplayEngine
        return ReplayEngine
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = ["ReplayEngine"]
