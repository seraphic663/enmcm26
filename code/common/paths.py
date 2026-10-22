"""Shared input locations and isolated local output paths."""
from pathlib import Path
import os

SUPPORT_ROOT = Path(__file__).resolve().parents[2]

def protected_output_path(path) -> Path:
    """Resolve a write target strictly inside project output/, including symlinks."""
    target = Path(path).resolve()
    allowed = SUPPORT_ROOT / "output"
    if target == allowed or not target.is_relative_to(allowed):
        raise ValueError("Write targets must stay strictly inside project output/")
    return target

def output_root():
    return protected_output_path(os.environ.get("MCM_OUTPUT_ROOT", SUPPORT_ROOT / "output" / "manual"))

def project_root():
    return SUPPORT_ROOT / "data"

def results_root():
    return output_root() / "result" / "来源证据"

def relative_path(path: Path, base: Path = SUPPORT_ROOT) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()
