"""Backward-compatible alias: older deploy configs call `SRC/prepare_data.py`.

Prefer `SRC/prepare_diabetes_data.py` in new scripts.
"""
from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).resolve().parent / "prepare_diabetes_data.py"), run_name="__main__")
