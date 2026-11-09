"""Run a source script without changing its historical path or arguments."""
import runpy
import sys
from pathlib import Path


def run(relative_path: str) -> None:
    source = Path(__file__).resolve().parent.parent / relative_path
    sys.path.insert(0, str(source.parent))
    runpy.run_path(str(source), run_name="__main__")
