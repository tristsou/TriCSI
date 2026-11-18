"""Resolve the original workflow name through the recorded source migration."""
import json
import runpy
import sys
from pathlib import Path


def run(relative_path: str) -> None:
    root = Path(__file__).resolve().parent.parent
    migration = json.loads((root / 'metadata/migration.json').read_text())
    mapping = {row['old']: row['new'] for row in migration['files']}
    source = root / mapping.get(relative_path, relative_path)
    sys.path[:0] = [str(source.parent), str(root), str(root / 'legacy/scripts')]
    runpy.run_path(str(source), run_name="__main__")
