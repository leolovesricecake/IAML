import json
from pathlib import Path
from typing import Iterable, Mapping


def ensure_dir(path) -> Path:
    """Create and return a directory path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path, payload) -> None:
    """Write JSON with deterministic formatting."""
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path, rows: Iterable[Mapping]) -> None:
    """Write JSON lines."""
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path):
    """Read JSON lines into dictionaries."""
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)
