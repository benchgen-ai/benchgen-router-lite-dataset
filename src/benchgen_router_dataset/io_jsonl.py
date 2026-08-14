"""JSONL read/write. Append-only by design so a partial run is never lost."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

M = TypeVar("M", bound=BaseModel)


def read_jsonl(path: Path) -> Iterator[dict]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno} is not valid JSON") from exc


def read_models(path: Path, model: type[M]) -> list[M]:
    return [model.model_validate(row) for row in read_jsonl(path)]


def write_jsonl(path: Path, rows: Iterable[BaseModel | dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with tmp.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(_dump(row) + "\n")
            count += 1
    tmp.replace(path)
    return count


def append_jsonl(path: Path, row: BaseModel | dict) -> None:
    """Flushed per line: a crash mid-run keeps everything already collected."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(_dump(row) + "\n")
        fh.flush()


def _dump(row: BaseModel | dict) -> str:
    data = row.model_dump(mode="json") if isinstance(row, BaseModel) else row
    return json.dumps(data, ensure_ascii=False, sort_keys=True)
