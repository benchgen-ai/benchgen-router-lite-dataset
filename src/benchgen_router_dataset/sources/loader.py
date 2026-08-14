"""Row loading. `datasets` is an optional extra so the package imports without network deps."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ..io_jsonl import read_jsonl
from ..paths import data_dir
from .base import Source

LOCAL_PREFIX = "local:"


class SourceUnavailable(RuntimeError):
    """Raised when a source cannot be read; the build reports it instead of silently skipping."""


def iter_rows(source: Source, cache_dir: Path | None = None) -> Iterator[dict[str, Any]]:
    spec = source.spec
    if spec.dataset.startswith(LOCAL_PREFIX):
        yield from _iter_local(spec.dataset[len(LOCAL_PREFIX):])
        return
    yield from _iter_hf(spec.dataset, spec.config, spec.split, cache_dir)


def _iter_local(relative: str) -> Iterator[dict[str, Any]]:
    path = data_dir().parent / relative
    if not path.exists():
        raise SourceUnavailable(
            f"{path} not found — export the bundle there before building this source"
        )
    yield from read_jsonl(path)


def _iter_hf(
    dataset: str, config: str | None, split: str, cache_dir: Path | None
) -> Iterator[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SourceUnavailable(
            "the `datasets` extra is required for Hugging Face sources: pip install -e .[sources]"
        ) from exc

    try:
        ds = load_dataset(
            dataset, config, split=split, cache_dir=str(cache_dir) if cache_dir else None
        )
    except Exception as exc:
        raise SourceUnavailable(f"{dataset} ({config}/{split}): {exc}") from exc

    for row in ds:
        yield dict(row)
