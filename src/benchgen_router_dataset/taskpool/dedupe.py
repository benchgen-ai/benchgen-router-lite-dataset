"""Deterministic dedupe on normalised prompt text."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

from ..graders.normalize import normalize_text
from ..schemas import Task


def prompt_fingerprint(task: Task) -> str:
    return hashlib.sha1(normalize_text(task.prompt).encode("utf-8")).hexdigest()


def dedupe(tasks: Iterable[Task]) -> tuple[list[Task], int]:
    """Keep the first occurrence of each fingerprint. Input order decides, so callers sort first."""
    seen: set[str] = set()
    kept: list[Task] = []
    dropped = 0
    for task in tasks:
        fp = prompt_fingerprint(task)
        if fp in seen:
            dropped += 1
            continue
        seen.add(fp)
        kept.append(task)
    return kept, dropped
