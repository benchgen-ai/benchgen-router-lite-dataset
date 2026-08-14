"""Resume support.

A resumed run that silently changes `agent_order` or `repetitions` corrupts every downstream
index, so mismatches are hard errors rather than warnings.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..io_jsonl import read_jsonl


class IncompatibleRewardsFile(RuntimeError):
    pass


@dataclass
class ResumeState:
    completed: set[str]
    records_seen: int

    @property
    def is_empty(self) -> bool:
        return self.records_seen == 0


def inspect(
    path: Path, pool_version: str, agent_order: list[str], repetitions: int
) -> ResumeState:
    completed: set[str] = set()
    seen = 0

    for row in read_jsonl(path):
        seen += 1
        _require(row, "pool_version", pool_version, path)
        _require(row, "agent_order", agent_order, path)
        _require(row, "repetitions", repetitions, path)
        task_id = row.get("task_id")
        if not task_id:
            raise IncompatibleRewardsFile(f"{path}: a record has no task_id")
        completed.add(str(task_id))

    return ResumeState(completed=completed, records_seen=seen)


def _require(row: dict, field: str, expected: object, path: Path) -> None:
    actual = row.get(field)
    if actual != expected:
        raise IncompatibleRewardsFile(
            f"{path}: existing records have {field}={actual!r} but this run uses {expected!r}. "
            "Refusing to append — start a new pool version instead."
        )
