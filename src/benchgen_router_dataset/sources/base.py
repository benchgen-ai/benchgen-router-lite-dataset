"""Source adapter contract. One module per upstream dataset, all normalising to `Task`."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..schemas import SourceRef, Task


@dataclass(frozen=True)
class SourceSpec:
    name: str
    dataset: str
    split: str
    domain: str
    difficulty: str
    license: str
    answer_type: str
    grader: str
    config: str | None = None
    task_group: str = "in_distribution"
    redistributable: bool = True
    subject_field: str | None = None
    notes: str = ""


RowMapper = Callable[[SourceSpec, int, dict[str, Any]], Task | None]


def first_field(row: dict[str, Any], names: Sequence[str]) -> Any:
    """First present, non-empty field among `names`.

    Upstream column names drift between mirrors of the same benchmark; guessing one and
    silently yielding zero rows is worse than accepting a short list of known spellings.
    """
    for name in names:
        value = row.get(name)
        if value not in (None, "", [], {}):
            return value
    return None


@dataclass(frozen=True)
class Source:
    spec: SourceSpec
    mapper: RowMapper


REGISTRY: dict[str, Source] = {}


def register(source: Source) -> Source:
    if source.spec.name in REGISTRY:
        raise ValueError(f"duplicate source {source.spec.name!r}")
    REGISTRY[source.spec.name] = source
    return source


def make_task(
    spec: SourceSpec,
    index: int,
    prompt: str,
    answer: str,
    subject: str | None = None,
    difficulty: str | None = None,
) -> Task:
    """Split is a placeholder here; `taskpool.split` assigns the real one with a fixed seed."""
    return Task(
        task_id=f"{spec.name}/{spec.split}/{index:06d}",
        prompt=prompt,
        answer=answer,
        answer_type=spec.answer_type,  # type: ignore[arg-type]
        grader=spec.grader,
        domain=spec.domain,  # type: ignore[arg-type]
        difficulty=(difficulty or spec.difficulty),  # type: ignore[arg-type]
        split="train",
        task_group=spec.task_group,  # type: ignore[arg-type]
        source=SourceRef(
            dataset=spec.dataset,
            config=spec.config,
            split=spec.split,
            index=index,
            license=spec.license,
            subject=subject,
            redistributable=spec.redistributable,
        ),
    )


@dataclass
class LoadStats:
    requested: int = 0
    yielded: int = 0
    skipped: int = 0
    reasons: dict[str, int] = field(default_factory=dict)

    def skip(self, reason: str) -> None:
        self.skipped += 1
        self.reasons[reason] = self.reasons.get(reason, 0) + 1


def map_rows(
    source: Source, rows: Iterator[dict[str, Any]], limit: int | None, stats: LoadStats
) -> Iterator[Task]:
    for index, row in enumerate(rows):
        if limit is not None and stats.yielded >= limit:
            return
        stats.requested += 1
        try:
            task = source.mapper(source.spec, index, row)
        except Exception as exc:  # a malformed upstream row must not kill the build
            stats.skip(f"mapper_error:{type(exc).__name__}")
            continue
        if task is None:
            stats.skip("mapper_returned_none")
            continue
        stats.yielded += 1
        yield task
