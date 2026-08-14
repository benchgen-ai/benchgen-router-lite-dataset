"""Task-pool build orchestration: load -> map -> dedupe -> balance -> split."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..schemas import Task
from ..sources import REGISTRY, LoadStats, SourceUnavailable, iter_rows, map_rows
from .balance import DEFAULT_MAX_SHARE, domain_shares, effective_max_share, enforce_balance
from .dedupe import dedupe
from .split import assert_no_leakage, assign_splits, split_counts


@dataclass
class BuildRequest:
    per_source_limit: dict[str, int]
    max_share: float = DEFAULT_MAX_SHARE
    seed: int = 42
    cache_dir: Path | None = None
    ratios: dict[str, float] | None = None


@dataclass
class BuildResult:
    tasks: list[Task]
    per_source: dict[str, LoadStats] = field(default_factory=dict)
    unavailable: dict[str, str] = field(default_factory=dict)
    duplicates_dropped: int = 0
    balance_removed: dict[str, int] = field(default_factory=dict)

    @property
    def shares(self) -> dict[str, float]:
        return domain_shares(self.tasks)

    @property
    def splits(self) -> dict[str, int]:
        return split_counts(self.tasks)

    def effective_share(self, max_share: float = DEFAULT_MAX_SHARE) -> float:
        return effective_max_share(len(self.shares), max_share)


def build_pool(request: BuildRequest) -> BuildResult:
    result = BuildResult(tasks=[])
    collected: list[Task] = []

    for name, limit in sorted(request.per_source_limit.items()):
        if limit <= 0:
            continue
        source = REGISTRY.get(name)
        if source is None:
            result.unavailable[name] = f"unknown source; known: {sorted(REGISTRY)}"
            continue
        stats = LoadStats()
        try:
            rows = iter_rows(source, cache_dir=request.cache_dir)
            collected.extend(map_rows(source, rows, limit, stats))
        except SourceUnavailable as exc:
            result.unavailable[name] = str(exc)
        result.per_source[name] = stats

    collected.sort(key=lambda t: t.task_id)
    deduped, dropped = dedupe(collected)
    result.duplicates_dropped = dropped

    balanced, removed = enforce_balance(deduped, request.max_share, request.seed)
    result.balance_removed = removed

    tasks = assign_splits(balanced, request.ratios, request.seed)
    assert_no_leakage(tasks)
    result.tasks = tasks
    return result


def gate_2(result: BuildResult, max_share: float = DEFAULT_MAX_SHARE) -> list[str]:
    """Gate 2 failures, as plain sentences. Empty list means the gate passes."""
    problems: list[str] = []
    if not result.tasks:
        return ["task pool is empty"]

    cap = result.effective_share(max_share)
    for domain, share in sorted(result.shares.items()):
        if share > cap + 1e-9:
            problems.append(f"domain {domain!r} is {share:.1%} of the pool, above {cap:.1%}")

    for task in result.tasks:
        if not task.answer.strip():
            problems.append(f"{task.task_id} has an empty answer")
        if not task.grader:
            problems.append(f"{task.task_id} has no grader")

    counts = result.splits
    for split, n in counts.items():
        if n == 0:
            problems.append(f"split {split!r} is empty")

    return problems
