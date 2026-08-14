"""Domain balance.

Fugu-Lite V1 was 92.7% one domain. A router cannot learn a specialisation it was never shown,
so no domain may exceed `max_share` of the pool.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict

from ..schemas import Task

DEFAULT_MAX_SHARE = 0.30


def domain_counts(tasks: list[Task]) -> Counter[str]:
    return Counter(t.domain for t in tasks)


def domain_shares(tasks: list[Task]) -> dict[str, float]:
    total = len(tasks)
    if not total:
        return {}
    return {d: c / total for d, c in domain_counts(tasks).items()}


def effective_max_share(n_domains: int, max_share: float = DEFAULT_MAX_SHARE) -> float:
    """With `n` domains no allocation can put every domain below `1/n`.

    Asking for 30% across 3 domains is arithmetically impossible; the cap becomes 33.3% and the
    report says so rather than the build silently failing its own gate.
    """
    if n_domains <= 0:
        return max_share
    return max(max_share, 1.0 / n_domains)


def _shares_under_cap(counts: dict[str, int], cap: int) -> tuple[dict[str, int], float]:
    kept = {d: min(n, cap) for d, n in counts.items()}
    total = sum(kept.values())
    worst = max(kept.values()) / total if total else 0.0
    return kept, worst


def enforce_balance(
    tasks: list[Task], max_share: float = DEFAULT_MAX_SHARE, seed: int = 42
) -> tuple[list[Task], dict[str, int]]:
    """Trim over-represented domains until every share <= the effective cap.

    Trimming is seeded and operates on task_id-sorted buckets, so the same input always yields
    the same pool.
    """
    if not tasks:
        return [], {}

    buckets: dict[str, list[Task]] = defaultdict(list)
    for task in tasks:
        buckets[task.domain].append(task)
    for bucket in buckets.values():
        bucket.sort(key=lambda t: t.task_id)

    counts = {d: len(v) for d, v in buckets.items()}
    share = effective_max_share(len(counts), max_share)

    # Largest per-domain cap that still satisfies the share constraint. Raising the cap can only
    # raise the biggest domain's share, so binary search is valid and keeps the most data.
    lo, hi, best = 1, max(counts.values()), 1
    while lo <= hi:
        mid = (lo + hi) // 2
        _, worst = _shares_under_cap(counts, mid)
        if worst <= share + 1e-9:
            best, lo = mid, mid + 1
        else:
            hi = mid - 1

    rng = random.Random(seed)
    kept: list[Task] = []
    removed: dict[str, int] = {}
    for domain, bucket in sorted(buckets.items()):
        if best < len(bucket):
            chosen = sorted(rng.sample(bucket, best), key=lambda t: t.task_id)
            removed[domain] = len(bucket) - best
        else:
            chosen = bucket
        kept.extend(chosen)

    kept.sort(key=lambda t: t.task_id)
    return kept, removed
