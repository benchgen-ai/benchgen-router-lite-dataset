"""Split assignment: stratified by (domain, difficulty), fixed seed, assigned exactly once."""

from __future__ import annotations

import random
from collections import defaultdict

from ..schemas import Task

DEFAULT_RATIOS = {"train": 0.6, "validation": 0.2, "test": 0.2}


def assign_splits(
    tasks: list[Task], ratios: dict[str, float] | None = None, seed: int = 42
) -> list[Task]:
    """Return new Task objects with `split` set. Never call this twice on the same pool.

    Held-out tasks (Trinity §4.3) go entirely to `test`; zero-shot transfer stops meaning
    anything the moment one of them is trained on.
    """
    ratios = ratios or DEFAULT_RATIOS
    if abs(sum(ratios.values()) - 1.0) > 1e-9:
        raise ValueError(f"split ratios must sum to 1.0, got {sum(ratios.values())}")

    out: list[Task] = [
        task.model_copy(update={"split": "test"})
        for task in tasks
        if task.task_group == "held_out"
    ]

    strata: dict[tuple[str, str], list[Task]] = defaultdict(list)
    for task in tasks:
        if task.task_group == "held_out":
            continue
        strata[(task.domain, task.difficulty)].append(task)

    for key in sorted(strata):
        bucket = sorted(strata[key], key=lambda t: t.task_id)
        random.Random(f"{seed}/{key}").shuffle(bucket)
        n = len(bucket)
        n_train = int(round(n * ratios["train"]))
        n_val = int(round(n * ratios["validation"]))
        # Remainder lands in test; with tiny strata that keeps test non-empty.
        assignments = (
            ["train"] * n_train + ["validation"] * n_val + ["test"] * (n - n_train - n_val)
        )
        for task, split in zip(bucket, assignments, strict=True):
            out.append(task.model_copy(update={"split": split}))

    out.sort(key=lambda t: t.task_id)
    return out


def split_counts(tasks: list[Task]) -> dict[str, int]:
    counts: dict[str, int] = {"train": 0, "validation": 0, "test": 0}
    for task in tasks:
        counts[task.split] = counts.get(task.split, 0) + 1
    return counts


def assert_no_leakage(tasks: list[Task]) -> None:
    seen: dict[str, str] = {}
    for task in tasks:
        prior = seen.get(task.task_id)
        if prior is not None and prior != task.split:
            raise ValueError(f"{task.task_id} appears in both {prior} and {task.split}")
        seen[task.task_id] = task.split
        if task.task_group == "held_out" and task.split != "test":
            raise ValueError(f"{task.task_id} is held out but landed in {task.split}")
