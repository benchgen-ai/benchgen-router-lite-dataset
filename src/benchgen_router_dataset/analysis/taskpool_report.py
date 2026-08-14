"""Task-pool report: counts by domain, difficulty, source, split, licence."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from ..schemas import Task
from ..taskpool import BuildResult
from .report import header, pct, table


def render_task_pool(
    result: BuildResult, inputs: dict[str, Path], seed: int, max_share: float
) -> str:
    tasks: list[Task] = result.tasks
    total = len(tasks)
    cap = result.effective_share(max_share)
    lines = header(
        "Task pool",
        inputs,
        {
            "tasks": str(total),
            "seed": str(seed),
            "requested max domain share": f"{max_share:.0%}",
            "effective max domain share": f"{cap:.1%}",
        },
    )

    lines += ["## By domain", ""]
    domain_counts = Counter(t.domain for t in tasks)
    lines += table(
        ["Domain", "Tasks", "Share", "Within cap"],
        [
            [d, str(n), pct(n / total if total else 0), "yes" if n / total <= cap + 1e-9 else "NO"]
            for d, n in sorted(domain_counts.items())
        ],
    )

    lines += ["## By difficulty", ""]
    diff_counts = Counter(t.difficulty for t in tasks)
    lines += table(
        ["Difficulty", "Tasks", "Share"],
        [[d, str(n), pct(n / total if total else 0)] for d, n in sorted(diff_counts.items())],
    )

    lines += ["## By split", ""]
    lines += table(
        ["Split", "Tasks", "Share"],
        [[s, str(n), pct(n / total if total else 0)] for s, n in sorted(result.splits.items())],
    )

    lines += ["## By source", ""]
    src_counts = Counter(t.task_id.split("/", 1)[0] for t in tasks)
    licences = {t.task_id.split("/", 1)[0]: t.source.license for t in tasks}
    redistributable = {t.task_id.split("/", 1)[0]: t.source.redistributable for t in tasks}
    lines += table(
        ["Source", "Tasks", "Licence (as recorded)", "Prompt text publishable"],
        [
            [s, str(n), licences.get(s, "?"), "yes" if redistributable.get(s) else "NO — ids only"]
            for s, n in sorted(src_counts.items())
        ],
    )

    lines += ["## Build losses", ""]
    loss_rows = [["duplicates removed", str(result.duplicates_dropped)]]
    loss_rows += [[f"balance trim: {d}", str(n)] for d, n in sorted(result.balance_removed.items())]
    for name, stats in sorted(result.per_source.items()):
        for reason, n in sorted(stats.reasons.items()):
            loss_rows.append([f"{name}: {reason}", str(n)])
    lines += table(["Reason", "Rows"], loss_rows)

    if result.unavailable:
        lines += ["## Unavailable sources", ""]
        lines += table(
            ["Source", "Why"], [[k, v] for k, v in sorted(result.unavailable.items())]
        )

    lines += [
        "Licences marked `CHECK-AT-COLLECTION` are unverified placeholders and block Stage 7.",
        "",
    ]
    return "\n".join(lines)
