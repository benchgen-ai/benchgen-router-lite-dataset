"""Markdown report rendering.

Reports contain **no timestamps and no commit hashes** — they must reproduce byte for byte from
the committed data. Provenance is carried by SHA-256 digests of the exact input files instead.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

from .baselines import Baselines
from .health import REDESIGN_OPTIONS, GateOutcome
from .metrics import HealthMetrics

VERDICT_MARK = {"continue": "PASS", "warn": "WARN", "stop": "FAIL", "missing": "MISSING"}


def digest(path: Path) -> str:
    if not path.exists():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def header(title: str, inputs: dict[str, Path], extra: dict[str, str] | None = None) -> list[str]:
    lines = [
        f"# {title}",
        "",
        "## Provenance",
        "",
        "| Input | SHA-256 (first 16) | Rows |",
        "| --- | --- | --- |",
    ]
    for name, path in inputs.items():
        rows = sum(1 for _ in path.open(encoding="utf-8")) if path.exists() else 0
        lines.append(f"| `{name}` | `{digest(path)}` | {rows} |")
    if extra:
        lines += ["", "| Setting | Value |", "| --- | --- |"]
        lines += [f"| {k} | {v} |" for k, v in extra.items()]
    return lines + [""]


def table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return out + [""]


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def num(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def render_health(metrics: HealthMetrics, outcome: GateOutcome, inputs: dict[str, Path]) -> str:
    lines = header(
        f"{outcome.name}: dataset health",
        inputs,
        {"tasks scored": str(metrics.n_tasks), "agents": str(len(metrics.agent_order))},
    )

    lines += ["## Decision", "", f"**{outcome.decision}**", ""]
    lines += table(
        ["Metric", "Value", "Continue at", "Stop at", "Verdict"],
        [
            [
                r.label,
                num(r.value),
                ("≥ " if r.direction == "min" else "≤ ") + num(r.continue_at, 2),
                ("≤ " if r.direction == "min" else "≥ ") + num(r.stop_at, 2),
                VERDICT_MARK[r.verdict],
            ]
            for r in outcome.rows
        ],
    )

    lines += ["## Headline numbers", ""]
    lines += table(
        ["Quantity", "Value", "Meaning"],
        [
            [
                "`S*` best fixed agent (per dataset, A.6)",
                num(metrics.s_star_dataset),
                metrics.best_dataset_agent,
            ],
            [
                "`Z` combination performance (A.6 eq. 13)",
                num(metrics.z_dataset),
                "each dataset routed to its own best agent",
            ],
            [
                "**`RER` (A.6 eq. 14)**",
                num(metrics.rer_dataset),
                "the paper's pool-selection criterion",
            ],
            [
                "Per-question-best oracle (Fig. 3)",
                num(metrics.oracle_per_question),
                "upper bound, not deployable",
            ],
            [
                "`RER` per question",
                num(metrics.rer_per_question),
                "same formula at question level; always the larger number",
            ],
            ["all-agents-equal tasks", pct(metrics.all_equal_rate), "no routing signal on these"],
            [
                "unique-winner tasks",
                pct(metrics.unique_winner_rate),
                "where routing can actually pay",
            ],
        ],
    )

    lines += ["## `E(D, M)` — accuracy per dataset per agent (A.6 eq. 13)", ""]
    lines += table(
        ["Dataset", "Tasks", *metrics.matrix.agents, "Best"],
        [
            [
                dataset,
                str(metrics.matrix.counts[dataset]),
                *(num(metrics.matrix.E(dataset, a), 3) for a in metrics.matrix.agents),
                max(metrics.matrix.agents, key=lambda a: metrics.matrix.E(dataset, a)),
            ]
            for dataset in metrics.matrix.datasets
        ],
    )

    lines += ["## Per agent", ""]
    lines += table(
        ["Agent", "Mean reward", "Uniquely best", "Empty", "Error", "Median latency (ms)"],
        [
            [
                agent,
                num(metrics.per_agent_mean[i]),
                pct(metrics.unique_best_share[agent]),
                pct(metrics.empty_rate[agent]),
                pct(metrics.error_rate[agent]),
                f"{metrics.median_latency_ms.get(agent, 0):.0f}",
            ]
            for i, agent in enumerate(metrics.agent_order)
        ],
    )

    lines += [f"Total collection cost: **${metrics.total_cost_usd:.4f}**", ""]

    if outcome.decision == "STOP":
        lines += [
            "## If the gate fails",
            "",
            "Do not collect more data with the same design. Re-running the same pilot larger",
            "will not help. Change one of:",
            "",
        ]
        lines += [f"- {opt}" for opt in REDESIGN_OPTIONS]
        lines += [""]

    return "\n".join(lines)


def render_baselines(baselines: Baselines, split: str, inputs: dict[str, Path]) -> str:
    lines = header(
        "Baselines",
        inputs,
        {"split": split, "tasks": str(baselines.n_tasks)},
    )
    rows = [[a, num(s), "single fixed agent"] for a, s in sorted(baselines.single_agent.items())]
    rows += [
        ["best fixed agent", num(baselines.best_fixed), baselines.best_fixed_agent],
        ["uniform random", num(baselines.uniform_random), "pick any agent with equal probability"],
        ["oracle (per-question best)", num(baselines.oracle), "upper bound, not deployable"],
    ]
    lines += table(["Policy", "Mean reward", "Note"], rows)
    lines += [
        f"Routing headroom (`Z - S*`): **{num(baselines.headroom)}**. "
        "Any trained router must land inside this band to have been worth training.",
        "",
    ]
    return "\n".join(lines)
