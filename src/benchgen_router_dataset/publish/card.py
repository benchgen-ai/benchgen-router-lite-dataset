"""Dataset card. Every limitation found during collection is stated here, not omitted."""

from __future__ import annotations

from collections import Counter

from ..schemas import AgentPool
from .redact import PublicRewardRow

FRONT_MATTER = """---
license: other
license_name: mixed-upstream
license_link: https://github.com/tahsinsoyak/benchgen-router-dataset
task_categories:
  - question-answering
  - text-classification
language:
  - en
tags:
  - llm-routing
  - model-selection
  - orchestration
  - agent-coordination
size_categories:
  - n<1K
configs:
  - config_name: rewards
    data_files: rewards.jsonl
---
"""


def _agent_table(pool: AgentPool) -> str:
    head = "| # | Agent | Model | Mode | $/1M out |\n| --- | --- | --- | --- | --- |\n"
    rows = "".join(
        f"| {a.index} | `{a.id}` | `{a.slug}` | "
        f"{'reasoning' if a.is_reasoning_model else 'direct'} | "
        f"{a.price_per_1m_output_usd} |\n"
        for a in pool.active
    )
    return head + rows


def _distribution(rows: list[PublicRewardRow]) -> str:
    by_source = Counter(r.dataset for r in rows)
    by_domain = Counter(r.domain for r in rows)
    by_diff = Counter(r.difficulty for r in rows)
    lines = ["| Facet | Breakdown |", "| --- | --- |"]
    for label, counter in (
        ("Source", by_source),
        ("Domain", by_domain),
        ("Difficulty", by_diff),
    ):
        parts = ", ".join(f"{k} {v}" for k, v in sorted(counter.items()))
        lines.append(f"| {label} | {parts} |")
    return "\n".join(lines)


def _excluded_note(excluded: dict[str, str]) -> str:
    if not excluded:
        return ""
    rows = "".join(f"| `{name}` | {why} |\n" for name, why in sorted(excluded.items()))
    return (
        "\n## Collected but excluded\n\n"
        "These were collected and are not in this release, because a label we cannot trust is "
        "worse than no label.\n\n"
        "| Source | Why |\n| --- | --- |\n" + rows
    )


def build_card(
    rows: list[PublicRewardRow],
    pool: AgentPool,
    title: str,
    excluded: dict[str, str] | None = None,
) -> str:
    n = len(rows)
    ties = sum(1 for r in rows if r.is_tie)
    n_agents = len(pool.active)
    reps = rows[0].repetitions if rows else 0
    return f"""{FRONT_MATTER}
# {title}

A **routing** dataset: for each question, how every agent in a fixed pool actually performed.
It is the input a model-selection policy trains on — not a question-answering dataset.

Each row is one task. `mean_reward[i]` is the accuracy of agent `agent_order[i]` over
{reps} independent attempts. Ties are preserved explicitly rather than collapsed by `argmax`,
because on easy questions several agents are genuinely equal and pretending otherwise teaches a
router a coin flip.

## Contents

{n} tasks x {n_agents} agents x {reps} repetitions = {n * n_agents * reps} graded calls.

{_distribution(rows)}

## The pool

{_agent_table(pool)}
{_excluded_note(excluded or {})}
## What is deliberately not here

- **No question text.** Upstream licences differ per source: ARC-Challenge is share-alike, and
  MMLU-Pro's licence was never verified. `task_id` identifies the upstream row so prompts can be
  rehydrated from the original datasets.
- **No raw completions.** Provider terms on republishing model outputs were not cleared. The
  routing signal — correctness, ties, empties, errors — survives without them.

## Limitations, stated plainly

- **High tie rate ({ties}/{n} tasks).** Most agents solve most of these questions, so the
  per-question best agent is often arbitrary. A router trained on this alone learns little; the
  dataset is most useful as evidence that routing does not pay on saturated benchmarks.
- **One agent dominates.** `frontier_a` or `frontier_b` is the best agent on almost every
  dataset here, so the pool behaves as a ranking rather than a set of specialists. That is a
  property of the pool, and no amount of extra data changes it.
- **The pool was hand-picked, not selected by measurement.** Trinity's Appendix A.6 chooses
  agents and datasets jointly from a full screening matrix; that screening was skipped for cost.
- **No direct-vs-reasoning contrast.** The intended pair of one base model in both modes was
  dropped: every candidate `-thinking` slug ignored `max_tokens` (24k-75k tokens against a 4,096
  cap), which breaks protocol comparability.
- **No code domain.** Execution-based grading needs a sandbox that does not exist here.
- **Small.** This is a pilot, sized to a strict budget.

## Protocol

Identical for every agent, because `E(D, M)` is not comparable otherwise: `max_tokens` 4096,
`temperature` 0.1, `top_p` 1.0, minimal reasoning effort, {reps} repetitions. Any response
exceeding the token cap aborts collection rather than being recorded, so no row mixes protocols.

## Grading

Answers are extracted by tagged answer, boxed LaTeX, labelled answer, then a last-number
fallback that strips markup first. An empty response is never scored correct.
"""
