"""Dataset card for the task pool (no rewards attached — see publish/card.py for that half)."""

from __future__ import annotations

from collections import Counter

from ..schemas import Task

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
  - 1K<n<10K
configs:
  - config_name: default
    data_files: tasks.jsonl
---
"""


def _breakdown(counter: Counter, label: str) -> str:
    parts = ", ".join(
        f"{k} {v} ({v / counter.total() * 100:.1f}%)" for k, v in sorted(counter.items())
    )
    return f"| {label} | {parts} |"


def _source_table(tasks: list[Task]) -> str:
    by_source: dict[str, dict] = {}
    for t in tasks:
        entry = by_source.setdefault(
            t.source.dataset,
            {"n": 0, "domain": t.domain, "license": t.source.license},
        )
        entry["n"] += 1
    head = "| Source | Count | Domain | Licence |\n| --- | --- | --- | --- |\n"
    rows = "".join(
        f"| `{name}` | {info['n']} | {info['domain']} | {info['license']} |\n"
        for name, info in sorted(by_source.items(), key=lambda kv: -kv[1]["n"])
    )
    return head + rows


def build_tasks_card(tasks: list[Task], title: str, rewards_repo: str, n_labelled: int) -> str:
    n = len(tasks)
    by_domain = Counter(t.domain for t in tasks)
    by_difficulty = Counter(t.difficulty for t in tasks)
    by_split = Counter(t.split for t in tasks)
    graders = Counter(t.grader for t in tasks)
    labelled = n_labelled

    return f"""{FRONT_MATTER}
# {title}

The **prompt pool** a BenchGen router head trains on: {n} normalized questions drawn from
public benchmark sources, balanced across domain and difficulty with a fixed seed, so results
reproduce. This is the *task* half of a paired release — the *reward* half (how well each pool
agent actually scored on a subset of these) is published separately at
[`{rewards_repo}`](https://huggingface.co/datasets/{rewards_repo}).

Only `task_id` is the join key between the two datasets — load both and match on it before
embedding anything.

## Contents

| Facet | Breakdown |
| --- | --- |
{_breakdown(by_domain, "Domain")}
{_breakdown(by_difficulty, "Difficulty")}
{_breakdown(by_split, "Split")}

## Sources

{_source_table(tasks)}

## Schema

Each row is one task:

```json
{{
  "task_id": "mmlu_pro/validation/000042",
  "prompt": "Question: ...\\n\\nA. ...\\nB. ...\\n\\nAnswer with the letter only.",
  "answer": "C",
  "answer_type": "multiple_choice",
  "grader": "exact_match_ci",
  "domain": "knowledge",
  "difficulty": "hard",
  "split": "train",
  "source": {{
    "dataset": "TIGER-Lab/MMLU-Pro",
    "split": "validation",
    "index": 42,
    "license": "MIT",
    "subject": "philosophy"
  }}
}}
```

| Field | Notes |
| --- | --- |
| `task_id` | `{{source}}/{{split}}/{{index}}` — stable, human-readable, sortable |
| `answer_type` | `multiple_choice` / `numeric` / `short_text` / `expression` |
| `grader` | Names the scoring function for this task; grading logic never lives in the data |
| `domain` | `math` / `knowledge` / `reasoning` / `science` / `code` / `turkish` |
| `difficulty` | `easy` / `medium` / `hard` |
| `split` | `train` / `validation` / `test` — assigned once with a fixed seed, never reassigned |
| `source` | Full provenance (dataset, upstream split/index, licence) |

Graders in this release: {', '.join(f'`{g}` ({c})' for g, c in sorted(graders.items()))}.

## Reward labels

Only **{labelled}** of these {n} tasks currently carry a measured reward row in
[`{rewards_repo}`](https://huggingface.co/datasets/{rewards_repo}) — the rest of the pool
exists so future collection rounds (a wider agent pool, or a bigger budget) can reuse the same
balanced task set instead of rebuilding it from scratch.

## Intended use

Training and evaluating LLM routers/coordinators — pairing a query with which pool agent should
answer it. **Not** a general capability benchmark: the task mix is selected for domain/difficulty
balance and disagreement potential between agents, so scores here are not comparable to standard
leaderboards.

## Limitations, stated plainly

- **Licence audit is per-source, not per-row.** `source.license` is recorded as observed at
  collection time; verify the licence for any source you redistribute further, particularly
  share-alike ones (e.g. ARC-Challenge).
- **Difficulty is source-derived where available, else assigned by rule** — it is not a
  human-calibrated difficulty score.
- **Domain coverage is deliberately narrow** (math / knowledge / reasoning), not representative
  of general use.
- Splits were fixed once at collection time and are never reassigned, even as more reward labels
  are added in later rounds.
"""
