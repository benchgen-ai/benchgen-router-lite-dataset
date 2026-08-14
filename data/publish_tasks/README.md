---
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

# BenchGen Router Pilot: task pool

The **prompt pool** a BenchGen router head trains on: 1110 normalized questions drawn from
public benchmark sources, balanced across domain and difficulty with a fixed seed, so results
reproduce. This is the *task* half of a paired release — the *reward* half (how well each pool
agent actually scored on a subset of these) is published separately at
[`benchgen/router-pilot`](https://huggingface.co/datasets/benchgen/router-pilot).

Only `task_id` is the join key between the two datasets — load both and match on it before
embedding anything.

## Contents

| Facet | Breakdown |
| --- | --- |
| Domain | knowledge 370 (33.3%), math 370 (33.3%), reasoning 370 (33.3%) |
| Difficulty | easy 179 (16.1%), hard 427 (38.5%), medium 504 (45.4%) |
| Split | test 245 (22.1%), train 648 (58.4%), validation 217 (19.5%) |

## Sources

| Source | Count | Domain | Licence |
| --- | --- | --- | --- |
| `openbmb/RLPR-Train-Dataset` | 250 | reasoning | Apache-2.0 |
| `HuggingFaceH4/MATH-500` | 233 | math | MIT |
| `TIGER-Lab/MMLU-Pro` | 186 | knowledge | CHECK-AT-COLLECTION |
| `cais/mmlu` | 184 | knowledge | MIT |
| `allenai/ai2_arc` | 120 | reasoning | CC-BY-SA-4.0 |
| `openai/gsm8k` | 109 | math | MIT |
| `opencompass/AIME2025` | 28 | math | MIT |


## Schema

Each row is one task:

```json
{
  "task_id": "mmlu_pro/validation/000042",
  "prompt": "Question: ...\n\nA. ...\nB. ...\n\nAnswer with the letter only.",
  "answer": "C",
  "answer_type": "multiple_choice",
  "grader": "exact_match_ci",
  "domain": "knowledge",
  "difficulty": "hard",
  "split": "train",
  "source": {
    "dataset": "TIGER-Lab/MMLU-Pro",
    "split": "validation",
    "index": 42,
    "license": "MIT",
    "subject": "philosophy"
  }
}
```

| Field | Notes |
| --- | --- |
| `task_id` | `{source}/{split}/{index}` — stable, human-readable, sortable |
| `answer_type` | `multiple_choice` / `numeric` / `short_text` / `expression` |
| `grader` | Names the scoring function for this task; grading logic never lives in the data |
| `domain` | `math` / `knowledge` / `reasoning` / `science` / `code` / `turkish` |
| `difficulty` | `easy` / `medium` / `hard` |
| `split` | `train` / `validation` / `test` — assigned once with a fixed seed, never reassigned |
| `source` | Full provenance (dataset, upstream split/index, licence) |

Graders in this release: `exact_match_ci` (250), `math_expression` (233), `mcq_letter` (490), `numeric_match` (137).

## Reward labels

Only **46** of these 1110 tasks currently carry a measured reward row in
[`benchgen/router-pilot`](https://huggingface.co/datasets/benchgen/router-pilot) — the rest of the pool
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
