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
  - n<1K
configs:
  - config_name: rewards
    data_files: rewards.jsonl
---

# BenchGen Router Pilot: per-question agent rewards

A **routing** dataset: for each question, how every agent in a fixed pool actually performed.
It is the input a model-selection policy trains on — not a question-answering dataset.

Each row is one task. `mean_reward[i]` is the accuracy of agent `agent_order[i]` over
3 independent attempts. Ties are preserved explicitly rather than collapsed by `argmax`,
because on easy questions several agents are genuinely equal and pretending otherwise teaches a
router a coin flip.

## Contents

46 tasks x 5 agents x 3 repetitions = 690 graded calls.

| Facet | Breakdown |
| --- | --- |
| Source | HuggingFaceH4/MATH-500 14, TIGER-Lab/MMLU-Pro 10, allenai/ai2_arc 6, cais/mmlu 10, openai/gsm8k 4, opencompass/AIME2025 2 |
| Domain | knowledge 20, math 20, reasoning 6 |
| Difficulty | easy 9, hard 18, medium 19 |

## The pool

| # | Agent | Model | Mode | $/1M out |
| --- | --- | --- | --- | --- |
| 0 | `frontier_a` | `openai/gpt-oss-120b` | reasoning | 0.17 |
| 1 | `frontier_b` | `deepseek/deepseek-v4-flash-0731` | reasoning | 0.18 |
| 2 | `frontier_c` | `google/gemma-3-27b-it` | direct | 0.45 |
| 4 | `open_mid` | `mistralai/mistral-nemo` | direct | 0.03 |
| 7 | `open_cheap_reasoning` | `inclusionai/ling-3.0-flash` | reasoning | 0.063 |


## Collected but excluded

These were collected and are not in this release, because a label we cannot trust is worse than no label.

| Source | Why |
| --- | --- |
| `rlpr` | Answers are free-form prose and tables. Our string-match grader scores correct answers as wrong, so these labels are not trustworthy. |

## What is deliberately not here

- **No question text.** Upstream licences differ per source: ARC-Challenge is share-alike, and
  MMLU-Pro's licence was never verified. `task_id` identifies the upstream row so prompts can be
  rehydrated from the original datasets.
- **No raw completions.** Provider terms on republishing model outputs were not cleared. The
  routing signal — correctness, ties, empties, errors — survives without them.

## Limitations, stated plainly

- **High tie rate (43/46 tasks).** Most agents solve most of these questions, so the
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
`temperature` 0.1, `top_p` 1.0, minimal reasoning effort, 3 repetitions. Any response
exceeding the token cap aborts collection rather than being recorded, so no row mixes protocols.

## Grading

Answers are extracted by tagged answer, boxed LaTeX, labelled answer, then a last-number
fallback that strips markup first. An empty response is never scored correct.
