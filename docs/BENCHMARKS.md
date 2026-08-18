# Benchmark results

Two questions, two benchmarks. Does the router actually route, or collapse to one agent, on data
from the exact distribution it trained on? And is a routed answer worth it: same accuracy, lower
cost, or is that a story that only holds up until you check the numbers?

- [Router Fidelity Benchmark](#router-fidelity-benchmark): in-distribution, does routing work at all
- [Router vs. a single frontier model](#router-vs-a-single-frontier-model): is it worth it
- [TR Benchmark](#tr-benchmark-out-of-distribution): out-of-distribution stress test

> Every result below is one run, against this one pool, on this one dataset. Run it against your
> own trained head, your own pool, or your own tasks on
> **[benchgen.com/benchmarks/platform/router-fidelity-benchmark](https://benchgen.com/benchmarks/platform/router-fidelity-benchmark)**,
> no eval harness required.

## Router Fidelity Benchmark

**[benchgen.com/benchmarks/platform/router-fidelity-benchmark](https://benchgen.com/benchmarks/platform/router-fidelity-benchmark)**

46 English math, knowledge, and reasoning tasks (MATH500, MMLU, MMLU-Pro, ARC-Challenge, GSM8K,
AIME2025) drawn from the exact same distribution `benchgen/router-pilot` /
`benchgen/router-pilot-tasks` trains a router head on, joined by `task_id`. Model connection is
env-var-first (`RPE_*` / `MODEL_*`, no `model.py`), scoring is deterministic and grader-aware per
task (`mcq_letter` / `numeric_match` / `math_expression`), no LLM-as-judge. If the evaluated
model is a router, the results page adds a per-question routing report: which pool member (and
the real underlying model, e.g. `anthropic/claude-sonnet-5`) answered each question, plus the
overall routing distribution.

### Why it exists

A router head embeds each query with a frozen backbone and picks a pool member based on where
that embedding falls relative to what it saw during training. Evaluate it on a wildly different
distribution (a different language, a different domain) and the embedding lands somewhere the
classifier never learned to discriminate, its pick becomes arbitrary rather than a genuine
routing decision, and it tends to collapse to a single agent. This benchmark's tasks come from
the exact same sources the router trained on, so a run here is a real fidelity check: does the
routing decision hold up on the data it was actually built for?

### Result: `head-Qwen3-1.7B-e99b449fde`

> **Pool update:** `frontier_a` has moved on from data-collection time. The reward matrix this
> repo publishes, and the TR Benchmark result below, were both collected against
> `openai/gpt-oss-120b`. This result routes to the pool's current `frontier_a`,
> `anthropic/claude-sonnet-5`. The `agent_order` identifiers are stable; the real model each one
> resolves to is a serving-time mapping, independent of the training data.

| Metric | Value |
| --- | ---: |
| Accuracy | **82.6%** (38/46) |
| Total cost | $0.0286 |
| Avg cost per task | $0.000622 |
| Total tokens | 14,483 |

| Difficulty | Score |
| --- | ---: |
| Easy | 88.9% (8/9) |
| Medium | 94.7% (18/19) |
| Hard | 66.7% (12/18) |

| Domain | Score |
| --- | ---: |
| Knowledge | 95.0% (19/20) |
| Math | 65.0% (13/20) |
| Reasoning | 100.0% (6/6) |

| Routed to | Real model | Questions | Share |
| --- | --- | ---: | ---: |
| `frontier_a` | `anthropic/claude-sonnet-5` | 36 | 78% |
| `frontier_b` | `deepseek/deepseek-v4-flash` | 7 | 15% |
| `open_cheap_reasoning` | `inclusionai/ling-3.0-flash` | 3 | 7% |

Three different agents actually receiving traffic, not one, is the concrete difference
in-distribution evaluation makes, and what a router doing its job is supposed to look like, in
contrast to the single-agent collapse [TR Benchmark](#tr-benchmark-out-of-distribution) below
surfaces on out-of-distribution data.

## Router vs. a single frontier model

The Cost KPI on Router Fidelity Benchmark exists precisely so a router's cost sits on the same
leaderboard as a single fixed model and can actually be compared. The baseline here is not an
arbitrary outside model: `anthropic/claude-sonnet-5` is `frontier_a`, the exact model the router
above picked 36 of 46 times. Calling it directly for every one of the same 46 questions, no
routing at all, answers a precise question: what does the pool's own strongest model cost when it
answers everything, against what it costs when the router only reaches for it when a question
actually calls for it?

| | BenchGen Router Lite | `claude-sonnet-5` alone |
| --- | ---: | ---: |
| Accuracy | **82.6%** (38/46) | 80.4% (37/46) |
| Total cost | **$0.0286** | $0.0889 |
| Avg cost per task | **$0.000622** | $0.001932 |
| Total tokens | 14,483 | 16,021 |

Both numbers move the same direction: the router is more accurate and **roughly 3.1x cheaper**
($0.0889 against $0.0286 in total cost) on the identical 46 questions. Total tokens are close,
14,483 against 16,021, only about 10% fewer for the router, so the savings are not coming from
shorter answers. They come from where the questions get sent: three of the five pool models cost
a fraction of a frontier model per token, and the router only reaches for `frontier_a` when a
question actually calls for it, 36 of 46 times here, not on every single one.

### By difficulty

| Difficulty | Router | `claude-sonnet-5` alone |
| --- | ---: | ---: |
| Easy | 88.9% (8/9) | 88.9% (8/9) |
| Medium | 94.7% (18/19) | 100.0% (19/19) |
| Hard | 66.7% (12/18) | 55.6% (10/18) |

### By domain

| Domain | Router | `claude-sonnet-5` alone |
| --- | ---: | ---: |
| Knowledge | 95.0% (19/20) | 90.0% (18/20) |
| Math | 65.0% (13/20) | 65.0% (13/20) |
| Reasoning | 100.0% (6/6) | 100.0% (6/6) |

The split is where it gets interesting rather than uniform: `claude-sonnet-5` alone wins medium
by one question, the router wins hard by two. The router leads knowledge, and the two are exactly
tied on math and reasoning. These are single runs of 6 to 20 questions per bucket, not a claim
that routing always wins every slice. The honest read: the router matched or beat a frontier
model used alone across every domain and most difficulty bands, at under a third of the cost.

## TR Benchmark (out-of-distribution)

70 Turkish multiple-choice general-knowledge and logic questions, chosen deliberately because
they look nothing like the training data (English math, knowledge, and reasoning sources). Run
against the pool as it stood at data-collection time (`frontier_a` = `openai/gpt-oss-120b`, since
moved to `anthropic/claude-sonnet-5`, see the pool update note above), the router scored 72.9%
overall, but **69 of 70 questions were routed to the same single pool member**, `open_mid`
(`mistralai/mistral-nemo`), the deliberately weak, cheap agent in the pool. That single-agent
collapse on unfamiliar data is exactly why Router Fidelity Benchmark above exists: it isolates
whether the routing decision itself works, separate from whether it holds up outside the exact
distribution it was trained on. Full walkthrough, with screenshots of both benchmarks:
[benchgen.com/docs/guides/router-head/benchgen-router-lite](https://benchgen.com/docs/guides/router-head/benchgen-router-lite).

---

Want to build and benchmark your own router instead of reading about this one? Collection,
gating, training, and benchmarking all run on **[benchgen.com](https://benchgen.com)**, against
your own task pool.
