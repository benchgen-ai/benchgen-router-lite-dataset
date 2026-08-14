# Gate 3 (pilot health): dataset health

## Provenance

| Input | SHA-256 (first 16) | Rows |
| --- | --- | --- |
| `rewards.v1-pilot.jsonl` | `3bc8b12c66dbb96e` | 60 |

| Setting | Value |
| --- | --- |
| tasks scored | 60 |
| agents | 5 |

## Decision

**STOP**

| Metric | Value | Continue at | Stop at | Verdict |
| --- | --- | --- | --- | --- |
| `RER` per dataset (Trinity A.6 eq. 14) | 0.0485 | ≥ 0.10 | ≤ 0.03 | WARN |
| `RER` per question (vs Per-Question-Best, Fig. 3) | 0.1731 | ≥ 0.15 | ≤ 0.08 | PASS |
| Datasets in the reward matrix | 8.0000 | ≥ 3.00 | ≤ 1.00 | PASS |
| Tasks where all agents score equal | 0.4333 | ≤ 0.45 | ≥ 0.65 | PASS |
| Tasks with a unique best agent | 0.1167 | ≥ 0.25 | ≤ 0.15 | FAIL |
| Weakest agent's uniquely-best share, x pool size | 0.0000 | ≥ 0.30 | ≤ 0.10 | FAIL |
| Worst agent's empty-response rate | 0.1556 | ≤ 0.02 | ≥ 0.10 | FAIL |
| Worst agent's call-error rate | 0.0000 | ≤ 0.02 | ≥ 0.10 | PASS |

## Headline numbers

| Quantity | Value | Meaning |
| --- | --- | --- |
| `S*` best fixed agent (per dataset, A.6) | 0.7548 | frontier_a |
| `Z` combination performance (A.6 eq. 13) | 0.7667 | each dataset routed to its own best agent |
| **`RER` (A.6 eq. 14)** | 0.0485 | the paper's pool-selection criterion |
| Per-question-best oracle (Fig. 3) | 0.7611 | upper bound, not deployable |
| `RER` per question | 0.1731 | same formula at question level; always the larger number |
| all-agents-equal tasks | 43.3% | no routing signal on these |
| unique-winner tasks | 11.7% | where routing can actually pay |

## `E(D, M)` — accuracy per dataset per agent (A.6 eq. 13)

| Dataset | Tasks | frontier_a | frontier_b | frontier_c | open_mid | open_cheap_reasoning | Best |
| --- | --- | --- | --- | --- | --- | --- | --- |
| aime2025_i | 1 | 1.000 | 1.000 | 0.667 | 0.000 | 1.000 | frontier_a |
| aime2025_ii | 1 | 0.333 | 0.000 | 0.000 | 0.000 | 0.000 | frontier_a |
| arc_challenge | 6 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | frontier_a |
| gsm8k | 4 | 1.000 | 1.000 | 0.583 | 0.583 | 1.000 | frontier_a |
| math500 | 14 | 0.810 | 0.857 | 0.762 | 0.262 | 0.810 | frontier_b |
| mmlu | 10 | 1.000 | 0.967 | 0.533 | 0.300 | 1.000 | frontier_a |
| mmlu_pro | 10 | 0.800 | 0.800 | 0.300 | 0.400 | 0.700 | frontier_a |
| rlpr | 14 | 0.095 | 0.143 | 0.048 | 0.000 | 0.119 | frontier_b |

## Per agent

| Agent | Mean reward | Uniquely best | Empty | Error | Median latency (ms) |
| --- | --- | --- | --- | --- | --- |
| frontier_a | 0.7000 | 3.3% | 0.0% | 0.0% | 6119 |
| frontier_b | 0.7111 | 5.0% | 5.0% | 0.0% | 4547 |
| frontier_c | 0.4778 | 0.0% | 0.0% | 0.0% | 1076 |
| open_mid | 0.3167 | 0.0% | 0.0% | 0.0% | 1759 |
| open_cheap_reasoning | 0.6833 | 3.3% | 15.6% | 0.0% | 1934 |

Total collection cost: **$0.0826**

## If the gate fails

Do not collect more data with the same design. Re-running the same pilot larger
will not help. Change one of:

- harder tasks (raise the MATH500 / GPQA-Diamond share)
- more heterogeneous agents (add a code specialist or a weak-but-cheap model)
- cost or latency weighting in the reward so ties break on price
- a different domain mix
