# Baselines

## Provenance

| Input | SHA-256 (first 16) | Rows |
| --- | --- | --- |
| `rewards.v1-pilot.jsonl` | `3bc8b12c66dbb96e` | 60 |

| Setting | Value |
| --- | --- |
| split | test |
| tasks | 13 |

| Policy | Mean reward | Note |
| --- | --- | --- |
| frontier_a | 0.6667 | single fixed agent |
| frontier_b | 0.6410 | single fixed agent |
| frontier_c | 0.5641 | single fixed agent |
| open_cheap_reasoning | 0.6667 | single fixed agent |
| open_mid | 0.2308 | single fixed agent |
| best fixed agent | 0.6667 | frontier_a |
| uniform random | 0.5538 | pick any agent with equal probability |
| oracle (per-question best) | 0.7692 | upper bound, not deployable |

Routing headroom (`Z - S*`): **0.1026**. Any trained router must land inside this band to have been worth training.
