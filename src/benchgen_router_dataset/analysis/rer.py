"""Relative Error Reduction, exactly as Trinity Appendix A.6 defines it.

The paper's `E(D, M)` is the accuracy of agent `M` on a whole **dataset** `D`, not on a single
question:

    Z(C, M')  = (1/|C|) Σ_{D ∈ C} max_{M ∈ M'} E(D, M)      # eq. 13, "combination performance"
    S*(C, M') = max_{M ∈ M'} (1/|C|) Σ_{D ∈ C} E(D, M)      # eq. 13, best single agent
    RER       = (Z - S*) / (1 - S*)                          # eq. 14

This is a different quantity from the per-question oracle the paper plots as "Per-Question-Best"
in Figure 3. Per-question max is always at least as large, usually much larger, so reporting a
per-question number as "RER (Appendix A.6)" overstates routing headroom. Both are computed and
reported separately; see `analysis/metrics.py`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ..schemas import RewardRecord


@dataclass(frozen=True)
class AccuracyMatrix:
    """`E(D, M)` for every dataset × agent pair, plus how many tasks backed each cell."""

    datasets: tuple[str, ...]
    agents: tuple[str, ...]
    values: dict[tuple[str, str], float]
    counts: dict[str, int]

    def E(self, dataset: str, agent: str) -> float:
        return self.values[(dataset, agent)]

    def is_complete(self) -> bool:
        return all((d, a) in self.values for d in self.datasets for a in self.agents)


def dataset_of(task_id: str) -> str:
    """`{source}/{split}/{index}` — the source name is the paper's `D`."""
    return task_id.split("/", 1)[0]


def build_matrix(
    records: Sequence[RewardRecord], group: Callable[[str], str] = dataset_of
) -> AccuracyMatrix:
    if not records:
        raise ValueError("no reward records")
    agents = tuple(records[0].agent_order)

    totals: dict[tuple[str, str], float] = {}
    counts: dict[str, int] = {}
    for rec in records:
        if tuple(rec.agent_order) != agents:
            raise ValueError(f"{rec.task_id}: agent_order differs from the first record")
        key = group(rec.task_id)
        counts[key] = counts.get(key, 0) + 1
        for agent, reward in zip(agents, rec.mean_reward, strict=True):
            totals[(key, agent)] = totals.get((key, agent), 0.0) + reward

    datasets = tuple(sorted(counts))
    values = {
        (d, a): totals.get((d, a), 0.0) / counts[d] for d in datasets for a in agents
    }
    return AccuracyMatrix(datasets=datasets, agents=agents, values=values, counts=counts)


def combination_performance(
    matrix: AccuracyMatrix, datasets: Sequence[str], agents: Sequence[str]
) -> float:
    """`Z`: route each dataset to its own best agent, then average over datasets."""
    if not datasets or not agents:
        return 0.0
    return sum(max(matrix.E(d, a) for a in agents) for d in datasets) / len(datasets)


def best_single(
    matrix: AccuracyMatrix, datasets: Sequence[str], agents: Sequence[str]
) -> tuple[str, float]:
    """`S*` and the agent that achieves it: one fixed agent for every dataset."""
    if not datasets or not agents:
        return "", 0.0
    scores = {
        a: sum(matrix.E(d, a) for d in datasets) / len(datasets) for a in agents
    }
    winner = max(scores, key=lambda a: (scores[a], a))
    return winner, scores[winner]


def relative_error_reduction(oracle: float, best_single_score: float) -> float:
    """Eq. 14. Zero when the best fixed agent is already perfect — no error left to remove."""
    denominator = 1.0 - best_single_score
    if denominator <= 1e-12:
        return 0.0
    return (oracle - best_single_score) / denominator


def rer(matrix: AccuracyMatrix, datasets: Sequence[str], agents: Sequence[str]) -> float:
    z = combination_performance(matrix, datasets, agents)
    _, s_star = best_single(matrix, datasets, agents)
    return relative_error_reduction(z, s_star)
