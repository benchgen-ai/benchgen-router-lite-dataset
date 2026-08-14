"""Health metrics.

Two different headroom numbers are reported, and conflating them is the easiest way to overstate
what routing can do here:

- `rer_dataset` — Trinity Appendix A.6, eq. 13-14, where `E(D, M)` is accuracy on a whole
  **dataset**. This is the criterion the paper uses to choose the agent pool and dataset mix.
- `rer_per_question` — the same formula against the paper's "Per-Question-Best" upper bound from
  Figure 3, where `E` is per **question**. Always the larger of the two.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean

from ..schemas import RewardRecord
from ..schemas.rewards import best_indices
from .rer import (
    AccuracyMatrix,
    best_single,
    build_matrix,
    combination_performance,
    relative_error_reduction,
)


@dataclass
class HealthMetrics:
    n_tasks: int
    agent_order: list[str]
    per_agent_mean: list[float]
    best_single_agent: str
    best_single_score: float
    oracle_per_question: float
    rer_per_question: float
    matrix: AccuracyMatrix
    z_dataset: float
    s_star_dataset: float
    best_dataset_agent: str
    rer_dataset: float
    all_equal_rate: float
    unique_winner_rate: float
    unique_best_share: dict[str, float]
    empty_rate: dict[str, float]
    error_rate: dict[str, float]
    total_cost_usd: float
    median_latency_ms: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, float]:
        """Flat view keyed the way `configs/gates.v1.json` refers to metrics."""
        n_agents = len(self.agent_order) or 1
        min_share = min(self.unique_best_share.values(), default=0.0)
        return {
            "rer_dataset": self.rer_dataset,
            "rer_per_question": self.rer_per_question,
            "all_equal_rate": self.all_equal_rate,
            "unique_winner_rate": self.unique_winner_rate,
            "min_unique_best_share": min_share,
            # Scaled by pool size: 1.0 means the weakest agent wins its uniform share, so one
            # threshold works for a 4-agent and a 7-agent pool alike.
            "min_unique_best_share_scaled": min_share * n_agents,
            "max_empty_rate": max(self.empty_rate.values(), default=0.0),
            "max_error_rate": max(self.error_rate.values(), default=0.0),
            "n_datasets": float(len(self.matrix.datasets)),
            "oracle_per_question": self.oracle_per_question,
            "best_single_score": self.best_single_score,
        }


def compute_health(records: list[RewardRecord]) -> HealthMetrics:
    if not records:
        raise ValueError("no reward records")

    agent_order = records[0].agent_order
    for rec in records:
        if rec.agent_order != agent_order:
            raise ValueError(f"{rec.task_id}: agent_order differs from the first record")

    n = len(records)
    per_agent = [mean(rec.mean_reward[i] for rec in records) for i in range(len(agent_order))]
    best_idx = max(range(len(agent_order)), key=lambda i: per_agent[i])
    oracle_q = mean(max(rec.mean_reward) for rec in records)

    matrix = build_matrix(records)
    z_dataset = combination_performance(matrix, matrix.datasets, matrix.agents)
    best_dataset_agent, s_star_dataset = best_single(matrix, matrix.datasets, matrix.agents)

    ties = sum(1 for rec in records if rec.is_tie)
    all_equal = sum(1 for rec in records if len(set(rec.mean_reward)) == 1)
    unique_wins = {a: 0 for a in agent_order}
    for rec in records:
        winners = best_indices(rec.mean_reward)
        if len(winners) == 1:
            unique_wins[agent_order[winners[0]]] += 1

    empties = {a: 0 for a in agent_order}
    errors = {a: 0 for a in agent_order}
    calls = {a: 0 for a in agent_order}
    latencies: dict[str, list[int]] = {a: [] for a in agent_order}
    cost = 0.0
    for rec in records:
        for call in rec.calls:
            if call.agent not in calls:
                continue
            calls[call.agent] += 1
            empties[call.agent] += int(call.empty)
            errors[call.agent] += int(call.error is not None)
            if call.latency_ms is not None:
                latencies[call.agent].append(call.latency_ms)
            cost += call.cost_usd or 0.0

    return HealthMetrics(
        n_tasks=n,
        agent_order=agent_order,
        per_agent_mean=per_agent,
        best_single_agent=agent_order[best_idx],
        best_single_score=per_agent[best_idx],
        oracle_per_question=oracle_q,
        rer_per_question=relative_error_reduction(oracle_q, per_agent[best_idx]),
        matrix=matrix,
        z_dataset=z_dataset,
        s_star_dataset=s_star_dataset,
        best_dataset_agent=best_dataset_agent,
        rer_dataset=relative_error_reduction(z_dataset, s_star_dataset),
        # "All agents equal" is the failure mode; a 2-way tie among 4 still carries signal.
        all_equal_rate=all_equal / n,
        unique_winner_rate=(n - ties) / n,
        unique_best_share={a: unique_wins[a] / n for a in agent_order},
        empty_rate={a: (empties[a] / calls[a] if calls[a] else 0.0) for a in agent_order},
        error_rate={a: (errors[a] / calls[a] if calls[a] else 0.0) for a in agent_order},
        total_cost_usd=cost,
        median_latency_ms={
            a: (sorted(v)[len(v) // 2] if v else 0.0) for a, v in latencies.items()
        },
    )
