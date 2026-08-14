"""Baselines. A routing number without its baselines invites exactly the Fugu-Lite mistake."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean

from ..schemas import RewardRecord
from ..schemas.rewards import TIE_ABS_TOL


@dataclass
class Baselines:
    n_tasks: int
    agent_order: list[str]
    single_agent: dict[str, float]
    best_fixed_agent: str
    best_fixed: float
    uniform_random: float
    oracle: float

    @property
    def headroom(self) -> float:
        return self.oracle - self.best_fixed


def compute_baselines(records: list[RewardRecord]) -> Baselines:
    if not records:
        raise ValueError("no reward records")
    agent_order = records[0].agent_order
    per_agent = {
        agent: mean(rec.mean_reward[i] for rec in records)
        for i, agent in enumerate(agent_order)
    }
    best_agent = max(per_agent, key=lambda a: per_agent[a])
    return Baselines(
        n_tasks=len(records),
        agent_order=agent_order,
        single_agent=per_agent,
        best_fixed_agent=best_agent,
        best_fixed=per_agent[best_agent],
        # Uniform random picks each agent equally often, so its expectation is the plain mean.
        uniform_random=mean(per_agent.values()),
        oracle=mean(max(rec.mean_reward) for rec in records),
    )


def route_accuracy(records: list[RewardRecord], choices: dict[str, str]) -> float:
    """Tie-aware: a route is correct when its reward equals the per-task maximum.

    Comparing against `argmax`'s first index instead makes every tied task look like a miss,
    which is how Fugu-Lite under-reported its own router.
    """
    if not records:
        return 0.0
    hits = 0
    counted = 0
    for rec in records:
        chosen = choices.get(rec.task_id)
        if chosen is None or chosen not in rec.agent_order:
            continue
        counted += 1
        got = rec.mean_reward[rec.agent_order.index(chosen)]
        if math.isclose(got, max(rec.mean_reward), abs_tol=TIE_ABS_TOL):
            hits += 1
    return hits / counted if counted else 0.0


def routed_reward(records: list[RewardRecord], choices: dict[str, str]) -> float:
    """Mean reward actually obtained by a routing policy — the number that matters in practice."""
    if not records:
        return 0.0
    scored = [
        rec.mean_reward[rec.agent_order.index(choices[rec.task_id])]
        for rec in records
        if choices.get(rec.task_id) in rec.agent_order
    ]
    return mean(scored) if scored else 0.0
