"""Reward-matrix records. Ties are represented explicitly, never collapsed by argmax."""

from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict, Field, model_validator

TIE_ABS_TOL = 1e-9


def best_indices(rewards: list[float], abs_tol: float = TIE_ABS_TOL) -> list[int]:
    """All positions within `abs_tol` of the maximum. A list, so ties survive."""
    if not rewards:
        return []
    top = max(rewards)
    return [i for i, r in enumerate(rewards) if math.isclose(r, top, abs_tol=abs_tol)]


class CallRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent: str
    repetition: int = Field(ge=0)
    role: str | None = None
    raw_response: str | None = None
    extracted: str | None = None
    correct: bool = False
    empty: bool = Field(
        default=False,
        description="HTTP 200 with no content. A distinct failure from `error`, and invisible "
        "if you only check `error`.",
    )
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost_usd: float | None = None
    latency_ms: int | None = None
    generation_id: str | None = None
    error: str | None = None

    @property
    def usable(self) -> bool:
        return self.error is None and not self.empty


class RewardRecord(BaseModel):
    """One record per task; every agent's outcome on one line makes tie analysis a single pass."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    pool_version: str
    repetitions: int = Field(ge=1)
    agent_order: list[str]
    mean_reward: list[float]
    best_agents: list[str] = Field(default_factory=list)
    is_tie: bool = False
    calls: list[CallRecord] = Field(default_factory=list)

    @model_validator(mode="after")
    def _recompute_bests(self) -> RewardRecord:
        if len(self.mean_reward) != len(self.agent_order):
            raise ValueError("mean_reward must be positionally aligned with agent_order")
        if len(set(self.agent_order)) != len(self.agent_order):
            raise ValueError("agent_order must not repeat an agent")
        winners = [self.agent_order[i] for i in best_indices(self.mean_reward)]
        # Derived, never trusted from the file: a stale best_agents silently corrupts the gate.
        self.best_agents = winners
        self.is_tie = len(winners) > 1
        return self

    @classmethod
    def from_calls(
        cls, task_id: str, pool_version: str, agent_order: list[str],
        repetitions: int, calls: list[CallRecord],
    ) -> RewardRecord:
        means: list[float] = []
        for agent in agent_order:
            got = [c for c in calls if c.agent == agent]
            means.append(sum(1.0 for c in got if c.correct) / repetitions if got else 0.0)
        return cls(
            task_id=task_id,
            pool_version=pool_version,
            repetitions=repetitions,
            agent_order=agent_order,
            mean_reward=means,
            calls=calls,
        )


class RoleRewardRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    pool_version: str
    agent: str
    role: str
    repetitions: int = Field(ge=1)
    mean_reward: float = 0.0
    verifier_judgment: str | None = None
    verifier_correct: bool | None = None
    calls: list[CallRecord] = Field(default_factory=list)
