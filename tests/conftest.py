"""Shared fixtures. Nothing here touches the network."""

from __future__ import annotations

import pytest

from benchgen_router_dataset.schemas import AgentCard, AgentPool, SourceRef, Task
from benchgen_router_dataset.schemas.rewards import CallRecord, RewardRecord


def make_task(index: int = 0, domain: str = "math", difficulty: str = "hard", **kw) -> Task:
    defaults = dict(
        task_id=f"fixture/test/{index:06d}",
        prompt=f"Problem: what is {index} + 1?",
        answer=str(index + 1),
        answer_type="numeric",
        grader="numeric_match",
        domain=domain,
        difficulty=difficulty,
        split="train",
        source=SourceRef(dataset="fixture", split="test", index=index, license="MIT"),
    )
    defaults.update(kw)
    return Task(**defaults)


def make_reward(task_id: str, means: list[float], agents: list[str]) -> RewardRecord:
    return RewardRecord(
        task_id=task_id,
        pool_version="v1",
        repetitions=3,
        agent_order=agents,
        mean_reward=means,
        calls=[
            CallRecord(agent=a, repetition=0, correct=m > 0.5, latency_ms=100)
            for a, m in zip(agents, means, strict=True)
        ],
    )


def make_rewards_over_datasets(
    agents: list[str], per_dataset: dict[str, list[list[float]]]
) -> list[RewardRecord]:
    """`{dataset: [mean_reward per task]}` -> reward records with realistic `{src}/{split}/{i}`."""
    out: list[RewardRecord] = []
    for dataset, rows in sorted(per_dataset.items()):
        for i, means in enumerate(rows):
            out.append(make_reward(f"{dataset}/test/{i:06d}", means, agents))
    return out


@pytest.fixture
def agents() -> list[str]:
    return ["fast_general", "reasoner", "coder", "open_local"]


@pytest.fixture
def pool(agents: list[str]) -> AgentPool:
    return AgentPool(
        pool_version="v1",
        agents=[
            AgentCard(
                index=i,
                id=agent_id,
                provider="fake",
                slug=f"fake/{agent_id}",
                description="fixture agent",
                verified_at="2026-01-01",
            )
            for i, agent_id in enumerate(agents)
        ],
    )
