"""Publishable projection of the reward matrix.

Question text and raw completions are deliberately absent. Upstream licences differ per source
(ARC-Challenge is share-alike, MMLU-Pro's is unverified) and provider terms on republishing
completions were never cleared, so the safe release is rewards plus metadata. `task_id` lets a
consumer rehydrate the prompt from upstream themselves.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from ..schemas import AgentPool, RewardRecord, Task


class PublicRewardRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    dataset: str
    domain: str
    difficulty: str
    split: str
    task_group: str
    pool_version: str
    agent_order: list[str]
    mean_reward: list[float]
    best_agents: list[str]
    is_tie: bool
    repetitions: int
    n_correct: list[int] = Field(description="Correct calls per agent, positionally aligned.")
    n_empty: list[int] = Field(description="HTTP 200 with no content, per agent.")
    n_error: list[int] = Field(description="Failed calls per agent, after retries.")


def _counts(record: RewardRecord, predicate) -> list[int]:
    return [
        sum(1 for c in record.calls if c.agent == a and predicate(c))
        for a in record.agent_order
    ]


def to_public_row(record: RewardRecord, task: Task) -> PublicRewardRow:
    return PublicRewardRow(
        task_id=record.task_id,
        dataset=task.source.dataset,
        domain=task.domain,
        difficulty=task.difficulty,
        split=task.split,
        task_group=task.task_group,
        pool_version=record.pool_version,
        agent_order=record.agent_order,
        mean_reward=record.mean_reward,
        best_agents=record.best_agents,
        is_tie=record.is_tie,
        repetitions=record.repetitions,
        n_correct=_counts(record, lambda c: c.correct),
        n_empty=_counts(record, lambda c: c.empty),
        n_error=_counts(record, lambda c: bool(c.error)),
    )


def public_rows(records: list[RewardRecord], tasks: dict[str, Task]) -> list[PublicRewardRow]:
    missing = [r.task_id for r in records if r.task_id not in tasks]
    if missing:
        raise KeyError(f"{len(missing)} reward rows have no matching task, e.g. {missing[0]}")
    return [to_public_row(r, tasks[r.task_id]) for r in records]


def agent_manifest(pool: AgentPool) -> list[dict[str, object]]:
    """Which model produced each column. Without this the reward vectors are meaningless."""
    return [
        {
            "index": a.index,
            "agent_id": a.id,
            "slug": a.slug,
            "is_reasoning_model": a.is_reasoning_model,
            "context_window": a.context_window,
            "price_per_1m_prompt_usd": a.price_per_1m_prompt_usd,
            "price_per_1m_output_usd": a.price_per_1m_output_usd,
            "verified_at": a.verified_at,
        }
        for a in pool.active
    ]
