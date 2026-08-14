"""Reward collection: concurrent, resumable, and honest about empty responses."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path

from ..graders import grade
from ..io_jsonl import append_jsonl
from ..providers.base import ChatResult
from ..schemas import AgentCard, AgentPool, CallRecord, RewardRecord, Role, Task
from .protocol import build_messages
from .resume import inspect

ChatFn = Callable[[AgentCard, list[dict[str, str]]], Awaitable[ChatResult]]

# Providers count reasoning and special tokens differently, so a few tokens over the cap is
# accounting noise. The failures this exists to catch overshot by 3.5x to 18x.
PROTOCOL_TOKEN_TOLERANCE = 1.05


class CollectionAborted(RuntimeError):
    """Out of credit or unauthorized. A row written now would record a billing failure as a
    wrong answer, and resume would then skip that task forever."""


class ProtocolViolation(CollectionAborted):
    """An upstream provider ignored max_tokens. On OpenRouter this varies per call, so a
    pre-flight probe cannot rule it out: E(D, M) stops being comparable and cost is unbounded."""


@dataclass
class CollectionProgress:
    tasks_total: int = 0
    tasks_done: int = 0
    tasks_skipped: int = 0
    calls_made: int = 0
    errors: int = 0
    empties: int = 0
    cost_usd: float = 0.0
    per_agent_empty: dict[str, int] = field(default_factory=dict)

    def note(self, call: CallRecord) -> None:
        self.calls_made += 1
        if call.error:
            self.errors += 1
        if call.empty:
            self.empties += 1
            self.per_agent_empty[call.agent] = self.per_agent_empty.get(call.agent, 0) + 1
        if call.cost_usd:
            self.cost_usd += call.cost_usd


def to_call_record(
    agent: AgentCard, repetition: int, task: Task, result: ChatResult,
    role: Role | None = None, store_raw: bool = True,
) -> CallRecord:
    graded = grade(task.grader, result.text, task.answer)
    return CallRecord(
        agent=agent.id,
        repetition=repetition,
        role=role.id if role else None,
        raw_response=result.text if store_raw else None,
        extracted=graded.extracted,
        correct=graded.correct,
        empty=result.empty,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        cost_usd=result.cost_usd,
        latency_ms=result.latency_ms,
        generation_id=result.generation_id,
        error=result.error,
    )


async def collect_rewards(
    tasks: list[Task],
    pool: AgentPool,
    chat: ChatFn,
    out_path: Path,
    repetitions: int = 3,
    concurrency: int = 8,
    store_raw: bool = True,
    resume: bool = True,
    on_progress: Callable[[CollectionProgress], None] | None = None,
) -> CollectionProgress:
    agents = pool.active
    agent_order = pool.agent_order
    progress = CollectionProgress(tasks_total=len(tasks))

    completed: set[str] = set()
    if resume:
        completed = inspect(out_path, pool.pool_version, agent_order, repetitions).completed
    elif out_path.exists():
        raise FileExistsError(f"{out_path} exists; pass resume=True or choose another path")

    pending = [t for t in tasks if t.task_id not in completed]
    progress.tasks_skipped = len(tasks) - len(pending)

    semaphore = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()

    async def one_call(agent: AgentCard, repetition: int, task: Task) -> CallRecord:
        async with semaphore:
            result = await chat(agent, build_messages(task))
        if result.fatal:
            raise CollectionAborted(f"{agent.id} ({agent.slug}): {result.error}")
        cap = pool.protocol.max_tokens
        if result.completion_tokens is not None and result.completion_tokens > cap * (
            PROTOCOL_TOKEN_TOLERANCE
        ):
            raise ProtocolViolation(
                f"{agent.id} ({agent.slug}) returned {result.completion_tokens} completion "
                f"tokens against a cap of {cap}"
            )
        return to_call_record(agent, repetition, task, result, store_raw=store_raw)

    async def one_task(task: Task) -> None:
        calls = await asyncio.gather(
            *(
                one_call(agent, rep, task)
                for agent in agents
                for rep in range(repetitions)
            )
        )
        record = RewardRecord.from_calls(
            task_id=task.task_id,
            pool_version=pool.pool_version,
            agent_order=agent_order,
            repetitions=repetitions,
            calls=list(calls),
        )
        async with write_lock:
            # Written per task so a crash never loses completed work.
            append_jsonl(out_path, record)
            for call in calls:
                progress.note(call)
            progress.tasks_done += 1
            if on_progress:
                on_progress(progress)

    for task in pending:
        await one_task(task)

    return progress
