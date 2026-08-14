from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from conftest import make_task

from benchgen_router_dataset.collection import (
    CollectionAborted,
    ProtocolViolation,
    collect_rewards,
)
from benchgen_router_dataset.collection.resume import IncompatibleRewardsFile
from benchgen_router_dataset.io_jsonl import read_models
from benchgen_router_dataset.providers import OpenRouterClient
from benchgen_router_dataset.providers.base import ChatResult
from benchgen_router_dataset.schemas import AgentCard, AgentPool, CallProtocol, RewardRecord


async def test_keep_alive_padded_body_is_parsed_not_fatal(monkeypatch) -> None:
    """Slow non-streaming requests come back padded with `: OPENROUTER PROCESSING` lines."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    good = {"choices": [{"message": {"content": "Answer: 7"}, "finish_reason": "stop"}]}
    padded = ": OPENROUTER PROCESSING\n" * 2000 + json.dumps(good)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=padded)

    agent = AgentCard(index=0, id="a", provider="openrouter", slug="v/m", description="d")
    async with OpenRouterClient(transport=httpx.MockTransport(handler)) as client:
        result = await client.chat(agent, [{"role": "user", "content": "hi"}], CallProtocol())

    assert result.error is None
    assert result.text == "Answer: 7"


async def test_upstream_failure_sent_as_http_200_is_retried(monkeypatch) -> None:
    """OpenRouter reports provider timeouts as 200 with an error body, so status is not enough."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    bodies = [
        {"error": {"message": "Provider timed out after 5011ms", "code": 504}},
        {"choices": [{"message": {"content": "Answer: 7"}, "finish_reason": "stop"}], "usage": {}},
    ]
    seen = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal seen
        body = bodies[min(seen, len(bodies) - 1)]
        seen += 1
        return httpx.Response(200, content=json.dumps(body))

    agent = AgentCard(index=0, id="a", provider="openrouter", slug="v/m", description="d")
    protocol = CallProtocol(max_retries=2)
    async with OpenRouterClient(transport=httpx.MockTransport(handler)) as client:
        result = await client.chat(agent, [{"role": "user", "content": "hi"}], protocol)

    assert seen == 2
    assert result.error is None
    assert result.text == "Answer: 7"


def fake_chat(answers: dict[str, str], calls: list[str] | None = None):
    async def chat(agent, messages):
        if calls is not None:
            calls.append(agent.id)
        return ChatResult(
            text=answers.get(agent.id, ""),
            prompt_tokens=10,
            completion_tokens=5,
            cost_usd=0.0001,
            latency_ms=50,
        )

    return chat


async def test_collect_writes_one_record_per_task(tmp_path: Path, pool: AgentPool) -> None:
    tasks = [make_task(i) for i in range(3)]
    answers = {"fast_general": "Answer: 1", "reasoner": "Answer: 2", "coder": "Answer: 3"}
    out = tmp_path / "rewards.jsonl"

    progress = await collect_rewards(tasks, pool, fake_chat(answers), out, repetitions=2)

    records = read_models(out, RewardRecord)
    assert len(records) == 3
    assert progress.calls_made == 3 * len(pool.active) * 2
    assert all(r.agent_order == pool.agent_order for r in records)
    assert all(len(r.mean_reward) == len(pool.agent_order) for r in records)


async def test_a_provider_ignoring_max_tokens_stops_the_run(
    tmp_path: Path, pool: AgentPool
) -> None:
    """Compliance follows the upstream provider and varies per call, so it is caught here."""
    out = tmp_path / "rewards.jsonl"

    async def chat(agent, messages):
        over = pool.protocol.max_tokens * 4
        return ChatResult(text="Answer: 1", completion_tokens=over, finish_reason="stop")

    with pytest.raises(ProtocolViolation, match="against a cap"):
        await collect_rewards([make_task(0)], pool, chat, out, repetitions=1)

    assert not out.exists()


async def test_one_token_over_the_cap_is_rounding_not_a_violation(
    tmp_path: Path, pool: AgentPool
) -> None:
    """Providers count reasoning and special tokens differently; a real breach is a multiple."""
    out = tmp_path / "rewards.jsonl"

    async def chat(agent, messages):
        return ChatResult(
            text="Answer: 1",
            completion_tokens=pool.protocol.max_tokens + 1,
            finish_reason="length",
        )

    await collect_rewards([make_task(0)], pool, chat, out, repetitions=1)
    assert len(read_models(out, RewardRecord)) == 1


async def test_running_out_of_credit_never_writes_a_zero_row(
    tmp_path: Path, pool: AgentPool
) -> None:
    """An all-zero row is indistinguishable from 'no agent solved it', and resume would skip it."""
    tasks = [make_task(i) for i in range(4)]
    out = tmp_path / "rewards.jsonl"
    seen = 0

    async def chat(agent, messages):
        nonlocal seen
        seen += 1
        if seen > len(pool.active):  # everything after the first task is unfunded
            return ChatResult(text="", error="HTTP 402: {'message': 'Insufficient credits'}")
        return ChatResult(text="Answer: 1", completion_tokens=5)

    with pytest.raises(CollectionAborted):
        await collect_rewards(tasks, pool, chat, out, repetitions=1)

    records = read_models(out, RewardRecord)
    assert len(records) == 1
    assert any(r > 0 for r in records[0].mean_reward)


async def test_empty_response_is_flagged_not_hidden(tmp_path: Path, pool: AgentPool) -> None:
    """HTTP 200 with empty content is invisible if you only check `error`."""
    out = tmp_path / "rewards.jsonl"
    await collect_rewards([make_task(0)], pool, fake_chat({}), out, repetitions=1)

    record = read_models(out, RewardRecord)[0]
    assert all(c.empty for c in record.calls)
    assert all(c.error is None for c in record.calls)


async def test_resume_skips_completed_tasks(tmp_path: Path, pool: AgentPool) -> None:
    tasks = [make_task(i) for i in range(4)]
    out = tmp_path / "rewards.jsonl"
    answers = {"fast_general": "Answer: 1"}

    await collect_rewards(tasks[:2], pool, fake_chat(answers), out, repetitions=1)
    seen: list[str] = []
    progress = await collect_rewards(
        tasks, pool, fake_chat(answers, seen), out, repetitions=1, resume=True
    )

    assert progress.tasks_skipped == 2
    assert progress.tasks_done == 2
    assert len(seen) == 2 * len(pool.active)
    assert len(read_models(out, RewardRecord)) == 4


async def test_resume_refuses_a_different_repetition_count(tmp_path: Path, pool: AgentPool) -> None:
    out = tmp_path / "rewards.jsonl"
    await collect_rewards([make_task(0)], pool, fake_chat({}), out, repetitions=1)
    with pytest.raises(IncompatibleRewardsFile, match="repetitions"):
        await collect_rewards([make_task(1)], pool, fake_chat({}), out, repetitions=3)


async def test_resume_refuses_a_reordered_pool(tmp_path: Path, pool: AgentPool) -> None:
    out = tmp_path / "rewards.jsonl"
    await collect_rewards([make_task(0)], pool, fake_chat({}), out, repetitions=1)

    reordered = AgentPool(
        pool_version=pool.pool_version,
        protocol=pool.protocol,
        agents=[
            a.model_copy(update={"index": i, "id": f"z_{a.id}" if i == 0 else a.id})
            for i, a in enumerate(pool.agents)
        ],
    )
    with pytest.raises(IncompatibleRewardsFile, match="agent_order"):
        await collect_rewards([make_task(1)], reordered, fake_chat({}), out, repetitions=1)


async def test_no_resume_refuses_to_clobber(tmp_path: Path, pool: AgentPool) -> None:
    out = tmp_path / "rewards.jsonl"
    await collect_rewards([make_task(0)], pool, fake_chat({}), out, repetitions=1)
    with pytest.raises(FileExistsError):
        await collect_rewards([make_task(1)], pool, fake_chat({}), out, repetitions=1, resume=False)
