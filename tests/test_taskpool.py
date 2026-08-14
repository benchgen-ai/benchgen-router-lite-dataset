from __future__ import annotations

import pytest
from conftest import make_task

from benchgen_router_dataset.schemas import RewardRecord, Task
from benchgen_router_dataset.taskpool.balance import (
    domain_shares,
    effective_max_share,
    enforce_balance,
)
from benchgen_router_dataset.taskpool.dedupe import dedupe
from benchgen_router_dataset.taskpool.split import assert_no_leakage, assign_splits, split_counts


def test_dedupe_keeps_first_occurrence() -> None:
    a = make_task(0)
    b = make_task(1, prompt=a.prompt)
    kept, dropped = dedupe([a, b])
    assert dropped == 1
    assert [t.task_id for t in kept] == [a.task_id]


def test_balance_caps_the_dominant_domain() -> None:
    """Fugu-Lite V1 was 92.7% one domain and the router had nothing to specialise on."""
    tasks = [make_task(i, domain="math") for i in range(90)]
    tasks += [make_task(100 + i, domain="knowledge") for i in range(5)]
    tasks += [make_task(200 + i, domain="reasoning") for i in range(5)]
    tasks += [make_task(300 + i, domain="science") for i in range(5)]
    balanced, removed = enforce_balance(tasks, max_share=0.30)
    shares = domain_shares(balanced)
    assert shares["math"] <= 0.30 + 1e-9
    assert removed["math"] > 0


def test_cap_below_one_over_n_domains_is_infeasible_and_reported() -> None:
    """30% across 3 domains is arithmetically impossible; the cap becomes 33.3%."""
    assert effective_max_share(3, 0.30) == pytest.approx(1 / 3)
    assert effective_max_share(6, 0.30) == pytest.approx(0.30)

    tasks = [make_task(i, domain="math") for i in range(90)]
    tasks += [make_task(100 + i, domain="knowledge") for i in range(5)]
    tasks += [make_task(200 + i, domain="reasoning") for i in range(5)]
    balanced, _ = enforce_balance(tasks, max_share=0.30)
    assert max(domain_shares(balanced).values()) <= 1 / 3 + 1e-9


def test_balance_keeps_as_much_data_as_the_cap_allows() -> None:
    tasks = [make_task(i, domain="math") for i in range(100)]
    tasks += [make_task(1000 + i, domain="knowledge") for i in range(100)]
    tasks += [make_task(2000 + i, domain="reasoning") for i in range(100)]
    tasks += [make_task(3000 + i, domain="science") for i in range(10)]
    balanced, _ = enforce_balance(tasks, max_share=0.30)
    counts = {d: 0 for d in ("math", "knowledge", "reasoning", "science")}
    for t in balanced:
        counts[t.domain] += 1
    assert counts["science"] == 10
    assert counts["math"] == counts["knowledge"] == counts["reasoning"]
    assert max(domain_shares(balanced).values()) <= 0.30 + 1e-9


def test_balance_is_deterministic() -> None:
    tasks = [make_task(i, domain="math") for i in range(50)]
    tasks += [make_task(100 + i, domain="knowledge") for i in range(50)]
    first, _ = enforce_balance(tasks, max_share=0.30, seed=42)
    second, _ = enforce_balance(tasks, max_share=0.30, seed=42)
    assert [t.task_id for t in first] == [t.task_id for t in second]


def test_balance_leaves_an_already_balanced_pool_alone() -> None:
    tasks = [
        make_task(i, domain=d)
        for d in ("math", "knowledge", "reasoning", "science")
        for i in range(10)
    ]
    tasks = [
        t.model_copy(update={"task_id": f"{t.domain}/test/{i:06d}"})
        for i, t in enumerate(tasks)
    ]
    balanced, removed = enforce_balance(tasks, max_share=0.30)
    assert removed == {}
    assert len(balanced) == len(tasks)


def test_splits_are_deterministic_and_disjoint() -> None:
    tasks = [make_task(i, domain="math" if i % 2 else "knowledge") for i in range(100)]
    first = assign_splits(tasks, seed=42)
    second = assign_splits(tasks, seed=42)
    assert {t.task_id: t.split for t in first} == {t.task_id: t.split for t in second}
    assert_no_leakage(first)
    counts = split_counts(first)
    assert sum(counts.values()) == 100
    assert all(n > 0 for n in counts.values())


def test_split_ratios_must_sum_to_one() -> None:
    with pytest.raises(ValueError, match="sum to 1.0"):
        assign_splits([make_task(0)], ratios={"train": 0.5, "validation": 0.2, "test": 0.2})


def test_task_id_shape_is_enforced() -> None:
    with pytest.raises(ValueError, match="task_id"):
        make_task(0, task_id="no-slashes")


def test_reward_record_recomputes_ties() -> None:
    """A stale best_agents in the file must never be trusted — it silently corrupts the gate."""
    record = RewardRecord(
        task_id="t/0",
        pool_version="v1",
        repetitions=3,
        agent_order=["a", "b", "c"],
        mean_reward=[1.0, 1.0, 0.0],
        best_agents=["c"],
        is_tie=False,
    )
    assert record.best_agents == ["a", "b"]
    assert record.is_tie is True


def test_reward_record_rejects_misaligned_vectors() -> None:
    with pytest.raises(ValueError, match="positionally aligned"):
        RewardRecord(
            task_id="t/0",
            pool_version="v1",
            repetitions=3,
            agent_order=["a", "b"],
            mean_reward=[1.0],
        )


def test_task_round_trips_through_json() -> None:
    task = make_task(7)
    assert Task.model_validate(task.model_dump(mode="json")) == task
