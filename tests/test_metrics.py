from __future__ import annotations

import pytest
from conftest import make_reward, make_rewards_over_datasets

from benchgen_router_dataset.analysis import compute_baselines, compute_health, evaluate
from benchgen_router_dataset.analysis.baselines import route_accuracy, routed_reward
from benchgen_router_dataset.analysis.rer import (
    best_single,
    build_matrix,
    combination_performance,
    relative_error_reduction,
    rer,
)
from benchgen_router_dataset.config_loader import load_gates


def test_rer_is_zero_when_best_agent_is_perfect() -> None:
    assert relative_error_reduction(1.0, 1.0) == 0.0


def test_rer_matches_the_paper_formula() -> None:
    # S* = 0.8, Z = 0.9 -> half of the remaining 0.2 error is removable.
    assert relative_error_reduction(0.9, 0.8) == pytest.approx(0.5)


def test_dataset_level_rer_uses_dataset_accuracy_not_question_accuracy(
    agents: list[str],
) -> None:
    """A.6 eq. 13: `E(D, M)` is accuracy on a whole dataset.

    Here every agent wins a quarter of the questions in every dataset, so per-dataset accuracy
    is identical for all of them and dataset-level RER is 0 — while per-question routing still
    looks like it has total headroom. Reporting the per-question number as "RER (A.6)" would
    overstate the case by that entire gap.
    """
    cycle = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    records = make_rewards_over_datasets(
        agents, {"alpha": cycle * 3, "beta": cycle * 3, "gamma": cycle * 3}
    )
    metrics = compute_health(records)
    assert metrics.rer_dataset == pytest.approx(0.0)
    assert metrics.rer_per_question == pytest.approx(1.0)


def test_dataset_level_rer_rewards_true_specialisation(agents: list[str]) -> None:
    """Different agents own different datasets — the case routing actually exists for."""
    records = make_rewards_over_datasets(
        agents,
        {
            "alpha": [[1.0, 0.0, 0.0, 0.0]] * 10,
            "beta": [[0.0, 1.0, 0.0, 0.0]] * 10,
            "gamma": [[0.0, 0.0, 1.0, 0.0]] * 10,
        },
    )
    metrics = compute_health(records)
    assert metrics.z_dataset == pytest.approx(1.0)
    assert metrics.s_star_dataset == pytest.approx(1 / 3)
    assert metrics.rer_dataset == pytest.approx(1.0)


def test_matrix_cells_are_dataset_means(agents: list[str]) -> None:
    records = make_rewards_over_datasets(
        agents,
        {
            "alpha": [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
            "beta": [[0.5, 1.0, 1.0, 1.0]],
        },
    )
    matrix = build_matrix(records)
    assert matrix.datasets == ("alpha", "beta")
    assert matrix.E("alpha", agents[0]) == pytest.approx(0.5)
    assert matrix.counts == {"alpha": 2, "beta": 1}
    assert combination_performance(matrix, matrix.datasets, matrix.agents) == pytest.approx(0.75)
    assert best_single(matrix, matrix.datasets, matrix.agents)[1] == pytest.approx(0.5)
    assert rer(matrix, matrix.datasets, matrix.agents) == pytest.approx(0.5)


def test_dominated_pool_produces_no_headroom(agents: list[str]) -> None:
    """The Fugu-Lite failure: one agent best everywhere, everyone else identical."""
    records = make_rewards_over_datasets(
        agents, {d: [[1.0, 0.0, 0.0, 0.0]] * 10 for d in ("alpha", "beta", "gamma")}
    )
    metrics = compute_health(records)
    assert metrics.best_single_agent == agents[0]
    assert metrics.rer_dataset == pytest.approx(0.0)
    assert metrics.rer_per_question == pytest.approx(0.0)
    assert metrics.unique_best_share[agents[1]] == 0.0


def test_all_equal_tasks_carry_no_signal(agents: list[str]) -> None:
    records = make_rewards_over_datasets(
        agents, {d: [[1.0, 1.0, 1.0, 1.0]] * 5 for d in ("alpha", "beta", "gamma")}
    )
    metrics = compute_health(records)
    assert metrics.all_equal_rate == 1.0
    assert metrics.unique_winner_rate == 0.0


def test_gate_stops_on_a_dominated_pool(agents: list[str]) -> None:
    records = make_rewards_over_datasets(
        agents, {d: [[1.0, 1.0, 1.0, 1.0]] * 5 for d in ("alpha", "beta", "gamma")}
    )
    outcome = evaluate(load_gates("v1"), compute_health(records))
    assert outcome.decision == "STOP"
    assert {row.metric for row in outcome.blocking} >= {"rer_dataset", "rer_per_question"}


def test_gate_stops_when_there_is_only_one_dataset(agents: list[str]) -> None:
    """`Z` and `S*` collapse together with one dataset, so A.6's RER means nothing."""
    records = make_rewards_over_datasets(
        agents, {"alpha": [[1.0, 0.0, 0.2, 0.2], [0.0, 1.0, 0.2, 0.2]] * 10}
    )
    outcome = evaluate(load_gates("v1"), compute_health(records))
    assert outcome.decision == "STOP"
    assert any(row.metric == "n_datasets" for row in outcome.blocking)


def test_gate_continues_on_a_specialised_pool(agents: list[str]) -> None:
    records = make_rewards_over_datasets(
        agents,
        {
            "alpha": [[1.0, 0.3, 0.2, 0.2]] * 10,
            "beta": [[0.2, 1.0, 0.3, 0.2]] * 10,
            "gamma": [[0.2, 0.2, 1.0, 0.3]] * 10,
            "delta": [[0.3, 0.2, 0.2, 1.0]] * 10,
        },
    )
    outcome = evaluate(load_gates("v1"), compute_health(records))
    assert outcome.decision == "CONTINUE"


def test_route_accuracy_is_tie_aware(agents: list[str]) -> None:
    """A tied pick is a correct route; comparing against argmax's first index says otherwise."""
    records = [make_reward("alpha/test/000000", [1.0, 1.0, 0.0, 0.0], agents)]
    assert route_accuracy(records, {"alpha/test/000000": agents[1]}) == 1.0
    assert route_accuracy(records, {"alpha/test/000000": agents[2]}) == 0.0


def test_baselines_order(agents: list[str]) -> None:
    records = []
    for i in range(20):
        means = [0.5, 0.5, 0.5, 0.5]
        means[i % 4] = 1.0
        records.append(make_reward(f"alpha/test/{i:06d}", means, agents))
    baselines = compute_baselines(records)
    assert baselines.uniform_random <= baselines.best_fixed <= baselines.oracle
    assert baselines.headroom > 0
    perfect = {r.task_id: r.agent_order[r.mean_reward.index(max(r.mean_reward))] for r in records}
    assert routed_reward(records, perfect) == pytest.approx(baselines.oracle)


def test_mismatched_agent_order_is_an_error(agents: list[str]) -> None:
    records = [
        make_reward("alpha/test/000000", [1.0, 0.0, 0.0, 0.0], agents),
        make_reward("alpha/test/000001", [1.0, 0.0, 0.0, 0.0], list(reversed(agents))),
    ]
    with pytest.raises(ValueError, match="agent_order"):
        compute_health(records)
