from __future__ import annotations

from pathlib import Path

from conftest import make_reward

from benchgen_router_dataset.analysis import (
    compute_baselines,
    compute_health,
    evaluate,
    render_baselines,
    render_health,
)
from benchgen_router_dataset.config_loader import load_gates
from benchgen_router_dataset.io_jsonl import write_jsonl


def _records(agents: list[str]):
    out = []
    for i in range(20):
        means = [0.2, 0.2, 0.2, 0.2]
        means[i % 4] = 1.0
        out.append(make_reward(f"fixture/test/{i:06d}", means, agents))
    return out


def test_reports_are_byte_identical_across_runs(tmp_path: Path, agents: list[str]) -> None:
    """No timestamps, no commit hashes: reports must reproduce from committed data alone."""
    records = _records(agents)
    path = tmp_path / "rewards.jsonl"
    write_jsonl(path, records)

    metrics = compute_health(records)
    outcome = evaluate(load_gates("v1"), metrics)
    first = render_health(metrics, outcome, {path.name: path})
    second = render_health(metrics, outcome, {path.name: path})
    assert first == second
    assert "sha256" not in first.lower() or "SHA-256" in first


def test_health_report_contains_the_decision_and_the_negative_numbers(
    tmp_path: Path, agents: list[str]
) -> None:
    records = [make_reward(f"t/{i}", [1.0, 1.0, 1.0, 1.0], agents) for i in range(10)]
    path = tmp_path / "rewards.jsonl"
    write_jsonl(path, records)
    metrics = compute_health(records)
    outcome = evaluate(load_gates("v1"), metrics)
    report = render_health(metrics, outcome, {path.name: path})

    assert "STOP" in report
    assert "all-agents-equal tasks" in report
    assert "Do not collect more data with the same design" in report


def test_baselines_report_prints_every_policy(tmp_path: Path, agents: list[str]) -> None:
    records = _records(agents)
    path = tmp_path / "rewards.jsonl"
    write_jsonl(path, records)
    report = render_baselines(compute_baselines(records), "test", {path.name: path})

    for expected in ("best fixed agent", "uniform random", "oracle (per-question best)"):
        assert expected in report
    for agent in agents:
        assert agent in report
