"""Stage 4: decide whether this pool deserves a router at all, then measure the bar.

    python pipeline/04_gate_and_baselines.py

Offline. Reads only `data/` and `configs/`, so it needs no network and no API key, and exits
non-zero when the health gate says STOP.

A routing dataset is only useful if no single agent dominates. If one agent is best almost
everywhere, the optimal router is "always pick that agent" and no amount of training helps. The
gate measures that before anything gets trained; the baselines then state the exact band a
trained router has to land inside to have been worth training.
"""

from __future__ import annotations

from benchgen_router_dataset.analysis import (
    compute_baselines,
    compute_health,
    evaluate,
    render_baselines,
    render_health,
)
from benchgen_router_dataset.config_loader import load_gates
from benchgen_router_dataset.io_jsonl import read_models
from benchgen_router_dataset.paths import reports_dir, rewards_path, tasks_path
from benchgen_router_dataset.schemas import RewardRecord, Task

VERSION = "v1"
REWARDS = rewards_path(f"{VERSION}-pilot")
HELD_OUT_SPLIT = "test"


def write_report(name: str, content: str) -> None:
    path = reports_dir() / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")
    print(f"wrote {path}")


def main() -> int:
    records = read_models(REWARDS, RewardRecord)
    tasks = read_models(tasks_path(VERSION), Task)

    metrics = compute_health(records)
    outcome = evaluate(load_gates(VERSION), metrics)
    write_report("dataset_health.md", render_health(metrics, outcome, {REWARDS.name: REWARDS}))

    print(f"\n{outcome.decision}")
    print(f"  RER per dataset   {metrics.rer_dataset:.4f}  (the pool-selection criterion)")
    print(f"  RER per question  {metrics.rer_per_question:.4f}")
    print(f"  all agents equal  {metrics.all_equal_rate:.1%}  no routing signal on these")
    print(f"  unique winner     {metrics.unique_winner_rate:.1%}  where routing can actually pay")
    for row in outcome.blocking:
        print(f"  BLOCKING: {row.label} = {row.value}")

    held_out = {t.task_id for t in tasks if t.split == HELD_OUT_SPLIT}
    baselines = compute_baselines([r for r in records if r.task_id in held_out])
    write_report(
        "baselines.md", render_baselines(baselines, HELD_OUT_SPLIT, {REWARDS.name: REWARDS})
    )

    print(f"\nbaselines on the {HELD_OUT_SPLIT} split ({baselines.n_tasks} tasks)")
    for agent, score in sorted(baselines.single_agent.items(), key=lambda kv: -kv[1]):
        print(f"  {agent:<22} {score:.4f}")
    print(f"  {'uniform random':<22} {baselines.uniform_random:.4f}")
    print(f"  {'best fixed agent':<22} {baselines.best_fixed:.4f} ({baselines.best_fixed_agent})")
    print(f"  {'oracle (upper bound)':<22} {baselines.oracle:.4f}")
    print(f"\nrouting headroom: {baselines.headroom:.4f}")

    return 1 if outcome.decision == "STOP" else 0


if __name__ == "__main__":
    raise SystemExit(main())
