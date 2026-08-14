"""Stage 2: build the task dataset. Load, map, dedupe, balance, split.

    python pipeline/02_build_task_dataset.py

Deterministic: the same config in gives a byte-identical `data/tasks.v1.jsonl` out. No API key
is needed, but the upstream benchmark sources are downloaded on the first run.

The chain, in order:

  load     one adapter per upstream source, each normalising to the same `Task` schema
  map      task_id, gold answer, named grader, domain, difficulty, and full source provenance
  dedupe   SHA-1 over normalised prompt text, first occurrence kept
  balance  no domain above `max_domain_share`, trimmed with a seeded binary search
  split    stratified by (domain, difficulty), held-out sources forced entirely into test
"""

from __future__ import annotations

from pathlib import Path

from benchgen_router_dataset.analysis.taskpool_report import render_task_pool
from benchgen_router_dataset.config_loader import load_collection
from benchgen_router_dataset.io_jsonl import write_jsonl
from benchgen_router_dataset.paths import reports_dir, tasks_path
from benchgen_router_dataset.taskpool import BuildRequest, build_pool, gate_2

VERSION = "v1"
CACHE_DIR = Path("data/raw/hf_cache")


def main() -> int:
    config = load_collection(VERSION)
    request = BuildRequest(
        per_source_limit=dict(config["per_source_limit"]),
        max_share=float(config["max_domain_share"]),
        seed=int(config["seed"]),
        cache_dir=CACHE_DIR,
        ratios=config["split_ratios"],
    )
    result = build_pool(request)

    out = tasks_path(VERSION)
    print(f"wrote {write_jsonl(out, result.tasks)} tasks to {out}")
    print(f"  duplicates dropped: {result.duplicates_dropped}")
    for domain, share in sorted(result.shares.items()):
        trimmed = result.balance_removed.get(domain, 0)
        print(f"  {domain:<10} {share:6.1%}  (balance trim: {trimmed})")
    for split, count in sorted(result.splits.items()):
        print(f"  split {split:<11} {count}")
    for name, why in sorted(result.unavailable.items()):
        print(f"  source unavailable: {name}: {why}")

    report = reports_dir() / "task_pool.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        render_task_pool(result, {out.name: out}, request.seed, request.max_share),
        encoding="utf-8",
        newline="\n",
    )
    print(f"wrote {report}")

    problems = gate_2(result, request.max_share)
    if problems:
        print("\nGate 2 FAILED:")
        for problem in problems[:20]:
            print(f"  - {problem}")
        return 1
    print("\nGate 2 passed: balanced, answered, graded, split.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
