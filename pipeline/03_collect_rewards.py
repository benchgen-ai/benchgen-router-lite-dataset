"""Stage 3: collect the reward matrix. Every agent answers every sampled task, three times.

    python pipeline/03_collect_rewards.py --dry-run   # call count and cost ceiling, zero calls
    python pipeline/03_collect_rewards.py             # the real run, resumable

Every agent sees byte-identical messages for a given task, so a score difference reflects the
model and not the prompt conditions. One JSONL row is written per task as soon as that task's
calls finish, so a crash never loses completed work and a rerun skips what is already there.
"""

from __future__ import annotations

import asyncio
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import load_env  # noqa: E402

from benchgen_router_dataset.collection import (  # noqa: E402
    CollectionAborted,
    ProtocolViolation,
    collect_rewards,
)
from benchgen_router_dataset.collection.protocol import build_messages  # noqa: E402
from benchgen_router_dataset.config_loader import (  # noqa: E402
    load_collection,
    load_pool,
    require_verified,
)
from benchgen_router_dataset.io_jsonl import read_models  # noqa: E402
from benchgen_router_dataset.paths import rewards_path, tasks_path  # noqa: E402
from benchgen_router_dataset.providers import OpenRouterClient  # noqa: E402
from benchgen_router_dataset.schemas import Task  # noqa: E402

VERSION = "v1"
OUT_PATH = rewards_path(f"{VERSION}-pilot")


def sample_per_domain(tasks: list[Task], per_domain: dict[str, int], seed: int) -> list[Task]:
    """Seeded and sorted, so the same seed always draws the same tasks."""
    buckets: dict[str, list[Task]] = defaultdict(list)
    for task in tasks:
        buckets[task.domain].append(task)

    chosen: list[Task] = []
    for domain, want in sorted(per_domain.items()):
        bucket = sorted(buckets.get(domain, []), key=lambda t: t.task_id)
        if len(bucket) < want:
            print(f"  warning: domain {domain!r} has {len(bucket)} tasks, wanted {want}")
        rng = random.Random(f"{seed}/{domain}")
        chosen.extend(rng.sample(bucket, min(want, len(bucket))))
    return sorted(chosen, key=lambda t: t.task_id)


def cost_ceiling(tasks: list[Task], pool, repetitions: int) -> float:
    """Upper bound before spending: prompts at 4 characters per token, replies at the cap."""
    total = 0.0
    for agent in pool.active:
        prompt_usd = (agent.price_per_1m_prompt_usd or 0.0) / 1_000_000
        output_usd = (agent.price_per_1m_output_usd or 0.0) / 1_000_000
        for task in tasks:
            in_tokens = len(build_messages(task)[1]["content"]) / 4
            out_tokens = pool.protocol.max_tokens / 4
            total += repetitions * (in_tokens * prompt_usd + out_tokens * output_usd)
    return total


async def run(dry_run: bool) -> int:
    config = load_collection(VERSION)
    pool = load_pool(VERSION)
    tasks = sample_per_domain(
        read_models(tasks_path(VERSION), Task),
        config["pilot"]["per_domain"],
        int(config["seed"]),
    )
    repetitions = int(config["repetitions"])

    print(f"tasks: {len(tasks)}  agents: {len(pool.active)}  repetitions: {repetitions}")
    print(f"calls: {len(tasks) * len(pool.active) * repetitions}")
    print(f"cost ceiling: ${cost_ceiling(tasks, pool, repetitions):.2f}")
    print(f"output: {OUT_PATH}")
    if dry_run:
        print("dry run, no calls made")
        return 0

    require_verified(pool)  # refuses to start against a slug Stage 1 never verified

    def report(progress) -> None:
        if progress.tasks_done % 10 == 0 or progress.tasks_done == len(tasks):
            print(
                f"  {progress.tasks_done}/{len(tasks) - progress.tasks_skipped} tasks  "
                f"calls={progress.calls_made} errors={progress.errors} "
                f"empty={progress.empties} cost=${progress.cost_usd:.4f}"
            )

    async with OpenRouterClient(timeout_s=pool.protocol.timeout_s) as client:
        try:
            progress = await collect_rewards(
                tasks=tasks,
                pool=pool,
                chat=lambda agent, messages: client.chat(agent, messages, pool.protocol),
                out_path=OUT_PATH,
                repetitions=repetitions,
                concurrency=int(config["concurrency"]),
                store_raw=bool(config["store_raw_response"]),
                resume=True,
                on_progress=report,
            )
        except ProtocolViolation as exc:
            # The agent's numbers are no longer comparable and its cost is unbounded.
            print(f"\nSTOPPED, protocol violation: {exc}")
            print("  Retire that agent and rerun; its measurements cannot be used.")
            return 2
        except CollectionAborted as exc:
            # A row written now would record a billing failure as a wrong answer, and resume
            # would then skip that task forever.
            print(f"\nSTOPPED, account-level failure: {exc}")
            print(f"  {OUT_PATH} holds only fully collected tasks; rerun to resume.")
            return 2

    print(
        f"done: {progress.tasks_done} collected, {progress.tasks_skipped} already present, "
        f"{progress.errors} errors, {progress.empties} empty, ${progress.cost_usd:.4f}"
    )
    if progress.per_agent_empty:
        print(f"  empty responses by agent: {progress.per_agent_empty}")
    return 0


def main() -> int:
    load_env()
    return asyncio.run(run(dry_run="--dry-run" in sys.argv))


if __name__ == "__main__":
    raise SystemExit(main())
