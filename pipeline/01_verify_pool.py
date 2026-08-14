"""Stage 1: prove every agent in the pool actually answers, before any money is spent.

    python pipeline/01_verify_pool.py

A dead or misbehaving slug produces a column of silent zeros in the reward matrix, and a router
trained on that learns to avoid a model that was never actually broken. So each active agent in
`configs/agents.v1.json` gets 21 live calls: 20 against a trivial prompt, plus one prompt that
cannot be satisfied inside `max_tokens`. Exits non-zero unless every agent passes.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import load_env  # noqa: E402

from benchgen_router_dataset.config_loader import load_pool  # noqa: E402
from benchgen_router_dataset.providers import OpenRouterClient  # noqa: E402

POOL_VERSION = "v1"
PROBE_CALLS = 20
PROBE_PROMPT = "Reply with exactly: OK"
# Deliberately unsatisfiable within max_tokens. A compliant agent stops at the cap with its
# partial text intact; silent truncation looks identical to a short answer once it reaches the
# reward matrix, so it has to be caught here.
LENGTH_PROBE_PROMPT = "Count from 1 to 5000. One number per line. Do not stop early."
MAX_EMPTY_RATE = 0.05


async def check(agent, protocol) -> tuple[str, str]:
    empties = errors = 0
    usage_seen = False

    async with OpenRouterClient(timeout_s=protocol.timeout_s) as client:
        for _ in range(PROBE_CALLS):
            result = await client.chat(
                agent, [{"role": "user", "content": PROBE_PROMPT}], protocol
            )
            errors += bool(result.error)
            empties += bool(result.empty)
            usage_seen = usage_seen or result.completion_tokens is not None
        long = await client.chat(
            agent, [{"role": "user", "content": LENGTH_PROBE_PROMPT}], protocol
        )

    detail = f"empty={empties / PROBE_CALLS:.0%} err={errors / PROBE_CALLS:.0%}"
    over_cap = (
        long.completion_tokens is not None and long.completion_tokens > protocol.max_tokens
    )
    if errors:
        return "FAIL", f"{detail} (call errors)"
    if empties / PROBE_CALLS > MAX_EMPTY_RATE:
        return "FAIL", f"{detail} (empty rate above {MAX_EMPTY_RATE:.0%})"
    if not usage_seen:
        return "FAIL", f"{detail} (no token counts, so cost cannot be tracked)"
    # A probe that errored or came back empty measured nothing, so it cannot pass: the point is
    # positive evidence that the cap is honoured.
    if long.error or over_cap or not long.text.strip():
        return "FAIL", f"{detail} (ignored the {protocol.max_tokens} token cap)"
    return "PASS", detail


def main() -> int:
    load_env()
    pool = load_pool(POOL_VERSION)
    failed: list[str] = []

    for agent in pool.active:
        verdict, detail = asyncio.run(check(agent, pool.protocol))
        print(f"{agent.id:<22} {agent.slug:<34} {detail:<22} {verdict}")
        if verdict != "PASS":
            failed.append(agent.id)

    if failed:
        print(f"\nGate 1 FAILED: {', '.join(failed)}. Collection must not start.")
        return 1
    print("\nGate 1 passed: every active agent returned a real completion.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
