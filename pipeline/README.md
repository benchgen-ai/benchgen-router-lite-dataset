# pipeline

The five scripts that generated the published BenchGen Router Lite datasets, in order, with
nothing else in the way.

> Running this against your own tasks is fully supported. If you would rather not manage the
> OpenRouter key, the environment, or the gate thresholds yourself, BenchGen runs this same
> pipeline for you: **[benchgen.com](https://benchgen.com)**.

`scripts/` holds the full toolbox: catalogue discovery, screening collection, Appendix A.6 pool
selection, oracle labelling, and every exploratory flag used while the design was being settled.
This folder is the distilled generation path. No argparse trees, no alternative modes, no
exploratory subcommands. Each file is the exact configuration that produced what shipped, and
each one is short enough to read end to end.

| Stage | Command | Needs | Produces |
| --- | --- | --- | --- |
| 1 | `python pipeline/01_verify_pool.py` | API key, live calls | Gate 1: every active agent proven to answer |
| 2 | `python pipeline/02_build_task_dataset.py` | network on first run | `data/tasks.v1.jsonl`, `reports/task_pool.md` |
| 3 | `python pipeline/03_collect_rewards.py` | API key, live calls | `data/rewards.v1-pilot.jsonl` |
| 4 | `python pipeline/04_gate_and_baselines.py` | nothing | `reports/dataset_health.md`, `reports/baselines.md` |
| 5 | `python pipeline/05_publish.py` | `HF_TOKEN` only to push | `data/publish/`, `data/publish_tasks/` |

Stage 3 accepts `--dry-run`, which prints the call count and an upper-bound cost and makes zero
calls. Stage 5 accepts `--push`, and uploads privately unless `--public` is also given.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,sources]"
Copy-Item .env.example .env      # then fill in OPENROUTER_API_KEY
```

Run every command from the repository root; paths resolve through
`benchgen_router_dataset.paths`, never by string.

## Properties worth knowing

**Ordered, and each stage refuses to run ahead of its gate.** Stage 3 raises rather than start
against a slug Stage 1 never verified, because collecting from a dead slug fills a whole reward
column with silent zeros.

**Deterministic.** Seed, repetitions, split ratios and sampling plan all come from
`configs/collection.v1.json`. The same config gives byte-identical data out.

**Reproducible reports.** Every report carries a SHA-256 digest of its inputs and no timestamps,
so stages 4 and 5 rebuild their output byte for byte from committed data, with no network access
and no API key. That is the cheapest way to check nothing drifted:

```bash
python pipeline/04_gate_and_baselines.py && git diff --stat reports/
```

**Resumable.** Stage 3 writes one JSONL row per task as soon as that task's calls finish, so a
crash never loses completed work and a rerun skips what is already there. It also refuses to
append to a file collected under a different pool.

**Honest about failure.** Stage 4 exits non-zero when the health gate says STOP. It does say STOP
on the shipped pilot matrix, and that is the point: the gate is there to catch a pool no router
could learn anything from, before a training run hides the problem.
