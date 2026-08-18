# train

The router head trainer, extracted from the same code BenchGen's **Train** tab runs when you
pick the **Head** training method. Not a rewrite: the embedding step, the head shapes, and the
sep-CMA-ES loop in `train_router_head.py` are the platform's own training worker, with only the
storage boundary changed (writes to a local folder here; the platform additionally uploads the
result to its own storage and lets you register it as a selectable model from the UI).

## Two ways to train this head

| | No code | This script |
| --- | --- | --- |
| Where | BenchGen's **[Train tab](https://benchgen.com/train)** | Your own machine |
| Steps | Point **Dataset** at the rewards dataset, **Tasks dataset** at the tasks dataset, pick **Head**, click **Start Training** | `python train/train_router_head.py` |
| Output | A registered model, selectable anywhere BenchGen lets you pick a model, ready to serve and benchmark immediately | `head_weights.npy` + `manifest.json`, local only |

Both run the identical algorithm below. The [full walkthrough](https://benchgen.com/docs/guides/router-head/benchgen-router-lite)
covers the platform path screenshot by screenshot; this folder is for reproducing it yourself,
or adapting it to your own reward matrix. If you just want a trained, servable head without
managing the run yourself, **[benchgen.com/train](https://benchgen.com/train)** is the faster
path, this script exists so the algorithm underneath it is auditable.

## Run it

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[train]"
.\.venv\Scripts\python.exe train\train_router_head.py
```

With no argument, it reads `train/config.example.json`, which reproduces the guide's own run:
`Qwen/Qwen3-1.7B` as the frozen backbone, `benchgen/router-pilot` and `benchgen/router-pilot-tasks`
as the paired dataset, `sigma0` 0.5, 60 generations, seed 42, a pure linear head
(`head_hidden_dim: 0`). Pass your own config to train against a different dataset or pool:

```bash
python train/train_router_head.py path/to/your-config.json
```

Verified against the real published data: joining the two datasets on `task_id` yields 46 rows,
0 missing, 35 train / 11 test after the split, embedding dim 2048 (Qwen3-1.7B's hidden size), and
`head_param_count(2048, 0, 5) == 10245`, the exact same numbers the guide's screenshots show.

> **Note:** `benchgen/router-pilot-tasks`'s own README currently declares `data_files: tasks.jsonl`
> in its config metadata, but the uploaded file is named `train.jsonl`, so
> `load_dataset("benchgen/router-pilot-tasks")` fails outright until that is fixed upstream.
> `config.example.json` works around it with `tasks_url` pointing straight at the resolvable file
> instead of `tasks_dataset`; swap back to `tasks_dataset` once the Hub metadata is corrected.

Set `tasks_dataset` (goes through the `datasets` library) instead of `tasks_url` (a direct JSONL
fetch) if you point this at a dataset with correct config metadata.

`device` defaults to `cpu` here so the example runs anywhere with no setup. Switch it to `cuda`
in your config for the same speed the guide reports (a few seconds instead of a few minutes) if
you have a GPU available; forward passes on a 1B+ parameter backbone are the slow part on CPU,
not the CMA-ES search itself.

## What it does not do

This script trains and writes local artifacts. It does not register a model anywhere, does not
serve one, and makes no API calls. What the platform's **Register as Model** step and its
`head_router` chat-completions gateway add on top (serving a trained head, picking an agent for
a live query, then actually calling that agent) is described in the guide's Benchmark section; it
is infrastructure this script deliberately leaves out so it stays runnable with nothing but a
Python environment.

## Why this is trustworthy as "the real algorithm"

Every number in it is one you can check against the guide: 10,245 parameters for the shipped
run's linear head, `sigma0=0.5`, 60 generations, `best_reward_so_far` climbing per generation in
the exact same log format the guide's own training-log screenshots show. Nothing here is a
simplified stand-in for what the platform actually runs.
