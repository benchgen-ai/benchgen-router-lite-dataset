"""Router head trainer: offline agent-selection head training.

This is the same algorithm BenchGen's Train tab runs when you pick the Head
training method there. It is not a rewrite or a simplified stand-in: the
embedding, the head shapes, and the sep-CMA-ES loop below are copied from the
platform's own training worker, with only the storage boundary changed (this
version writes the trained head to a local folder; the platform additionally
uploads it to its own storage and lets you register it as a model from the
UI). See the "Train the router head" section of the docs for the full story
this script is one half of:
https://benchgen.com/docs/guides/router-head/benchgen-router-lite

Unlike an RL or SFT loop, this never generates text and never touches
gradients on the backbone:

    query text -> [frozen backbone, mean-pooled] -> embedding
                                                        |
                                              head (evolved by sep-CMA-ES)
                                                        |
                                        logits over N pool agents -> argmax pick

The reward for (task, agent) is already known for every row in the dataset,
so training needs no GPU generation and makes zero API calls.

Two data sources are joined before anything is embedded:
  - rewards_dataset: a dataset of reward rows (task_id, agent_order,
    mean_reward, ...), e.g. "benchgen/router-pilot". Deliberately has NO
    prompt text (upstream licensing).
  - tasks_dataset: supplies prompt text per task_id.
A head trained against `agent_order` A is meaningless when served against
pool order B, so the manifest written alongside the weights pins the exact
order it was trained on.

Usage:

    python train/train_router_head.py                      # uses config.example.json
    python train/train_router_head.py path/to/config.json  # your own config
"""
from __future__ import annotations

import json
import sys
import time
import zipfile
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.example.json"


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _resolve_split(ds_id, split, ds_config=""):
    from datasets import get_dataset_split_names

    try:
        if ds_config:
            avail = get_dataset_split_names(ds_id, ds_config)
        else:
            avail = get_dataset_split_names(ds_id)
    except Exception:
        return split
    base = (split or "train").split("[", 1)[0]
    if not avail or base in avail:
        return split or "train"
    for cand in ("train", "test", "validation", "dev"):
        if cand in avail:
            print(f"    Requested split {split!r} unavailable (have {avail}); using {cand!r}")
            return cand
    return avail[0]


def load_rewards(cfg: dict) -> list:
    from datasets import load_dataset

    ds_id = cfg["rewards_dataset"]
    split = cfg.get("rewards_dataset_split") or "train"
    print(f"Loading rewards dataset {ds_id} (split={split})...")
    ds = load_dataset(ds_id, split=_resolve_split(ds_id, split))
    rows = [dict(r) for r in ds]
    print(f"  {len(rows)} reward rows loaded.")
    return rows


def load_tasks(cfg: dict) -> dict:
    """Returns {task_id: {"prompt": str, "split": str|None}}.

    Two loading paths, same as the platform's own trainer: `tasks_url` fetches a JSONL file
    directly, `tasks_dataset` goes through the `datasets` library. Prefer `tasks_url` if a
    dataset's own README config metadata is wrong (`benchgen/router-pilot-tasks`'s README
    currently declares `data_files: tasks.jsonl`, but the uploaded file is `train.jsonl`, so
    `load_dataset("benchgen/router-pilot-tasks")` fails outright until that is fixed upstream).
    """
    tasks_url = cfg.get("tasks_url")
    tasks = {}
    if tasks_url:
        import httpx

        print(f"Loading tasks from {tasks_url} ...")
        resp = httpx.get(tasks_url, timeout=120, follow_redirects=True)
        resp.raise_for_status()
        for line in resp.text.splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            tid = row.get("task_id")
            if tid:
                tasks[tid] = row
    elif cfg.get("tasks_dataset"):
        from datasets import load_dataset

        tasks_dataset = cfg["tasks_dataset"]
        split = cfg.get("tasks_dataset_split") or "train"
        print(f"Loading tasks dataset {tasks_dataset} (split={split}) ...")
        ds = load_dataset(tasks_dataset, split=_resolve_split(tasks_dataset, split))
        for row in ds:
            row = dict(row)
            tid = row.get("task_id")
            if tid:
                tasks[tid] = row
    else:
        raise ValueError(
            "Config must set either tasks_url or tasks_dataset; rewards_dataset "
            "intentionally omits question text (licensing)."
        )
    print(f"  {len(tasks)} tasks with prompt text loaded.")
    return tasks


def join_rows(reward_rows: list, tasks: dict) -> tuple:
    joined, missing = [], 0
    for r in reward_rows:
        t = tasks.get(r.get("task_id"))
        if not t or not t.get("prompt"):
            missing += 1
            continue
        joined.append({**r, "prompt": t["prompt"], "split": t.get("split")})
    return joined, missing


def embed_texts(texts: list, backbone_model: str, device: str):
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer

    print(
        f"Embedding {len(texts)} prompts with frozen backbone {backbone_model} on {device} ...",
        flush=True,
    )
    tok = AutoTokenizer.from_pretrained(backbone_model)
    model = AutoModel.from_pretrained(backbone_model, torch_dtype=torch.float32).to(device).eval()

    out_chunks = []
    batch_size = 16
    n_batches = (len(texts) + batch_size - 1) // batch_size
    with torch.no_grad():
        for bi, i in enumerate(range(0, len(texts), batch_size)):
            batch_started = time.time()
            batch = texts[i : i + batch_size]
            enc = tok(
                batch, padding=True, truncation=True, max_length=1024, return_tensors="pt"
            ).to(device)
            out = model(**enc)
            hidden = out.last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
            out_chunks.append(pooled.cpu().numpy())
            # CPU forward passes on a >1B param model can take minutes per batch
            # with no other output in between, so progress is printed per batch
            # rather than at the end, so a slow run never looks like a hang.
            print(
                f"  batch {bi + 1}/{n_batches} embedded ({len(batch)} prompts, "
                f"{time.time() - batch_started:.1f}s)",
                flush=True,
            )
    return np.concatenate(out_chunks, axis=0)


def head_param_count(in_dim: int, hidden_dim: int, n_agents: int) -> int:
    if hidden_dim <= 0:
        return in_dim * n_agents + n_agents
    return in_dim * hidden_dim + hidden_dim + hidden_dim * n_agents + n_agents


def build_head(theta, in_dim: int, hidden_dim: int, n_agents: int):
    """Unflatten a CMA-ES candidate vector into a callable head(x) -> logits."""
    import numpy as np

    if hidden_dim <= 0:
        w = theta[: in_dim * n_agents].reshape(in_dim, n_agents)
        b = theta[in_dim * n_agents :]

        def head(x):
            return x @ w + b

        return head

    i = 0
    w1 = theta[i : i + in_dim * hidden_dim].reshape(in_dim, hidden_dim)
    i += in_dim * hidden_dim
    b1 = theta[i : i + hidden_dim]
    i += hidden_dim
    w2 = theta[i : i + hidden_dim * n_agents].reshape(hidden_dim, n_agents)
    i += hidden_dim * n_agents
    b2 = theta[i : i + n_agents]

    def head(x):
        h = np.tanh(x @ w1 + b1)
        return h @ w2 + b2

    return head


def train_cma(E_train, R_train, in_dim: int, n_agents: int, cfg: dict):
    import cma
    import numpy as np

    hidden_dim = int(cfg.get("head_hidden_dim") or 0)
    n_params = head_param_count(in_dim, hidden_dim, n_agents)
    seed = int(cfg.get("cma_seed", 42))
    sigma0 = float(cfg.get("cma_sigma0", 0.5))
    generations = int(cfg.get("cma_generations", 60))

    opts = {"seed": seed, "verbose": -9}
    popsize = cfg.get("cma_population_size")
    if popsize:
        opts["popsize"] = int(popsize)

    print(
        f"sep-CMA-ES: {n_params} head parameters, sigma0={sigma0}, generations={generations}, "
        f"popsize={opts.get('popsize', 'auto')}"
    )

    x0 = np.zeros(n_params)
    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

    best_theta, best_score = x0, -1.0
    history = []
    gen = 0
    while not es.stop() and gen < generations:
        population = es.ask()
        neg_scores = []
        for theta in population:
            head = build_head(np.asarray(theta), in_dim, hidden_dim, n_agents)
            picks = np.argmax(head(E_train), axis=1)
            reward = float(R_train[np.arange(len(picks)), picks].mean())
            neg_scores.append(-reward)  # CMA-ES minimizes
        es.tell(population, neg_scores)

        gen_best_idx = int(np.argmin(neg_scores))
        gen_best_reward = -neg_scores[gen_best_idx]
        if gen_best_reward > best_score:
            best_score = gen_best_reward
            best_theta = np.asarray(population[gen_best_idx])
        history.append({"generation": gen, "best_reward": gen_best_reward})
        # This search space is tiny (a few thousand params, a few dozen rows)
        # so every generation is cheap, so all of them are logged rather than
        # every 10th, so a short run (it often stops itself in under 20 gens)
        # never goes silent for its entire duration.
        print(f"  gen {gen}: best_reward_so_far={best_score:.4f}", flush=True)
        gen += 1

    return best_theta, best_score, history, hidden_dim


def evaluate(theta, E, R, in_dim: int, hidden_dim: int, n_agents: int) -> float:
    import numpy as np

    if len(E) == 0:
        return None
    head = build_head(np.asarray(theta), in_dim, hidden_dim, n_agents)
    picks = np.argmax(head(E), axis=1)
    return float(R[np.arange(len(picks)), picks].mean())


def compute_baselines(R) -> dict:
    return {
        "random_agent": float(R.mean()),
        "best_fixed_agent": float(R.mean(axis=0).max()),
        "oracle": float(R.max(axis=1).mean()),
    }


def split_train_test(joined: list, cfg: dict):
    import numpy as np

    test_split_name = cfg.get("test_split") or "test"
    is_test = [row.get("split") == test_split_name for row in joined]
    if not any(is_test):
        # No usable split column, so use a deterministic held-out split instead.
        rng = np.random.RandomState(int(cfg.get("cma_seed", 42)))
        idx = rng.permutation(len(joined))
        n_test = max(1, int(round(0.2 * len(joined))))
        test_idx = set(idx[:n_test].tolist())
        is_test = [i in test_idx for i in range(len(joined))]
        print(
            f"  No rows matched test_split='{test_split_name}'; using a deterministic "
            f"{n_test}/{len(joined)} held-out split instead (seed={cfg.get('cma_seed', 42)})."
        )
    return is_test


def main() -> int:
    import numpy as np

    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CONFIG
    cfg = load_config(config_path)
    model_id = cfg["model_id"]
    print(f"=== Router head training: {model_id} ===")
    started = time.time()

    reward_rows = load_rewards(cfg)
    tasks = load_tasks(cfg)
    joined, missing = join_rows(reward_rows, tasks)
    if missing:
        print(
            f"WARNING: {missing}/{len(reward_rows)} reward rows had no matching prompt "
            f"in the tasks source and were skipped."
        )
    if len(joined) < 10:
        raise RuntimeError(
            f"Only {len(joined)} joined rows have prompt text, too few to train a head. "
            f"Check that tasks_dataset actually covers these task_ids."
        )

    agent_order = cfg.get("agent_order") or joined[0].get("agent_order")
    if not agent_order:
        raise RuntimeError("No agent_order found on the reward rows or in the config.")
    for row in joined:
        if row.get("agent_order") != agent_order:
            raise RuntimeError(
                f"agent_order mismatch on task {row.get('task_id')}, pool_version drift. "
                f"A head trained on one agent order is silently wrong against another."
            )
    n_agents = len(agent_order)
    print(f"Pool ({n_agents} agents): {agent_order}")

    R_all = np.array([row["mean_reward"] for row in joined], dtype=np.float64)
    if R_all.shape[1] != n_agents:
        raise RuntimeError(f"mean_reward width {R_all.shape[1]} != len(agent_order) {n_agents}")

    is_test = np.array(split_train_test(joined, cfg))

    prompts = [row["prompt"] for row in joined]
    E_all = embed_texts(prompts, cfg["model"], cfg.get("device", "cpu"))
    in_dim = E_all.shape[1]

    E_train, R_train = E_all[~is_test], R_all[~is_test]
    E_test, R_test = E_all[is_test], R_all[is_test]
    print(f"Train rows: {len(E_train)}, test rows: {len(E_test)}, embedding dim: {in_dim}")

    theta, train_reward, history, hidden_dim = train_cma(E_train, R_train, in_dim, n_agents, cfg)
    test_reward = evaluate(theta, E_test, R_test, in_dim, hidden_dim, n_agents)
    baseline_source_R = R_test if len(E_test) else R_train
    baselines = compute_baselines(baseline_source_R)

    beats_best_fixed = test_reward is not None and test_reward > baselines["best_fixed_agent"]
    report = {
        "model_id": model_id,
        "backbone_model": cfg["model"],
        "rewards_dataset": cfg["rewards_dataset"],
        "agent_order": agent_order,
        "embedding_dim": int(in_dim),
        "head_hidden_dim": int(hidden_dim),
        "n_head_params": int(head_param_count(in_dim, hidden_dim, n_agents)),
        "n_total": int(len(joined)),
        "n_train": int(len(E_train)),
        "n_test": int(len(E_test)),
        "train_reward": train_reward,
        "test_reward": test_reward,
        "baselines": baselines,
        "beats_best_fixed_agent": bool(beats_best_fixed),
        "cma_generations_run": len(history),
        "history": history,
        "duration_seconds": round(time.time() - started, 1),
    }
    print(json.dumps({k: v for k, v in report.items() if k != "history"}, indent=2))
    if test_reward is not None and not beats_best_fixed:
        print(
            "WARNING: the trained head does NOT beat the best-fixed-agent baseline on the "
            "held-out split. This router is not deployable as-is, more or complementary "
            "agents, or a wider-spread dataset, are needed before it pays."
        )

    out_dir = Path(cfg.get("output_dir", f"train/output/{model_id}"))
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "head_weights.npy", theta)
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    zip_path = out_dir.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(out_dir / "head_weights.npy", "head_weights.npy")
        zf.write(out_dir / "manifest.json", "manifest.json")

    print(f"=== Done: wrote {out_dir}/ and {zip_path} ===")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)
