"""Stage 5: export the publishable projection of both datasets, then upload.

    python pipeline/05_publish.py           # export to data/publish and data/publish_tasks
    python pipeline/05_publish.py --push    # export, then upload (private by default)

Question text and raw completions are never exported. Upstream licences differ per source and
provider terms on republishing completions were never cleared, so the safe release is rewards
plus metadata; `task_id` lets a consumer rehydrate the prompt from upstream themselves.

Two independent safeguards enforce that. `PublicRewardRow` has no prompt or completion field and
forbids extra fields, so one cannot be added by accident, and the finished export is then scanned
for anything that looks like question text before a single byte is uploaded.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import load_env  # noqa: E402

from benchgen_router_dataset.config_loader import load_pool  # noqa: E402
from benchgen_router_dataset.io_jsonl import read_models, write_jsonl  # noqa: E402
from benchgen_router_dataset.paths import data_dir, rewards_path, tasks_path  # noqa: E402
from benchgen_router_dataset.publish import (  # noqa: E402
    agent_manifest,
    build_card,
    public_rows,
)
from benchgen_router_dataset.publish.tasks_card import build_tasks_card  # noqa: E402
from benchgen_router_dataset.schemas import RewardRecord, Task  # noqa: E402

VERSION = "v1"
REWARDS = rewards_path(f"{VERSION}-pilot")
REWARDS_DIR = data_dir() / "publish"
TASKS_DIR = data_dir() / "publish_tasks"
REWARDS_REPO = "benchgen/router-pilot"
TASKS_REPO = "benchgen/router-pilot-tasks"

# Free-form prose and table answers. Our normalised string-match grader scores genuinely correct
# answers as wrong, and a label that cannot be trusted is worse than no label.
UNGRADEABLE = {
    "rlpr": "Answers are free-form prose and tables. Our string-match grader scores correct "
    "answers as wrong, so these labels are not trustworthy."
}
LEAK_MARKERS = ('"prompt"', '"raw_response"', '"answer"')


def assert_no_prompt_text(out_dir: Path) -> None:
    """Cheap guard against a future change re-introducing prompts into the export."""
    for path in out_dir.iterdir():
        if path.suffix not in {".json", ".jsonl"}:
            continue
        body = path.read_text(encoding="utf-8")
        if any(marker in body for marker in LEAK_MARKERS):
            raise SystemExit(f"refusing to publish: {path.name} appears to contain question text")


def export() -> int:
    records = read_models(REWARDS, RewardRecord)
    tasks = read_models(tasks_path(VERSION), Task)
    by_id = {t.task_id: t for t in tasks}
    pool = load_pool(VERSION)

    rows = public_rows(records, by_id)
    kept = [r for r in rows if not r.task_id.startswith(tuple(UNGRADEABLE))]
    if not kept:
        raise SystemExit("every row was excluded")

    REWARDS_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(REWARDS_DIR / "rewards.jsonl", kept)
    # Without this manifest the reward vectors are just numbers: it says which model produced
    # each column, at what price, and when it was last verified.
    (REWARDS_DIR / "agents.json").write_text(
        json.dumps(agent_manifest(pool), indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    (REWARDS_DIR / "README.md").write_text(
        build_card(kept, pool, "BenchGen Router Pilot: per-question agent rewards", UNGRADEABLE),
        encoding="utf-8",
        newline="\n",
    )
    assert_no_prompt_text(REWARDS_DIR)
    print(f"exported {len(kept)} reward rows to {REWARDS_DIR} ({len(rows) - len(kept)} excluded)")

    TASKS_DIR.mkdir(parents=True, exist_ok=True)
    (TASKS_DIR / "README.md").write_text(
        build_tasks_card(tasks, "BenchGen Router Pilot: task pool", REWARDS_REPO, len(kept)),
        encoding="utf-8",
        newline="\n",
    )
    print(f"wrote the card for {len(tasks)} tasks to {TASKS_DIR / 'README.md'}")
    return 0


def push(public: bool) -> int:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise SystemExit("HF_TOKEN is not set")
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(
        repo_id=REWARDS_REPO, repo_type="dataset", private=not public, exist_ok=True
    )
    api.upload_folder(
        repo_id=REWARDS_REPO, repo_type="dataset", folder_path=str(REWARDS_DIR)
    )
    api.upload_file(
        repo_id=TASKS_REPO,
        repo_type="dataset",
        path_or_fileobj=str(TASKS_DIR / "README.md"),
        path_in_repo="README.md",
    )
    print(f"pushed {REWARDS_REPO} and {TASKS_REPO} ({'public' if public else 'private'})")
    return 0


def main() -> int:
    load_env()
    code = export()
    if code or "--push" not in sys.argv:
        return code
    # Private by default: a release is reviewed on the Hub before it is made public.
    return push(public="--public" in sys.argv)


if __name__ == "__main__":
    raise SystemExit(main())
