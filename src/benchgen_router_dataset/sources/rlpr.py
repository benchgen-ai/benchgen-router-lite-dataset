"""RLPR — one of Trinity's four in-distribution training tasks (§4.1).

77k general-domain (deliberately non-mathematical) reasoning prompts derived from WebInstruct,
with a reference answer per row. The paper reports RLPR scores in the 0.28–0.41 band, i.e. it is
the hardest of its four training tasks and therefore the one with the most routing headroom.
"""

from __future__ import annotations

from typing import Any

from ..schemas import Task
from .base import Source, SourceSpec, first_field, make_task, register
from .prompts import format_short_answer

RLPR_SPEC = SourceSpec(
    name="rlpr",
    dataset="openbmb/RLPR-Train-Dataset",
    config=None,
    split="train",
    domain="reasoning",
    difficulty="hard",
    license="Apache-2.0",
    answer_type="short_text",
    grader="exact_match_ci",
    task_group="in_distribution",
    subject_field="ability",
    notes=(
        "Answers are free-form short strings ('2.74%', 'Business'). RLPR itself uses a "
        "verifier-free probability reward; we grade by normalised string match, which is "
        "stricter and will understate every agent equally."
    ),
)

_DIFFICULTY = {
    "primary school": "easy",
    "junior high school": "easy",
    "senior high school": "medium",
    "university": "hard",
    "graduate": "hard",
    "phd": "hard",
}


def _user_message(prompt: Any) -> str:
    """`prompt` is a chat list; the system turn only states the <think>/<answer> format."""
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        for turn in reversed(prompt):
            if isinstance(turn, dict) and turn.get("role") == "user":
                return str(turn.get("content") or "")
    return ""


def _rlpr(spec: SourceSpec, index: int, row: dict[str, Any]) -> Task | None:
    question = _user_message(row.get("prompt")).strip()
    reward_model = row.get("reward_model") or {}
    answer = str(first_field(reward_model, ["ground_truth", "answer"]) or "").strip()
    if not question or not answer:
        return None

    extra = row.get("extra_info") or {}
    difficulty = _DIFFICULTY.get(str(extra.get("difficulty") or "").strip().lower())
    return make_task(
        spec,
        index,
        prompt=format_short_answer(question),
        answer=answer,
        subject=row.get("ability") or extra.get("category"),
        difficulty=difficulty,
    )


RLPR = register(Source(RLPR_SPEC, _rlpr))
