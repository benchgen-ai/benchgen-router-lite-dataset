"""AIME 2025 — one of Trinity's four held-out zero-shot transfer tasks (§4.3).

30 problems, integer answers in 0–999, split across AIME I and II. Small, so it never dominates
the pool, but it is where the difficulty ceiling is.
"""

from __future__ import annotations

from typing import Any

from ..schemas import Task
from .base import Source, SourceSpec, first_field, make_task, register
from .prompts import format_numeric

_QUESTION_FIELDS = ("question", "problem", "Question", "Problem")
_ANSWER_FIELDS = ("answer", "Answer", "solution", "final_answer")


def _spec(name: str, config: str) -> SourceSpec:
    return SourceSpec(
        name=name,
        dataset="opencompass/AIME2025",
        config=config,
        split="test",
        domain="math",
        difficulty="hard",
        license="MIT",
        answer_type="numeric",
        grader="numeric_match",
        task_group="held_out",
        notes="Held out. Never assigned to train or validation.",
    )


def _aime(spec: SourceSpec, index: int, row: dict[str, Any]) -> Task | None:
    question = str(first_field(row, _QUESTION_FIELDS) or "").strip()
    answer = str(first_field(row, _ANSWER_FIELDS) or "").strip()
    if not question or not answer:
        return None
    return make_task(spec, index, prompt=format_numeric(question), answer=answer)


AIME2025_I = register(Source(_spec("aime2025_i", "AIME2025-I"), _aime))
AIME2025_II = register(Source(_spec("aime2025_ii", "AIME2025-II"), _aime))
