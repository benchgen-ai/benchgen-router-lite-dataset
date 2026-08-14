"""Math sources: MATH500 (hard, Trinity's own) and GSM8K (easy, the difficulty contrast)."""

from __future__ import annotations

from typing import Any

from ..schemas import Task
from .base import Source, SourceSpec, make_task, register
from .prompts import format_expression, format_numeric

MATH500_SPEC = SourceSpec(
    name="math500",
    dataset="HuggingFaceH4/MATH-500",
    config=None,
    split="test",
    domain="math",
    difficulty="hard",
    license="MIT",
    answer_type="expression",
    grader="math_expression",
    notes="Trinity trains on MATH500; keeping it makes our numbers comparable to the paper.",
)

GSM8K_SPEC = SourceSpec(
    name="gsm8k",
    dataset="openai/gsm8k",
    config="main",
    split="test",
    domain="math",
    difficulty="easy",
    license="MIT",
    answer_type="numeric",
    grader="numeric_match",
    notes="Present as the easy end of the difficulty axis, not for volume.",
)

_LEVEL_TO_DIFFICULTY = {1: "easy", 2: "easy", 3: "medium", 4: "hard", 5: "hard"}


def _math500(spec: SourceSpec, index: int, row: dict[str, Any]) -> Task | None:
    problem = (row.get("problem") or "").strip()
    answer = str(row.get("answer") or "").strip()
    if not problem or not answer:
        return None
    level = row.get("level")
    difficulty = _LEVEL_TO_DIFFICULTY.get(int(level)) if isinstance(level, (int, float)) else None
    return make_task(
        spec,
        index,
        prompt=format_expression(problem),
        answer=answer,
        subject=row.get("subject"),
        difficulty=difficulty,
    )


def _gsm8k(spec: SourceSpec, index: int, row: dict[str, Any]) -> Task | None:
    question = (row.get("question") or "").strip()
    raw = str(row.get("answer") or "")
    if "####" not in raw:
        return None
    answer = raw.split("####")[-1].strip().replace(",", "")
    if not question or not answer:
        return None
    return make_task(spec, index, prompt=format_numeric(question), answer=answer)


MATH500 = register(Source(MATH500_SPEC, _math500))
GSM8K = register(Source(GSM8K_SPEC, _gsm8k))
