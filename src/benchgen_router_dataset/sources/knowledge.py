"""Knowledge sources: MMLU (Trinity's own) and MMLU-Pro (10 options, materially harder)."""

from __future__ import annotations

from typing import Any

from ..schemas import Task
from .base import Source, SourceSpec, make_task, register
from .prompts import format_mcq, letter_for

MMLU_SPEC = SourceSpec(
    name="mmlu",
    dataset="cais/mmlu",
    config="all",
    split="test",
    domain="knowledge",
    difficulty="medium",
    license="MIT",
    answer_type="multiple_choice",
    grader="mcq_letter",
    subject_field="subject",
)

MMLU_PRO_SPEC = SourceSpec(
    name="mmlu_pro",
    dataset="TIGER-Lab/MMLU-Pro",
    config=None,
    split="test",
    domain="knowledge",
    difficulty="hard",
    license="CHECK-AT-COLLECTION",
    answer_type="multiple_choice",
    grader="mcq_letter",
    subject_field="category",
    notes="Aggregates several upstreams; the aggregate licence is what binds. Verify on the page.",
)


def _mmlu(spec: SourceSpec, index: int, row: dict[str, Any]) -> Task | None:
    question = (row.get("question") or "").strip()
    options = [str(c) for c in (row.get("choices") or [])]
    if not question or len(options) < 2:
        return None
    letter = letter_for(row.get("answer"), options)
    if letter is None:
        return None
    return make_task(
        spec, index, prompt=format_mcq(question, options), answer=letter,
        subject=row.get("subject"),
    )


def _mmlu_pro(spec: SourceSpec, index: int, row: dict[str, Any]) -> Task | None:
    question = (row.get("question") or "").strip()
    options = [str(c) for c in (row.get("options") or []) if str(c).strip().upper() != "N/A"]
    if not question or len(options) < 2:
        return None
    letter = letter_for(row.get("answer") or row.get("answer_index"), options)
    if letter is None:
        return None
    return make_task(
        spec, index, prompt=format_mcq(question, options), answer=letter,
        subject=row.get("category"),
    )


MMLU = register(Source(MMLU_SPEC, _mmlu))
MMLU_PRO = register(Source(MMLU_PRO_SPEC, _mmlu_pro))
