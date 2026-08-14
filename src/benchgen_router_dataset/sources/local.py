"""BenchGen Turkish benchmarks — the differentiator, loaded from a local export.

Expected at `data/raw/benchgen_turkish.jsonl`, one object per line:
    {"question": "...", "options": ["...", ...] | null, "answer": "...", "subject": "..."}
Kept local because the bundles live in the internal `all-turkish-benchmarks` repo; publication
needs internal sign-off (docs/04-publishing.md).
"""

from __future__ import annotations

from typing import Any

from ..schemas import Task
from .base import Source, SourceSpec, make_task, register
from .prompts import format_mcq, format_numeric, letter_for

TURKISH_SPEC = SourceSpec(
    name="benchgen_turkish",
    dataset="local:data/raw/benchgen_turkish.jsonl",
    config=None,
    split="test",
    domain="turkish",
    difficulty="medium",
    license="INTERNAL-PENDING-APPROVAL",
    answer_type="multiple_choice",
    grader="mcq_letter",
    task_group="held_out",
    redistributable=False,
    notes="No public routing dataset covers Turkish. Confirm ownership before publishing text.",
)


def _turkish(spec: SourceSpec, index: int, row: dict[str, Any]) -> Task | None:
    question = (row.get("question") or row.get("soru") or "").strip()
    answer = str(row.get("answer") or row.get("cevap") or "").strip()
    if not question or not answer:
        return None

    options = [str(o) for o in (row.get("options") or row.get("secenekler") or [])]
    if options:
        letter = letter_for(answer, options)
        if letter is None:
            return None
        task = make_task(
            spec, index, prompt=format_mcq(question, options), answer=letter,
            subject=row.get("subject"),
            difficulty=row.get("difficulty"),
        )
        return task

    task = make_task(
        spec, index, prompt=format_numeric(question), answer=answer,
        subject=row.get("subject"), difficulty=row.get("difficulty"),
    )
    return task.model_copy(update={"answer_type": "short_text", "grader": "exact_match_ci"})


BENCHGEN_TURKISH = register(Source(TURKISH_SPEC, _turkish))
