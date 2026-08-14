"""Reasoning: ARC-Challenge.

CC-BY-SA is share-alike and can propagate to the whole release. Decide before collection
whether to accept share-alike, ship ARC as its own config, or drop it (see docs/04-publishing.md).
"""

from __future__ import annotations

from typing import Any

from ..schemas import Task
from .base import Source, SourceSpec, make_task, register
from .prompts import format_mcq, letter_for

ARC_SPEC = SourceSpec(
    name="arc_challenge",
    dataset="allenai/ai2_arc",
    config="ARC-Challenge",
    split="test",
    domain="reasoning",
    difficulty="medium",
    license="CC-BY-SA-4.0",
    answer_type="multiple_choice",
    grader="mcq_letter",
    notes="Share-alike. Isolate as its own config if the release licence must stay permissive.",
)


def _arc(spec: SourceSpec, index: int, row: dict[str, Any]) -> Task | None:
    question = (row.get("question") or "").strip()
    choices = row.get("choices") or {}
    texts = [str(t) for t in (choices.get("text") or [])]
    labels = [str(v) for v in (choices.get("label") or [])]
    if not question or len(texts) < 2 or len(texts) != len(labels):
        return None

    key = str(row.get("answerKey") or "").strip()
    if key in labels:
        # ARC labels are sometimes 1..4; re-letter so the grader only ever sees A..J.
        answer = letter_for(labels.index(key), texts)
    else:
        answer = letter_for(key, texts)
    if answer is None:
        return None
    return make_task(spec, index, prompt=format_mcq(question, texts), answer=answer)


ARC_CHALLENGE = register(Source(ARC_SPEC, _arc))
