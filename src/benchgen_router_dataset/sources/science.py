"""Science: GPQA-Diamond. Gated upstream, so its prompts are marked non-redistributable."""

from __future__ import annotations

import random
from typing import Any

from ..schemas import Task
from .base import Source, SourceSpec, make_task, register
from .prompts import LETTERS, format_mcq

GPQA_SPEC = SourceSpec(
    name="gpqa_diamond",
    dataset="Idavidrein/gpqa",
    config="gpqa_diamond",
    split="train",
    domain="science",
    difficulty="hard",
    license="CC-BY-4.0 (gated)",
    answer_type="multiple_choice",
    grader="mcq_letter",
    task_group="held_out",
    subject_field="subject",
    redistributable=False,
    notes="Gated on HF. Publish task_id + rewards only; users rehydrate prompts themselves.",
)

_DISTRACTOR_FIELDS = ("Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3")


def _gpqa(spec: SourceSpec, index: int, row: dict[str, Any]) -> Task | None:
    question = (row.get("Question") or "").strip()
    correct = str(row.get("Correct Answer") or "").strip()
    distractors = [str(row.get(f) or "").strip() for f in _DISTRACTOR_FIELDS]
    if not question or not correct or not all(distractors):
        return None

    # Option order is shuffled per row with a fixed seed: the raw file always has the correct
    # answer in the same field, which would otherwise make position the answer.
    options = [correct, *distractors]
    random.Random(f"gpqa/{index}").shuffle(options)
    letter = LETTERS[options.index(correct)]

    return make_task(
        spec, index, prompt=format_mcq(question, options), answer=letter,
        subject=row.get("Subdomain") or row.get("High-level domain"),
    )


GPQA_DIAMOND = register(Source(GPQA_SPEC, _gpqa))
