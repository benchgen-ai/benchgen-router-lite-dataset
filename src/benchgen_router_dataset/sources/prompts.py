"""Shared prompt shapes. Every agent sees the identical prompt for a task."""

from __future__ import annotations

from collections.abc import Sequence

LETTERS = "ABCDEFGHIJ"

MCQ_INSTRUCTION = (
    "Answer with the letter of the single best option, in the form `Answer: X`. "
    "Do not explain after the answer line."
)
NUMERIC_INSTRUCTION = (
    "Give the final numeric answer in the form `Answer: <number>` on the last line."
)
SHORT_ANSWER_INSTRUCTION = (
    "Give the final answer in the form `Answer: <answer>` on the last line. "
    "Answer as briefly as the question allows."
)
EXPRESSION_INSTRUCTION = (
    "Put the final answer inside \\boxed{} on the last line."
)


def format_mcq(question: str, options: Sequence[str]) -> str:
    if not options or len(options) > len(LETTERS):
        raise ValueError(f"unsupported option count: {len(options)}")
    lines = [f"{LETTERS[i]}. {opt}" for i, opt in enumerate(options)]
    return f"Question: {question.strip()}\n\n" + "\n".join(lines) + f"\n\n{MCQ_INSTRUCTION}"


def format_numeric(question: str) -> str:
    return f"Problem: {question.strip()}\n\n{NUMERIC_INSTRUCTION}"


def format_short_answer(question: str) -> str:
    return f"Question: {question.strip()}\n\n{SHORT_ANSWER_INSTRUCTION}"


def format_expression(question: str) -> str:
    return f"Problem: {question.strip()}\n\n{EXPRESSION_INSTRUCTION}"


def letter_for(answer: object, options: Sequence[str]) -> str | None:
    """Upstreams encode the key as an index, a letter or the option text. Accept all three."""
    if isinstance(answer, bool):
        return None
    if isinstance(answer, int):
        return LETTERS[answer] if 0 <= answer < len(options) else None
    text = str(answer).strip()
    if len(text) == 1 and text.upper() in LETTERS[: len(options)]:
        return text.upper()
    if text.isdigit():
        idx = int(text)
        for candidate in (idx, idx - 1):
            if 0 <= candidate < len(options):
                return LETTERS[candidate]
        return None
    for i, opt in enumerate(options):
        if str(opt).strip() == text:
            return LETTERS[i]
    return None
