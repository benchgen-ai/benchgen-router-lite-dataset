"""Named graders. Data references a grader by name; grading logic never lives in the data."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from .extract import (
    extract_choice,
    extract_expression,
    extract_number,
    extract_short_text,
)
from .normalize import normalize_latex, normalize_number, normalize_text, strip_markup

NUMERIC_REL_TOL = 1e-6


@dataclass(frozen=True)
class GradeResult:
    extracted: str | None
    correct: bool


GraderFn = Callable[[str, str], GradeResult]


def mcq_letter(response: str, answer: str) -> GradeResult:
    got = extract_choice(response)
    return GradeResult(got, got is not None and got.upper() == answer.strip().upper())


def exact_match_ci(response: str, answer: str) -> GradeResult:
    got = extract_short_text(response)
    if got is None:
        return GradeResult(None, False)
    return GradeResult(got, normalize_text(got) == normalize_text(answer))


def numeric_match(response: str, answer: str) -> GradeResult:
    got = extract_number(response)
    if got is None:
        return GradeResult(None, False)
    lhs, rhs = normalize_number(got), normalize_number(answer)
    if lhs is None or rhs is None:
        return GradeResult(got, normalize_text(got) == normalize_text(answer))
    return GradeResult(got, math.isclose(lhs, rhs, rel_tol=NUMERIC_REL_TOL, abs_tol=1e-9))


def math_expression(response: str, answer: str) -> GradeResult:
    got = extract_expression(response)
    if got is None:
        return GradeResult(None, False)
    if normalize_latex(got) == normalize_latex(answer):
        return GradeResult(got, True)
    lhs, rhs = normalize_number(got), normalize_number(answer)
    if lhs is not None and rhs is not None:
        return GradeResult(got, math.isclose(lhs, rhs, rel_tol=NUMERIC_REL_TOL, abs_tol=1e-9))
    return GradeResult(got, False)


def verifier_judgment(response: str, answer: str) -> GradeResult:
    """Unparseable verifier output is REVISE, never ACCEPT.

    A coordinator that can be stopped early by malformed text will learn to emit malformed text.
    """
    body = strip_markup(response).strip().upper()
    got = "ACCEPT" if body.startswith("ACCEPT") else "REVISE"
    return GradeResult(got, got == answer.strip().upper())


GRADERS: dict[str, GraderFn] = {
    "mcq_letter": mcq_letter,
    "exact_match_ci": exact_match_ci,
    "numeric_match": numeric_match,
    "math_expression": math_expression,
    "verifier_judgment": verifier_judgment,
}


def grade(grader_name: str, response: str | None, answer: str) -> GradeResult:
    if grader_name not in GRADERS:
        raise KeyError(f"unknown grader {grader_name!r}; known: {sorted(GRADERS)}")
    if response is None or not response.strip():
        return GradeResult(None, False)
    return GRADERS[grader_name](response, answer)


def known_graders() -> list[str]:
    return sorted(GRADERS)
