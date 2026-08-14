from __future__ import annotations

import pytest

from benchgen_router_dataset.graders import grade


@pytest.mark.parametrize(
    ("response", "answer", "expected"),
    [
        ("Answer: C", "C", True),
        ("The reasoning is long.\nAnswer: b", "B", True),
        ("Answer: D", "C", False),
        ("<answer>A</answer>", "A", True),
        ("I think the answer is E.", "E", True),
    ],
)
def test_mcq_letter(response: str, answer: str, expected: bool) -> None:
    assert grade("mcq_letter", response, answer).correct is expected


@pytest.mark.parametrize(
    ("response", "answer", "expected"),
    [
        ("Answer: 42", "42", True),
        ("Answer: 1,234", "1234", True),
        ("... #### 18", "18", True),
        ("Answer: $3.50", "3.5", True),
        ("Answer: 41", "42", False),
    ],
)
def test_numeric_match(response: str, answer: str, expected: bool) -> None:
    assert grade("numeric_match", response, answer).correct is expected


def test_numeric_ignores_model_slug_digits() -> None:
    """The lenient last-number fallback once 'extracted' 731 from a model slug."""
    response = '<delegate model="deepseek/deepseek-v4-flash-0731">Answer: 12</delegate>'
    result = grade("numeric_match", response, "12")
    assert result.extracted == "12"
    assert result.correct is True


@pytest.mark.parametrize(
    ("response", "answer", "expected"),
    [
        (r"So \boxed{\frac{1}{2}}", r"\frac{1}{2}", True),
        (r"\boxed{\dfrac{1}{2}}", r"\frac{1}{2}", True),
        (r"\boxed{\frac{x}{y+1}}", r"\frac{x}{y+1}", True),
        (r"\boxed{7}", "7.0", True),
        (r"\boxed{3}", "4", False),
    ],
)
def test_math_expression(response: str, answer: str, expected: bool) -> None:
    assert grade("math_expression", response, answer).correct is expected


def test_boxed_handles_nested_braces() -> None:
    result = grade("math_expression", r"\boxed{\frac{a}{b}}", r"\frac{a}{b}")
    assert result.extracted == r"\frac{a}{b}"


def test_verifier_unparseable_is_revise_never_accept() -> None:
    """A coordinator that can be stopped by malformed text will learn to emit malformed text."""
    assert grade("verifier_judgment", "hmm, looks fine to me", "ACCEPT").extracted == "REVISE"
    assert grade("verifier_judgment", "ACCEPT - complete", "ACCEPT").correct is True
    assert grade("verifier_judgment", "REVISE - wrong sign", "REVISE").correct is True


def test_empty_response_is_never_correct() -> None:
    assert grade("mcq_letter", "", "A").correct is False
    assert grade("numeric_match", None, "1").correct is False


def test_answer_label_with_nothing_after_it_does_not_crash() -> None:
    """A reply of bare "Answer:" matches the label but has no line to take; it killed a run."""
    for response in ("Answer:", "Answer: ", "Final answer:\n"):
        assert grade("numeric_match", response, "42").correct is False
        assert grade("mcq_letter", response, "A").correct is False


def test_unknown_grader_raises() -> None:
    with pytest.raises(KeyError):
        grade("no_such_grader", "x", "y")
