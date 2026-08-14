from __future__ import annotations

from benchgen_router_dataset.sources import REGISTRY
from benchgen_router_dataset.sources.prompts import letter_for


def build(name: str, row: dict, index: int = 0):
    source = REGISTRY[name]
    return source.mapper(source.spec, index, row)


def test_gsm8k_takes_the_hash_answer() -> None:
    task = build("gsm8k", {"question": "2+2?", "answer": "reasoning...\n#### 4"})
    assert task is not None
    assert task.answer == "4"
    assert task.domain == "math"
    assert task.task_id == "gsm8k/test/000000"


def test_gsm8k_skips_rows_without_a_final_answer() -> None:
    assert build("gsm8k", {"question": "2+2?", "answer": "no marker"}) is None


def test_math500_maps_level_to_difficulty() -> None:
    easy = build("math500", {"problem": "1+1", "answer": "2", "level": 1})
    hard = build("math500", {"problem": "hard", "answer": r"\frac{1}{2}", "level": 5})
    assert easy is not None and easy.difficulty == "easy"
    assert hard is not None and hard.difficulty == "hard"


def test_mmlu_maps_an_integer_key_to_a_letter() -> None:
    task = build(
        "mmlu",
        {"question": "q", "choices": ["w", "x", "y", "z"], "answer": 2, "subject": "law"},
    )
    assert task is not None
    assert task.answer == "C"
    assert "A. w" in task.prompt
    assert task.source.subject == "law"


def test_mmlu_pro_drops_na_padding() -> None:
    task = build(
        "mmlu_pro",
        {
            "question": "q",
            "options": ["a", "b", "c", "N/A", "N/A"],
            "answer": "C",
            "category": "math",
        },
    )
    assert task is not None
    assert task.answer == "C"
    assert "D." not in task.prompt


def test_arc_reletters_numeric_labels() -> None:
    task = build(
        "arc_challenge",
        {"question": "q", "choices": {"text": ["w", "x", "y", "z"], "label": ["1", "2", "3", "4"]},
         "answerKey": "3"},
    )
    assert task is not None
    assert task.answer == "C"


def test_gpqa_shuffles_options_deterministically() -> None:
    row = {
        "Question": "q",
        "Correct Answer": "right",
        "Incorrect Answer 1": "w1",
        "Incorrect Answer 2": "w2",
        "Incorrect Answer 3": "w3",
    }
    first = build("gpqa_diamond", row, index=5)
    second = build("gpqa_diamond", row, index=5)
    assert first is not None and second is not None
    assert first.prompt == second.prompt
    letter = first.answer
    correct_line = [ln for ln in first.prompt.splitlines() if ln.startswith(f"{letter}. ")]
    assert correct_line == [f"{letter}. right"]


def test_turkish_falls_back_to_short_text_without_options() -> None:
    task = build("benchgen_turkish", {"question": "Baskent neresi?", "answer": "Ankara"})
    assert task is not None
    assert task.answer_type == "short_text"
    assert task.grader == "exact_match_ci"
    assert task.domain == "turkish"


def test_letter_for_accepts_index_letter_and_text() -> None:
    options = ["alpha", "beta", "gamma"]
    assert letter_for(1, options) == "B"
    assert letter_for("B", options) == "B"
    assert letter_for("gamma", options) == "C"
    assert letter_for("zzz", options) is None
