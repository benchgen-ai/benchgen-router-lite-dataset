"""Answer extraction. Ordered from most explicit signal to weakest fallback."""

from __future__ import annotations

import re

from .normalize import normalize_number, strip_markup

_ANSWER_TAG_RE = re.compile(r"<answer>\s*(.+?)\s*</answer>", re.IGNORECASE | re.DOTALL)
# Either a separator or the word "is" must follow the label, otherwise "answer the following"
# in an echoed prompt would match and swallow the question.
_ANSWER_LABEL_RE = re.compile(
    r"(?:final\s+answer|answer|cevap|sonu[çc])\s*(?:\s+is\b|\s*[:\-])\s*(.+)", re.IGNORECASE
)
_BOXED_RE = re.compile(r"\\boxed\s*\{")
_CHOICE_RE = re.compile(r"\b([A-J])\b")
_NUMBER_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")
_GSM8K_RE = re.compile(r"####\s*(-?[\d,.]+)")


def extract_boxed(text: str) -> str | None:
    """Brace-balanced \\boxed{...}; a regex alone breaks on nested braces like \\frac."""
    match = _BOXED_RE.search(text)
    if not match:
        return None
    depth, start = 1, match.end()
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i].strip()
    return None


def extract_tagged(text: str) -> str | None:
    match = _ANSWER_TAG_RE.search(text)
    return match.group(1).strip() if match else None


def extract_labelled(text: str) -> str | None:
    matches = _ANSWER_LABEL_RE.findall(text)
    if not matches:
        return None
    # A reply ending in a bare "Answer:" matches but carries no line to take.
    lines = matches[-1].strip().splitlines()
    return lines[0].strip() if lines else None


def extract_choice(text: str, valid: str = "ABCDEFGHIJ") -> str | None:
    body = strip_markup(text)
    for candidate in (extract_tagged(body), extract_boxed(body), extract_labelled(body)):
        if candidate:
            found = _CHOICE_RE.search(candidate.upper())
            if found and found.group(1) in valid:
                return found.group(1)
    tail = body.strip().splitlines()
    for line in reversed(tail[-3:] if tail else []):
        # Last standalone letter on the line: a leading "I" in "I think the answer is E" would
        # otherwise be extracted as the choice.
        found = [m for m in _CHOICE_RE.findall(line.upper()) if m in valid]
        if found:
            return found[-1]
    return None


def extract_number(text: str) -> str | None:
    body = strip_markup(text)
    gsm = _GSM8K_RE.search(body)
    if gsm:
        return gsm.group(1).strip()
    for candidate in (extract_tagged(body), extract_boxed(body), extract_labelled(body)):
        if candidate:
            nums = _NUMBER_RE.findall(candidate)
            if nums:
                return nums[-1]
            if normalize_number(candidate) is not None:
                return candidate.strip()
    nums = _NUMBER_RE.findall(body)
    return nums[-1] if nums else None


def extract_expression(text: str) -> str | None:
    body = strip_markup(text)
    for candidate in (extract_boxed(body), extract_tagged(body), extract_labelled(body)):
        if candidate:
            return candidate
    lines = [ln.strip() for ln in body.strip().splitlines() if ln.strip()]
    return lines[-1] if lines else None


def extract_short_text(text: str) -> str | None:
    body = strip_markup(text)
    for candidate in (extract_tagged(body), extract_boxed(body), extract_labelled(body)):
        if candidate:
            return candidate
    lines = [ln.strip() for ln in body.strip().splitlines() if ln.strip()]
    return lines[-1] if lines else None
