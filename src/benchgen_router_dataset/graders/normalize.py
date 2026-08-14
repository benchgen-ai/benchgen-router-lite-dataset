"""Text normalisation shared by every grader. Pure functions, no I/O."""

from __future__ import annotations

import re
import unicodedata

_MARKUP_RE = re.compile(r"</?(?:delegate|answer|think|reasoning)[^>]*>", re.IGNORECASE)
_WS_RE = re.compile(r"\s+")
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)

_LATEX_WRAPPERS = (
    (re.compile(r"\\left|\\right"), ""),
    (re.compile(r"\\!|\\,|\\;|\\ "), ""),
    (re.compile(r"\\text\{([^}]*)\}"), r"\1"),
    (re.compile(r"\\mathrm\{([^}]*)\}"), r"\1"),
    (re.compile(r"\\dfrac|\\tfrac"), r"\\frac"),
)


def strip_markup(text: str) -> str:
    """Remove tool/answer tags before any fallback extraction.

    Without this a last-number fallback happily "extracts" 731 from a model slug like
    `deepseek-v4-flash-0731` that appeared inside a delegate tag.
    """
    text = _THINK_BLOCK_RE.sub(" ", text)
    return _MARKUP_RE.sub(" ", text)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u2212", "-").replace("\u00a0", " ")
    text = _WS_RE.sub(" ", text).strip().strip(".").strip()
    return text.casefold()


def normalize_number(text: str) -> float | None:
    """Parse a human-written number: commas, currency, percent, trailing punctuation."""
    cleaned = text.strip().strip("$").replace(",", "").replace("%", "").rstrip(".")
    cleaned = cleaned.replace("\u2212", "-")
    if cleaned.endswith(")") and cleaned.startswith("("):
        cleaned = "-" + cleaned[1:-1]
    try:
        return float(cleaned)
    except ValueError:
        return None


def normalize_latex(text: str) -> str:
    """Collapse cosmetic LaTeX differences so `\\dfrac{1}{2}` == `\\frac{1}{2}`."""
    out = text.strip().strip("$").strip()
    for pattern, repl in _LATEX_WRAPPERS:
        out = pattern.sub(repl, out)
    out = out.replace("^{\\circ}", "").replace("\\%", "")
    out = _WS_RE.sub("", out)
    return out.casefold()
