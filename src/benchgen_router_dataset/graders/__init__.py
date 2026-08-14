"""Grading. Import `grade` and reference graders by name."""

from __future__ import annotations

from .registry import GRADERS, GradeResult, grade, known_graders

__all__ = ["GRADERS", "GradeResult", "grade", "known_graders"]
