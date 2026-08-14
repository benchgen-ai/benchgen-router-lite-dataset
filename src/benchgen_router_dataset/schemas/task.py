"""Task pool records."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

AnswerType = Literal["multiple_choice", "numeric", "short_text", "expression"]
Domain = Literal["math", "knowledge", "reasoning", "science", "code", "turkish"]
Difficulty = Literal["easy", "medium", "hard"]
Split = Literal["train", "validation", "test"]
# Trinity trains on four in-distribution tasks and measures zero-shot transfer on four held-out
# ones (§4.3). Held-out sources never appear in train or validation.
TaskGroup = Literal["in_distribution", "held_out"]


class SourceRef(BaseModel):
    """Full provenance. A row without this cannot pass the licence audit."""

    model_config = ConfigDict(extra="forbid")

    dataset: str
    config: str | None = None
    split: str
    index: int
    license: str = Field(description="Licence as observed on the source page, not assumed.")
    license_checked_on: str | None = Field(
        default=None, description="ISO date the licence was actually read."
    )
    subject: str | None = None
    redistributable: bool = Field(
        default=True,
        description="False means publish task_id + rewards only, never the prompt text.",
    )


class Task(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    prompt: str
    answer: str
    answer_type: AnswerType
    grader: str
    domain: Domain
    difficulty: Difficulty
    split: Split
    task_group: TaskGroup = "in_distribution"
    source: SourceRef

    @field_validator("prompt", "answer", "task_id")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("must be non-empty")
        return v

    @field_validator("task_id")
    @classmethod
    def _id_shape(cls, v: str) -> str:
        if v.count("/") != 2:
            raise ValueError("task_id must be '{source}/{split}/{index}'")
        return v
