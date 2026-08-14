"""Health-gate specification. Thresholds live in config so the gate is data-driven."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Direction = Literal["min", "max"]


class GateThreshold(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric: str
    label: str
    direction: Direction = Field(
        description="'min' = continue when value >= continue_at; 'max' = when value <= continue_at"
    )
    continue_at: float
    stop_at: float
    note: str | None = None

    def verdict(self, value: float | None) -> str:
        if value is None:
            return "missing"
        if self.direction == "min":
            if value >= self.continue_at:
                return "continue"
            return "stop" if value <= self.stop_at else "warn"
        if value <= self.continue_at:
            return "continue"
        return "stop" if value >= self.stop_at else "warn"


class GateSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gate_version: str
    name: str
    thresholds: list[GateThreshold]
