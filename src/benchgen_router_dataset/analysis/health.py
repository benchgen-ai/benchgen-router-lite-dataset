"""Gate evaluation: turn metrics + thresholds into a continue / redesign decision."""

from __future__ import annotations

from dataclasses import dataclass

from ..schemas import GateSpec
from .metrics import HealthMetrics

VERDICT_ORDER = {"continue": 0, "warn": 1, "missing": 2, "stop": 3}


@dataclass
class GateRow:
    label: str
    metric: str
    value: float | None
    continue_at: float
    stop_at: float
    direction: str
    verdict: str
    note: str | None = None


@dataclass
class GateOutcome:
    name: str
    rows: list[GateRow]

    @property
    def decision(self) -> str:
        worst = max((VERDICT_ORDER[r.verdict] for r in self.rows), default=0)
        return {0: "CONTINUE", 1: "CONTINUE WITH CAUTION", 2: "INCOMPLETE", 3: "STOP"}[worst]

    @property
    def blocking(self) -> list[GateRow]:
        return [r for r in self.rows if r.verdict in ("stop", "missing")]


def evaluate(spec: GateSpec, metrics: HealthMetrics) -> GateOutcome:
    values = metrics.as_dict()
    rows = [
        GateRow(
            label=t.label,
            metric=t.metric,
            value=values.get(t.metric),
            continue_at=t.continue_at,
            stop_at=t.stop_at,
            direction=t.direction,
            verdict=t.verdict(values.get(t.metric)),
            note=t.note,
        )
        for t in spec.thresholds
    ]
    return GateOutcome(name=spec.name, rows=rows)


REDESIGN_OPTIONS = (
    "harder tasks (raise the MATH500 / GPQA-Diamond share)",
    "more heterogeneous agents (add a code specialist or a weak-but-cheap model)",
    "cost or latency weighting in the reward so ties break on price",
    "a different domain mix",
)
