"""Analysis: metrics, gates, baselines, reports."""

from __future__ import annotations

from .baselines import Baselines, compute_baselines, route_accuracy, routed_reward
from .health import GateOutcome, GateRow, evaluate
from .metrics import HealthMetrics, compute_health
from .report import render_baselines, render_health
from .rer import AccuracyMatrix, build_matrix, relative_error_reduction

__all__ = [
    "AccuracyMatrix",
    "Baselines",
    "GateOutcome",
    "GateRow",
    "HealthMetrics",
    "build_matrix",
    "compute_baselines",
    "compute_health",
    "evaluate",
    "relative_error_reduction",
    "render_baselines",
    "render_health",
    "route_accuracy",
    "routed_reward",
]
