"""Pydantic record schemas — the single source of truth for every artifact."""

from __future__ import annotations

from .agents import UNRESOLVED_SLUG, AgentCard, AgentPool, CallProtocol, Role, RoleSet
from .gates import GateSpec, GateThreshold
from .rewards import CallRecord, RewardRecord, RoleRewardRecord
from .task import SourceRef, Task

__all__ = [
    "UNRESOLVED_SLUG",
    "AgentCard",
    "AgentPool",
    "CallProtocol",
    "CallRecord",
    "GateSpec",
    "GateThreshold",
    "RewardRecord",
    "Role",
    "RoleRewardRecord",
    "RoleSet",
    "SourceRef",
    "Task",
]
