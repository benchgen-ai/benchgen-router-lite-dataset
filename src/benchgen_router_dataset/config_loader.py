"""Config loading. Configs are JSON on disk and typed models in memory."""

from __future__ import annotations

import json
from pathlib import Path

from .paths import configs_dir
from .schemas import AgentPool, GateSpec, RoleSet


def _load(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"missing config: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_pool(version: str = "v1", root: Path | None = None) -> AgentPool:
    base = root or configs_dir()
    pool = AgentPool.model_validate(_load(base / f"agents.{version}.json"))
    if pool.pool_version != version:
        raise ValueError(
            f"pool_version {pool.pool_version!r} does not match file version {version!r}"
        )
    return pool


def load_roles(version: str = "v1", root: Path | None = None) -> RoleSet:
    base = root or configs_dir()
    return RoleSet.model_validate(_load(base / f"roles.{version}.json"))


def load_gates(version: str = "v1", root: Path | None = None) -> GateSpec:
    base = root or configs_dir()
    return GateSpec.model_validate(_load(base / f"gates.{version}.json"))


def load_collection(version: str = "v1", root: Path | None = None) -> dict:
    base = root or configs_dir()
    return _load(base / f"collection.{version}.json")


def require_verified(pool: AgentPool) -> None:
    """Stage 1 gate. Collecting against an unverified slug produces silent zeros."""
    unresolved = [a.id for a in pool.active if not a.resolved]
    if unresolved:
        raise RuntimeError(
            "agent slots still hold a placeholder slug: "
            + ", ".join(unresolved)
            + " — run scripts/preflight_agents.py discover, then apply"
        )
    unverified = [a.slug for a in pool.active if not a.verified]
    if unverified:
        raise RuntimeError(
            "unverified agent slugs: "
            + ", ".join(unverified)
            + " — run scripts/preflight_agents.py before collecting"
        )
