"""Canonical locations. Every script resolves paths through here, never by string."""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    """Repo root, overridable with BRD_ROOT so tests can run against a temp tree."""
    env = os.environ.get("BRD_ROOT")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parents[2]


def configs_dir() -> Path:
    return repo_root() / "configs"


def data_dir() -> Path:
    return repo_root() / "data"


def reports_dir() -> Path:
    return repo_root() / "reports"


def tasks_path(version: str = "v1") -> Path:
    return data_dir() / f"tasks.{version}.jsonl"


def rewards_path(version: str = "v1") -> Path:
    return data_dir() / f"rewards.{version}.jsonl"


def role_rewards_path(version: str = "v1") -> Path:
    return data_dir() / f"role_rewards.{version}.jsonl"


def preflight_path(pool_version: str = "v1") -> Path:
    """Measured pre-flight stats. Data, not a report: latencies cannot reproduce byte for byte."""
    return data_dir() / f"preflight.{pool_version}.json"


def ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
