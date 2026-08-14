"""Source adapters. Importing this module populates the registry."""

from __future__ import annotations

# mathematics.py is deliberately not called math.py — it would shadow the stdlib module.
from . import (  # noqa: F401  (imported for registration side effect)
    aime,
    knowledge,
    local,
    mathematics,
    reasoning,
    rlpr,
    science,
)
from .base import REGISTRY, LoadStats, Source, SourceSpec, first_field, map_rows
from .loader import SourceUnavailable, iter_rows

__all__ = [
    "REGISTRY",
    "LoadStats",
    "Source",
    "SourceSpec",
    "SourceUnavailable",
    "first_field",
    "iter_rows",
    "map_rows",
]
