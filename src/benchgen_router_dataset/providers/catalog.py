"""Live OpenRouter catalogue: discovery + a transparent, complementarity-first pool proposal."""

from __future__ import annotations

import os
from dataclasses import dataclass

import httpx

from ..schemas.agents import DIRECT_MARKER, PAIRED_SLOTS, REASONING_MARKER
from .openrouter import DEFAULT_BASE_URL

CODE_HINTS = ("coder", "code", "devstral", "codestral", "swe")
OPEN_WEIGHT_VENDORS = ("qwen", "meta-llama", "mistralai", "deepseek", "google/gemma", "nvidia")
DISTILL_HINTS = ("r1", "distill", "qwq", "thinking", "reason")
# `:batch` is asynchronous, `:free` is rate-limited and silently truncates, the rest change
# routing. Any of them makes E(D, M) measure the variant rather than the model.
UNSTABLE_VARIANTS = ("batch", "free", "extended", "online", "nitro", "floor", "beta")


@dataclass(slots=True)
class CatalogEntry:
    slug: str
    name: str
    description: str
    context_length: int
    prompt_usd_per_1m: float
    output_usd_per_1m: float
    supports_reasoning: bool

    @property
    def vendor(self) -> str:
        return self.slug.split("/", 1)[0]

    @property
    def is_free(self) -> bool:
        """Also true for `openrouter/auto`, which prices at a negative sentinel."""
        return self.output_usd_per_1m <= 0.0

    @property
    def is_meta_router(self) -> bool:
        """A router masquerading as a model. Including one in a routing dataset is circular."""
        return self.vendor == "openrouter"

    @property
    def looks_like_coder(self) -> bool:
        blob = f"{self.slug} {self.name}".lower()
        return any(h in blob for h in CODE_HINTS)

    @property
    def looks_like_distill(self) -> bool:
        blob = f"{self.slug} {self.name}".lower()
        return any(h in blob for h in DISTILL_HINTS)

    @property
    def is_open_weight(self) -> bool:
        return any(self.slug.lower().startswith(v) for v in OPEN_WEIGHT_VENDORS)

    @property
    def is_pinned(self) -> bool:
        """A slug that will still mean the same model when the dataset is republished."""
        slug = self.slug.lower()
        variant = slug.split(":", 1)[1] if ":" in slug else ""
        return (
            not slug.startswith("~")  # floating alias; also hides the real vendor
            and "latest" not in slug
            and variant not in UNSTABLE_VARIANTS
        )


def fetch_catalog(timeout_s: float = 60.0) -> list[CatalogEntry]:
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    resp = httpx.get(f"{DEFAULT_BASE_URL}/models", headers=headers, timeout=timeout_s)
    resp.raise_for_status()
    return [_entry(raw) for raw in resp.json().get("data", [])]


def _entry(raw: dict) -> CatalogEntry:
    pricing = raw.get("pricing") or {}
    return CatalogEntry(
        slug=raw.get("id", ""),
        name=raw.get("name", ""),
        description=(raw.get("description") or "")[:400],
        context_length=int(raw.get("context_length") or 0),
        prompt_usd_per_1m=_per_million(pricing.get("prompt")),
        output_usd_per_1m=_per_million(pricing.get("completion")),
        supports_reasoning="reasoning" in (raw.get("supported_parameters") or []),
    )


def _per_million(value: object) -> float:
    try:
        # Rounded: the API sends per-token strings, and the raw product writes binary noise
        # like 2.1500000000000004 straight into the agent config.
        return round(float(value) * 1_000_000, 6)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


SLOT_ORDER = (
    "frontier_a",
    "frontier_b",
    "frontier_c",
    "open_distill_reasoning",
    "open_mid",
    "open_direct",
    "open_reasoning",
    "open_cheap_reasoning",
    "open_coder",
)
FRONTIER_SLOTS = ("frontier_a", "frontier_b", "frontier_c")


def propose_pool(
    catalog: list[CatalogEntry],
    max_output_usd: float = 5.0,
    max_frontier_output_usd: float = 15.0,
) -> dict[str, list[CatalogEntry]]:
    """Rank candidates per slot of the seven-slot pool in `configs/agents.v1.json`.

    Complementarity, not leaderboard rank: seven excellent similar models make ties explode
    and leave the router nowhere to win.
    """
    usable = [
        c
        for c in catalog
        if c.slug
        and c.context_length >= 16384
        and not c.is_free
        and not c.is_meta_router
        and c.is_pinned
    ]
    open_usable = [c for c in usable if c.is_open_weight and c.output_usd_per_1m <= max_output_usd]
    closed_usable = [
        c
        for c in usable
        if not c.is_open_weight
        and c.output_usd_per_1m <= max_frontier_output_usd
        and c.context_length >= 128_000
    ]

    # Price is the only tier signal the catalogue exposes, so frontier slots rank it descending.
    frontier = sorted(closed_usable, key=lambda c: (-c.output_usd_per_1m, -c.context_length))
    slots: dict[str, list[CatalogEntry]] = {slot: frontier for slot in FRONTIER_SLOTS}
    slots["open_distill_reasoning"] = sorted(
        (c for c in open_usable if c.supports_reasoning and c.looks_like_distill),
        key=lambda c: (c.output_usd_per_1m, -c.context_length),
    )
    slots["open_mid"] = sorted(
        (c for c in open_usable if not c.supports_reasoning and not c.looks_like_coder),
        key=lambda c: (c.output_usd_per_1m, -c.context_length),
    )
    slots["open_cheap_reasoning"] = sorted(
        (c for c in usable if c.supports_reasoning and c.output_usd_per_1m <= max_output_usd),
        key=lambda c: (c.output_usd_per_1m, -c.context_length),
    )
    slots["open_coder"] = sorted(
        (c for c in usable if c.looks_like_coder and c.output_usd_per_1m <= max_output_usd),
        key=lambda c: (c.output_usd_per_1m, -c.context_length),
    )
    paired = _mode_pairs(open_usable)
    slots["open_direct"] = [direct for direct, _ in paired]
    slots["open_reasoning"] = [reasoning for _, reasoning in paired]
    return {slot: slots[slot][:8] for slot in SLOT_ORDER}


def _mode_pairs(catalog: list[CatalogEntry]) -> list[tuple[CatalogEntry, CatalogEntry]]:
    """Sibling slugs of one base model: `...-instruct-2507` against `...-thinking-2507`.

    A single hybrid slug is not usable here. `reasoning: {enabled: false}` is advisory and at
    least one provider ignores it, which would silently make both slots the same agent.
    """
    by_stem: dict[str, dict[str, CatalogEntry]] = {}
    for entry in catalog:
        for marker, mode in ((DIRECT_MARKER, "direct"), (REASONING_MARKER, "reasoning")):
            if marker in entry.slug:
                by_stem.setdefault(entry.slug.replace(marker, ""), {})[mode] = entry
    pairs = [
        (modes["direct"], modes["reasoning"])
        for modes in by_stem.values()
        if "direct" in modes and "reasoning" in modes
    ]
    pairs.sort(key=lambda p: (p[0].output_usd_per_1m + p[1].output_usd_per_1m, p[0].slug))
    return pairs


def distinct_vendor_pick(proposals: dict[str, list[CatalogEntry]]) -> dict[str, CatalogEntry]:
    """One model per slot, preferring an unused vendor so failure profiles differ."""
    chosen: dict[str, CatalogEntry] = {}
    used: set[str] = set()
    for slot in SLOT_ORDER:
        candidates = proposals.get(slot, [])
        if slot == PAIRED_SLOTS[1] and PAIRED_SLOTS[0] in chosen:
            index = proposals[PAIRED_SLOTS[0]].index(chosen[PAIRED_SLOTS[0]])
            chosen[slot] = candidates[index]  # the sibling of the direct-mode pick
            continue
        pick = next((c for c in candidates if c.vendor not in used), None) or (
            candidates[0] if candidates else None
        )
        if pick:
            chosen[slot] = pick
            used.add(pick.vendor)
    return chosen
