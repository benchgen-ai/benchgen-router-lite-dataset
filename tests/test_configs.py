from __future__ import annotations

import pytest

from benchgen_router_dataset.config_loader import (
    load_gates,
    load_pool,
    load_roles,
    require_verified,
)
from benchgen_router_dataset.graders import known_graders
from benchgen_router_dataset.providers.catalog import (
    PAIRED_SLOTS,
    SLOT_ORDER,
    CatalogEntry,
    propose_pool,
)
from benchgen_router_dataset.schemas import UNRESOLVED_SLUG, AgentCard, AgentPool
from benchgen_router_dataset.sources import REGISTRY


def _entry(slug: str, reasoning: bool, out_usd: float = 1.0) -> CatalogEntry:
    return CatalogEntry(
        slug=slug,
        name=slug,
        description="",
        context_length=131072,
        prompt_usd_per_1m=out_usd / 4,
        output_usd_per_1m=out_usd,
        supports_reasoning=reasoning,
    )


def test_unpinned_slugs_never_reach_the_pool() -> None:
    """A floating alias or a `:batch` variant makes E(D, M) unreproducible."""
    assert not _entry("~anthropic/claude-opus-latest", True).is_pinned
    assert not _entry("openai/gpt-5:batch", True).is_pinned
    assert _entry("openai/gpt-5", True).is_pinned


def test_meta_routers_are_never_proposed() -> None:
    """`openrouter/auto` prices at a negative sentinel, so it sorts cheapest for every slot."""
    auto = _entry("openrouter/auto", True, out_usd=-1_000_000.0)
    assert auto.is_meta_router and auto.is_free
    proposals = propose_pool([auto, _entry("mistralai/mistral-nemo", False, out_usd=0.03)])
    assert all(auto not in candidates for candidates in proposals.values())


def test_paired_slots_resolve_to_sibling_modes_of_one_base_model() -> None:
    catalog = [
        _entry("qwen/qwen3-30b-a3b-instruct-2507", False),
        _entry("qwen/qwen3-30b-a3b-thinking-2507", True),
    ]
    proposals = propose_pool(catalog)
    direct, reasoning = (proposals[slot][0] for slot in PAIRED_SLOTS)
    assert direct.slug.replace("-instruct", "") == reasoning.slug.replace("-thinking", "")
    assert not direct.supports_reasoning and reasoning.supports_reasoning


def test_shipped_configs_validate() -> None:
    pool = load_pool("v1")
    roles = load_roles("v1")
    gates = load_gates("v1")
    assert [a.index for a in pool.agents] == list(range(len(pool.agents)))
    assert [r.id for r in roles.roles] == ["thinker", "worker", "verifier"]
    assert gates.thresholds


def test_discovery_slots_match_the_shipped_pool_ids() -> None:
    """`apply --slug <id>=...` silently no-ops on an unknown id, so drift here is invisible."""
    assert list(SLOT_ORDER) == [a.id for a in load_pool("v1").agents]
    assert all(slot in SLOT_ORDER for slot in PAIRED_SLOTS)


def test_verifier_role_has_an_enforced_grammar() -> None:
    """ACCEPT terminates the loop, so its output must be machine-parseable, not merely requested."""
    verifier = load_roles("v1").by_id("verifier")
    grammar = verifier.compiled_grammar()
    assert grammar is not None
    assert grammar.match("ACCEPT - looks complete")
    assert not grammar.match("Sure, that seems right")


def test_shipped_pool_is_unverified_until_preflight() -> None:
    pool = load_pool("v1")
    unverified = pool.model_copy(
        update={"agents": [a.model_copy(update={"verified_at": None}) for a in pool.agents]}
    )
    with pytest.raises(RuntimeError, match="unverified"):
        require_verified(unverified)


def test_duplicate_agent_indices_are_rejected() -> None:
    cards = [
        AgentCard(index=0, id="a", provider="p", slug="p/a", description="d"),
        AgentCard(index=0, id="b", provider="p", slug="p/b", description="d"),
    ]
    with pytest.raises(ValueError, match="unique"):
        AgentPool(pool_version="v1", agents=cards)


def test_two_slots_on_one_slug_are_rejected() -> None:
    """The direct/reasoning contrast needs sibling slugs; a per-call toggle is advisory."""
    cards = [
        AgentCard(index=0, id="direct", provider="p", slug="p/a", description="d"),
        AgentCard(
            index=1,
            id="reasoning",
            provider="p",
            slug="p/a",
            description="d",
            is_reasoning_model=True,
        ),
    ]
    with pytest.raises(ValueError, match="distinct slugs"):
        AgentPool(pool_version="v1", agents=cards)


def _paired_cards(direct_slug: str, reasoning_slug: str) -> list[AgentCard]:
    return [
        AgentCard(index=0, id="open_direct", provider="p", slug=direct_slug, description="d"),
        AgentCard(
            index=1,
            id="open_reasoning",
            provider="p",
            slug=reasoning_slug,
            description="d",
            is_reasoning_model=True,
        ),
    ]


def test_paired_slots_on_two_unrelated_models_are_rejected() -> None:
    """Two different models is not the single-axis contrast either — it confounds it."""
    with pytest.raises(ValueError, match="sibling modes"):
        AgentPool(pool_version="v1", agents=_paired_cards("p/a-instruct", "q/b-thinking"))


def test_paired_slots_accept_sibling_modes_of_one_base_model() -> None:
    pool = AgentPool(
        pool_version="v1", agents=_paired_cards("p/a-instruct-2507", "p/a-thinking-2507")
    )
    assert pool.agent_order == ["open_direct", "open_reasoning"]


def test_unfilled_slots_may_share_the_placeholder_but_never_count_as_verified() -> None:
    pool = AgentPool(pool_version="v1", agents=_paired_cards(UNRESOLVED_SLUG, UNRESOLVED_SLUG))
    assert not any(a.resolved or a.verified for a in pool.active)
    stamped = pool.model_copy(
        update={"agents": [a.model_copy(update={"verified_at": "2026-01-01"}) for a in pool.agents]}
    )
    # A date on a placeholder is a stamp for a model nobody ever called.
    assert not any(a.verified for a in stamped.active)
    with pytest.raises(RuntimeError, match="placeholder slug"):
        require_verified(stamped)


def test_shipped_paired_slots_are_sibling_modes() -> None:
    pool = load_pool("v1")
    direct, reasoning = (pool.by_id(slot) for slot in PAIRED_SLOTS)
    assert direct.slug != reasoning.slug
    assert direct.is_reasoning_model is False and reasoning.is_reasoning_model is True


def test_retired_agents_leave_their_index_untouched() -> None:
    pool = AgentPool(
        pool_version="v2",
        agents=[
            AgentCard(index=0, id="a", provider="p", slug="p/a", description="d"),
            AgentCard(index=1, id="b", provider="p", slug="p/b", description="d", retired=True),
            AgentCard(index=2, id="c", provider="p", slug="p/c", description="d"),
        ],
    )
    assert pool.agent_order == ["a", "c"]
    assert pool.by_id("b").retired is True


def test_every_source_names_a_known_grader() -> None:
    graders = set(known_graders())
    for name, source in REGISTRY.items():
        assert source.spec.grader in graders, f"{name} names an unknown grader"


def test_gated_sources_are_marked_non_redistributable() -> None:
    """Publishing gated items may breach the access agreement; ids + rewards only."""
    assert REGISTRY["gpqa_diamond"].spec.redistributable is False
    assert REGISTRY["benchgen_turkish"].spec.redistributable is False
