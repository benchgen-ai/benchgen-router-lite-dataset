"""Agent pool, call protocol and role contracts."""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field, model_validator

# A slot that pre-flight has not filled yet. Several slots may hold it at once, so it is the
# one slug value exempt from the distinctness rule below.
UNRESOLVED_SLUG = "TBD-run-scripts/preflight_agents.py"
UNRESOLVED_PREFIX = "TBD"

# The paper's single-axis contrast: one base model in two modes. The provider publishes those
# as sibling slugs; a per-call reasoning toggle is advisory and at least one provider ignores
# it, which silently collapses the two slots into one agent.
PAIRED_SLOTS = ("open_direct", "open_reasoning")
DIRECT_MARKER = "-instruct"
REASONING_MARKER = "-thinking"


def are_mode_siblings(direct_slug: str, reasoning_slug: str) -> bool:
    """Two published modes of one base model, not one hybrid slug used twice."""
    if direct_slug == reasoning_slug:
        return False
    return (
        DIRECT_MARKER in direct_slug
        and REASONING_MARKER in reasoning_slug
        and direct_slug.replace(DIRECT_MARKER, "") == reasoning_slug.replace(REASONING_MARKER, "")
    )


class CallProtocol(BaseModel):
    """Identical for every agent — a reward matrix is meaningless without it."""

    model_config = ConfigDict(extra="forbid")

    max_tokens: int = 4096
    temperature: float = 0.1
    top_p: float = 1.0
    reasoning_effort: str | None = "minimal"
    timeout_s: float = 180.0
    max_retries: int = 3


class AgentCard(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int = Field(ge=0)
    id: str
    provider: str
    slug: str
    description: str
    capabilities: list[str] = Field(default_factory=list)
    is_reasoning_model: bool = False
    context_window: int | None = None
    price_per_1m_prompt_usd: float | None = None
    price_per_1m_output_usd: float | None = None
    base_url: str | None = Field(
        default=None, description="Set for self-hosted agents; None means the provider default."
    )
    api_key_env: str = "OPENROUTER_API_KEY"
    verified_at: str | None = Field(
        default=None, description="ISO date a live completion was observed. None = unverified."
    )
    retired: bool = False

    @property
    def resolved(self) -> bool:
        """False while the slot still holds a pre-flight placeholder instead of a real slug."""
        return not self.slug.startswith(UNRESOLVED_PREFIX)

    @property
    def verified(self) -> bool:
        return self.resolved and bool(self.verified_at)


class AgentPool(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pool_version: str
    notes: str | None = None
    paper_reference: list[str] = Field(
        default_factory=list, description="The pool this one mirrors, for provenance only."
    )
    protocol: CallProtocol = Field(default_factory=CallProtocol)
    agents: list[AgentCard]

    @model_validator(mode="after")
    def _indices_are_a_contract(self) -> AgentPool:
        indices = [a.index for a in self.agents]
        if len(set(indices)) != len(indices):
            raise ValueError("agent indices must be unique — indices are the head's contract")
        if indices != sorted(indices):
            raise ValueError("agents must be listed in ascending index order")
        ids = [a.id for a in self.agents]
        if len(set(ids)) != len(ids):
            raise ValueError("agent ids must be unique")
        slugs = [a.slug for a in self.agents if not a.retired and a.resolved]
        duplicate = next((s for s in slugs if slugs.count(s) > 1), None)
        if duplicate is not None:
            # Two slots on one slug is never a real contrast: the pool silently shrinks by one
            # agent and the reward matrix fills with ties instead of raising an error.
            raise ValueError(
                f"active agents must have distinct slugs; {duplicate!r} fills more than one slot"
            )
        return self

    @model_validator(mode="after")
    def _paired_slots_are_sibling_modes(self) -> AgentPool:
        by_id = {a.id: a for a in self.agents if not a.retired}
        direct, reasoning = (by_id.get(slot) for slot in PAIRED_SLOTS)
        if direct is None or reasoning is None:
            return self
        if not (direct.resolved and reasoning.resolved):
            return self
        if not are_mode_siblings(direct.slug, reasoning.slug):
            raise ValueError(
                f"{PAIRED_SLOTS[0]}={direct.slug!r} and {PAIRED_SLOTS[1]}={reasoning.slug!r} are "
                f"not sibling modes of one base model ('{DIRECT_MARKER}' against "
                f"'{REASONING_MARKER}'); a per-call reasoning toggle is advisory and does not "
                "make two agents"
            )
        return self

    @property
    def active(self) -> list[AgentCard]:
        return [a for a in self.agents if not a.retired]

    @property
    def agent_order(self) -> list[str]:
        """Positional contract for every reward vector."""
        return [a.id for a in self.active]

    def by_id(self, agent_id: str) -> AgentCard:
        for a in self.agents:
            if a.id == agent_id:
                return a
        raise KeyError(agent_id)


class Role(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int = Field(ge=0)
    id: str
    contract: str
    prompt_template: str
    output_grammar: str | None = None

    def compiled_grammar(self) -> re.Pattern[str] | None:
        return re.compile(self.output_grammar) if self.output_grammar else None


class RoleSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    roles_version: str
    roles: list[Role]

    def by_id(self, role_id: str) -> Role:
        for r in self.roles:
            if r.id == role_id:
                return r
        raise KeyError(role_id)
