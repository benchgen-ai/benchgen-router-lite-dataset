"""Provider-neutral call result. Everything downstream reads this, not raw HTTP."""

from __future__ import annotations

from dataclasses import dataclass

# Auth, credit and key-limit failures. Neither retrying nor moving to the next task can help.
FATAL_STATUS = ("HTTP 401", "HTTP 402", "HTTP 403")


@dataclass(slots=True)
class ChatResult:
    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost_usd: float | None = None
    latency_ms: int | None = None
    generation_id: str | None = None
    finish_reason: str | None = None
    error: str | None = None

    @property
    def empty(self) -> bool:
        """HTTP 200 with no content. Tracked separately because `error` stays None here."""
        return self.error is None and not self.text.strip()

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"

    @property
    def fatal(self) -> bool:
        return bool(self.error and self.error.startswith(FATAL_STATUS))
