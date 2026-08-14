"""OpenRouter chat client: async, retrying, cost-aware, never logs the key."""

from __future__ import annotations

import asyncio
import json
import os
import random
import time

import httpx

from ..schemas import AgentCard, CallProtocol
from .base import ChatResult

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}


class MissingApiKey(RuntimeError):
    pass


def _api_key(agent: AgentCard) -> str:
    key = os.environ.get(agent.api_key_env, "").strip()
    if not key:
        raise MissingApiKey(f"{agent.api_key_env} is not set (agent {agent.id!r})")
    return key


class OpenRouterClient:
    """One client per collection run; shares a connection pool across agents."""

    def __init__(
        self,
        timeout_s: float = 180.0,
        referer: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        headers = {"Content-Type": "application/json"}
        if referer:
            headers["HTTP-Referer"] = referer
            headers["X-Title"] = "benchgen-router-dataset"
        self._client = httpx.AsyncClient(timeout=timeout_s, headers=headers, transport=transport)

    async def __aenter__(self) -> OpenRouterClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def chat(
        self, agent: AgentCard, messages: list[dict[str, str]], protocol: CallProtocol
    ) -> ChatResult:
        url = (agent.base_url or DEFAULT_BASE_URL).rstrip("/") + "/chat/completions"
        payload: dict[str, object] = {
            "model": agent.slug,
            "messages": messages,
            "max_tokens": protocol.max_tokens,
            "temperature": protocol.temperature,
            "top_p": protocol.top_p,
            "usage": {"include": True},
        }
        if agent.is_reasoning_model and protocol.reasoning_effort:
            payload["reasoning"] = {"effort": protocol.reasoning_effort}

        headers = {"Authorization": f"Bearer {_api_key(agent)}"}
        started = time.perf_counter()
        last_error = "no attempt made"

        for attempt in range(protocol.max_retries + 1):
            try:
                resp = await self._client.post(url, json=payload, headers=headers)
            except httpx.HTTPError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            else:
                if resp.status_code == 200:
                    body = _decode(resp.text)
                    if body is None:
                        # Unparseable body: retryable, and never fatal to the run.
                        last_error = f"unparseable 200 body: {resp.text[:200]!r}"
                    else:
                        code = _body_error_code(body)
                        if code is None:
                            return _parse(body, started)
                        # OpenRouter reports upstream failures as 200 with an error object, so a
                        # retryable provider timeout is invisible to a status-code-only check.
                        last_error = f"HTTP {code}: {str(body.get('error'))[:300]}"
                        if code not in RETRY_STATUS:
                            break
                else:
                    # Body can echo the request; truncate so a key can never reach a log or the
                    # data.
                    last_error = f"HTTP {resp.status_code}: {resp.text[:300]}"
                    if resp.status_code not in RETRY_STATUS:
                        break
            if attempt < protocol.max_retries:
                await asyncio.sleep(min(2**attempt, 8) + random.uniform(0, 0.5))

        return ChatResult(
            text="",
            latency_ms=int((time.perf_counter() - started) * 1000),
            error=last_error,
        )


def _decode(text: str) -> dict | None:
    """Parse a response body, tolerating OpenRouter's SSE-style keep-alive padding.

    Slow non-streaming requests come back as thousands of `: OPENROUTER PROCESSING` lines
    followed by the real payload, which `resp.json()` rejects outright.
    """
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    payload = "\n".join(
        line for line in text.splitlines() if line.strip() and not line.startswith(":")
    )
    try:
        decoded = json.loads(payload)
    except (json.JSONDecodeError, ValueError):
        return None
    return decoded if isinstance(decoded, dict) else None


def _body_error_code(body: dict) -> int | None:
    error = body.get("error")
    if not error:
        return None
    code = error.get("code") if isinstance(error, dict) else None
    return code if isinstance(code, int) else -1


def _parse(body: dict, started: float) -> ChatResult:
    latency_ms = int((time.perf_counter() - started) * 1000)
    if "error" in body and body.get("error"):
        return ChatResult(text="", latency_ms=latency_ms, error=str(body["error"])[:300])
    choices = body.get("choices") or []
    message = (choices[0].get("message") or {}) if choices else {}
    text = message.get("content") or ""
    if not text and message.get("reasoning"):
        # Reasoning models occasionally return only the reasoning channel; that is still empty.
        text = ""
    usage = body.get("usage") or {}
    return ChatResult(
        text=text,
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
        cost_usd=usage.get("cost"),
        latency_ms=latency_ms,
        generation_id=body.get("id"),
        finish_reason=(choices[0].get("finish_reason") if choices else None),
    )
