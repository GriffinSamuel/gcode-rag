"""Pluggable LLM clients: Anthropic, OpenAI-compatible, and a scripted fake (SPEC 6.1).

`get_llm_from_env()` reads the process environment only; call `gcoderag.config.load_settings()` first to pull in `.env`.
"""
from __future__ import annotations

import logging
import os
import re
import time
from collections.abc import Callable, Mapping
from typing import Protocol, TypeVar, runtime_checkable

logger = logging.getLogger(__name__)

DEFAULT_ANTHROPIC_MODEL = "claude-haiku-5-5"
PROVIDERS = ("anthropic", "openai_compatible")
MAX_RETRIES = 3
OPENAI_TIMEOUT_S = 120.0   # local models can take a while to load on the first call
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
BACKOFF_BASE_S = 2.0
T = TypeVar("T")


@runtime_checkable
class LLM(Protocol):
    """A text completion model."""

    name: str

    def complete(self, system: str, user: str, max_tokens: int = 800, temperature: float = 0.0) -> str:
        """Return the model's text reply to one system + user message."""
        ...


def with_retries(call: Callable[[], T], retries: int = MAX_RETRIES, base_delay: float = BACKOFF_BASE_S,
                 sleep: Callable[[float], None] = time.sleep) -> T:
    """Run `call`, retrying up to `retries` times with exponential backoff; re-raise the last error."""
    for attempt in range(retries + 1):
        try:
            return call()
        except Exception as exc:  # noqa: BLE001 - provider SDKs raise many error types
            if attempt == retries:
                raise
            delay = base_delay * (2 ** attempt)
            logger.warning("LLM call failed (%s); retrying in %.1fs", type(exc).__name__, delay)
            sleep(delay)
    raise RuntimeError("unreachable")


def strip_think(text: str) -> str:
    """Remove <think>...</think> blocks (emitted by reasoning models) and trim."""
    return _THINK_RE.sub("", text).strip()


class AnthropicLLM:
    """Claude via the official `anthropic` SDK."""

    def __init__(self, api_key: str, model: str | None = None) -> None:
        import anthropic

        self.model = model or DEFAULT_ANTHROPIC_MODEL
        self.name = f"anthropic:{self.model}"
        self._client = anthropic.Anthropic(api_key=api_key, max_retries=0)

    def complete(self, system: str, user: str, max_tokens: int = 800, temperature: float = 0.0) -> str:
        """Send one message and join the text blocks of the reply."""
        response = with_retries(lambda: self._client.messages.create(
            model=self.model, max_tokens=max_tokens, temperature=temperature, system=system,
            messages=[{"role": "user", "content": user}],
        ))
        return "".join(block.text for block in response.content if getattr(block, "type", "") == "text").strip()


class OpenAICompatibleLLM:
    """Any OpenAI-compatible chat endpoint (OpenAI, Groq, Gemini, Ollama) via the `openai` SDK."""

    def __init__(self, api_key: str, model: str, base_url: str | None = None) -> None:
        from openai import OpenAI

        if not model:
            raise ValueError("LLM_MODEL must be set for the openai_compatible provider")
        self.model = model
        self.name = f"openai_compatible:{model}"
        self._client = OpenAI(api_key=api_key, base_url=base_url or None, max_retries=0,
                              timeout=OPENAI_TIMEOUT_S)

    def complete(self, system: str, user: str, max_tokens: int = 800, temperature: float = 0.0) -> str:
        """Send one chat completion request; return the first choice's text without any <think> block."""
        response = with_retries(lambda: self._client.chat.completions.create(
            model=self.model, max_tokens=max_tokens, temperature=temperature,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        ))
        return strip_think(response.choices[0].message.content or "")


class FakeLLM:
    """Scripted LLM for tests: returns responses in order, then repeats the last one; records every call."""

    def __init__(self, responses: list[str], name: str = "fake-llm") -> None:
        if not responses:
            raise ValueError("FakeLLM needs at least one scripted response")
        self.name = name
        self._responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def complete(self, system: str, user: str, max_tokens: int = 800, temperature: float = 0.0) -> str:
        """Return the next scripted response."""
        self.calls.append({"system": system, "user": user, "max_tokens": max_tokens, "temperature": temperature})
        return self._responses[min(len(self.calls) - 1, len(self._responses) - 1)]


def get_llm_from_env(env: Mapping[str, str] | None = None) -> LLM | None:
    """Build the configured LLM from LLM_PROVIDER / LLM_API_KEY / LLM_MODEL / LLM_BASE_URL, or None if unconfigured."""
    values = os.environ if env is None else env
    provider = (values.get("LLM_PROVIDER") or "").strip().lower()
    if provider not in PROVIDERS:
        return None
    key = values.get("LLM_API_KEY") or (values.get("ANTHROPIC_API_KEY") if provider == "anthropic" else "")
    if not key:
        return None
    model = values.get("LLM_MODEL") or ""
    if provider == "anthropic":
        return AnthropicLLM(key, model or None)
    return OpenAICompatibleLLM(key, model, values.get("LLM_BASE_URL") or None)


def provider_and_model(llm: LLM) -> tuple[str, str]:
    """Split an LLM's `name` ('provider:model') into its parts."""
    provider, _, model = llm.name.partition(":")
    return provider, model or provider
