"""Tests for the LLM layer: FakeLLM scripting, env detection, retry backoff (no network)."""
from __future__ import annotations

import pytest

from gcoderag.llm import LLM, FakeLLM, get_llm_from_env, provider_and_model, with_retries


def test_fake_llm_scripts_and_records() -> None:
    llm = FakeLLM(["first", "second"])
    assert [llm.complete("s", "u1"), llm.complete("s", "u2"), llm.complete("s", "u3")] == ["first", "second", "second"]
    assert [c["user"] for c in llm.calls] == ["u1", "u2", "u3"]
    assert llm.calls[0]["temperature"] == 0.0
    assert isinstance(llm, LLM)


def test_get_llm_from_env_none_when_unconfigured(monkeypatch) -> None:
    for key in ("LLM_PROVIDER", "LLM_API_KEY", "ANTHROPIC_API_KEY", "LLM_MODEL", "LLM_BASE_URL"):
        monkeypatch.delenv(key, raising=False)
    assert get_llm_from_env() is None
    assert get_llm_from_env({}) is None
    assert get_llm_from_env({"LLM_PROVIDER": "anthropic", "LLM_API_KEY": ""}) is None
    assert get_llm_from_env({"LLM_PROVIDER": "mystery", "LLM_API_KEY": "x"}) is None


def test_anthropic_key_alias_builds_client() -> None:
    llm = get_llm_from_env({"LLM_PROVIDER": "anthropic", "ANTHROPIC_API_KEY": "test-key-not-real"})
    assert llm is not None and provider_and_model(llm) == ("anthropic", "claude-haiku-5-5")


def test_with_retries_backs_off_then_succeeds() -> None:
    delays: list[float] = []
    attempts = iter([RuntimeError("rate limit"), RuntimeError("rate limit"), "ok"])

    def call() -> str:
        item = next(attempts)
        if isinstance(item, Exception):
            raise item
        return item

    assert with_retries(call, retries=3, base_delay=1.0, sleep=delays.append) == "ok"
    assert delays == [1.0, 2.0]


def test_with_retries_reraises_after_last_attempt() -> None:
    def boom() -> str:
        raise ValueError("auth")

    with pytest.raises(ValueError):
        with_retries(boom, retries=2, base_delay=0.0, sleep=lambda _: None)


def test_strip_think_removes_reasoning_blocks() -> None:
    from gcoderag.llm import strip_think

    assert strip_think("<think>plan\nmore</think>\n  PONG ") == "PONG"
    assert strip_think("A <think>x</think>B <think>y</think>") == "A B"
    assert strip_think("no tags") == "no tags"


def test_openai_client_uses_long_timeout() -> None:
    from gcoderag.llm import OPENAI_TIMEOUT_S, OpenAICompatibleLLM

    llm = OpenAICompatibleLLM("ollama", "qwen2.5-coder:7b", "http://localhost:11434/v1")
    assert OPENAI_TIMEOUT_S >= 120 and llm._client.timeout == OPENAI_TIMEOUT_S
