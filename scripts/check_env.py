#!/usr/bin/env python
"""Report the project's configuration WITHOUT printing any secret values.

Prints one of:
  MODE: llm (provider=<anthropic|openai_compatible>, model=<name>, base_url=<set|unset>)
  MODE: retrieval-only (no LLM key configured)
and whether HF_TOKEN and EMBEDDING_MODEL are set. Standard library only; reads .env manually so it works before
requirements are installed. Exit code is always 0.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        val = val.split(" #")[0].strip().strip('"').strip("'")
        values[key.strip()] = val
    return values


def main() -> int:
    env = load_env_file(ROOT / ".env")
    for key in ("LLM_PROVIDER", "LLM_API_KEY", "LLM_MODEL", "LLM_BASE_URL", "ANTHROPIC_API_KEY", "HF_TOKEN", "EMBEDDING_MODEL"):
        if os.environ.get(key):
            env[key] = os.environ[key]

    provider = env.get("LLM_PROVIDER", "").strip().lower()
    key = env.get("LLM_API_KEY") or (env.get("ANTHROPIC_API_KEY", "") if provider == "anthropic" else "")
    model = env.get("LLM_MODEL") or ("claude-haiku-5-5" if provider == "anthropic" else "")
    base_url_state = "set" if env.get("LLM_BASE_URL") else "unset"

    print(f".env present: {(ROOT / '.env').exists()}")
    if provider in {"anthropic", "openai_compatible"} and key:
        print(f"MODE: llm (provider={provider}, model={model or 'unset'}, base_url={base_url_state})")
        if provider == "openai_compatible" and not env.get("LLM_BASE_URL"):
            print("WARNING: openai_compatible provider without LLM_BASE_URL; the openai default endpoint will be used")
        if provider == "openai_compatible" and not model:
            print("WARNING: LLM_MODEL is empty; set it to a model your endpoint serves")
    else:
        reason = "LLM_PROVIDER not set" if not provider else ("unknown LLM_PROVIDER" if provider not in {"anthropic", "openai_compatible"} else "no API key")
        print(f"MODE: retrieval-only (no LLM key configured: {reason})")
    print(f"HF_TOKEN: {'set' if env.get('HF_TOKEN') else 'unset'}")
    print(f"EMBEDDING_MODEL: {env.get('EMBEDDING_MODEL') or 'BAAI/bge-small-en-v1.5 (default)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
