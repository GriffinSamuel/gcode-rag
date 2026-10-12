#!/usr/bin/env python
"""Deploy the Gradio app to a Hugging Face Space (docs/DEPLOY.md section 4). Needs HF_TOKEN (env/.env) or `hf auth login`.

Uploads the code (never data/, .env or .venv/) and uses deploy/README_HF.md as the Space README.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gcoderag.config import load_settings  # noqa: E402

IGNORE = [".venv/*", "data/*", ".git/*", ".claude/*", ".env", "results/answers.jsonl", "*.pyc", "__pycache__/*",
          "**/__pycache__/*", ".pytest_cache/*", "README.md", "KICKOFF.md"]


def main() -> int:
    """Create (or reuse) <user>/gcode-rag as a Gradio Space and upload the repo to it."""
    from huggingface_hub import HfApi, get_token

    load_settings()
    token = os.environ.get("HF_TOKEN") or get_token()
    if not token:
        print("HF_TOKEN is not set: add it to .env or run `hf auth login`, then re-run.", file=sys.stderr)
        return 2

    api = HfApi(token=token)
    repo_id = f"{api.whoami()['name']}/gcode-rag"
    api.create_repo(repo_id, repo_type="space", space_sdk="gradio", exist_ok=True)
    api.upload_folder(folder_path=str(ROOT), repo_id=repo_id, repo_type="space", ignore_patterns=IGNORE)
    api.upload_file(path_or_fileobj=str(ROOT / "deploy" / "README_HF.md"), path_in_repo="README.md",
                    repo_id=repo_id, repo_type="space")
    print(f"https://huggingface.co/spaces/{repo_id}")
    print("Next: in the Space settings add the variable SKIP_HAAS=1 and (optionally) the LLM_* secrets.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
