# Deployment: Hugging Face Spaces (free, Gradio)

Deploy only if `HF_TOKEN` is set (check with `scripts/check_env.py`). Otherwise complete steps 1-3 and put the manual
steps in the README so the user can deploy later.

## 1. Files the Space needs

- `app.py` at the repo root (entry point), `requirements.txt`, the `gcoderag/` package, `eval/` (optional), and a
  Space README with YAML front matter. Keep the GitHub README separate: write the Space header to `deploy/README_HF.md`:

```markdown
---
title: CNC G-Code Assistant
emoji: 🛠️
colorFrom: gray
colorTo: blue
sdk: gradio
sdk_version: <installed gradio version, e.g. 5.49.1; read it with: .venv/Scripts/python.exe -c "import gradio; print(gradio.__version__)">
app_file: app.py
pinned: false
license: mit
---
Short description, link to the GitHub repo, the Safety sentence.
```

## 2. Data at startup

The Space has no `data/` folder (gitignored). `app.py` must build what is missing on first start:
1. If `data/corpus/chunks.jsonl` is missing: run the LinuxCNC ingest + chunker (network OK on Spaces; Haas is skipped
   on Spaces by setting env `SKIP_HAAS=1` in the Space settings).
2. If `data/index/` is missing: build it with the embedding model (bge-small downloads in ~20 s on the free CPU tier).
3. Cache both so restarts are fast. Keep startup under the Space timeout by building sequentially with logging.

## 3. Secrets

Set `LLM_PROVIDER`, `LLM_API_KEY`, `LLM_MODEL`, `LLM_BASE_URL` as Space **secrets** (Settings > Variables and secrets).
Without them the app runs in retrieval-only mode, which is acceptable for a public demo.

## 4. Push (only with HF_TOKEN)

`scripts/deploy_hf.py` (write it only if deploying):

```python
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
repo_id = f"{api.whoami()['name']}/gcode-rag"
api.create_repo(repo_id, repo_type="space", space_sdk="gradio", exist_ok=True)
api.upload_folder(folder_path=".", repo_id=repo_id, repo_type="space",
                  ignore_patterns=[".venv/*", "data/*", ".git/*", ".claude/*", "results/answers.jsonl", "*.pyc", "__pycache__/*", "README.md"])
api.upload_file(path_or_fileobj="deploy/README_HF.md", path_in_repo="README.md", repo_id=repo_id, repo_type="space")
print(f"https://huggingface.co/spaces/{repo_id}")
```

Record the Space URL in README and `docs/DECISIONS.md`. The build log is at the Space page > Logs; if it fails on a
dependency, pin the offending package and re-run.

## 5. Manual instructions for the README (when not deployed)

1. Create a free account at huggingface.co and a Write token (Settings > Access Tokens).
2. `pip install huggingface_hub` then `huggingface-cli login`.
3. Run `.venv/Scripts/python.exe scripts/deploy_hf.py` (or create the Space in the web UI with SDK = Gradio and upload the files above).
