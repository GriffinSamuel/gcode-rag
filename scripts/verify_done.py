#!/usr/bin/env python
"""Definition of done for gcode-rag.

Standard library only (the Stop hook may run this with a Python that has no packages installed).
Prints one line per check and ends with either `ALL CHECKS PASSED` (exit 0) or `CHECKS FAILED (<n>)` (exit 1).

Usage:  python scripts/verify_done.py [--quick]
  --quick  skips the pytest run (used for fast status; the Stop hook always runs the full version)
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Thresholds (keep in sync with docs/SPEC.md section 1)
MIN_DOCS = 120
MIN_CHUNKS = 150
MIN_DISTINCT_CODES = 60
MIN_QUESTIONS = 40
MIN_CATEGORIES = 4
MIN_RECALL5 = 0.75
MIN_KEYWORD_ACC = 0.60
MIN_TESTS_PASSED = 25
MIN_COMMITS = 8
MIN_VALID_EXAMPLES = 3
MIN_INVALID_EXAMPLES = 3
MIN_FINAL_REPORT_WORDS = 300
PYTEST_TIMEOUT_S = 600

REQUIRED_FILES = [
    "CLAUDE.md", "PROGRESS.md", "README.md", "FINAL_REPORT.md", "requirements.txt", "pytest.ini", ".gitignore",
    "app.py",
    "gcoderag/__init__.py", "gcoderag/config.py", "gcoderag/textnorm.py", "gcoderag/embed.py", "gcoderag/index.py",
    "gcoderag/llm.py", "gcoderag/rag.py", "gcoderag/generator.py", "gcoderag/validator.py", "gcoderag/evaluate.py",
    "gcoderag/cli.py", "gcoderag/ingest/__init__.py", "gcoderag/ingest/linuxcnc.py", "gcoderag/ingest/haas.py",
    "gcoderag/ingest/chunker.py",
    "scripts/check_env.py", "scripts/build_corpus.py", "scripts/build_index.py", "scripts/run_eval.py",
    "scripts/validate_examples.py",
    "tests/test_validator.py", "tests/test_index.py", "tests/test_evaluate.py", "tests/test_rag.py",
    "tests/test_generator.py", "tests/test_linuxcnc_parser.py", "tests/test_chunker.py", "tests/test_app.py",
    "eval/questions.jsonl",
    "data/corpus/docs.jsonl", "data/corpus/chunks.jsonl", "data/corpus/stats.json",
    "data/index/embeddings.npy", "data/index/meta.json",
    "data/raw/haas/STATUS.json",
    "results/results.json", "results/results.md", "results/validator_report.json",
    "deploy/README_HF.md",
    "docs/DECISIONS.md",
]

README_SECTIONS = ["architecture", "how to run", "evaluation", "results", "safety", "decisions"]
SAFETY_SENTENCE = ("Generated G-code is for learning and must be checked in a simulator (such as NC Viewer) and "
                   "reviewed by a qualified machinist before it is run on a real machine.")
FINAL_REPORT_SECTIONS = ["what was built", "results", "limitations", "demo"]
ALLOWED_CATEGORIES = {"code_lookup", "parameters", "modal_groups", "syntax", "canned_cycles", "safety_practice",
                      "haas_specific"}
REQUIRED_CONFIGS = {"dense_baseline", "bm25_only", "hybrid_rrf"}
SECRET_PATTERNS = [
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}"),
    re.compile(r"\bgsk_[A-Za-z0-9]{20,}"),
    re.compile(r"\bAIza[0-9A-Za-z_\-]{30,}"),
    re.compile(r"\bhf_[A-Za-z0-9]{20,}"),
    re.compile(r"\bsk-[A-Za-z0-9]{40,}"),
]
TEXT_EXT = {".py", ".md", ".txt", ".json", ".jsonl", ".toml", ".ini", ".cfg", ".yaml", ".yml", ".nc", ".ngc", ".html",
            ".adoc", ".example", ".ps1"}


class Checker:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.passes = 0

    def ok(self, name: str, detail: str = "") -> None:
        self.passes += 1
        print(f"PASS  {name}" + (f"  ({detail})" if detail else ""))

    def fail(self, name: str, detail: str) -> None:
        self.failures.append(f"{name}: {detail}")
        print(f"FAIL  {name}: {detail}")

    def check(self, name: str, cond: bool, detail_ok: str = "", detail_fail: str = "") -> bool:
        if cond:
            self.ok(name, detail_ok)
        else:
            self.fail(name, detail_fail or "condition not met")
        return cond


def read_text(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")


def read_jsonl(rel: str) -> list[dict]:
    out: list[dict] = []
    with (ROOT / rel).open(encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{rel} line {i}: invalid JSON ({exc})") from exc
    return out


def run(cmd: list[str], timeout: int = 60, env: dict | None = None) -> tuple[int, str]:
    try:
        proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=timeout, env=env,
                              encoding="utf-8", errors="replace")
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")
    except FileNotFoundError as exc:
        return 127, f"command not found: {cmd[0]} ({exc})"
    except subprocess.TimeoutExpired:
        return 124, f"timed out after {timeout}s: {' '.join(cmd)}"


def venv_python() -> Path | None:
    for candidate in (ROOT / ".venv" / "Scripts" / "python.exe", ROOT / ".venv" / "bin" / "python"):
        if candidate.exists():
            return candidate
    return None


def main(argv: list[str]) -> int:
    quick = "--quick" in argv
    c = Checker()
    print(f"verify_done: root={ROOT}")

    # 1. Required files
    missing = [f for f in REQUIRED_FILES if not (ROOT / f).exists()]
    c.check("required files", not missing, f"{len(REQUIRED_FILES)} present", "missing: " + ", ".join(missing[:12])
            + (" ..." if len(missing) > 12 else ""))

    # 2. venv
    py = venv_python()
    c.check("virtualenv", py is not None, str(py) if py else "", ".venv not found (python -m venv .venv)")

    # 3. PROGRESS.md fully checked
    if (ROOT / "PROGRESS.md").exists():
        unchecked = [ln.strip() for ln in read_text("PROGRESS.md").splitlines() if ln.strip().startswith("- [ ]")]
        c.check("PROGRESS.md all checked", not unchecked, "no unchecked items",
                f"{len(unchecked)} unchecked: " + " | ".join(u[:70] for u in unchecked[:5]))

    # 4. Corpus
    doc_ids: set[str] = set()
    num_chunks = 0
    if (ROOT / "data/corpus/docs.jsonl").exists() and (ROOT / "data/corpus/chunks.jsonl").exists():
        try:
            docs = read_jsonl("data/corpus/docs.jsonl")
            chunks = read_jsonl("data/corpus/chunks.jsonl")
            num_chunks = len(chunks)
            doc_keys = {"doc_id", "source", "title", "codes", "url", "text"}
            chunk_keys = {"chunk_id", "doc_id", "source", "title", "codes", "url", "chunk_index", "text"}
            bad_docs = [d for d in docs if not doc_keys <= set(d)]
            bad_chunks = [ch for ch in chunks if not chunk_keys <= set(ch)]
            doc_ids = {d.get("doc_id") for d in docs}
            codes = {code for d in docs for code in (d.get("codes") or [])}
            sources = {d.get("source") for d in docs}
            c.check("corpus docs", len(docs) >= MIN_DOCS and not bad_docs and len(doc_ids) == len(docs),
                    f"{len(docs)} docs", f"docs={len(docs)} (min {MIN_DOCS}), bad records={len(bad_docs)}, "
                    f"duplicate ids={len(docs) - len(doc_ids)}")
            c.check("corpus chunks", len(chunks) >= MIN_CHUNKS and not bad_chunks,
                    f"{len(chunks)} chunks", f"chunks={len(chunks)} (min {MIN_CHUNKS}), bad records={len(bad_chunks)}")
            c.check("corpus codes", len(codes) >= MIN_DISTINCT_CODES, f"{len(codes)} distinct codes",
                    f"{len(codes)} distinct codes (min {MIN_DISTINCT_CODES})")
            c.check("corpus has linuxcnc source", "linuxcnc" in sources, "", f"sources={sorted(map(str, sources))}")
        except Exception as exc:  # noqa: BLE001
            c.fail("corpus parse", str(exc))

    # 5. Index
    if (ROOT / "data/index/meta.json").exists() and (ROOT / "data/index/embeddings.npy").exists():
        try:
            meta = json.loads(read_text("data/index/meta.json"))
            nv, dim = int(meta.get("num_vectors", 0)), int(meta.get("dim", 0))
            size = (ROOT / "data/index/embeddings.npy").stat().st_size
            ok = (nv == num_chunks and nv > 0 and dim >= 64 and len(meta.get("chunk_ids", [])) == nv
                  and size >= nv * dim * 4 and bool(meta.get("embedding_model")))
            c.check("dense index", ok, f"{nv} vectors x {dim} ({meta.get('embedding_model')})",
                    f"num_vectors={nv}, chunks={num_chunks}, dim={dim}, chunk_ids={len(meta.get('chunk_ids', []))}, "
                    f"npy bytes={size}")
        except Exception as exc:  # noqa: BLE001
            c.fail("dense index", str(exc))

    # 6. Questions
    n_questions = 0
    if (ROOT / "eval/questions.jsonl").exists():
        try:
            qs = read_jsonl("eval/questions.jsonl")
            n_questions = len(qs)
            ids = [q.get("id") for q in qs]
            problems: list[str] = []
            for q in qs:
                if not all(k in q for k in ("id", "category", "question", "gold_doc_ids", "answer_keywords")):
                    problems.append(f"{q.get('id')}: missing field")
                    continue
                if q["category"] not in ALLOWED_CATEGORIES:
                    problems.append(f"{q['id']}: bad category {q['category']}")
                if not q["gold_doc_ids"] or not q["answer_keywords"]:
                    problems.append(f"{q['id']}: empty list")
                if doc_ids and any(g not in doc_ids for g in q["gold_doc_ids"]):
                    problems.append(f"{q['id']}: gold id not in corpus")
            cats = {q.get("category") for q in qs}
            c.check("eval questions", n_questions >= MIN_QUESTIONS and len(cats) >= MIN_CATEGORIES
                    and len(set(ids)) == len(ids) and not problems,
                    f"{n_questions} questions, {len(cats)} categories",
                    f"n={n_questions} (min {MIN_QUESTIONS}), categories={len(cats)} (min {MIN_CATEGORIES}), "
                    f"dup ids={len(ids) - len(set(ids))}, problems={problems[:5]}")
        except Exception as exc:  # noqa: BLE001
            c.fail("eval questions", str(exc))

    # 7. Results
    if (ROOT / "results/results.json").exists():
        try:
            res = json.loads(read_text("results/results.json"))
            cfgs = {cfg.get("name"): cfg for cfg in res.get("configs", [])}
            have = REQUIRED_CONFIGS <= set(cfgs)
            numeric_ok = all(isinstance(cfgs[n].get(m), (int, float)) and 0 <= cfgs[n][m] <= 1
                             for n in REQUIRED_CONFIGS if n in cfgs for m in ("recall_at_1", "recall_at_5", "mrr"))
            best = max((cfgs[n].get("recall_at_5", 0) for n in REQUIRED_CONFIGS if n in cfgs), default=0)
            c.check("results configs", have and numeric_ok and res.get("n_questions") == n_questions
                    and res.get("n_questions", 0) >= MIN_QUESTIONS,
                    f"configs={sorted(cfgs)}, n={res.get('n_questions')}",
                    f"configs={sorted(cfgs)} need {sorted(REQUIRED_CONFIGS)}, numeric_ok={numeric_ok}, "
                    f"n_questions={res.get('n_questions')} vs file {n_questions}")
            c.check("retrieval quality", best >= MIN_RECALL5, f"best recall@5={best}",
                    f"best recall@5={best} < {MIN_RECALL5}")
            ae = res.get("answer_eval") or {}
            llm_configured = env_has_llm_key()
            if llm_configured:
                c.check("answer eval", ae.get("status") == "done" and (ae.get("keyword_accuracy") or 0) >= MIN_KEYWORD_ACC
                        and ae.get("n") == n_questions,
                        f"keyword_accuracy={ae.get('keyword_accuracy')}",
                        f"LLM key is configured so answer_eval must be done with accuracy >= {MIN_KEYWORD_ACC}; got {ae}")
            else:
                c.check("answer eval", ae.get("status") in {"done", "skipped_no_llm_key"}, f"status={ae.get('status')}",
                        f"answer_eval.status must be 'done' or 'skipped_no_llm_key'; got {ae}")
        except Exception as exc:  # noqa: BLE001
            c.fail("results parse", str(exc))

    # 8. Validator report + examples
    if (ROOT / "results/validator_report.json").exists():
        try:
            rep = json.loads(read_text("results/validator_report.json"))
            rows = rep.get("results", [])
            valid = [r for r in rows if r.get("expected") == "valid"]
            invalid = [r for r in rows if r.get("expected") == "invalid"]
            all_match = all(r.get("matches_expectation") is True for r in rows) and bool(rows)
            c.check("validator examples", len(valid) >= MIN_VALID_EXAMPLES and len(invalid) >= MIN_INVALID_EXAMPLES
                    and all_match, f"{len(valid)} valid, {len(invalid)} invalid, all match",
                    f"valid={len(valid)} (min {MIN_VALID_EXAMPLES}), invalid={len(invalid)} (min {MIN_INVALID_EXAMPLES}), "
                    f"all_match={all_match}")
            example_files = sorted(p.name for p in (ROOT / "examples").glob("*.nc")) if (ROOT / "examples").exists() else []
            c.check("example files exist", len(example_files) >= MIN_VALID_EXAMPLES + MIN_INVALID_EXAMPLES,
                    f"{len(example_files)} .nc files", f"found {example_files}")
        except Exception as exc:  # noqa: BLE001
            c.fail("validator report", str(exc))

    # 9. README
    if (ROOT / "README.md").exists():
        readme = read_text("README.md")
        headings = [ln.lstrip("#").strip().lower() for ln in readme.splitlines() if ln.startswith("#")]
        missing_sections = [s for s in README_SECTIONS if not any(s in h for h in headings)]
        c.check("README sections", not missing_sections, ", ".join(README_SECTIONS),
                f"missing headings: {missing_sections}")
        c.check("README safety sentence", SAFETY_SENTENCE in readme, "", "exact Safety sentence from CLAUDE.md section 8 not found")
        c.check("README results table", "recall@5" in readme.lower() and "mrr" in readme.lower()
                and "hybrid_rrf" in readme and "dense_baseline" in readme and "bm25_only" in readme,
                "", "README must contain the results table with recall@5, MRR and all three config names")

    # 10. FINAL_REPORT.md
    if (ROOT / "FINAL_REPORT.md").exists():
        fr = read_text("FINAL_REPORT.md")
        words = len(fr.split())
        fr_heads = [ln.lstrip("#").strip().lower() for ln in fr.splitlines() if ln.startswith("#")]
        missing_fr = [s for s in FINAL_REPORT_SECTIONS if not any(s in h for h in fr_heads)]
        c.check("FINAL_REPORT.md", words >= MIN_FINAL_REPORT_WORDS and not missing_fr, f"{words} words",
                f"words={words} (min {MIN_FINAL_REPORT_WORDS}), missing headings={missing_fr}")

    # 11. app.py parses and mentions gradio
    if (ROOT / "app.py").exists():
        import ast
        try:
            src = read_text("app.py")
            ast.parse(src)
            c.check("app.py", "gradio" in src or "streamlit" in src, "parses", "app.py must use gradio (or streamlit fallback)")
        except SyntaxError as exc:
            c.fail("app.py", f"syntax error: {exc}")

    # 12. Git
    code, out = run(["git", "rev-parse", "--is-inside-work-tree"])
    if code == 0 and "true" in out:
        _, porcelain = run(["git", "status", "--porcelain"])
        c.check("git clean tree", porcelain.strip() == "", "", "uncommitted changes:\n" + porcelain.strip()[:800])
        _, count = run(["git", "rev-list", "--count", "HEAD"])
        try:
            n_commits = int(count.strip().splitlines()[-1])
        except (ValueError, IndexError):
            n_commits = 0
        c.check("git commits", n_commits >= MIN_COMMITS, f"{n_commits} commits", f"{n_commits} commits (min {MIN_COMMITS})")
        _, tracked_env = run(["git", "ls-files", ".env"])
        c.check(".env not tracked", tracked_env.strip() == "", "", ".env is tracked by git; remove it from the index")
        _, files = run(["git", "ls-files"])
        leaks: list[str] = []
        for rel in files.splitlines():
            p = ROOT / rel.strip()
            if p.suffix.lower() in TEXT_EXT and p.exists() and p.stat().st_size < 5_000_000:
                txt = p.read_text(encoding="utf-8", errors="replace")
                if any(pat.search(txt) for pat in SECRET_PATTERNS):
                    leaks.append(rel.strip())
        c.check("no secrets in tracked files", not leaks, "", f"possible secrets in: {leaks[:5]}")
    else:
        c.fail("git repository", "not a git repository (run git init)")

    # 13. pytest
    if quick:
        print("SKIP  pytest (--quick)")
    elif py is None:
        c.fail("pytest", "no virtualenv to run tests with")
    else:
        env = dict(os.environ)
        env["PYTHONUTF8"] = "1"
        env.pop("RUN_SLOW", None)
        code, out = run([str(py), "-m", "pytest", "-q", "-p", "no:cacheprovider", "--tb=short"], timeout=PYTEST_TIMEOUT_S, env=env)
        m = re.search(r"(\d+) passed", out)
        passed = int(m.group(1)) if m else 0
        failed = re.search(r"(\d+) (failed|error)", out)
        c.check("pytest", code == 0 and passed >= MIN_TESTS_PASSED and failed is None, f"{passed} passed",
                f"exit={code}, passed={passed} (min {MIN_TESTS_PASSED}); tail:\n" + "\n".join(out.strip().splitlines()[-15:]))

    print("-" * 60)
    if c.failures:
        print(f"CHECKS FAILED ({len(c.failures)})")
        for f in c.failures:
            print(f" - {f.splitlines()[0][:300]}")
        return 1
    print(f"{c.passes} checks passed")
    print("ALL CHECKS PASSED")
    return 0


def env_has_llm_key() -> bool:
    """True if .env (or the environment) configures an LLM provider with a non-empty key."""
    values: dict[str, str] = {}
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            v = v.split(" #")[0].strip().strip('"').strip("'")
            values[k.strip()] = v
    for k in ("LLM_PROVIDER", "LLM_API_KEY", "ANTHROPIC_API_KEY"):
        if os.environ.get(k):
            values[k] = os.environ[k]
    provider = values.get("LLM_PROVIDER", "").strip().lower()
    key = values.get("LLM_API_KEY") or (values.get("ANTHROPIC_API_KEY") if provider == "anthropic" else "")
    return provider in {"anthropic", "openai_compatible"} and bool(key)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
