# KICKOFF: how to launch the one-shot build

Everything Claude Code needs is in this folder. You do steps 1-5 once; the session runs unattended after that.

## 1. Prerequisites (2 minutes)

Open Git Bash or PowerShell in this folder and check:

```bash
python --version        # need 3.10 or newer (if "python" is not found, try: py -3 --version)
git --version
claude update           # the loop uses /goal, --effort, auto mode and Fable subagents: be on the newest version
claude --version
```

## 2. Optional but recommended: LLM key

Without a key the project still builds, tests, and evaluates retrieval, but the "Generate G-code" tab and the
answer-accuracy number will be skipped. With a key you get the full project.

```bash
cp .env.example .env
```

Edit `.env` and fill in `LLM_API_KEY`. Cheapest reliable choice: an Anthropic API key with `LLM_MODEL=claude-haiku-5-5`
($5 of credit is far more than this project uses). Free alternative: Groq (`LLM_PROVIDER=openai_compatible`,
`LLM_BASE_URL=https://api.groq.com/openai/v1`, a current Groq model name). Optional: `HF_TOKEN` to auto-deploy the demo.

## 3. Install the loop config, then launch

The Stop-hook loop and the two Fable subagents live in `.claude/`, which could not be written remotely, so they are
staged in `claude-setup/`. Move them into place with ONE of these (run from this folder), then delete the staging folder:

PowerShell:
```powershell
New-Item -ItemType Directory -Force .claude | Out-Null; Copy-Item claude-setup\settings.json, claude-setup\hooks, claude-setup\agents -Destination .claude -Recurse -Force; Remove-Item claude-setup -Recurse -Force; Get-ChildItem -Recurse .claude | Select-Object FullName
```

Git Bash:
```bash
mkdir -p .claude && cp -r claude-setup/settings.json claude-setup/hooks claude-setup/agents .claude/ && rm -rf claude-setup && find .claude -type f
```

You should see `.claude/settings.json`, `.claude/hooks/stop_gate.ps1`, `.claude/hooks/stop_gate.py`,
`.claude/hooks/session_start.ps1`, `.claude/agents/fable-escalation.md`, `.claude/agents/final-reviewer.md`.
Do this BEFORE launching: Claude Code only discovers a new `.claude/agents/` folder at startup.

```bash
claude --model opus --effort high --permission-mode auto
```

- When Claude Code asks whether to trust this folder, accept: the Stop-hook loop in `.claude/settings.json` only runs in a trusted folder.
- If the status bar does not show `auto mode on`, press Shift+Tab until it does (auto mode approves routine commands;
  manual mode would stall the run waiting for you).

## 4. Paste this as your first and only message

```text
/goal Build the gcode-rag project in this folder to completion, unattended, following CLAUDE.md exactly. The goal is met only when ALL of these are shown in the conversation: (1) every checkbox in PROGRESS.md is checked; (2) the final command you ran was `.venv/Scripts/python.exe scripts/verify_done.py` and its output ends with the exact line ALL CHECKS PASSED; (3) `git status --porcelain` printed nothing; (4) FINAL_REPORT.md exists. Working rules: first read CLAUDE.md, PROGRESS.md, docs/SPEC.md and docs/EVAL.md in full; then work through PROGRESS.md strictly in order, ticking each box the moment its check passes and committing after every item. Never stop to ask me anything: decide, record the decision in docs/DECISIONS.md, and continue. When the same step fails twice, immediately delegate the blocker to the fable-escalation subagent with a complete brief (PROGRESS item, exact commands, full error text, what you tried, file paths, acceptance check), apply its fix, and continue. Before declaring completion run the final-reviewer subagent and fix every MUST-FIX finding. This goal is never impossible: CLAUDE.md section 5 gives a fallback for every external dependency (Haas scraping, embedding backend, LLM key, Hugging Face token), so apply the fallback instead of giving up. Whenever you think you are done, run verify_done.py and paste its full output; if any line starts with FAIL, fix the first failure and continue.
```

## 5. What happens next

Three layers keep it running:

1. `/goal`: after every turn a small model checks the condition above and sends Claude back to work if it is not met.
2. The Stop hook (`.claude/hooks/stop_gate.ps1` -> `stop_gate.py`): every time Claude tries to end a turn it runs
   `scripts/verify_done.py` (which includes the test suite) and refuses the stop until it prints `ALL CHECKS PASSED`,
   feeding the failing checks and the unchecked PROGRESS items back as the next instruction. If PROGRESS.md and git
   stop changing for 4 stops in a row it orders an escalation to the `fable-escalation` subagent (Fable 5.1, max
   effort); after 10 no-progress stops it lets the session end so it cannot burn budget forever.
3. `.claude/agents/final-reviewer.md`: an independent Fable review before completion.

Expect several hours. Progress is visible in `PROGRESS.md`, `docs/DECISIONS.md`, and `git log --oneline`.
`/goal` with no arguments shows the goal status and token spend.

## 6. Controls

- Pause / stop: `/goal clear` then Ctrl+C. Resume later with `claude --continue` (the goal is restored automatically).
- If you see `Goal cleared after an unrecoverable error` (usage limit exhausted, model unavailable, context overflow):
  fix the cause and paste the `/goal` message again. A usage-limit pause normally resumes by itself at the reset.
- To re-arm the Stop gate after it has passed once: delete `.claude/hooks/state/DONE`.
- Usage: this session and your other Claude Code session share one usage pool. Running both at once is fine, but the
  build will slow down or pause if the pool runs out; it resumes automatically when the limit resets.

## 7. When it is finished

1. Read `FINAL_REPORT.md` (what to say in interviews, which numbers to quote).
2. Run the app: `.venv/Scripts/python.exe app.py` and open http://127.0.0.1:7860.
3. Publish (the agent never pushes): create an empty GitHub repo, then

```bash
git remote add origin https://github.com/<you>/gcode-rag.git
git push -u origin main
```

## 8. Model and effort choices (why these)

- Main session `--model opus --effort high`: Opus 5.5 is the strongest workhorse tier for long autonomous builds; its
  default effort is `medium`, and an unattended run has nobody to catch sloppy steps, so `high` is worth the extra cost.
  `max`/`xhigh` on the main session mostly adds cost on routine implementation; the hard parts go to Fable anyway.
- Escalation `fable-escalation`: Fable 5.1 at `max` effort, only invoked when something fails twice.
- Review `final-reviewer`: Fable 5.1 at `high` effort, once.
- If budget is not a concern: `claude --model fable --effort high --permission-mode auto` runs the whole build on Fable
  and will need fewer escalations.

## 9. Troubleshooting

| Symptom | Fix |
| --- | --- |
| `/goal` says auto mode unavailable or Claude keeps asking permission | Shift+Tab to auto mode; or `claude update` (auto mode is the default on current versions) |
| `/hooks` does not list the Stop hook | Trust the folder (restart Claude Code in the folder and accept), check `.claude/settings.json` is valid JSON |
| Stop hook error about PowerShell | Run `powershell.exe -NoProfile -ExecutionPolicy Bypass -File .claude/hooks/stop_gate.ps1 < NUL` manually and read `.claude/hooks/state/stop_gate.stderr.log` |
| Build ended with "no progress for 10 consecutive stops" | Read `.claude/hooks/state/stop_gate.log` and the tail of `docs/DECISIONS.md`, fix or decide the blocker yourself, delete `.claude/hooks/state/streak`, paste the `/goal` message again |
| Python not found inside Claude Code | Install Python 3.12 from python.org with "Add to PATH" checked, restart the terminal |
