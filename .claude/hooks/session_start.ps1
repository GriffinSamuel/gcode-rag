# Claude Code SessionStart hook (matcher: compact|resume). Plain text on stdout is added to Claude's context.
# Re-injects the operating rules and the current state after context compaction or a session resume.
$ErrorActionPreference = "Continue"
$root = $env:CLAUDE_PROJECT_DIR
if ([string]::IsNullOrWhiteSpace($root)) { $root = (Get-Location).Path }

Write-Output "POST-COMPACTION / RESUME REMINDER (gcode-rag unattended build):"
Write-Output "1. Re-read CLAUDE.md and PROGRESS.md before touching code. docs/SPEC.md is the contract."
Write-Output "2. Never stop to ask the user anything. Decide, log the decision in docs/DECISIONS.md, continue."
Write-Output "3. Work the first unchecked PROGRESS item. Tick it when its check passes. Commit after each item."
Write-Output "4. Same step failed twice -> delegate to the fable-escalation subagent with a full brief."
Write-Output "5. Finished only when .venv/Scripts/python.exe scripts/verify_done.py prints ALL CHECKS PASSED."

$progress = Join-Path $root "PROGRESS.md"
if (Test-Path $progress) {
    $unchecked = Get-Content $progress -Encoding UTF8 | Where-Object { $_ -match '^- \[ \]' } | Select-Object -First 25
    if ($unchecked) {
        Write-Output ""
        Write-Output "Unchecked PROGRESS.md items:"
        $unchecked | ForEach-Object { Write-Output $_ }
    } else {
        Write-Output ""
        Write-Output "PROGRESS.md: all items checked. Run scripts/verify_done.py to confirm completion."
    }
}

$decisions = Join-Path $root "docs\DECISIONS.md"
if (Test-Path $decisions) {
    Write-Output ""
    Write-Output "Most recent decisions (docs/DECISIONS.md tail):"
    Get-Content $decisions -Encoding UTF8 | Select-Object -Last 12 | ForEach-Object { Write-Output $_ }
}
exit 0
