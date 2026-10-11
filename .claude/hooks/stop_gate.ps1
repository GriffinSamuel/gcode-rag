# Claude Code Stop hook wrapper for Windows.
# Finds a working Python (project .venv first, then the py launcher, then python on PATH) and runs stop_gate.py,
# passing the hook's stdin JSON through and returning the script's stdout and exit code unchanged.
# Registered in .claude/settings.json in exec form:
#   powershell.exe -NoProfile -ExecutionPolicy Bypass -File <project>/.claude/hooks/stop_gate.ps1
$ErrorActionPreference = "Continue"

$root = $env:CLAUDE_PROJECT_DIR
if ([string]::IsNullOrWhiteSpace($root)) { $root = (Get-Location).Path }
$script = Join-Path $root ".claude\hooks\stop_gate.py"
$stateDir = Join-Path $root ".claude\hooks\state"
if (-not (Test-Path $stateDir)) { New-Item -ItemType Directory -Path $stateDir -Force | Out-Null }
$errLog = Join-Path $stateDir "stop_gate.stderr.log"

if (-not (Test-Path $script)) {
    # Nothing to run: allow the stop silently.
    exit 0
}

$stdinText = [Console]::In.ReadToEnd()

# Candidate interpreters in priority order: @(executable, extra-args)
$candidates = @()
$venvPy = Join-Path $root ".venv\Scripts\python.exe"
if (Test-Path $venvPy) { $candidates += ,@($venvPy, @()) }
$pyLauncher = Get-Command "py" -ErrorAction SilentlyContinue
if ($pyLauncher) { $candidates += ,@($pyLauncher.Source, @("-3")) }
foreach ($name in @("python", "python3")) {
    $cmd = Get-Command $name -ErrorAction SilentlyContinue
    if ($cmd) { $candidates += ,@($cmd.Source, @()) }
}

if ($candidates.Count -eq 0) {
    # No Python at all: we cannot gate. Allow the stop; /goal still applies.
    exit 0
}

foreach ($cand in $candidates) {
    $exe = $cand[0]
    $extra = $cand[1]
    $allArgs = @()
    if ($extra) { $allArgs += $extra }
    $allArgs += $script
    try {
        $output = $stdinText | & $exe @allArgs 2>> $errLog
        $code = $LASTEXITCODE
    } catch {
        Add-Content -Path $errLog -Value ("wrapper: failed to start {0}: {1}" -f $exe, $_)
        continue
    }
    # 9009 = Windows "not recognized" / Microsoft Store python stub; try the next candidate.
    if ($code -eq 9009) { continue }
    if ($null -ne $output) { $output | ForEach-Object { Write-Output $_ } }
    exit $code
}

# Every candidate failed to start: allow the stop rather than wedge the session.
exit 0
