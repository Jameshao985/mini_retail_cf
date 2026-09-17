param(
    [ValidateSet('smoke', 'pilot', 'study')][string]$Profile = 'smoke',
    [string]$RunDir = '',
    [string]$ModelPath = '',
    [int]$Seed = 42,
    [int]$TaskSeed = 20260914,
    [int]$Rounds = 1
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$taskPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) {
    throw 'Project .venv is missing. Follow 完整流程与运行说明.md to create it.'
}
$taskArgs = @('-u', '-m', 'experiment.run', '--profile', $Profile, '--seed', "$Seed",
              '--task-seed', "$TaskSeed", '--rounds', "$Rounds")
if ($RunDir) { $taskArgs += @('--run-dir', $RunDir) }
if ($ModelPath) { $taskArgs += @('--model-path', $ModelPath) }
& $taskPython @taskArgs
if ($LASTEXITCODE -ne 0) { throw "Experiment failed with exit code $LASTEXITCODE. See console and LAST_ERROR.json." }
