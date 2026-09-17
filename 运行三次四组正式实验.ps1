param(
    [ValidateSet('pilot', 'study')][string]$Profile = 'study',
    [int[]]$Seeds = @(42, 43, 44),
    [int]$TaskSeed = 20260914,
    [string]$OutputRoot = 'outputs/experiments/four_group_study',
    [ValidateSet('rule', 'deepseek')][string]$Teacher = 'rule',
    [switch]$UseLegacyKey
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$taskPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) {
    throw 'Project .venv is missing. Follow 第二版四组实验说明.md to create it.'
}

$completed = @()
foreach ($trainSeed in $Seeds) {
    $runDir = "${OutputRoot}_s${trainSeed}"
    $taskArgs = @(
        '-u', '-m', 'experiment.run', '--profile', $Profile,
        '--seed', "$trainSeed", '--task-seed', "$TaskSeed",
        '--teacher', $Teacher, '--arms', 'random', 'targeted', 'random_cf', 'targeted_cf',
        '--run-dir', $runDir
    )
    if ($Teacher -eq 'deepseek') {
        $taskArgs += @('--teacher-max-calls', '240')
        if ($UseLegacyKey) { $taskArgs += '--use-legacy-key' }
    }
    & $taskPython @taskArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Experiment failed for training seed $trainSeed. See $runDir/LAST_ERROR.json."
    }
    $completed += $runDir
}

$summary = "${OutputRoot}_summary.csv"
& $taskPython scripts/aggregate_results.py @completed --output $summary
if ($LASTEXITCODE -ne 0) { throw 'Multi-seed aggregation failed.' }
Write-Host "All runs completed. Summary: $summary"
