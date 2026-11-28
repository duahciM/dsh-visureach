param([string]$RunName = ('partial_modules_stage8_replay_' + (Get-Date -Format 'yyyyMMdd_HHmmss')))
$ErrorActionPreference = 'Stop'
if ($RunName -cnotmatch '^partial_modules_stage8_replay_[a-zA-Z0-9_]+$') {
    throw 'Use a distinct Python package name beginning with partial_modules_stage8_replay_.'
}
$project = Split-Path -Parent $PSScriptRoot
$target = Join-Path $project $RunName
if (Test-Path -LiteralPath $target) { throw "Replay destination exists: $target" }
New-Item -ItemType Directory -Path $target | Out-Null
New-Item -ItemType Directory -Path (Join-Path $target 'inputs') | Out-Null
New-Item -ItemType Directory -Path (Join-Path $target 'assessment') | Out-Null
Get-ChildItem -LiteralPath $PSScriptRoot -File |
    Where-Object { $_.Extension -eq '.py' -or $_.Name -in @('freeze.json', 'amr_probe.mjs') } |
    ForEach-Object { Copy-Item -LiteralPath $_.FullName -Destination $target }
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'inputs/cases.json') -Destination (Join-Path $target 'inputs')
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'assessment/sealed.json') -Destination (Join-Path $target 'assessment')
$python = Join-Path $project '.venv/Scripts/python.exe'
Push-Location -LiteralPath $project
try {
    foreach ($stage in @('run bind', 'run automatic', 'replay_binding', 'run m4', 'refine', 'evaluate', 'diagnostics', 'regression')) {
        $parts = $stage.Split(' ')
        $module = "$RunName.$($parts[0])"
        $arguments = @('-B', '-X', 'utf8', '-m', $module)
        if ($parts.Length -gt 1) { $arguments += $parts[1] }
        & $python @arguments
        if ($LASTEXITCODE -ne 0) { throw "Replay stage failed: $stage ($LASTEXITCODE)" }
    }
    Write-Output $target
} finally {
    Pop-Location
}
