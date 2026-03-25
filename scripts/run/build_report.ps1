$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoDir = Resolve-Path (Join-Path $scriptDir "..\\..")
$reportScript = Join-Path $repoDir "experiments\\report\\build_report.ps1"

& $reportScript @args
exit $LASTEXITCODE
