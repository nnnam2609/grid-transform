$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoDir = Resolve-Path (Join-Path $scriptDir "..\\..")
$outputDir = Join-Path $repoDir "outputs\\reports"
$reportTex = Join-Path $scriptDir "grid_transformation_report.tex"
$reportPdf = Join-Path $outputDir "grid_transformation_report.pdf"

$venvPython = Join-Path $repoDir ".venv\Scripts\python.exe"
if (Test-Path $venvPython) {
    $pythonExe = $venvPython
} else {
    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if (-not $pythonCmd) {
        throw "Python was not found. Create .venv or install Python first."
    }
    $pythonExe = $pythonCmd.Source
}

if ($env:TECTONIC_EXE -and (Test-Path $env:TECTONIC_EXE)) {
    $tectonicExe = $env:TECTONIC_EXE
} else {
    $tectonicCmd = Get-Command tectonic -ErrorAction SilentlyContinue
    if ($tectonicCmd) {
        $tectonicExe = $tectonicCmd.Source
    } else {
        $localTectonic = Join-Path $env:LOCALAPPDATA "Programs\Tectonic\tectonic.exe"
        if (Test-Path $localTectonic) {
            $tectonicExe = $localTectonic
        } else {
            throw "tectonic was not found. Install it or set TECTONIC_EXE before running this script."
        }
    }
}

New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

& $pythonExe (Join-Path $scriptDir "generate_report_assets.py")
if ($LASTEXITCODE -ne 0) {
    throw "Asset generation failed."
}

Push-Location $scriptDir
try {
    & $tectonicExe "--keep-logs" "--keep-intermediates" "--outdir" $outputDir $reportTex
    if ($LASTEXITCODE -ne 0) {
        throw "tectonic failed to compile the report."
    }
} finally {
    Pop-Location
}

if (-not (Test-Path $reportPdf)) {
    throw "Expected PDF was not produced: $reportPdf"
}

Write-Host "Saved PDF:" $reportPdf
