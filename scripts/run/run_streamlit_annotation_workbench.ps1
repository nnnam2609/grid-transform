$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$streamlitExe = Join-Path $repoRoot ".venv\Scripts\streamlit.exe"
$appPath = Join-Path $repoRoot "grid_transform\apps\streamlit_annotation_workbench.py"
$env:PYTHONIOENCODING = "utf-8"

if (-not (Test-Path $streamlitExe)) {
    throw "Streamlit executable not found at $streamlitExe. Install the repo requirements first."
}

& $streamlitExe run $appPath @args
