Param(
  [string]$OutDir = "dist",
  [string]$WorkDir = "build",
  [string]$Version = "",
  [switch]$Clean
)

$ErrorActionPreference = "Stop"

if ($Clean) {
  if (Test-Path $OutDir) { Remove-Item -Recurse -Force $OutDir }
  if (Test-Path $WorkDir) { Remove-Item -Recurse -Force $WorkDir }
  Get-ChildItem -Path . -Filter "*.spec" -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
}

# Install build-time tooling only (kept out of requirements.txt)
python -m pip install --upgrade pip | Out-Null
python -m pip install pyinstaller | Out-Null
python -m pip install pillow | Out-Null

# Ensure runtime deps are present (uses requirements.txt)
python -m pip install -r requirements.txt | Out-Null

# Ensure the local project package itself is importable for PyInstaller analysis.
python -m pip install -e . | Out-Null

# Build the executable (folder-based /onedir for reliability)
# Note: AI engines (torch/diffusers/transformers) make these builds large.
$baseAppName = "SÖNDBÖUND"
$appName = $baseAppName
if ($Version -and $Version.Trim().Length -gt 0) {
  $ver = $Version.Trim()
  $appName = "$baseAppName-$ver"
}

$commonArgs = @(
  "--noconfirm",
  "--clean",
  "--onedir",
  "--paths", "src"
)

# Optional app icon (Windows): PyInstaller expects .ico
$iconPng = ".examples/icon.png"
$iconIco = Join-Path $WorkDir "app.ico"
if (Test-Path $iconPng) {
  if (!(Test-Path $WorkDir)) { New-Item -ItemType Directory -Path $WorkDir | Out-Null }
  python -c "from PIL import Image; import os; p=r'$iconPng'; o=r'$iconIco'; im=Image.open(p); im.save(o, sizes=[(256,256),(128,128),(64,64),(48,48),(32,32),(16,16)])"
  if (Test-Path $iconIco) {
    $commonArgs += @("--icon", $iconIco)
  }

  # Bundle the original PNG as data so the UI can use it as a background watermark.
  # (PyInstaller expects SRC;DEST on Windows.)
  $commonArgs += @("--add-data", "$iconPng;.examples")
}

$commonCollect = @(
  "--collect-all", "soundgen",
  "--collect-all", "numpy",
  "--collect-all", "scipy",
  "--collect-all", "soundfile",
  "--collect-all", "gradio",
  "--collect-all", "safehttpx",
  "--collect-all", "groovy",
  "--collect-all", "webview",
  "--collect-all", "torch",
  "--collect-all", "diffusers",
  "--collect-all", "transformers",
  "--collect-all", "accelerate",
  "--collect-all", "safetensors"
)

function Invoke-PyInstaller {
  param(
    [Parameter(Mandatory=$true)][string[]]$Args
  )

  python -m PyInstaller @Args
  if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller failed with exit code $LASTEXITCODE"
  }
}

$appArgs = @()
$appArgs += $commonArgs
$appArgs += @("--name", $baseAppName)
$appArgs += $commonCollect
$appArgs += @(
  "--distpath", $OutDir,
  "--workpath", $WorkDir,
  "sndbund_entry.py"
)

Invoke-PyInstaller -Args $appArgs

# If versioned, rename the output folder, but keep the EXE name stable.
if ($appName -ne $baseAppName) {
  $src = Join-Path $OutDir $baseAppName
  $dst = Join-Path $OutDir $appName
  if (Test-Path $dst) { Remove-Item -Recurse -Force $dst }
  if (Test-Path $src) { Move-Item -Force $src $dst }
}

Write-Host "Built executable into $OutDir\\$appName"