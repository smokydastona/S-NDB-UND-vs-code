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
  if (Test-Path "soundgen.spec") { Remove-Item -Force "soundgen.spec" }
}

# Install build-time tooling only (kept out of requirements.txt)
python -m pip install --upgrade pip | Out-Null
python -m pip install pyinstaller | Out-Null

# Ensure runtime deps are present (uses requirements.txt)
python -m pip install -r requirements.txt | Out-Null

# Build two executables (folder-based /onedir for reliability)
# Note: AI engines (torch/diffusers/transformers) make these builds large.
$genName = "soundgen-generate"
$webName = "soundgen-web"
$desktopName = "soundgen-desktop"
if ($Version -and $Version.Trim().Length -gt 0) {
  $ver = $Version.Trim()
  $genName = "$genName-$ver"
  $webName = "$webName-$ver"
  $desktopName = "$desktopName-$ver"
}

$commonArgs = @(
  "--noconfirm",
  "--clean",
  "--onedir"
)

$commonCollect = @(
  "--collect-all", "soundgen",
  "--collect-all", "numpy",
  "--collect-all", "scipy",
  "--collect-all", "soundfile",
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

$genArgs = @()
$genArgs += $commonArgs
$genArgs += @("--name", $genName)
$genArgs += $commonCollect
$genArgs += @(
  "--distpath", $OutDir,
  "--workpath", $WorkDir,
  "src/soundgen/generate.py"
)

Invoke-PyInstaller -Args $genArgs

$webArgs = @()
$webArgs += $commonArgs
$webArgs += @("--name", $webName)
$webArgs += @(
  "--collect-all", "gradio"
)
$webArgs += $commonCollect
$webArgs += @(
  "--distpath", $OutDir,
  "--workpath", $WorkDir,
  "src/soundgen/web.py"
)

Invoke-PyInstaller -Args $webArgs

$desktopArgs = @()
$desktopArgs += $commonArgs
$desktopArgs += @("--noconsole")
$desktopArgs += @("--name", $desktopName)
$desktopArgs += @(
  "--collect-all", "gradio",
  "--collect-all", "webview"
)
$desktopArgs += $commonCollect
$desktopArgs += @(
  "--distpath", $OutDir,
  "--workpath", $WorkDir,
  "src/soundgen/desktop.py"
)

Invoke-PyInstaller -Args $desktopArgs

Write-Host "Built executables into $OutDir\\$genName, $OutDir\\$webName, and $OutDir\\$desktopName"