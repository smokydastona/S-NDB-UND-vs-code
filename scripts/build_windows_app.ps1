Param(
  [string]$PythonVersion = "3.12",
  [string]$Version = "",
  [switch]$Clean,
  [switch]$SkipBackend,
  [switch]$SkipElectron
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot ".."))

function Invoke-BackendBuild {
  param(
    [string]$PythonVersion,
    [string]$Version,
    [switch]$Clean
  )

  Write-Output "[build_windows_app] Building backend (PyInstaller)..."

  $backendCliParams = @()
  if ($Clean) { $backendCliParams += "-Clean" }
  if ($Version -and $Version.Trim().Length -gt 0) { $backendCliParams += @("-Version", $Version.Trim()) }
  if ($PythonVersion -and $PythonVersion.Trim().Length -gt 0) { $backendCliParams += @("-PythonVersion", $PythonVersion.Trim()) }

  $buildExe = Join-Path $repoRoot "scripts\build_exe.ps1"
  & powershell -ExecutionPolicy Bypass -File $buildExe @backendCliParams
  if ($LASTEXITCODE -ne 0) { throw "Backend build failed (exit $LASTEXITCODE)." }
}

function Invoke-ElectronBuild {
  Write-Output "[build_windows_app] Building Electron installer (electron-builder)..."

  Push-Location (Join-Path $repoRoot "electron")
  try {
    if (!(Get-Command npm -ErrorAction SilentlyContinue)) {
      throw "npm not found on PATH. Install Node.js LTS and try again."
    }

    # No lockfile in this repo; use install (ci would fail).
    npm install
    if ($LASTEXITCODE -ne 0) { throw "npm install failed (exit $LASTEXITCODE)." }

    npm run dist
    if ($LASTEXITCODE -ne 0) { throw "npm run dist failed (exit $LASTEXITCODE)." }

    Write-Output "[build_windows_app] Output folder: electron\\dist\\"
  }
  finally {
    Pop-Location
  }
}

Push-Location $repoRoot
try {
  if (-not $SkipBackend) {
    Invoke-BackendBuild -PythonVersion $PythonVersion -Version $Version -Clean:$Clean
  } else {
    Write-Output "[build_windows_app] Skipping backend build (-SkipBackend)."
  }

  if (-not $SkipElectron) {
    Invoke-ElectronBuild
  } else {
    Write-Output "[build_windows_app] Skipping Electron build (-SkipElectron)."
  }

  Write-Output "[build_windows_app] Done."
}
finally {
  Pop-Location
}
