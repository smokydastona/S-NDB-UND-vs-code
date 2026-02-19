[CmdletBinding()]
param(
    # Download a specific tag (e.g. "5.0"). If omitted, uses latest release.
    [Parameter(Mandatory = $false)]
    [string]$Tag = "",

    # Where to install rfxgen.exe in this repo.
    [Parameter(Mandatory = $false)]
    [string]$InstallDir = "tools/rfxgen",

    # Regex used to pick the right Windows asset from the release.
    # Examples you can try if selection fails: "win", "windows", "win64", "x64".
    [Parameter(Mandatory = $false)]
    [string]$AssetRegex = "win.*(x64|64)|windows|win64|win_64|win-x64"
)

$ErrorActionPreference = "Stop"

function Write-Info([string]$Message) {
    Write-Host "[get_rfxgen] $Message"
}

function Get-ReleaseJson([string]$Tag) {
    $headers = @{
        "Accept" = "application/vnd.github+json"
        "User-Agent" = "sound-generator-get-rfxgen"
    }

    if ([string]::IsNullOrWhiteSpace($Tag)) {
        $url = "https://api.github.com/repos/raysan5/rfxgen/releases/latest"
    } else {
        $url = "https://api.github.com/repos/raysan5/rfxgen/releases/tags/$Tag"
    }

    Write-Info "Fetching release metadata: $url"
    return Invoke-RestMethod -Uri $url -Headers $headers -Method Get
}

function Select-Asset($release, [string]$AssetRegex) {
    if (-not $release.assets -or $release.assets.Count -eq 0) {
        throw "No release assets found. Try building rfxgen from source instead."
    }

    $assets = @($release.assets)

    # Prefer zip assets first
    $zipAssets = $assets | Where-Object { $_.name -match "\.zip$" }
    $candidates = if ($zipAssets.Count -gt 0) { $zipAssets } else { $assets }

    $picked = $candidates | Where-Object { $_.name -match $AssetRegex } | Select-Object -First 1
    if (-not $picked) {
        $names = ($candidates | Select-Object -ExpandProperty name) -join ", "
        throw "Could not find a Windows asset matching regex '$AssetRegex'. Available assets: $names"
    }

    return $picked
}

# ---- main ----

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot ".."))
Push-Location $repoRoot
try {
    $release = Get-ReleaseJson -Tag $Tag
    $asset = Select-Asset -release $release -AssetRegex $AssetRegex

    $cacheDir = Join-Path $env:TEMP "rfxgen_downloads"
    $zipPath = Join-Path $cacheDir $asset.name

    if (-not (Test-Path $zipPath)) {
        Write-Info "Downloading: $($asset.browser_download_url)"
        Write-Info "To: $zipPath"

        $tmpDir = Split-Path -Parent $zipPath
        if (-not (Test-Path $tmpDir)) {
            New-Item -ItemType Directory -Path $tmpDir | Out-Null
        }

        Invoke-WebRequest -Uri $asset.browser_download_url -OutFile $zipPath -UseBasicParsing
    } else {
        Write-Info "Using cached download: $zipPath"
    }

    $installPath = Resolve-Path -LiteralPath (Join-Path $InstallDir ".") -ErrorAction SilentlyContinue
    if (-not $installPath) {
        New-Item -ItemType Directory -Path $InstallDir | Out-Null
        $installPath = Resolve-Path -LiteralPath (Join-Path $InstallDir ".")
    }

    $extractDir = Join-Path $env:TEMP ("rfxgen_extract_" + [Guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Path $extractDir | Out-Null

    try {
        Write-Info "Extracting archive"
        Expand-Archive -Path $zipPath -DestinationPath $extractDir -Force

        $exe = Get-ChildItem -Path $extractDir -Recurse -Filter "rfxgen.exe" | Select-Object -First 1
        if (-not $exe) {
            throw "Could not find rfxgen.exe inside the downloaded archive."
        }

        $exePath = Join-Path $installPath "rfxgen.exe"
        Copy-Item -Path $exe.FullName -Destination $exePath -Force
        Write-Info "Installed: $exePath"
    } finally {
        Remove-Item -Path $extractDir -Recurse -Force -ErrorAction SilentlyContinue
    }

    Write-Info "Quick check: rfxgen.exe --help"
    & $exePath --help | Select-Object -First 20

    Write-Info "Done. You can now run: python -m soundgen.generate --engine rfxgen ..."
} finally {
    Pop-Location
}
