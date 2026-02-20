# SÖNDBÖUND — Electron shell

This folder contains an Electron wrapper that launches the existing Python/Gradio UI backend and embeds it in a native desktop window.

## Why

- Native window frame (no browser chrome)
- App menu hooks (reload / devtools / quit)
- A straightforward place to add tray + auto-update later

## Dev run (repo checkout)

Prereqs:
- Node.js (LTS recommended)
- A working Python environment for this repo (`.venv` recommended)

From repo root:

```powershell
cd electron
npm install
npm run dev
```

By default, it uses the repo venv at `../.venv/Scripts/python.exe` on Windows.

### Override Python

```powershell
$env:SOUNDGEN_PYTHON = "C:\Path\To\python.exe"
npm run dev
```

## How it works

- Electron starts: `python -m soundgen.app serve --host 127.0.0.1 --port 0`
- The backend prints a line like: `SOUNDGEN_URL=http://127.0.0.1:7860`
- Electron reads that URL from stdout and loads it in a `BrowserWindow`.

Backend logs are written under the Electron user data folder (see `electron-backend.log`).

## Models on first run

When running as a packaged app, Electron checks whether the default Hugging Face models are already cached.

- If models are missing, it prompts to download them and shows a simple progress window.
- Downloads go into the per-user app cache (Electron `userData` → `cache/hf/...`) via `SOUNDGEN_DATA_DIR`.

You can disable the prompt (for CI or debugging) by setting:

```powershell
$env:SOUNDGEN_SKIP_MODEL_CHECK = "1"
```

## Auto-update (packaged builds)

When running as a packaged app, SÖNDBÖUND can check GitHub Releases for updates (via `electron-updater`).

To disable update checks:

```powershell
$env:SOUNDGEN_AUTO_UPDATE = "0"
```

## Crash dumps / crash reporting

Packaged builds write Windows crash dumps under the Electron user data folder:

- `%APPDATA%\\SÖNDBÖUND\\crashdumps`

Optional: if you have a crash upload endpoint (Crashpad-compatible), set:

```powershell
$env:SOUNDGEN_CRASH_SUBMIT_URL = "https://your-crash-endpoint.example/upload"
```

## System tray (optional)

To enable a Windows-style system tray icon (click to show/hide, right-click for Quit):

```powershell
$env:SOUNDGEN_TRAY = "1"
```

When tray is enabled, closing the window will hide it instead of quitting (use the tray menu to Quit).

## Build a self-contained installer (no Python required)

This bundles the existing PyInstaller backend (`dist/SÖNDBÖUND/`) into the Electron app as `extraResources`, then produces a Windows installer.

### One-command build (recommended)

From repo root:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/build_windows_app.ps1 -Clean
```

To build a Windows Store AppX instead of an NSIS installer:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/build_windows_app.ps1 -Clean -Store
```

To build a portable EXE (no installer):

```powershell
powershell -ExecutionPolicy Bypass -File scripts/build_windows_app.ps1 -Clean -Portable
```

1) Build the backend EXE (from repo root):

```powershell
powershell -ExecutionPolicy Bypass -File scripts/build_exe.ps1 -Clean
```

Notes:
- The backend build targets Python 3.12 by default (some Windows wheels — especially `pythonnet` used by `pywebview` — can lag very new Python releases).
- Desktop wrapper diagnostics are written to `%LOCALAPPDATA%\SÖNDBÖUND\desktop.log`.

2) Build Electron (from `electron/`):

```powershell
cd electron
npm install
npm run dist
```

The output will be in `electron/dist/`.

## Build a portable EXE

From repo root:

```powershell
cd electron
npm install
npm run dist:portable
```

## Build a Windows Store package (AppX)

Electron Builder supports AppX (Windows Store) packaging.

From repo root:

```powershell
cd electron
npm install
npm run dist:store
```

Notes:
- AppX assets can be provided under `electron/build/appx/` (the build script creates basic placeholders by copying `.examples/icon.png`).
- For actual Store submission you will need to reserve your app identity in Partner Center; you may also need to adjust AppX metadata in the Electron Builder `build.appx` config.

## Code signing (optional but strongly recommended)

Unsigned installers often trigger Windows SmartScreen warnings.

Electron Builder supports signing via environment variables:

```powershell
$env:CSC_LINK = "C:\path\to\certificate.p12"  # or a base64 data URI
$env:CSC_KEY_PASSWORD = "your-password"
npm run dist
```

If you don’t have a certificate yet, you can still build unsigned for local testing.
