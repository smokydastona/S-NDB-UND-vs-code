# SÖNDBÖUND (VS Code Extension)

SÖNDBÖUND is a VS Code extension that embeds the SÖNDBÖUND editor UI in a Webview and exposes **command-palette + Copilot-friendly commands** to generate/export game audio using the bundled Python backend (`soundgen`).

This repository is an **extension fork/version** of SÖNDBÖUND: it contains both the VS Code extension (TypeScript) and the backend (Python, `src/soundgen/`).

## What you get

- **Embedded UI**: open the built-in SÖNDBÖUND editor inside VS Code.
- **Commands (automation-friendly)**:
  - Generate a WAV from a prompt
  - Export a pack ZIP from a manifest
- **Copilot-friendly**: commands accept structured JSON-like args and can run headless (no popups) when args are provided.

## Install

### Install from VSIX (recommended right now)

1. Build the VSIX:

```powershell
npm install
npx --yes @vscode/vsce package
```

2. In VS Code: **Extensions → ⋯ → Install from VSIX…** and select `sondbound-0.0.1.vsix`.

## Quickstart (in VS Code)

Open Command Palette and run one of:

- **SÖNDBÖUND: Open UI**
- **SÖNDBÖUND: Open Editor**
- **SÖNDBÖUND: Generate Sound**
- **SÖNDBÖUND: Export Pack**

## Commands (Copilot usage)

These commands are designed so Copilot can invoke them deterministically.

### `sondbound.openUI`

Opens the embedded editor Webview.

Args:

```ts
{ wavPath?: string }
```

### `sondbound.generate` / `sondbound.generateSound`

Generates a WAV using the Python backend.

Args:

```ts
{
  prompt: string;
  engine?: string;
  seconds?: number;
  outputPath?: string;
  post?: boolean;
  edit?: boolean;
}
```

Notes:

- If args are provided, the command runs **headless** (no dialogs).
- If args are omitted, it falls back to interactive prompts for humans.

Return value (machine-readable):

```ts
{ ok: true; outputPath: string; engine: string; seconds: number | null; post: boolean }
// or
{ ok: false; error: string }
```

### `sondbound.exportPack`

Exports a ZIP pack from a manifest (JSON/CSV) using `soundgen.batch`.

Args:

```ts
{
  manifestPath: string;
  zipPath?: string;
}
```

Return value:

```ts
{ ok: true; zipPath: string }
// or
{ ok: false; error: string }
```

## Configuration

Settings (VS Code → Settings → search “SÖNDBÖUND”):

- `sondbound.pythonPath`: override Python executable path.
- `sondbound.defaultEngine`: used when a headless generate call omits `engine`.
- `sondbound.defaultSeconds`: used when a headless generate call omits `seconds`.
- `sondbound.defaultPost`: used when a headless generate call omits `post`.
- `sondbound.defaultOutputSubdir`: default output folder (relative to workspace root).

Environment variable alternative:

- `SOUNDGEN_PYTHON`: optional Python override (used if `sondbound.pythonPath` is empty).

## Backend prerequisites (Windows)

The extension shells out to Python to run the `soundgen` backend.

Minimal setup:

```powershell
python -m venv .venv
\.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For `.ogg` export you may need `ffmpeg` on PATH.

## Development

Build the extension:

```powershell
npm install
npm run compile
```

Run locally:

- Press `F5` to launch an **Extension Development Host**
- Use Command Palette to run the SÖNDBÖUND commands

## Packaging

```powershell
npx --yes @vscode/vsce package
```

This repo uses the `files` allow-list in `package.json` to keep the VSIX small and to avoid shipping caches like `__pycache__`.

## Troubleshooting

- If generation fails, open **Output → SÖNDBÖUND** for the backend logs.
- If Python isn’t found, set `sondbound.pythonPath` (or `SOUNDGEN_PYTHON`).
