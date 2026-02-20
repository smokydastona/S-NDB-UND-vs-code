# SÖNDBÖUND (VS Code Extension)

![SÖNDBÖUND banner](https://raw.githubusercontent.com/smokydastona/S-NDB-UND-vs-code/main/.examples/Banner.png)

## What this does

SÖNDBÖUND adds a “prompt → sound” workflow to VS Code (Generate WAV, Export Pack, and an embedded editor UI), powered by the bundled Python backend (`soundgen`).

This repository is an **extension fork/version** of SÖNDBÖUND: it contains both the VS Code extension (TypeScript) and the backend (Python, `src/soundgen/`).

## What you get

- **Embedded UI**: open the built-in SÖNDBÖUND editor inside VS Code.
- **Commands (automation-friendly)**:
  - Generate a WAV from a prompt
  - Export a pack ZIP from a manifest
- **Copilot-friendly**: commands accept structured JSON-like args and can run headless (no popups) when args are provided.

## Install

### Install from VS Code Marketplace (one-click)

- In VS Code: open **Extensions** and search for **SÖNDBÖUND**.
- Or via CLI:

```powershell
code --install-extension smokydastona.sondbound
```

### Install from VSIX (dev / offline)

1. Build the VSIX:

```powershell
npm install
npx --yes @vscode/vsce package
```

2. In VS Code: **Extensions → ⋯ → Install from VSIX…** and select `sondbound-0.0.1.vsix`.

## Quickstart (in VS Code)

### Use (60 seconds)

1. Click the Status Bar button **“SÖNDBÖUND: Generate Sound”** (or run **SÖNDBÖUND: Generate Sound** from Command Palette).
2. Enter a prompt (example: `coin pickup`).

If you get a Python/dependency error on first run, use:

- **SÖNDBÖUND: Setup Backend (Create .venv + pip install)**

### Other commands

Open Command Palette and run:

- **SÖNDBÖUND: Open UI**
- **SÖNDBÖUND: Open Web UI (Control Panel)**
- **SÖNDBÖUND: Open Editor**
- **SÖNDBÖUND: Generate Sound**
- **SÖNDBÖUND: Export Pack**

## Commands (Copilot usage)

These commands are designed so Copilot can invoke them deterministically.

### Copilot rules of engagement

- Prefer calling commands with an `options` object (headless mode). If `options` is omitted, commands may show interactive prompts for humans.
- Commands return machine-readable objects like `{ ok: true, ... }` or `{ ok: false, error }`.
- For logs and backend output, check **Output → SÖNDBÖUND**.

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
  seed?: number;
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
{ ok: true; outputPath: string; engine: string; seconds: number | null; seed: number | null; post: boolean }
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

### `sondbound.openWebUI`

Starts the Python Gradio UI and embeds it inside a VS Code Webview.

Args:

```ts
{
  host?: string;
  port?: number;
  mode?: "control-panel" | "legacy";
  embed?: "proxy" | "direct";
  proxyPort?: number;
}
```

Notes:

- Default embedding is `proxy` (recommended): runs a local reverse proxy that strips frame-blocking headers and tunnels WebSockets.
- Use `embed: "direct"` if you want to iframe the Gradio server URL directly.
```

## Configuration

Settings (VS Code → Settings → search “SÖNDBÖUND”):

- `sondbound.pythonPath`: override Python executable path.
- `sondbound.defaultEngine`: used when a headless generate call omits `engine`.
- `sondbound.defaultSeconds`: used when a headless generate call omits `seconds`.
- `sondbound.defaultPost`: used when a headless generate call omits `post`.
- `sondbound.defaultOutputSubdir`: default output folder (relative to workspace root).
- `sondbound.deterministic`: if true, the extension will pass a stable seed when possible.
- `sondbound.defaultSeed`: seed used when `deterministic=true` and a call omits `seed`.
- `sondbound.webUiHost`: bind host for the local web UI (default `127.0.0.1`).
- `sondbound.webUiPort`: bind port for the local web UI (default `7860`).
- `sondbound.webUiMode`: `control-panel` (default) or `legacy`.
- `sondbound.webUiEmbed`: `proxy` (default, most reliable) or `direct` (iframe Gradio directly).
- `sondbound.webUiProxyPort`: proxy port when `webUiEmbed=proxy` (`0` = auto-pick a free port).

Environment variable alternative:

- `SOUNDGEN_PYTHON`: optional Python override (used if `sondbound.pythonPath` is empty).

## Backend prerequisites (Windows)

The extension shells out to Python to run the `soundgen` backend.

For the Web UI, the backend also needs `gradio` (included via `requirements.txt`).

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

## Tasks (automation)

This repo includes two first-class VS Code Tasks that mirror real workflows (useful for automation, CI, and Copilot/agent chaining):

- `sondbound: generate wav` → runs `python -m soundgen.app generate` (prompt/seconds/seed/out)
- `sondbound: export pack` → runs `python -m soundgen.batch` (manifest/zip)

Run them via Command Palette → **Tasks: Run Task**.

## Packaging

```powershell
npx --yes @vscode/vsce package
```

This repo uses the `files` allow-list in `package.json` to keep the VSIX small and to avoid shipping caches like `__pycache__`.

## Troubleshooting

- If generation fails, open **Output → SÖNDBÖUND** for the backend logs.
- If Python isn’t found, set `sondbound.pythonPath` (or `SOUNDGEN_PYTHON`).
