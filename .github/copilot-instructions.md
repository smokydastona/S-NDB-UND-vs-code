# Copilot instructions (soundgen)

## Big picture
- This is a **Python (src-layout)** project for generating **Minecraft-ready SFX** from prompts.
- Main data flow: **prompt → engine generates mono WAV (float32) → optional post-process → export**
  - WAV write: `src/soundgen/io_utils.py`
  - Post-process chain + QA flags: `src/soundgen/postprocess.py`, `src/soundgen/qa.py`
  - Minecraft export (`.ogg` + `sounds.json` + optional `lang/en_us.json`): `src/soundgen/minecraft.py`

## Entry points (use these patterns)
- CLI (single sound): `src/soundgen/generate.py`
  - `python -m soundgen.generate --engine rfxgen --prompt "coin pickup" --post`
  - Minecraft: `--minecraft --namespace mymod --event ui.coin --variants 5 --subtitle "Coin"`
- Web UI (Gradio): `src/soundgen/web.py`
- Batch from manifest (CSV/JSON): `src/soundgen/batch.py`
- Doc inputs (drop docs in `pre_gen_sound/`): `src/soundgen/from_docs.py`

## Engines (how they’re wired)
- `diffusers` engine: `src/soundgen/audiogen_backend.py`
  - Uses `AudioLDM2Pipeline`; includes a GPT-2 LM head workaround.
- `rfxgen` engine: `src/soundgen/rfxgen_backend.py`
  - External `rfxgen.exe`; Windows helper: [scripts/get_rfxgen.ps1](../scripts/get_rfxgen.ps1)
  - External `rfxgen.exe`; Windows helper: `scripts/get_rfxgen.ps1`
- Optional paid API engine: `replicate` in `src/soundgen/replicate_backend.py`
  - Token via `REPLICATE_API_TOKEN` or `--replicate-token`.

## Minecraft 1.20.1 conventions
- `sounds.json` key is the **event** (e.g. `ui.coin`); the full id is `<namespace>:<event>`.
- Variants are written as **object entries** with `name/weight/volume/pitch`.
- Subtitles: writes `subtitle` in `sounds.json` and adds `assets/<ns>/lang/en_us.json`.

## External tools / dependencies
- `.ogg` export requires **ffmpeg**. The code finds it on PATH or WinGet’s install path.
  - Typical install: `winget install Gyan.FFmpeg`

## Repo conventions (important for agents)
- Keep audio arrays **mono 1D float32 in [-1, 1]** between stages.
- Prefer adding new functionality as small modules under `src/soundgen/` and then wiring into:
  - CLI flags in `soundgen.generate`
  - (Optionally) Gradio controls in `soundgen.web`
- Local/generated folders are intentionally gitignored: `outputs/`, `resourcepack/`, `pre_gen_sound/`, `tools/rfxgen/`, `library/`, `*.egg-info/`.

## Workflow after every change (required)

After implementing any requested change, ALWAYS run a full workspace scan/validation, then commit and push.

### Full workspace scan (do this every time)
- Check VS Code Problems across the workspace.
- Run a fast syntax/import validation: `python -m compileall -q src/soundgen`.
- If you touched CLI flags or behavior, ensure `README.md` examples/help text are updated.

### Fix + validate loop
1. **Scan first** (full scan above)
2. **Fix errors systematically**
  - Don’t stop after fixing one file; iterate until the workspace is clean for the area you changed.
3. **Re-validate after each fix**
  - Re-run the relevant smoke checks (below). If behavior changed, refresh docs immediately.
4. **Keep outputs out of git**
  - Do not commit generated audio, packs, or local catalogs. Extend `.gitignore` instead.

### Commit + push (do this after changes)
- Stage only real source/config/doc changes (never generated audio/outputs).
- Commit with a short, descriptive message (prefer `feat:`, `fix:`, `docs:`, `chore:`).
- Push to the current branch.

### User intent: “push”
- When the user says **“push”**, interpret it as: **run the full workspace scan → stage → commit → push**.

### Explain changes
- State what was wrong, what changed, and what to run to verify.

### “Scan likely impact radius” definition
After the workspace-wide scan, proactively review the connected modules/configs that must remain consistent so we don’t ship a change that breaks generation/export.

### Impact radius checklists
- **Audio format / I/O changes** (`src/soundgen/io_utils.py`)
  - Ensure all intermediate arrays remain mono `float32` in `[-1, 1]`.
  - WAV write stays Minecraft-friendly (`PCM_16`) unless there’s a strong reason.
- **Post-process / QA changes** (`src/soundgen/postprocess.py`, `src/soundgen/qa.py`)
  - Confirm no NaNs/inf introduced; confirm final clip to `[-1, 1]` remains.
  - Ensure trimming never returns empty audio (downstream code expects non-empty).
- **Minecraft export changes** (`src/soundgen/minecraft.py`)
  - Keep event ids sanitized and stable: `<namespace>:<event>`.
  - Preserve `sounds.json` schema: event → `{sounds: [...]}` with object entries for `name/weight/volume/pitch`.
  - Verify subtitle writes both `sounds.json` and `assets/<ns>/lang/en_us.json`.
  - Re-check ffmpeg discovery and the `wav->ogg` conversion command.
- **Engine wiring changes** (`src/soundgen/generate.py`, `src/soundgen/audiogen_backend.py`, `src/soundgen/rfxgen_backend.py`, `src/soundgen/replicate_backend.py`)
  - Keep heavy deps lazily imported so `--help` stays fast.
  - Maintain deterministic behavior with `--seed` where supported.
  - Ensure rfxgen executable resolution still supports PATH + `tools/rfxgen/rfxgen.exe` + `--rfxgen-path`.
- **Web UI changes** (`src/soundgen/web.py`)
  - Keep UI controls in sync with CLI meaning (post-processing, Minecraft export fields).
  - Verify generated file + playsound string + QA text still return correctly.
- **Batch / docs workflow changes** (`src/soundgen/batch.py`, `src/soundgen/from_docs.py`)
  - Keep manifest schema backwards compatible when possible.
  - Confirm `pre_gen_sound/` is still gitignored and doc parsing remains robust.

## Copilot behavior rules (repo-specific)
- Don’t introduce stereo processing unless explicitly requested; default is mono.
- Don’t change Minecraft id sanitization rules casually; they affect pack compatibility.
- Don’t add new engine dependencies that break `--help` (keep imports lazy).
- If you add a new feature, wire it through the CLI first (`soundgen.generate`), then optionally the Gradio UI.

## Quick smoke checks
- CLI generation: `python -m soundgen.generate --engine rfxgen --prompt "coin pickup" --out outputs\\test.wav --post`
- Minecraft export: `python -m soundgen.generate --engine rfxgen --minecraft --namespace mymod --event ui.coin --prompt "coin pickup" --post`
- Batch: `python -m soundgen.batch --manifest example_manifest.json --zip outputs\\resourcepack.zip`
