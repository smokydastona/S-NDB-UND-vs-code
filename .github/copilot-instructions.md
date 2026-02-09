# Copilot instructions (soundgen)

## Big picture
- This is a **Python (src-layout)** project for generating **Minecraft-ready SFX** from prompts.
- Main data flow: **prompt → engine generates mono WAV (float32) → optional post-process → export**
  - WAV write: [src/soundgen/io_utils.py](../src/soundgen/io_utils.py)
  - Post-process chain + QA flags: [src/soundgen/postprocess.py](../src/soundgen/postprocess.py), [src/soundgen/qa.py](../src/soundgen/qa.py)
  - Minecraft export (`.ogg` + `sounds.json` + optional `lang/en_us.json`): [src/soundgen/minecraft.py](../src/soundgen/minecraft.py)

## Entry points (use these patterns)
- CLI (single sound): [src/soundgen/generate.py](../src/soundgen/generate.py)
  - `python -m soundgen.generate --engine rfxgen --prompt "coin pickup" --post`
  - Minecraft: `--minecraft --namespace mymod --event ui.coin --variants 5 --subtitle "Coin"`
- Web UI (Gradio): [src/soundgen/web.py](../src/soundgen/web.py)
- Batch from manifest (CSV/JSON): [src/soundgen/batch.py](../src/soundgen/batch.py)
- Doc inputs (drop docs in `pre_gen_sound/`): [src/soundgen/from_docs.py](../src/soundgen/from_docs.py)

## Engines (how they’re wired)
- `diffusers` engine: [src/soundgen/audiogen_backend.py](../src/soundgen/audiogen_backend.py)
  - Uses `AudioLDM2Pipeline`; includes a GPT-2 LM head workaround.
- `rfxgen` engine: [src/soundgen/rfxgen_backend.py](../src/soundgen/rfxgen_backend.py)
  - External `rfxgen.exe`; Windows helper: [scripts/get_rfxgen.ps1](../scripts/get_rfxgen.ps1)
- Optional paid API engine: `replicate` in [src/soundgen/replicate_backend.py](../src/soundgen/replicate_backend.py)
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

## Quick smoke checks
- CLI generation: `python -m soundgen.generate --engine rfxgen --prompt "coin pickup" --out outputs\\test.wav --post`
- Minecraft export: `python -m soundgen.generate --engine rfxgen --minecraft --namespace mymod --event ui.coin --prompt "coin pickup" --post`
- Batch: `python -m soundgen.batch --manifest example_manifest.json --zip outputs\\resourcepack.zip`
