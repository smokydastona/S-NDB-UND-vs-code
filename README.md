# Sound Generator (Prompt → SFX WAV)

Generate short sound effects from a text prompt.

This project supports two engines:

- **diffusers**: AI prompt-to-audio (AudioLDM2)
- **rfxgen**: procedural chiptune-style SFX presets (coin/laser/explosion/etc)

## Setup (Windows)

1) Create a virtual env:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Install dependencies:

```powershell
pip install -r requirements.txt
```

> Note: The first run will download model weights.

## Generate from the CLI

```powershell
	python -m soundgen.generate --prompt "laser zap" --seconds 2.5 --out outputs\laser.wav
```

### Use rfxgen presets (optional)

1) Download or build `rfxgen.exe` from https://github.com/raysan5/rfxgen

Quickest on Windows (downloads the latest release asset and installs to `tools/rfxgen/rfxgen.exe`):

```powershell
./scripts/get_rfxgen.ps1
```

2) Put it at `tools/rfxgen/rfxgen.exe` (or add it to your `PATH`)

3) Generate using the procedural engine:

```powershell
python -m soundgen.generate --engine rfxgen --prompt "coin pickup" --out outputs\coin.wav
python -m soundgen.generate --engine rfxgen --preset explosion --prompt "boom" --out outputs\boom.wav
```

Common options:
- `--seconds` duration
- `--seed` for repeatable results
- `--device` `cpu` or `cuda`
- `--model` model id (default `cvssp/audioldm2`)

## Minecraft resource pack output (.ogg)

Minecraft resource packs use `.ogg` sound files under `assets/<namespace>/sounds/` and a `sounds.json` file.

This project can export directly into a pack folder:

```powershell
python -m soundgen.generate --engine rfxgen --minecraft --namespace mymod --prompt "coin pickup"
```

That creates/updates:
- `resourcepack/pack.mcmeta`
- `resourcepack/assets/mymod/sounds/generated/<slug>.ogg`
- `resourcepack/assets/mymod/sounds.json`

In-game you can test with:

```mcfunction
/playsound mymod:generated.<slug> master @s
```

### Variants + subtitles (recommended)

Generate multiple variants under one event id so Minecraft randomly picks:

```powershell
python -m soundgen.generate --engine rfxgen --minecraft --namespace mymod --event ui.coin --variants 5 --subtitle "Coin" --prompt "coin pickup"
```

You can also enable the Minecraft-friendly post-processing chain:

```powershell
python -m soundgen.generate --engine rfxgen --prompt "coin pickup" --post --out outputs\coin.wav
```

Then:

```mcfunction
/playsound mymod:ui.coin master @s
```

### Forge mod export

Point `--pack-root` at your mod resources folder and set `--mc-target forge`:

```powershell
python -m soundgen.generate --engine rfxgen --minecraft --mc-target forge --pack-root "C:\path\to\YourMod\src\main\resources" --namespace yourmodid --event ui.coin --prompt "coin pickup"
```

## Batch generation (manifest)

Create a JSON manifest (example: `sounds.json`):

```json
[
	{"engine": "rfxgen", "namespace": "mymod", "event": "ui.coin", "prompt": "coin pickup", "variants": 5, "subtitle": "Coin"},
	{"engine": "diffusers", "namespace": "mymod", "event": "sfx.magic", "prompt": "short magical sparkle", "seconds": 2.0, "post": true}
]
```

Run:

```powershell
python -m soundgen.batch --manifest sounds.json --pack-root resourcepack --mc-target resourcepack --zip outputs\resourcepack.zip
```

This appends a local catalog to `library/catalog.jsonl`.

## Optional paid API engine: Replicate

Set your token:

```powershell
$env:REPLICATE_API_TOKEN = "<your token>"
```

Then run (example only; model inputs vary by model):

```powershell
python -m soundgen.generate --engine replicate --replicate-model "owner/model" --prompt "coin pickup" --seconds 2 --out outputs\replicate.wav
```

### Requirements

- `ffmpeg` must be installed and available on your `PATH` for WAV → OGG conversion.
	- Quick install (Windows): `winget install Gyan.FFmpeg`

## Run the Web UI (Gradio)

```powershell
python -m soundgen.web
```

Then open the local URL printed in the terminal.

## Troubleshooting

- **Slow on CPU**: Try shorter durations (1–3 seconds) or use a GPU (`--device cuda`).
- **CUDA not found**: Install a CUDA-enabled PyTorch build and ensure your NVIDIA drivers are installed.
- **Audio saving issues**: This project writes WAV via `soundfile`.

## Output

Generated files go to `outputs/` by default.
