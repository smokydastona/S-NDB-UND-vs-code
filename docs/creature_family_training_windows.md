# Creature family LoRA training (Windows)

This repo supports **inference-time LoRA loading** for `stable_audio_open`. Training is intentionally *external* (you can use any LoRA trainer that outputs a diffusers-compatible LoRA).

On Windows, the most reliable path is **WSL2 + Ubuntu** (still “Windows”, but with a real Linux CUDA stack). Native Windows training can work, but it’s more fragile.

## 0) Prepare the dataset (inside this repo)

Export a training-ready folder from your existing generations:

```powershell
# From the repo root
python -m soundgen.creature_finetune prepare --in outputs --out datasets\ghoul_family --family ghoul --copy-audio --convert-to-wav
```

Output:
- `datasets/ghoul_family/audio/…`
- `datasets/ghoul_family/metadata.jsonl`

Each line includes `file_name` + `text` (the prompt), plus embedded credits.

## 1) Decide how “Windows” you mean

### Option A — WSL2 (recommended)

Use WSL2 if you have an NVIDIA GPU and want the least pain.

High level steps:
1) Install WSL2 + Ubuntu
2) Install NVIDIA’s WSL driver + CUDA toolkit support
3) Create a Python venv in WSL
4) Install your chosen LoRA trainer (diffusers+accelerate recommended)
5) Train a LoRA using `audio/` + `metadata.jsonl`

Why this is best:
- CUDA + PyTorch + accelerate are most stable on Linux.
- Fewer edge cases with dependencies.

### Option B — Native Windows (fallback)

High level steps:
1) Install Python 3.12 (recommended) or use conda
2) Install a matching PyTorch CUDA build for your GPU/driver
3) Install your LoRA trainer stack
4) Train from `datasets\...`

Native Windows usually fails due to one of:
- CUDA/torch build mismatch
- wheels missing for one dependency
- path/long-path issues

## 2) Training stack recommendation

### Best default: diffusers + accelerate

- Trainer stack: `diffusers`, `accelerate`, `transformers`, `peft`, `datasets`
- Output goal: a LoRA file that can be loaded by diffusers’ `load_lora_weights()` (often `.safetensors`).

Notes:
- Training scripts for *text-to-audio* diffusion models vary by model and diffusers version.
- Use a trainer that explicitly supports your target pipeline/model.

## 3) A complete, working training recipe (WSL2)

This recipe is designed to be:
- **trainer-agnostic** (works with any text-to-audio LoRA trainer that accepts `audio/` + `metadata.jsonl`)
- **explicit** (no “tell me your GPU” step)

### 3.1 Install WSL2 + Ubuntu (one-time)

In an elevated PowerShell:

```powershell
wsl --install
```

Reboot if prompted, then open Ubuntu.

### 3.2 Create a training venv in Ubuntu

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip git

python3 -m venv ~/venvs/sndbund-train
source ~/venvs/sndbund-train/bin/activate

python -m pip install --upgrade pip
```

### 3.3 Install your trainer stack

Pick one trainer stack that supports **text-to-audio LoRA** for your base model.

Two common shapes:

1) **Diffusers-based trainer**
  - Typical deps: `torch`, `diffusers`, `transformers`, `accelerate`, `peft`, `datasets`, `soundfile`

2) **Model-vendor trainer** (Stable Audio Open community trainers, etc.)
  - Follow that repo’s instructions, but keep your dataset format from this repo.

For diffusers-based stacks, start with:

```bash
python -m pip install "torch" "diffusers" "transformers<5" "accelerate" "peft" "datasets" "soundfile"
accelerate config default
```

### 3.4 Connect your exported dataset

Copy the dataset folder produced by `soundgen.creature_finetune prepare` into WSL (or access it under `/mnt/c/...`).

Expected structure:
- `datasets/<family>/audio/*.wav`
- `datasets/<family>/metadata.jsonl`

### 3.5 Recommended hyperparameters (good defaults)

Use these as starting values when your trainer asks:
- LoRA rank: `8` (try 16 if you have lots of data)
- LoRA alpha: `8` or `16`
- LR: `1e-4` to `2e-4`
- Batch size: `1` to `4` (use gradient accumulation to reach an effective batch of 8–16)
- Steps: `2k–10k` depending on dataset size
- Clip length: use the typical duration you generate (e.g. 0.8–2.0s for vocalizations)

### 3.6 Output expectation

At the end of training, you should have a LoRA weights file (commonly `.safetensors`) that your inference stack can load via diffusers’ LoRA loading APIs.

## 4) Bring the LoRA back into S‑NDB‑UND

1) Put the LoRA file somewhere (example): `lora/ghoul.safetensors`
2) Add a creature family entry:

Create `configs/creature_families.json` (or use `library/creature_families.json` as a local override):

```json
{
  "ghoul": {
    "lora_path": "lora/ghoul.safetensors",
    "trigger": "ghoul",
    "scale": 0.8,
    "negative_prompt": "music, singing, speech, vocals"
  }
}
```

3) Generate using the family:

```powershell
python -m soundgen.generate --engine stable_audio_open --creature-family ghoul --prompt "creature screech" --seconds 1.6 --seed 123 --post --out outputs\ghoul_screech.wav
```

## 6) Optional: model versions + validation preview

If you want stable, named references to fine-tuned models (instead of remembering file paths), add a model versions file:

- `configs/model_versions.json` (project config; commit-friendly)
- or `library/model_versions.json` (local override)

Example starter: `configs/model_versions.example.json`.

Then you can validate a model version by generating a small preview set:

```powershell
python -m soundgen.finetune validate --model-version example_ghoul_v1 --prompt "creature screech" --seconds 1.6 --variants 6 --seed 123 --out outputs\validate_ghoul --post
```

This writes a small set of WAVs (plus a minimal Minecraft pack stub) into the output folder.

## 5) Practical tips for creature families

- Start with ~50–200 clean examples per family.
- Keep prompts consistent (same style words) to teach a tight “family identity”.
- Prefer short clips (0.6–2.5s) for vocalizations; longer clips for ambience.
- Keep post-processing conservative during dataset creation; heavy polish can bake in artifacts.
