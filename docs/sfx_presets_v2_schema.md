# SFX Presets v2 (Smart Presets) — schema

This repo supports a "smart" SFX preset library format that adds:

- **Families** (shared base settings)
- **Inheritance** (`inherits` base → variants)
- **Prompt template variables** (`vars`) with deterministic random choices driven by `--seed`

The v1 preset list format still works.

## File format

Top-level JSON object:

- `version` (optional): `2`
- `families` (optional): object mapping family key → family definition
- `presets` (required): list of preset definitions

## Family definition

A family is a reusable base block.

Supported keys (all optional, but effective presets must end up with `engine` + `prompt`):

- `engine`: e.g. `stable_audio_open`, `diffusers`, `rfxgen`, `samplelib`, `layered`, `hybrid`
- `prompt`: a template string using `{var}` placeholders
- `vars`: object mapping variable name → either a scalar or a list of options
- `negative_prompt`, `seconds`, `seed`, `variation_strength`, `fx_chain`, `post`
- `engine_params`: object (argparse dest → value)
- `post_params`: object (argparse dest → value)

### Fine-tuned model linking

You can link a preset to a fine-tuned model (LoRA) by setting either:

- `engine_params.model_version`: a key from `configs/model_versions.json` or `library/model_versions.json`.
- OR the low-level Stable Audio Open LoRA args:
  - `engine_params.stable_audio_lora_path`
  - `engine_params.stable_audio_lora_scale`
  - `engine_params.stable_audio_lora_trigger`

If you use `model_version`, it fills those Stable Audio Open LoRA fields unless you explicitly override them.

## Preset definition

Each preset entry in `presets` must include:

- `name`: unique key (used by `--sfx-preset`)

Optional keys:

- `family`: family key to inherit from first
- `inherits`: another preset name to inherit from (base → variant)

Plus the same keys as a family (`engine`, `prompt`, `vars`, etc).

## Merge/precedence rules

When resolving a preset:

1. Start from the referenced `family` (if present)
2. Merge the `inherits` preset (if present)
3. Merge the preset’s own fields (the child wins)

For dict fields like `engine_params`, `post_params`, and `vars`, merging is deep.

## Prompt template rendering

- Placeholders use Python-style formatting: `{size}`, `{texture}`, `{emotion}`
- For each entry in `vars`:
  - If the value is a list, one option is chosen using the effective seed.
  - If the value is a scalar, it is used directly.

Batch generation renders templates using the per-variant seed, so randomized fragments are deterministic per variant.

## Example

See [configs/sfx_presets_v2.example.json](../configs/sfx_presets_v2.example.json).
