# Pro Mode spec (controls, ranges, precedence)

This spec documents the **advanced post / polish controls** (“Pro Mode”) and how they interact with:
- `--pro-preset`
- `--polish-profile`
- explicit per-flag overrides

Applies to:
- CLI: `python -m soundgen.generate`
- Web UI: `python -m soundgen.web`

## Precedence (who wins)

Order of application is:
1. **User explicit values** (CLI flags / Web widget values)
2. **Pro Mode overrides** (the optional overrides like `--denoise-amount`, `--transient-attack`, …) if provided
3. `--polish-profile` (applies conservatively: only sets knobs still at defaults)
4. `--pro-preset` (applies conservatively: only sets knobs still at defaults)
5. Built-in defaults

Notes:
- “Conservative” means presets/profiles only override values that are still at their parser/UI defaults.
- Credits always record: `pro_preset`, `polish_profile`, `loop_clean`, `loop_crossfade_ms`.

## Pro DSP modules (post stage)

These work across engines when `--post` or `--polish` is enabled.

### Multi-band cleanup/dynamics
- Flag: `--multiband` (bool)
- Crossovers:
  - `--mb-low-hz` (float, default 250)
  - `--mb-high-hz` (float, default 3000)
- Band gains (dB):
  - `--mb-low-gain-db` (default 0)
  - `--mb-mid-gain-db` (default 0)
  - `--mb-high-gain-db` (default 0)
- Optional per-band compression:
  - `--mb-comp-threshold-db` (float or omitted; omitted = no multiband compression)
  - `--mb-comp-ratio` (default 2.0)

Web UI ranges (current):
- low hz: 80..600
- high hz: 1200..8000
- gains: -6..+6
- ratio: 1..6

### Creature size / formant color
- `--creature-size` (float, intended -1..+1; default 0)
  - -1 = smaller/brighter
  - +1 = larger/darker
- `--formant-shift` (float factor, default 1.0)
  - 1.0 disables
  - If set != 1.0, it overrides `--creature-size`

Web UI range: 0.5..2.0 (formant) and -1..+1 (creature size).

### Procedural texture overlay
- `--texture-preset` (choices: `off|auto|chitter|rasp|buzz|screech`, default `off`)
- `--texture-amount` (0..1, default 0)
- `--texture-grain-ms` (default 22)
- `--texture-spray` (0..1, default 0.55)

Web UI behavior:
- Texture amount/grain/spray only appear when preset != `off`.

### Synthetic convolution reverb
- `--reverb` (choices: `off|room|cave|forest|nether`, default `off`)
- `--reverb-mix` (0..1, default 0)
- `--reverb-time` (seconds, default 1.2)

Web UI behavior:
- Reverb mix/time only appear when preset != `off`.

## Polish Mode DSP (polish stage)

Polish mode is enabled via:
- CLI: `--polish` (implies `--post`)
- Web UI: “Polish mode (denoise/transients/compress/limit)”

### Optional Pro Mode overrides (exact knobs)

These **override** the usual “polish defaults” and any conditioning nudges.

- `--denoise-amount` (0..1, optional)
  - Controls spectral denoise strength.
- `--transient-attack` (-1..+1, optional)
  - Positive values emphasize attack transients in the high band.
- `--transient-sustain` (-1..+1, optional)
  - Positive values emphasize sustain/body relative to transient.
- `--exciter-amount` (0..1, optional)
  - Harmonic exciter (upper-band saturation mixback).
- `--compressor-attack-ms` (ms, optional)
- `--compressor-release-ms` (ms, optional)

Web UI accepts these as blank/None to mean “auto/default”.

## Loop cleaning (ambience)
- CLI: `--loop` (bool)
- CLI: `--loop-crossfade-ms` (int, default 100)
- Web UI: “Loop-clean ambience (100ms seam crossfade)” (fixed at 100ms)

## Credits requirements

Sidecar credits JSON always includes:
- `pro_preset` (string, `off` or key)
- `polish_profile` (string, `off` or key)
- `loop_clean` (boolean)
- `loop_crossfade_ms` (int)

Plus typical generation fields like `engine`, `prompt`, `seed`, and engine-specific extras.
