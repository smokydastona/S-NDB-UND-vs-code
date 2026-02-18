# FX Chains v2 (Modular JSON)

FX Chains v2 are **offline**, **modular** effect stacks stored as JSON.

They are meant for workflows like:

- generate sound → post-process → apply FX chain v2 → export
- editor region export → apply chain v2 (via JSON) → export

## Format

A v2 chain is a JSON object:

```json
{
  "format": "sndbund_fx_chain",
  "version": 2,
  "name": "my_chain",
  "description": "Optional",
  "steps": [
    {"id": "spectral_denoise", "params": {"strength": 0.15}},
    {"id": "transient_shaper", "params": {"attack": 0.25, "sustain": -0.05}},
    {"id": "multiband_eq", "params": {"low_hz": 250, "high_hz": 3000, "low_gain_db": 1.0}},
    {"id": "distortion", "params": {"drive": 0.2, "mode": "tanh", "mix": 0.6}},
    {"id": "pitch_shift", "params": {"semitones": -1.0}}
  ]
}
```

Unknown step `id`s are ignored for forward compatibility.

## Built-in modules

These module IDs are implemented in `soundgen.fx_chain_v2`.

### `pitch_shift`

- Purpose: pitch shift without changing duration.
- Params:
  - `semitones` (float, default `0.0`)
  - `n_fft` (int, default `2048`)

### `time_stretch`

- Purpose: time stretch without changing pitch.
- Params:
  - `rate` (float, default `1.0`) — `>1` faster/shorter, `<1` slower/longer
  - `n_fft` (int, default `2048`)
  - `hop_length` (int, optional)

### `distortion`

- Purpose: waveshaper distortion.
- Params:
  - `drive` (float 0..1, default `0.0`)
  - `mode` (string, `tanh|softclip|hardclip`, default `tanh`)
  - `mix` (float 0..1, default `1.0`)

### `convolution_reverb`

- Purpose: convolution reverb using an **external IR WAV**.
- Params:
  - `ir_path` (string, required) — path to impulse response WAV (mono recommended)
  - `mix` (float 0..1, default `0.25`)
  - `tail_s` (float, default `1.0`) — tail length kept after the dry signal
  - `normalize_ir` (bool, default `true`)
  - `pre_delay_ms` (float, default `0.0`)

### `multiband_eq`

- Purpose: simple 3-band EQ via crossover splits.
- Params:
  - `low_hz` (float, default `250.0`)
  - `high_hz` (float, default `3000.0`)
  - `low_gain_db` (float, default `0.0`)
  - `mid_gain_db` (float, default `0.0`)
  - `high_gain_db` (float, default `0.0`)

### `transient_shaper`

- Purpose: attack/sustain shaping (high-band focused).
- Params:
  - `attack` (float -1..+1, default `0.0`)
  - `sustain` (float -1..+1, default `0.0`)
  - `split_hz` (float, default `1200.0`)

### `spectral_denoise`

- Purpose: gentle spectral subtraction noise reduction.
- Params:
  - `strength` (float 0..1, default `0.0`)
  - `n_fft` (int, default `2048`)
  - `hop_length` (int, default `n_fft/4`)

## Editor

Interactive editor:

```powershell
python -m soundgen.fx_chain_editor --chain configs\fx_chain_v2.example.json --wav outputs\test.wav
```

Within the editor:

- `modules` lists available module IDs
- `add`, `del`, `up`, `down`, `set` modify the chain
- `audition` plays the processed WAV (Windows winsound)
- `save` writes the chain JSON
