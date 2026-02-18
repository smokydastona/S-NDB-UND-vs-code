# S‑NDB‑UND v2 — The Professional Sound‑Design Environment

v2 is about workflow depth, editor power, and modder‑first features.

## 1) Advanced built‑in audio editor (v2 upgrade)

v1 gave a minimal waveform editor.

v2 turns it into a real sound‑design workstation.

### New editor features

- Spectrogram view
  - adjustable FFT size
  - frequency zoom
  - click‑removal targeting
- Multi‑region editing
  - slice one file into multiple exports
  - region naming
  - region‑based FX
- Layering (2–4 layers max)
  - overlay multiple generated sounds
  - per‑layer gain + pan
  - per‑layer fade
  - per‑layer pitch shift
- Transient detection
  - auto‑mark transients
  - snap selection to transient
- Loop tools
  - auto‑crossfade loop creation
  - seamless loop preview
  - loop point metadata export

### Why this matters

You can now:

- build complex creature roars
- layer textures
- clean up AI artifacts
- create seamless ambience loops

…all inside S‑NDB‑UND.

## 2) Preset system v2 — Smart presets

v1 presets were static.

v2 presets become dynamic, parameterized, and context‑aware.

### New preset features

- Parameter variables
- Randomized prompt fragments
  - choose from lists to add variation
- Preset families
  - small_creature
  - medium_creature
  - large_creature
  - ethereal
  - mechanical
- Preset inheritance
  - base preset → specialized variants

## 3) FX chains v2 — Modular, editable, visual

v1 FX chains were JSON lists.

v2 introduces visual editing + new FX modules.

### New FX modules

- Pitch shift
- Time stretch
- Distortion
- Convolution reverb (IR support)
- Multi‑band EQ
- Transient shaper
- Noise reduction (spectral gating)

### FX chain editor

- drag‑and‑drop modules
- reorder FX
- tweak parameters live
- audition changes instantly
- save as new chain

## 4) Project system — Sound packs & mobs

v2 introduces projects, which group sounds, presets, and metadata.

### Project features

- Create a “sound pack” project
- Add categories:
  - creature
  - UI
  - ambience
  - footsteps
- Track:
  - generated sounds
  - edited versions
  - metadata
  - presets used
- Export entire pack at once

### Minecraft project mode

- auto‑generate all mob sounds
- edit them in the editor
- export pack + `sounds.json`
- versioning for each sound

## 5) Hybrid engine mode — AI + procedural layering

Hybrid engine features:

- Generate AI base layer
- Add procedural transient layer
- Add noise texture layer
- Auto‑mix layers
- Apply FX chain to final composite

## 6) Model fine‑tuning (optional)

v2 introduces fine‑tuning support for Stable Audio Open.

Fine‑tuning features:

- dataset folder structure
- training script
- validation preview
- model versioning
- preset linking to fine‑tuned models

Use cases:

- creature families
- biome ambience sets
- UI sound packs

## 7) CLI v2 — More power, more control

New CLI features:

- project creation
- project export
- hybrid engine mode
- preset variables
- FX chain override
- region export from editor

## 8) GUI v1 — The first real interface

v2 introduces the first GUI, focused on:

- preset browser
- generate panel
- waveform editor
- FX chain editor
- project browser
- export panel

Not a full DAW — a focused sound‑design workstation.

## What v2 achieves

With v2, S‑NDB‑UND becomes:

- a sound generator
- a sound editor
- a sound pack manager
- a mini‑DAW for creature SFX
- a Minecraft soundset builder
- a hybrid AI/procedural engine
- a GUI application
