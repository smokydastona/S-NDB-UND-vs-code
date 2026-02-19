# Audio FX plugins (CLAP + LV2)

This page is about **audio effect plugins** (EQ, compressors, reverbs, etc.) that process audio during playback/export.

> Note: This is **not** the same thing as SoundGen “engine plugins”. Engine plugins add **generation backends** (see [docs/plugins.md](plugins.md)).

## What we’re targeting

### LV2

Best for:
- Linux-first open audio ecosystems
- Deep extensibility (audio, MIDI, CV, UI extensions)
- Experimental / modular DSP work

Common hosts include Ardour, Audacity, and Carla.

Key GitHub resources:
- Core spec: https://github.com/lv2/lv2
- Example plugins: https://github.com/swh/lv2
- UI toolkit (GTK): https://github.com/lv2/lv2kit

Why it matters:
- LV2 is very hackable. If you want experimental DSP, modular control, or academic-grade audio tools, it’s a great ecosystem.

### CLAP (CLever Audio Plug-in API)

Best for:
- Modern, high-performance cross-platform plugins
- Thread-safe design and sample-accurate automation

Created by u-he + Bitwig as a more modern alternative to older plugin APIs.

Key GitHub resources:
- Spec + SDK: https://github.com/free-audio/clap
- Example plugins: https://github.com/free-audio/clap-plugins
- Plugin validator: https://github.com/free-audio/clap-validator

Why it matters:
- CLAP is designed for modern host/plugin architecture (scalable, performant, automation-friendly).

## Open-source plugin collections to study

- x42 Plugins (LV2): https://github.com/x42
  - High-quality LV2 plugins (EQ, compressors, meters)

- CALF Studio Gear (LV2 + LADSPA): https://github.com/calf-studio-gear/calf
  - Large suite of effects and synths

- ZamAudio (LV2): https://github.com/zamaudio
  - Mixing/mastering plugins (compressors, EQs)

- Airwindows (mostly VST; DSP concepts still valuable): https://github.com/airwindows/airwindows
  - Hundreds of minimalist DSP plugins; great learning resource

## Practical host architecture advice

If you’re building an editor/engine, a durable approach is:
- Keep your DSP “graph” and audio buffer plumbing **plugin-agnostic**
- Add thin wrappers / adapters for:
  - CLAP
  - LV2

This avoids lock-in and lets you support multiple ecosystems.

## SÖNDBÖUND status in this repo (current)

- Native helper (Windows dev): `native/pluginhost/`
  - Scans CLAP/LV2 locations
  - Can **offline-render** a mono WAV through a selected CLAP plugin for preview

- Electron Editor integration:
  - The editor can load a `.clap`, list contained plugin descriptors, and render a preview WAV for playback.

See:
- `native/pluginhost/README.md`
- `electron/main.js` (IPC glue)
- `electron/editor/editor.js` (UI flow)
