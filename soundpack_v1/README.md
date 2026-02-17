# Sound Generator — Demo Sound Pack v1

This folder is an **in-repo demo pack** built with `soundgen`.

## What you get
- `wav/` — source WAVs (good for editing)
- `resourcepack/` — Minecraft-ready resource pack output (`.ogg` + `sounds.json` + optional subtitles)
- Per-file credits JSON sidecars next to both `.wav` and `.ogg`

## Build / regenerate
From repo root:

- Generate pack:
  - `C:/Users/smoky/OneDrive/Desktop/Homemade Mods/sound generator/.venv/Scripts/python.exe scripts/build_soundpack_v1.py`

(Or run `python scripts/build_soundpack_v1.py` in your activated venv.)

## Minecraft IDs
- Namespace: `soundgen`
- Each item writes to `assets/soundgen/sounds.json` with an event key like:
  - `soundgen:soundpack_v1.ui.click_soft`

## Credits
Each audio file has a `*.credits.json` sidecar including (at minimum):
- `engine`, `prompt`, `seed`
- `pro_preset`, `polish_profile`
- `loop_clean`, `loop_crossfade_ms`
- `namespace`, `event`, `sound_path`
- `soundpack_id`, `soundpack_version`, `item_id`

The source of truth for the pack contents is `manifest.json`.
