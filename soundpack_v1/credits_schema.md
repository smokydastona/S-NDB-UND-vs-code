# Credits schema (per-file sidecar)

Each generated audio file in this pack has a `*.credits.json` next to it.

## Required fields
- `soundpack_id`: string (e.g. `soundpack_v1`)
- `soundpack_version`: string (e.g. `1.0.0`)
- `item_id`: string (manifest `items[].id`)
- `namespace`: string (Minecraft namespace)
- `event`: string (sounds.json event key)
- `sound_path`: string (path under `assets/<ns>/sounds/` without extension)

## Generation fields
- `engine`: string (`rfxgen`, `diffusers`, `layered`, ...)
- `prompt`: string
- `seed`: int or null

## Traceability fields (always present)
- `pro_preset`: string (`off` or a preset key)
- `polish_profile`: string (`off` or a profile key)
- `loop_clean`: boolean
- `loop_crossfade_ms`: int

## Optional fields
May appear depending on engine/workflow:
- `sources`: list (e.g. sample sources)
- Any engine-specific fields included via `credits_extra`
