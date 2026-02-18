# soundgen.editor (built-in destructive editor)

This folder contains the built-in destructive waveform editor used for quick SFX cleanup (trim/fade/normalize/loop audition) without leaving S‑NDB‑UND.

- UI layout + interactions: `docs/editor_v1_ui_layout.md`
- v1 goal: replace the “open Audacity to trim/fade/normalize/loop-check” loop.

Launch:

- `python -c "from soundgen.editor import launch_editor; launch_editor('outputs/out.wav')"`
- `S-NDB-UND.exe edit outputs\out.wav`

Notes:
- Audio stays mono `float32` in `[-1, 1]`.
- Keybinds print to the console when the editor starts (`h` to reprint).
- v2 (early) multi-region: create regions from selections (`R`) and export slices (`O`). Each region can have an FX chain key set (`C`) that is applied on export.
- v2 (early) loop metadata export: set per-region loop points (`{`/`}`) and region exports include `region.loop_s` in the `.edits.json` sidecar (relative to the exported slice).
