# soundgen.editor (v1 scaffold)

This folder is the planned home for the built-in destructive waveform editor.

- UI layout + interactions: `docs/editor_v1_ui_layout.md`
- v1 goal: replace the “open Audacity to trim/fade/normalize/loop-check” loop.

Notes:
- Keep audio as mono `float32` in `[-1, 1]`.
- Keep GUI deps optional (avoid importing PySide6 at module import time).
