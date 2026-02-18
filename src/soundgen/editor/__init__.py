"""Built-in destructive editor (v1 scaffolding).

This package intentionally avoids importing GUI dependencies at import time.

The v1 plan is a small, single-file waveform editor:
- open last render / open WAV
- trim/cut/delete
- fades
- gain/normalize
- loop audition
- export overwrite / export variation

See docs/editor_v1_ui_layout.md for the UI + interaction spec.
"""

from __future__ import annotations

from .launch import launch_editor

__all__ = ["launch_editor"]
