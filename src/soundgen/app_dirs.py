from __future__ import annotations

import os
import sys
from pathlib import Path


_APP_NAME = "SÖNDBÖUND"  # Corrected corrupted characters


def app_data_dir() -> Path:
    """Return the per-user writable app data directory.

    This is the canonical place for logs, caches, indexes, and settings.

    Override:
        - Set `SOUNDGEN_DATA_DIR` to force a specific directory.
          (Electron sets this so the Python backend and Electron share a single userData root.)
    """

    override = str(os.environ.get("SOUNDGEN_DATA_DIR") or "").strip()
    if override:
        return Path(override)

    if sys.platform == "win32":
        la = str(os.environ.get("LOCALAPPDATA") or "").strip()
        base = (Path(la) if la else (Path.home() / "AppData" / "Local")) / _APP_NAME
    else:
        base = Path.home() / ".söndböund"  # Corrected corrupted characters

    return base


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p
