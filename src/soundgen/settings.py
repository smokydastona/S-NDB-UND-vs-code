from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .app_dirs import app_data_dir, ensure_dir
from .json_utils import load_json_file_lenient


@dataclass(frozen=True)
class Settings:
    data: dict[str, Any]


def settings_path(*, data_dir: Path | None = None) -> Path:
    base = Path(data_dir) if data_dir is not None else app_data_dir()
    return base / "settings.json"


def load_settings(*, data_dir: Path | None = None) -> Settings:
    p = settings_path(data_dir=data_dir)
    try:
        obj = load_json_file_lenient(p, context=f"Settings JSON file: {p}")
        if isinstance(obj, dict):
            return Settings(data=dict(obj))
    except Exception:
        pass
    return Settings(data={})


def save_settings(settings: Settings, *, data_dir: Path | None = None) -> None:
    base = Path(data_dir) if data_dir is not None else app_data_dir()
    ensure_dir(base)
    p = settings_path(data_dir=base)
    p.write_text(json.dumps(settings.data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def update_settings(patch: dict[str, Any], *, data_dir: Path | None = None) -> Settings:
    cur = load_settings(data_dir=data_dir)
    merged = dict(cur.data)
    merged.update(dict(patch or {}))
    out = Settings(data=merged)
    save_settings(out, data_dir=data_dir)
    return out
