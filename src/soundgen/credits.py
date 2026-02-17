from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .minecraft import sanitize_id


def write_sidecar_credits(audio_path: Path, credits: dict[str, Any]) -> Path:
    """Write credits next to an audio file as <file>.<ext>.credits.json."""

    if "created_utc" not in credits:
        credits["created_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    audio_path = Path(audio_path)
    out = audio_path.with_name(audio_path.name + ".credits.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(credits, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def upsert_pack_credits(
    *,
    pack_root: Path,
    namespace: str,
    event: str,
    sound_path: str,
    credits: dict[str, Any],
) -> Path:
    """Write/update a credits file inside the pack.

    Output path: assets/<namespace>/soundgen_credits.json

    Schema (stable, simple):
      {
        "<event>": {
          "<sound_path>": { ... credits ... }
        }
      }
    """

    if "created_utc" not in credits:
        credits["created_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    pack_root = Path(pack_root)
    namespace = sanitize_id(namespace, kind="namespace")
    event = sanitize_id(event, kind="event")
    sound_path = sanitize_id(sound_path, kind="sound_path")

    out = pack_root / "assets" / namespace / "soundgen_credits.json"
    if out.exists():
        data = json.loads(out.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            data = {}
    else:
        data = {}

    ev = data.get(event)
    if not isinstance(ev, dict):
        ev = {}

    ev[sound_path] = credits
    data[event] = ev

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out
