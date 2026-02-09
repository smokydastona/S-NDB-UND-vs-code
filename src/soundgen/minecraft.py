from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class MinecraftExportParams:
    pack_root: Path
    namespace: str
    event: str
    sound_path: str  # path under assets/<ns>/sounds WITHOUT extension
    out_ogg_path: Path
    ogg_quality: int = 5  # 0-10, ffmpeg qscale
    sample_rate: int = 44100
    channels: int = 1


@dataclass(frozen=True)
class MinecraftSoundDef:
    """One entry inside a sounds.json event's `sounds` list."""

    name: str  # path under sounds/ without extension
    weight: int = 1
    volume: float = 1.0
    pitch: float = 1.0


_ID_RE = re.compile(r"^[a-z0-9_\-./]+$")


def sanitize_id(value: str, *, kind: str) -> str:
    v = value.strip().lower().replace("\\", "/")
    v = re.sub(r"\s+", "_", v)
    v = re.sub(r"[^a-z0-9_\-./]", "_", v)
    v = re.sub(r"/+", "/", v).strip("/")
    if not v:
        raise ValueError(f"Empty {kind}")
    if not _ID_RE.match(v):
        raise ValueError(f"Invalid {kind}: {value!r}")
    return v


def ensure_pack_mcmeta(pack_root: Path, *, description: str = "Generated sounds") -> None:
    pack_root.mkdir(parents=True, exist_ok=True)
    mcmeta = pack_root / "pack.mcmeta"
    if mcmeta.exists():
        return

    # pack_format varies by MC version; 15 is for 1.20.1.
    data = {
        "pack": {
            "pack_format": 15,
            "description": description,
        }
    }
    mcmeta.write_text(json.dumps(data, indent=2), encoding="utf-8")


def upsert_lang_entry(pack_root: Path, namespace: str, key: str, value: str, *, lang: str = "en_us") -> Path:
    namespace = sanitize_id(namespace, kind="namespace")
    lang_path = pack_root / "assets" / namespace / "lang" / f"{lang}.json"
    data = _load_json(lang_path)
    data[str(key)] = str(value)
    _save_json(lang_path, data)
    return lang_path


def _ffmpeg_path() -> str:
    found = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
    if found:
        return found

    # WinGet installs (e.g. Gyan.FFmpeg) may not be visible until a new shell.
    local_appdata = os.getenv("LOCALAPPDATA")
    if local_appdata:
        base = Path(local_appdata) / "Microsoft" / "WinGet" / "Packages"
        if base.exists():
            for p in base.glob("Gyan.FFmpeg_*/*/bin/ffmpeg.exe"):
                return str(p)

    raise FileNotFoundError(
        "ffmpeg not found on PATH. Install ffmpeg (or add it to PATH) to export Minecraft .ogg files."
    )


def wav_to_minecraft_ogg(
    wav_path: Path,
    ogg_path: Path,
    *,
    sample_rate: int = 44100,
    channels: int = 1,
    ogg_quality: int = 5,
) -> None:
    ogg_path.parent.mkdir(parents=True, exist_ok=True)

    # Use Vorbis (most compatible for Minecraft resource packs).
    # -qscale:a sets VBR quality (0-10)
    cmd = [
        _ffmpeg_path(),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(wav_path),
        "-ar",
        str(sample_rate),
        "-ac",
        str(channels),
        "-c:a",
        "libvorbis",
        "-qscale:a",
        str(int(ogg_quality)),
        str(ogg_path),
    ]

    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "ffmpeg failed converting wav->ogg:\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout: {completed.stdout}\n"
            f"stderr: {completed.stderr}"
        )


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def upsert_sounds_json(
    pack_root: Path,
    namespace: str,
    event: str,
    sound_defs: list[MinecraftSoundDef],
    *,
    subtitle_key: Optional[str] = None,
) -> Path:
    """Update assets/<namespace>/sounds.json to include event definitions.

    For a file at assets/<namespace>/sounds.json, the full in-game id becomes:
      <namespace>:<event>
    """

    namespace = sanitize_id(namespace, kind="namespace")
    event = sanitize_id(event, kind="event")
    sound_defs = [
        MinecraftSoundDef(
            name=sanitize_id(sd.name, kind="sound_path"),
            weight=int(sd.weight),
            volume=float(sd.volume),
            pitch=float(sd.pitch),
        )
        for sd in sound_defs
    ]

    sounds_json = pack_root / "assets" / namespace / "sounds.json"
    data = _load_json(sounds_json)

    # Minecraft expects: { "event": { "sounds": ["path" | {name,weight,volume,pitch}] } }
    entry = data.get(event)
    if not isinstance(entry, dict):
        entry = {"sounds": []}
    sounds = entry.get("sounds")
    if not isinstance(sounds, list):
        sounds = []

    existing_names: set[str] = set()
    for s in sounds:
        if isinstance(s, str):
            existing_names.add(s)
        elif isinstance(s, dict) and isinstance(s.get("name"), str):
            existing_names.add(s["name"])

    for sd in sound_defs:
        if sd.name in existing_names:
            continue
        # Use object format so we can attach weights/volume/pitch.
        sounds.append(
            {
                "name": sd.name,
                "weight": max(1, int(sd.weight)),
                "volume": float(sd.volume),
                "pitch": float(sd.pitch),
            }
        )

    if subtitle_key:
        entry["subtitle"] = str(subtitle_key)

    entry["sounds"] = sounds
    data[event] = entry
    _save_json(sounds_json, data)
    return sounds_json


def export_wav_to_minecraft_pack(
    wav_path: Path,
    *,
    pack_root: Path,
    namespace: str,
    event: str,
    sound_path: str,
    weight: int = 1,
    volume: float = 1.0,
    pitch: float = 1.0,
    subtitle: Optional[str] = None,
    subtitle_key: Optional[str] = None,
    ogg_quality: int = 5,
    sample_rate: int = 44100,
    channels: int = 1,
    description: str = "Generated sounds",
    write_pack_mcmeta: bool = True,
) -> Path:
    namespace = sanitize_id(namespace, kind="namespace")
    event = sanitize_id(event, kind="event")
    sound_path = sanitize_id(sound_path, kind="sound_path")

    if write_pack_mcmeta:
        ensure_pack_mcmeta(pack_root, description=description)

    ogg_path = pack_root / "assets" / namespace / "sounds" / f"{sound_path}.ogg"
    wav_to_minecraft_ogg(
        wav_path,
        ogg_path,
        sample_rate=sample_rate,
        channels=channels,
        ogg_quality=ogg_quality,
    )

    final_subtitle_key = subtitle_key
    if subtitle and not final_subtitle_key:
        # Common convention used by vanilla and many mods.
        final_subtitle_key = f"subtitles.{namespace}.{event}"
    if subtitle and final_subtitle_key:
        upsert_lang_entry(pack_root, namespace, final_subtitle_key, subtitle)

    upsert_sounds_json(
        pack_root,
        namespace,
        event,
        [MinecraftSoundDef(name=sound_path, weight=weight, volume=volume, pitch=pitch)],
        subtitle_key=final_subtitle_key,
    )
    return ogg_path
