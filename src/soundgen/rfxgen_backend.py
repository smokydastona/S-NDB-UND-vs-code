from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Optional


SUPPORTED_PRESETS: tuple[str, ...] = (
    "coin",
    "laser",
    "explosion",
    "powerup",
    "hit",
    "jump",
    "blip",
)


@dataclass(frozen=True)
class RfxGenParams:
    prompt: str
    out_path: Path
    rfxgen_path: Optional[Path] = None  # if None, uses PATH lookup
    preset: Optional[str] = None
    sample_rate: int = 44100
    sample_size: int = 16
    channels: int = 1


def _infer_preset_from_prompt(prompt: str) -> str:
    p = prompt.lower()

    if any(k in p for k in ("coin", "pickup", "collect", "money", "cash")):
        return "coin"
    if any(k in p for k in ("laser", "blaster", "zap", "pew", "sci-fi")):
        return "laser"
    if any(k in p for k in ("explosion", "boom", "blast", "grenade", "thunder", "storm")):
        return "explosion"
    if any(k in p for k in ("powerup", "power up", "upgrade", "level up")):
        return "powerup"
    if any(
        k in p
        for k in (
            "hit",
            "hurt",
            "damage",
            "impact",
            "punch",
            "slam",
            "smash",
            "thud",
            "crash",
            "collapse",
            "debris",
            "rumble",
            "grind",
            "grinding",
            "stone",
        )
    ):
        return "hit"
    if any(k in p for k in ("jump", "hop", "leap")):
        return "jump"
    if any(k in p for k in ("blip", "beep", "ui", "click")):
        return "blip"

    # default: a short UI-ish sound is often the safest
    return "blip"


def _resolve_rfxgen_exe(explicit: Optional[Path]) -> str:
    env_override = os.environ.get("SONDBOUND_RFXGEN_EXE") or os.environ.get("SOUNDGEN_RFXGEN_EXE")
    if env_override:
        p = Path(env_override)
        if p.exists():
            return str(p)

    if explicit is not None:
        if not explicit.exists():
            raise FileNotFoundError(f"rfxgen not found at: {explicit}")
        return str(explicit)

    # Prefer 'rfxgen' in PATH
    found = shutil.which("rfxgen") or shutil.which("rfxgen.exe")
    if found:
        return found

    # Common local convention
    local = Path("tools") / "rfxgen" / "rfxgen.exe"
    if local.exists():
        return str(local)

    # VS Code extension convention: install under SOUNDGEN_DATA_DIR
    data_dir = os.environ.get("SOUNDGEN_DATA_DIR")
    if data_dir:
        p = Path(data_dir) / "tools" / "rfxgen" / "rfxgen.exe"
        if p.exists():
            return str(p)

    raise FileNotFoundError(
        "rfxgen executable not found. Put rfxgen.exe in PATH, or at tools/rfxgen/rfxgen.exe, or pass --rfxgen-path."
    )


def generate_with_rfxgen(params: RfxGenParams) -> Path:
    exe = _resolve_rfxgen_exe(params.rfxgen_path)

    preset = params.preset or _infer_preset_from_prompt(params.prompt)
    if preset not in SUPPORTED_PRESETS:
        raise ValueError(f"Unsupported preset '{preset}'. Supported: {', '.join(SUPPORTED_PRESETS)}")

    params.out_path.parent.mkdir(parents=True, exist_ok=True)

    fmt = f"{params.sample_rate},{params.sample_size},{params.channels}"

    # rfxgen CLI (from upstream README):
    #   rfxgen --generate <preset> --output out.wav --format 44100,16,1
    cmd = [
        exe,
        "--generate",
        preset,
        "--output",
        str(params.out_path),
        "--format",
        fmt,
    ]

    # Capture output for debugging on failures.
    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "rfxgen failed:\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout: {completed.stdout}\n"
            f"stderr: {completed.stderr}"
        )

    return params.out_path
