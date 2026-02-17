from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from soundgen.minecraft import (
    MinecraftSoundDef,
    ensure_pack_mcmeta,
    upsert_lang_entry,
    upsert_sounds_json,
    wav_to_minecraft_ogg,
)


@dataclass(frozen=True)
class PackItem:
    item_id: str
    prompt: str
    seconds: float
    engine: str
    pro_preset: str
    polish_profile: str
    event: str
    sound_path: str
    subtitle: str | None


def _load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_generate(*, python_exe: str, repo_root: Path, args: list[str]) -> None:
    cmd = [python_exe, "-m", "soundgen.generate", *args]
    completed = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "soundgen.generate failed:\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout: {completed.stdout}\n"
            f"stderr: {completed.stderr}"
        )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _pack_items(manifest: dict[str, Any]) -> list[PackItem]:
    defaults = manifest.get("defaults", {})
    out: list[PackItem] = []
    for raw in manifest.get("items", []):
        out.append(
            PackItem(
                item_id=str(raw["id"]),
                prompt=str(raw["prompt"]),
                seconds=float(raw.get("seconds", defaults.get("seconds", 2.5))),
                engine=str(raw.get("engine", defaults.get("engine", "rfxgen"))),
                pro_preset=str(raw.get("pro_preset", "off")),
                polish_profile=str(raw.get("polish_profile", "off")),
                event=str(raw["event"]),
                sound_path=str(raw["sound_path"]),
                subtitle=str(raw["subtitle"]) if raw.get("subtitle") else None,
            )
        )
    return out


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    manifest_path = repo_root / "soundpack_v1" / "manifest.json"
    manifest = _load_manifest(manifest_path)

    python_exe = sys.executable

    soundpack_id = str(manifest.get("soundpack_id", "soundpack_v1"))
    soundpack_version = str(manifest.get("version", "1.0.0"))
    namespace = str(manifest.get("namespace", "soundgen"))
    defaults = manifest.get("defaults", {})

    wav_dir = repo_root / "soundpack_v1" / "wav"
    pack_root = repo_root / "soundpack_v1" / "resourcepack"
    ogg_quality = int(defaults.get("ogg_quality", 5))
    mc_sr = int(defaults.get("mc_sample_rate", 44100))
    mc_ch = int(defaults.get("mc_channels", 1))

    ensure_pack_mcmeta(pack_root, description=str(manifest.get("title", "Soundpack")))

    items = _pack_items(manifest)
    if not items:
        raise ValueError("No items in soundpack_v1/manifest.json")

    rfxgen_path = str(defaults.get("rfxgen_path", "tools/rfxgen/rfxgen.exe"))
    post_default = bool(defaults.get("post", True))
    polish_default = bool(defaults.get("polish", True))
    loop_default = bool(defaults.get("loop", False))

    for it in items:
        wav_path = wav_dir / f"{it.item_id}.wav"

        gen_args: list[str] = [
            "--engine",
            it.engine,
            "--prompt",
            it.prompt,
            "--seconds",
            str(it.seconds),
            "--out",
            str(wav_path),
            "--out-format",
            "wav",
        ]

        if it.engine == "rfxgen":
            gen_args += ["--rfxgen-path", rfxgen_path]

        # Always apply our pack's intended chain.
        if post_default:
            gen_args += ["--post"]
        if polish_default:
            gen_args += ["--polish"]
        if loop_default:
            gen_args += ["--loop"]

        if it.pro_preset and it.pro_preset != "off":
            gen_args += ["--pro-preset", it.pro_preset]
        if it.polish_profile and it.polish_profile != "off":
            gen_args += ["--polish-profile", it.polish_profile]

        _run_generate(python_exe=python_exe, repo_root=repo_root, args=gen_args)

        wav_credits_path = Path(str(wav_path) + ".credits.json")
        if not wav_credits_path.exists():
            raise FileNotFoundError(f"Missing credits sidecar: {wav_credits_path}")

        credits = _read_json(wav_credits_path)
        credits["soundpack_id"] = soundpack_id
        credits["soundpack_version"] = soundpack_version
        credits["item_id"] = it.item_id
        credits["namespace"] = namespace
        credits["event"] = it.event
        credits["sound_path"] = it.sound_path
        _write_json(wav_credits_path, credits)

        # Convert to Minecraft-ready OGG inside the resource pack structure.
        ogg_path = pack_root / "assets" / namespace / "sounds" / f"{it.sound_path}.ogg"
        wav_to_minecraft_ogg(wav_path, ogg_path, sample_rate=mc_sr, channels=mc_ch, ogg_quality=ogg_quality)

        ogg_credits_path = Path(str(ogg_path) + ".credits.json")
        _write_json(ogg_credits_path, credits)

        # Register in sounds.json + subtitles
        upsert_sounds_json(
            pack_root,
            namespace,
            it.event,
            [MinecraftSoundDef(name=it.sound_path, weight=1, volume=1.0, pitch=1.0)],
            subtitle_key=(f"subtitles.{namespace}.{it.event}" if it.subtitle else None),
        )
        if it.subtitle:
            upsert_lang_entry(pack_root, namespace, f"subtitles.{namespace}.{it.event}", it.subtitle)

        print(f"OK: {it.item_id} -> {ogg_path}")

    print("\nSoundpack build complete:")
    print(f"- WAV: {wav_dir}")
    print(f"- Resourcepack: {pack_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
