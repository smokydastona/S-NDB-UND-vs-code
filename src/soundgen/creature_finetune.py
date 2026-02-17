from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from .io_utils import convert_audio_with_ffmpeg


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Prepare a creature-family fine-tuning dataset from existing S-NDB-UND outputs.\n\n"
            "This tool is intentionally training-framework-agnostic: it exports a simple audio folder + metadata.jsonl "
            "that can be consumed by diffusers/accelerate scripts or other LoRA trainers."
        )
    )

    sub = p.add_subparsers(dest="cmd", required=True)

    prep = sub.add_parser("prepare", help="Export audio + metadata.jsonl from .credits.json sidecars")
    prep.add_argument("--in", dest="inputs", action="append", required=True, help="Input folder to scan (repeatable)")
    prep.add_argument("--out", required=True, help="Output dataset folder")
    prep.add_argument("--family", required=True, help="Creature family key/name (e.g. ghoul, spiderling)")
    prep.add_argument(
        "--copy-audio",
        action="store_true",
        help="Copy audio into the dataset folder (recommended for portability)",
    )
    prep.add_argument(
        "--convert-to-wav",
        action="store_true",
        help="If set, convert non-.wav inputs to mono 44.1kHz WAV using ffmpeg.",
    )
    prep.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of items to export",
    )

    return p


def _iter_credits_sidecars(inputs: list[Path]) -> list[Path]:
    out: list[Path] = []
    for root in inputs:
        if not root.exists():
            continue
        for p in root.rglob("*.credits.json"):
            if p.is_file():
                out.append(p)
    out.sort()
    return out


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _infer_audio_path(sidecar: Path) -> Path | None:
    # Sidecar naming: <audio>.<ext>.credits.json
    name = sidecar.name
    if not name.endswith(".credits.json"):
        return None
    audio_name = name[: -len(".credits.json")]
    audio_path = sidecar.with_name(audio_name)
    return audio_path if audio_path.exists() else None


def run_prepare(argv: list[str] | None = None) -> int:
    p = build_parser()
    args = p.parse_args(argv)
    if args.cmd != "prepare":
        raise SystemExit(2)

    inputs = [Path(x) for x in args.inputs]
    out_root = Path(args.out)
    family = str(args.family).strip()
    if not family:
        raise ValueError("--family is required")

    audio_dir = out_root / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    sidecars = _iter_credits_sidecars(inputs)
    if args.limit is not None:
        sidecars = sidecars[: max(0, int(args.limit))]

    meta_path = out_root / "metadata.jsonl"
    exported = 0

    with meta_path.open("w", encoding="utf-8") as f:
        for sc in sidecars:
            audio_src = _infer_audio_path(sc)
            if audio_src is None:
                continue

            credits = _load_json(sc)
            prompt = str(credits.get("prompt") or "").strip()
            engine = str(credits.get("engine") or "").strip()

            if not prompt:
                # Keep dataset usable even if older outputs missed prompt.
                continue

            # Stable file name (avoid collisions): <family>__<stem>.wav
            safe_stem = audio_src.stem
            rel_name = f"{family}__{safe_stem}.wav" if args.convert_to_wav else f"{family}__{audio_src.name}"
            audio_dst = audio_dir / rel_name

            if args.copy_audio:
                if args.convert_to_wav and audio_src.suffix.lower() != ".wav":
                    convert_audio_with_ffmpeg(
                        audio_src,
                        audio_dst.with_suffix(".wav"),
                        sample_rate=44100,
                        channels=1,
                        out_format="wav",
                    )
                    audio_dst = audio_dst.with_suffix(".wav")
                else:
                    audio_dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(audio_src, audio_dst)
            else:
                # Reference original path.
                audio_dst = audio_src

            rec = {
                "file_name": str(audio_dst.relative_to(out_root) if args.copy_audio else audio_dst),
                "text": prompt,
                "family": family,
                "engine": engine or None,
                "seed": credits.get("seed"),
                "created_utc": credits.get("created_utc"),
                "credits": credits,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            exported += 1

    print(f"Wrote {exported} items to {meta_path}")
    if args.copy_audio:
        print(f"Audio folder: {audio_dir}")
    return 0


def main() -> None:
    raise SystemExit(run_prepare())


if __name__ == "__main__":
    main()
