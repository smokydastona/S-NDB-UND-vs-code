from __future__ import annotations

import argparse
import tempfile
import zipfile
from pathlib import Path

from .engine_registry import generate_wav
from .library import append_record, make_record
from .manifest import ManifestItem, load_manifest
from .minecraft import export_wav_to_minecraft_pack
from .postprocess import PostProcessParams, post_process_audio
from .credits import upsert_pack_credits


def _default_sound_path(namespace: str, event: str) -> str:
    # event can contain dots; use folder style
    safe = event.replace(":", ".")
    return f"generated/{safe.replace('.', '/')}"


def _pp_params() -> PostProcessParams:
    # Reasonable defaults for Minecraft.
    return PostProcessParams()


def _maybe_postprocess_wav(wav_path: Path, *, enabled: bool) -> None:
    if not enabled:
        return
    from .io_utils import read_wav_mono, write_wav

    audio, sr = read_wav_mono(wav_path)
    processed, _ = post_process_audio(audio, sr, _pp_params())
    write_wav(wav_path, processed, sr)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Batch-generate Minecraft sounds from a CSV/JSON manifest.")
    p.add_argument("--manifest", required=True, help="Path to .json/.csv manifest")

    p.add_argument("--pack-root", default="resourcepack", help="Resource pack root or Forge resources root")
    p.add_argument("--mc-target", choices=["resourcepack", "forge"], default="resourcepack")
    p.add_argument("--ogg-quality", type=int, default=5)
    p.add_argument("--mc-sample-rate", type=int, default=44100)
    p.add_argument("--mc-channels", type=int, default=1)

    p.add_argument("--catalog", default="library/catalog.jsonl", help="Append outputs to a local catalog")
    p.add_argument("--zip", default=None, help="If set, write a zip of the pack folder to this path")

    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="Diffusers device")
    p.add_argument("--model", default="cvssp/audioldm2", help="Diffusers model id")
    p.add_argument("--rfxgen-path", default=None, help="Optional path to rfxgen.exe")

    # Sample library engine
    p.add_argument(
        "--library-zip",
        action="append",
        default=None,
        help="Path to a ZIP sound library (for engine=samplelib). Can be specified multiple times. Default: .examples/sound libraies/*.zip",
    )
    p.add_argument("--library-pitch-min", type=float, default=0.85)
    p.add_argument("--library-pitch-max", type=float, default=1.20)

    return p


def _zip_folder(src_dir: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in src_dir.rglob("*"):
            if p.is_dir():
                continue
            z.write(p, arcname=str(p.relative_to(src_dir)))


def run_item(item: ManifestItem, *, args: argparse.Namespace) -> list[Path]:
    pack_root = Path(args.pack_root)
    write_pack_mcmeta = args.mc_target == "resourcepack"

    namespace = item.namespace
    event = item.event

    base_sound_path = item.sound_path or _default_sound_path(namespace, event)
    out_files: list[Path] = []

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        default_zips = sorted(Path(".examples").joinpath("sound libraies").glob("*.zip"))
        zip_args = args.library_zip if args.library_zip else [str(p) for p in default_zips]

        for i in range(max(1, int(item.variants))):
            suffix = f"_{i+1:02d}" if item.variants > 1 else ""
            sound_path = f"{base_sound_path}{suffix}"

            tmp_wav = tmp_dir / f"{namespace}_{event.replace('.', '_')}{suffix}.wav"

            if item.engine == "samplelib" and not zip_args:
                raise FileNotFoundError(
                    "engine=samplelib but no --library-zip provided and no default zips found at .examples/sound libraies/*.zip"
                )

            def _pp(audio, sr):
                processed, _ = post_process_audio(audio, sr, _pp_params())
                return processed, "post"

            generated = generate_wav(
                item.engine,
                prompt=item.prompt,
                seconds=float(item.seconds),
                seed=(int(item.seed) + i) if item.seed is not None else None,
                out_wav=tmp_wav,
                postprocess_fn=(_pp if item.post else None),
                device=str(args.device),
                model=str(args.model),
                preset=item.preset,
                layered_preset=(item.preset or "auto"),
                rfxgen_path=(Path(args.rfxgen_path) if args.rfxgen_path else None),
                library_zips=tuple(Path(p) for p in zip_args),
                library_pitch_min=float(args.library_pitch_min),
                library_pitch_max=float(args.library_pitch_max),
                sample_rate=44100,
            )

            sources: tuple[dict, ...] = tuple(generated.sources)

            ogg_path = export_wav_to_minecraft_pack(
                generated.wav_path,
                pack_root=pack_root,
                namespace=namespace,
                event=event,
                sound_path=sound_path,
                weight=int(item.weight),
                volume=float(item.volume),
                pitch=float(item.pitch),
                subtitle=item.subtitle,
                subtitle_key=item.subtitle_key,
                ogg_quality=int(args.ogg_quality),
                sample_rate=int(args.mc_sample_rate),
                channels=int(args.mc_channels),
                description="Sound Generator pack",
                write_pack_mcmeta=write_pack_mcmeta,
            )
            out_files.append(ogg_path)

            # Pack credits for all engines.
            credits: dict = {"engine": item.engine, "prompt": item.prompt}
            credits.update({k: v for k, v in generated.credits_extra.items() if v is not None})
            if sources:
                credits["sources"] = list(sources)

            upsert_pack_credits(
                pack_root=pack_root,
                namespace=namespace,
                event=event,
                sound_path=sound_path,
                credits=credits,
            )

            append_record(
                Path(args.catalog),
                make_record(
                    engine=item.engine,
                    prompt=item.prompt,
                    namespace=namespace,
                    event=event,
                    sound_path=sound_path,
                    output_file=ogg_path,
                    tags=item.tags,
                    sources=sources,
                    seconds=item.seconds,
                    seed=item.seed,
                    preset=item.preset,
                ),
            )

    return out_files


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    items = load_manifest(Path(args.manifest))

    total = 0
    for item in items:
        outs = run_item(item, args=args)
        total += len(outs)
        for o in outs:
            print(f"Wrote {o}")

    if args.zip:
        _zip_folder(Path(args.pack_root), Path(args.zip))
        print(f"Zipped pack to {args.zip}")

    print(f"Done. Exported {total} file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
