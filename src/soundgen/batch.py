from __future__ import annotations

import argparse
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from .engine_registry import generate_wav
from .library import append_record, make_record
from .manifest import ManifestItem, load_manifest
from .minecraft import export_wav_to_minecraft_pack
from .postprocess import PostProcessParams, post_process_audio
from .credits import upsert_pack_credits
from .pro_presets import PRO_PRESETS, apply_pro_preset, pro_preset_keys
from .polish_profiles import apply_polish_profile, polish_profile_keys


def _default_sound_path(namespace: str, event: str) -> str:
    # event can contain dots; use folder style
    safe = event.replace(":", ".")
    return f"generated/{safe.replace('.', '/')}"


def _pp_params(*, args: argparse.Namespace, post_seed: int | None, prompt_hint: str | None) -> PostProcessParams:
    # Reasonable defaults for Minecraft, plus optional "pro" DSP controls.
    intensity = float(np.clip(float(getattr(args, "intensity", 0.0)), 0.0, 1.0))
    variation = float(np.clip(float(getattr(args, "variation", 0.0)), 0.0, 1.0))
    emotion = str(getattr(args, "emotion", "neutral") or "neutral").strip().lower()

    denoise = (0.25 if bool(getattr(args, "polish", False)) else 0.0)
    transient = (0.25 if bool(getattr(args, "polish", False)) else 0.0)
    comp_thr = (-18.0 if bool(getattr(args, "polish", False)) else None)
    comp_makeup = (3.0 if bool(getattr(args, "polish", False)) else 0.0)
    limiter = (-1.0 if bool(getattr(args, "polish", False)) else None)

    if intensity > 0.0:
        transient = float(np.clip(transient + 0.35 * intensity, -1.0, 1.0))
        comp_thr = float(-18.0 - 8.0 * intensity)
        comp_makeup = float(comp_makeup + 2.0 * intensity)
        denoise = float(np.clip(denoise + (0.10 * intensity), 0.0, 1.0))
        if emotion == "aggressive":
            transient = float(np.clip(transient + 0.20, -1.0, 1.0))
        elif emotion == "calm":
            transient = float(np.clip(transient - 0.15, -1.0, 1.0))
        elif emotion == "scared":
            transient = float(np.clip(transient + 0.10, -1.0, 1.0))

    texture_preset = str(getattr(args, "texture_preset", "off") or "off")
    texture_amount = float(getattr(args, "texture_amount", 0.0) or 0.0)
    if intensity > 0.0 and (texture_preset == "off") and texture_amount <= 0.0:
        if emotion in {"scared", "aggressive"}:
            texture_preset = "auto"
            texture_amount = float(np.clip(0.18 + 0.25 * intensity + 0.10 * variation, 0.0, 1.0))

    reverb_preset = str(getattr(args, "reverb", "off") or "off")
    reverb_mix = float(getattr(args, "reverb_mix", 0.0) or 0.0)
    reverb_time = float(getattr(args, "reverb_time", 1.0) or 1.0)
    if intensity > 0.0 and reverb_preset == "off" and reverb_mix <= 0.0 and emotion == "calm":
        reverb_preset = "room"
        reverb_mix = float(np.clip(0.06 + 0.10 * intensity, 0.0, 0.35))

    mb_thr = getattr(args, "mb_comp_threshold_db", None)
    mb_thr_f = float(mb_thr) if mb_thr is not None else (-24.0 if (bool(getattr(args, "polish", False)) and intensity > 0.0) else None)

    return PostProcessParams(
        denoise_strength=float(denoise),
        transient_amount=float(transient),
        transient_split_hz=1200.0,
        multiband=bool(getattr(args, "multiband", False) or (bool(getattr(args, "polish", False)) and intensity > 0.0)),
        multiband_low_hz=float(getattr(args, "mb_low_hz", 180.0)),
        multiband_high_hz=float(getattr(args, "mb_high_hz", 3800.0)),
        multiband_low_gain_db=float(getattr(args, "mb_low_gain_db", 0.0)),
        multiband_mid_gain_db=float(getattr(args, "mb_mid_gain_db", 0.0)),
        multiband_high_gain_db=float(getattr(args, "mb_high_gain_db", 0.0)),
        multiband_comp_threshold_db=mb_thr_f,
        multiband_comp_ratio=float(getattr(args, "mb_comp_ratio", 2.0)),
        formant_shift=float(getattr(args, "formant_shift", 0.0)),
        creature_size=float(getattr(args, "creature_size", 0.0)),
        texture_preset=str(texture_preset),
        texture_amount=float(texture_amount),
        texture_grain_ms=float(getattr(args, "texture_grain_ms", 28.0)),
        texture_spray=float(getattr(args, "texture_spray", 0.35)),
        reverb_preset=str(reverb_preset),
        reverb_mix=float(reverb_mix),
        reverb_time_s=float(reverb_time),
        random_seed=(int(post_seed) if post_seed is not None else None),
        prompt_hint=(str(prompt_hint) if prompt_hint else None),
        compressor_threshold_db=(float(comp_thr) if comp_thr is not None else None),
        compressor_makeup_db=float(comp_makeup),
        limiter_ceiling_db=(float(limiter) if limiter is not None else None),
        loop_clean=bool(getattr(args, "loop", False)),
        loop_crossfade_ms=int(getattr(args, "loop_crossfade_ms", 100)),
    )


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

    # Diffusers multi-band mode (model-side; slower)
    p.add_argument("--diffusers-multiband", action="store_true")
    p.add_argument("--diffusers-mb-mode", choices=["auto", "2band", "3band"], default="auto")
    p.add_argument("--diffusers-mb-low-hz", type=float, default=250.0)
    p.add_argument("--diffusers-mb-high-hz", type=float, default=3000.0)

    # Post + pro controls (applies to all items where manifest sets post=true)
    p.add_argument(
        "--pro-preset",
        choices=["off", *pro_preset_keys()],
        default="off",
        help="High-level preset that sets sensible defaults (polish/conditioning/DSP). Only overrides values still at their defaults.",
    )
    p.add_argument(
        "--polish-profile",
        choices=["off", *polish_profile_keys()],
        default="off",
        help="Named post/polish profile (AAA-style chain). Only overrides values still at their defaults.",
    )
    p.add_argument("--polish", action="store_true", help="Enable conservative denoise/transient/compress/limit defaults")
    p.add_argument("--emotion", choices=["neutral", "aggressive", "calm", "scared"], default="neutral")
    p.add_argument("--intensity", type=float, default=0.0, help="0..1")
    p.add_argument("--variation", type=float, default=0.0, help="0..1")
    p.add_argument("--pitch-contour", choices=["flat", "rise", "fall", "updown", "downup"], default="flat")

    p.add_argument("--multiband", action="store_true")
    p.add_argument("--mb-low-hz", type=float, default=180.0)
    p.add_argument("--mb-high-hz", type=float, default=3800.0)
    p.add_argument("--mb-low-gain-db", type=float, default=0.0)
    p.add_argument("--mb-mid-gain-db", type=float, default=0.0)
    p.add_argument("--mb-high-gain-db", type=float, default=0.0)
    p.add_argument("--mb-comp-threshold-db", type=float, default=None)
    p.add_argument("--mb-comp-ratio", type=float, default=2.0)

    p.add_argument("--creature-size", type=float, default=0.0, help="-1..+1")
    p.add_argument("--formant-shift", type=float, default=0.0)

    p.add_argument("--texture-preset", choices=["off", "auto", "chitter", "rasp", "buzz", "screech"], default="off")
    p.add_argument("--texture-amount", type=float, default=0.0)
    p.add_argument("--texture-grain-ms", type=float, default=28.0)
    p.add_argument("--texture-spray", type=float, default=0.35)

    p.add_argument("--reverb", choices=["off", "room", "cave", "forest", "nether"], default="off")
    p.add_argument("--reverb-mix", type=float, default=0.0)
    p.add_argument("--reverb-time", type=float, default=1.0)

    p.add_argument(
        "--loop",
        action="store_true",
        help="Loop-clean output (ambience): blend the end into the start to reduce seam clicks.",
    )
    p.add_argument(
        "--loop-crossfade-ms",
        type=int,
        default=100,
        help="Loop-clean crossfade window in milliseconds (default 100).",
    )

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

            seed_i = (int(item.seed) + i) if item.seed is not None else None

            if item.engine == "samplelib" and not zip_args:
                raise FileNotFoundError(
                    "engine=samplelib but no --library-zip provided and no default zips found at .examples/sound libraies/*.zip"
                )

            pitch_contour = str(getattr(args, "pitch_contour", "flat") or "flat")
            effective_prompt = item.prompt

            # Optional pro preset prompt augmentation (only for AI/sample selection engines).
            preset_key = str(getattr(args, "pro_preset", "off") or "off").strip()
            preset_obj = PRO_PRESETS.get(preset_key) if preset_key.lower() != "off" else None
            if preset_obj is not None and preset_obj.prompt_suffix and item.engine in {"diffusers", "replicate", "samplelib", "layered"}:
                suf = str(preset_obj.prompt_suffix).strip()
                if suf and suf.lower() not in effective_prompt.lower():
                    effective_prompt = f"{effective_prompt}, {suf}"

            if pitch_contour != "flat":
                effective_prompt = f"{effective_prompt}, pitch contour {pitch_contour}"

            def _pp(audio, sr):
                processed, _ = post_process_audio(
                    audio,
                    sr,
                    _pp_params(args=args, post_seed=seed_i, prompt_hint=effective_prompt),
                )
                return processed, "post"

            generated = generate_wav(
                item.engine,
                prompt=effective_prompt,
                seconds=float(item.seconds),
                seed=seed_i,
                out_wav=tmp_wav,
                postprocess_fn=(_pp if item.post else None),
                device=str(args.device),
                model=str(args.model),
                diffusers_multiband=bool(getattr(args, "diffusers_multiband", False)),
                diffusers_multiband_mode=str(getattr(args, "diffusers_mb_mode", "auto")),
                diffusers_multiband_low_hz=float(getattr(args, "diffusers_mb_low_hz", 250.0)),
                diffusers_multiband_high_hz=float(getattr(args, "diffusers_mb_high_hz", 3000.0)),
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
            credits: dict = {
                "engine": item.engine,
                "prompt": item.prompt,
                "emotion": str(getattr(args, "emotion", "neutral") or "neutral"),
                "intensity": float(getattr(args, "intensity", 0.0) or 0.0),
                "variation": float(getattr(args, "variation", 0.0) or 0.0),
                "pitch_contour": pitch_contour,
            }
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
    parser = build_parser()
    args = parser.parse_args(argv)

    # Apply pro preset after parsing so we can compare against argparse defaults.
    apply_pro_preset(preset_key=str(getattr(args, "pro_preset", "off")), args=args, parser=parser)
    apply_polish_profile(profile_key=str(getattr(args, "polish_profile", "off")), args=args, parser=parser)
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
