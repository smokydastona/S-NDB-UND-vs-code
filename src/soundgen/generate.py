from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np

from .audiogen_backend import GenerationParams, generate_audio
from .io_utils import read_wav_mono, write_wav
from .postprocess import PostProcessParams, post_process_audio
from .qa import compute_metrics, detect_long_tail
from .rfxgen_backend import RfxGenParams, generate_with_rfxgen
from .replicate_backend import ReplicateParams, generate_with_replicate
from .minecraft import export_wav_to_minecraft_pack


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate a sound effect WAV from a text prompt.")
    p.add_argument(
        "--engine",
        choices=["diffusers", "rfxgen", "replicate"],
        default="diffusers",
        help="Generation engine: diffusers (AI), rfxgen (procedural presets), or replicate (paid API).",
    )
    p.add_argument("--prompt", required=True, help="Text prompt describing the sound.")
    p.add_argument("--seconds", type=float, default=3.0, help="Duration in seconds.")
    p.add_argument("--seed", type=int, default=None, help="Random seed for repeatability.")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="Compute device.")
    p.add_argument(
        "--model",
        default="cvssp/audioldm2",
        help="Diffusers pretrained model id (e.g. cvssp/audioldm2).",
    )
    p.add_argument(
        "--preset",
        default=None,
        help="rfxgen preset override (coin, laser, explosion, powerup, hit, jump, blip). If omitted, inferred from prompt.",
    )
    p.add_argument(
        "--rfxgen-path",
        default=None,
        help="Path to rfxgen executable (e.g. tools/rfxgen/rfxgen.exe). If omitted, uses PATH.",
    )

    # Replicate (optional paid API backend)
    p.add_argument("--replicate-model", default=None, help="Replicate model id (e.g. owner/model)")
    p.add_argument("--replicate-token", default=None, help="Replicate API token (or set REPLICATE_API_TOKEN)")
    p.add_argument(
        "--replicate-input-json",
        default=None,
        help="Extra JSON object merged into Replicate input (model-specific).",
    )

    # Minecraft resource pack export
    p.add_argument(
        "--minecraft",
        action="store_true",
        help="Export as a Minecraft resource pack sound (.ogg) + update sounds.json.",
    )
    p.add_argument(
        "--pack-root",
        default="resourcepack",
        help="Root folder for output. For resource packs: the pack root. For Forge: usually <mod>/src/main/resources.",
    )
    p.add_argument(
        "--mc-target",
        choices=["resourcepack", "forge"],
        default="resourcepack",
        help="Export target layout. 'resourcepack' creates pack.mcmeta. 'forge' writes under a mod resources folder.",
    )
    p.add_argument(
        "--namespace",
        default="soundgen",
        help="Minecraft namespace (modid) for assets/<namespace>/... (e.g. yourmodid).",
    )
    p.add_argument(
        "--event",
        default=None,
        help="sounds.json event key (e.g. laser.fire). Full id becomes <namespace>:<event>. Default: generated.<slug>",
    )
    p.add_argument(
        "--sound-path",
        default=None,
        help="Path under assets/<namespace>/sounds/ WITHOUT extension (e.g. sfx/laser_01). Default: generated/<slug>",
    )
    p.add_argument(
        "--variants",
        type=int,
        default=1,
        help="Number of sound variants to generate and register under the same event (default 1).",
    )
    p.add_argument(
        "--weight",
        type=int,
        default=1,
        help="Weight for each variant entry in sounds.json (default 1).",
    )
    p.add_argument(
        "--volume",
        type=float,
        default=1.0,
        help="Volume for sounds.json entry (default 1.0).",
    )
    p.add_argument(
        "--pitch",
        type=float,
        default=1.0,
        help="Pitch for sounds.json entry (default 1.0).",
    )
    p.add_argument(
        "--subtitle",
        default=None,
        help="Subtitle text to add to sounds.json and assets/<ns>/lang/en_us.json (optional).",
    )
    p.add_argument(
        "--subtitle-key",
        default=None,
        help="Translation key to use for subtitle (optional). Default: subtitles.<namespace>.<event>",
    )
    p.add_argument(
        "--ogg-quality",
        type=int,
        default=5,
        help="Vorbis VBR quality 0-10 for Minecraft .ogg export (default 5).",
    )
    p.add_argument(
        "--mc-sample-rate",
        type=int,
        default=44100,
        help="Sample rate for Minecraft .ogg export (default 44100).",
    )
    p.add_argument(
        "--mc-channels",
        type=int,
        default=1,
        help="Channels for Minecraft .ogg export: 1 mono or 2 stereo (default 1).",
    )

    # Post-processing / QA
    p.add_argument("--post", action="store_true", help="Enable post-processing (trim/fade/normalize/EQ).")
    p.add_argument("--no-trim", action="store_true", help="Disable trimming silence (post-processing).")
    p.add_argument("--silence-threshold-db", type=float, default=-40.0, help="Trim threshold in dBFS.")
    p.add_argument("--silence-padding-ms", type=int, default=30, help="Padding kept around trimmed audio.")
    p.add_argument("--fade-ms", type=int, default=8, help="Fade in/out duration to prevent clicks.")
    p.add_argument(
        "--normalize-rms-db",
        type=float,
        default=-18.0,
        help="Approx loudness target (RMS). Use 0 to disable RMS normalization.",
    )
    p.add_argument("--normalize-peak-db", type=float, default=-1.0, help="Peak cap target in dBFS.")
    p.add_argument("--highpass-hz", type=float, default=40.0, help="Highpass cutoff. Use 0 to disable.")
    p.add_argument("--lowpass-hz", type=float, default=16000.0, help="Lowpass cutoff. Use 0 to disable.")

    p.add_argument("--out", default="outputs/out.wav", help="Output WAV path.")
    return p


def _slug_from_prompt(prompt: str) -> str:
    import re

    s = prompt.strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_\-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:48] or "sound"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    out_path = Path(args.out)
    slug = _slug_from_prompt(args.prompt)

    def _pp_params() -> PostProcessParams:
        return PostProcessParams(
            trim_silence=(not args.no_trim),
            silence_threshold_db=float(args.silence_threshold_db),
            silence_padding_ms=int(args.silence_padding_ms),
            fade_ms=int(args.fade_ms),
            normalize_rms_db=(None if float(args.normalize_rms_db) == 0.0 else float(args.normalize_rms_db)),
            normalize_peak_db=float(args.normalize_peak_db),
            highpass_hz=(None if float(args.highpass_hz) == 0.0 else float(args.highpass_hz)),
            lowpass_hz=(None if float(args.lowpass_hz) == 0.0 else float(args.lowpass_hz)),
        )

    def _qa_info(audio: np.ndarray, sr: int) -> str:
        m = compute_metrics(audio, sr)
        flags: list[str] = []
        if m.clipped:
            flags.append("CLIPPING")
        if detect_long_tail(audio, sr):
            flags.append("LONG_TAIL")
        flag_s = (" " + " ".join(flags)) if flags else ""
        return f"qa: peak={m.peak:.3f} rms={m.rms:.3f}{flag_s}".strip()

    def _maybe_postprocess(audio: np.ndarray, sr: int) -> tuple[np.ndarray, str]:
        if not args.post:
            return audio, _qa_info(audio, sr)
        processed, rep = post_process_audio(audio, sr, _pp_params())
        info = _qa_info(processed, sr)
        return processed, f"post: trimmed={rep.trimmed} {info}".strip()

    def export_if_minecraft(wav_path: Path, *, sound_path: str) -> None:
        if not args.minecraft:
            return

        pack_root = Path(args.pack_root)
        namespace = args.namespace
        effective_sound_path = args.sound_path or sound_path
        event = args.event or f"generated.{slug}"
        write_pack_mcmeta = args.mc_target == "resourcepack"

        ogg_path = export_wav_to_minecraft_pack(
            wav_path,
            pack_root=pack_root,
            namespace=namespace,
            event=event,
            sound_path=effective_sound_path,
            weight=int(args.weight),
            volume=float(args.volume),
            pitch=float(args.pitch),
            subtitle=args.subtitle,
            subtitle_key=args.subtitle_key,
            ogg_quality=args.ogg_quality,
            sample_rate=args.mc_sample_rate,
            channels=args.mc_channels,
            description="Sound Generator pack",
            write_pack_mcmeta=write_pack_mcmeta,
        )
        print(f"Minecraft export: {ogg_path}")
        print(f"Minecraft playsound id: {namespace}:{event}")

    if args.engine == "rfxgen":
        if args.minecraft:
            with tempfile.TemporaryDirectory() as tmp:
                base_sound_path = args.sound_path or f"generated/{slug}"
                variants = max(1, int(args.variants))

                for i in range(variants):
                    suffix = f"_{i+1:02d}" if variants > 1 else ""
                    tmp_wav = Path(tmp) / f"rfxgen{suffix}.wav"
                    rfx_params = RfxGenParams(
                        prompt=args.prompt,
                        out_path=tmp_wav,
                        rfxgen_path=Path(args.rfxgen_path) if args.rfxgen_path else None,
                        preset=args.preset,
                    )
                    written = generate_with_rfxgen(rfx_params)

                    if args.post:
                        a, sr = read_wav_mono(written)
                        a, info = _maybe_postprocess(a, sr)
                        write_wav(written, a, sr)
                        print(info)

                    export_if_minecraft(written, sound_path=f"{base_sound_path}{suffix}")
        else:
            rfx_params = RfxGenParams(
                prompt=args.prompt,
                out_path=out_path,
                rfxgen_path=Path(args.rfxgen_path) if args.rfxgen_path else None,
                preset=args.preset,
            )
            written = generate_with_rfxgen(rfx_params)
            if args.post:
                a, sr = read_wav_mono(written)
                a, info = _maybe_postprocess(a, sr)
                write_wav(written, a, sr)
                print(info)
            print(f"Wrote {written}")
        return 0

    if args.engine == "replicate":
        # Replicate returns a file (typically wav) at out_path.
        rp = ReplicateParams(
            prompt=args.prompt,
            seconds=float(args.seconds),
            out_path=out_path,
            model=str(args.replicate_model or "").strip(),
            api_token=(str(args.replicate_token).strip() if args.replicate_token else None),
            extra_input_json=(str(args.replicate_input_json) if args.replicate_input_json else None),
        )
        written = generate_with_replicate(rp)

        # Optional post-processing (works if written is WAV)
        if args.post and written.suffix.lower() == ".wav":
            a, sr = read_wav_mono(written)
            a, info = _maybe_postprocess(a, sr)
            write_wav(written, a, sr)
            print(info)

        if args.minecraft:
            # Replicate export requires ffmpeg (wav->ogg) for Minecraft.
            export_if_minecraft(written, sound_path=args.sound_path or f"generated/{slug}")
        else:
            print(f"Wrote {written}")
        return 0

    params = GenerationParams(
        prompt=args.prompt,
        seconds=args.seconds,
        seed=args.seed,
        device=args.device,
        model=args.model,
    )

    audio, sr = generate_audio(params)
    audio, info = _maybe_postprocess(audio, sr)
    if args.minecraft:
        with tempfile.TemporaryDirectory() as tmp:
            base_sound_path = args.sound_path or f"generated/{slug}"
            variants = max(1, int(args.variants))

            # If user supplied a seed, use it as base; otherwise, use sequential seeds.
            base_seed = args.seed if args.seed is not None else 1337

            for i in range(variants):
                suffix = f"_{i+1:02d}" if variants > 1 else ""
                tmp_wav = Path(tmp) / f"diffusers{suffix}.wav"

                if variants > 1:
                    v_params = GenerationParams(
                        prompt=args.prompt,
                        seconds=args.seconds,
                        seed=int(base_seed) + i,
                        device=args.device,
                        model=args.model,
                    )
                    v_audio, v_sr = generate_audio(v_params)
                    v_audio, v_info = _maybe_postprocess(v_audio, v_sr)
                    print(v_info)
                    write_wav(tmp_wav, v_audio, v_sr)
                else:
                    write_wav(tmp_wav, audio, sr)

                export_if_minecraft(tmp_wav, sound_path=f"{base_sound_path}{suffix}")
    else:
        write_wav(out_path, audio, sr)
        print(info)
        print(f"Wrote {out_path} ({len(audio)/sr:.2f}s @ {sr}Hz)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
