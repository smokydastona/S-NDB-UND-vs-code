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
from .fx_chains import apply_fx_chain, fx_chain_keys
from .sfx_presets import apply_sfx_preset, render_prompt_template


def _default_sound_path(namespace: str, event: str) -> str:
    # event can contain dots; use folder style
    safe = event.replace(":", ".")
    return f"generated/{safe.replace('.', '/')}"


def _slug(s: str) -> str:
    import re

    s = str(s or "").strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_\-.]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:64] or "sound"


def _pp_params(*, args: argparse.Namespace, post_seed: int | None, prompt_hint: str | None) -> PostProcessParams:
    # Reasonable defaults for Minecraft, plus optional "pro" DSP controls.
    intensity = float(np.clip(float(getattr(args, "intensity", 0.0)), 0.0, 1.0))
    variation = float(np.clip(float(getattr(args, "variation", 0.0)), 0.0, 1.0))
    emotion = str(getattr(args, "emotion", "neutral") or "neutral").strip().lower()

    denoise = (0.25 if bool(getattr(args, "polish", False)) else 0.0)
    transient = (0.25 if bool(getattr(args, "polish", False)) else 0.0)
    transient_sustain = 0.0
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

    denoise_ovr = getattr(args, "denoise_amount", None)
    if denoise_ovr is not None:
        denoise = float(np.clip(float(denoise_ovr), 0.0, 1.0))

    trans_attack_ovr = getattr(args, "transient_attack", None)
    if trans_attack_ovr is not None:
        transient = float(np.clip(float(trans_attack_ovr), -1.0, 1.0))

    trans_sustain_ovr = getattr(args, "transient_sustain", None)
    if trans_sustain_ovr is not None:
        transient_sustain = float(np.clip(float(trans_sustain_ovr), -1.0, 1.0))

    exciter = 0.0
    exciter_ovr = getattr(args, "exciter_amount", None)
    if exciter_ovr is not None:
        exciter = float(np.clip(float(exciter_ovr), 0.0, 1.0))

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
        transient_sustain=float(transient_sustain),
        transient_split_hz=1200.0,
        exciter_amount=float(exciter),
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
        compressor_attack_ms=float(getattr(args, "compressor_attack_ms", 5.0) if getattr(args, "compressor_attack_ms", None) is not None else (5.0 if bool(getattr(args, "polish", False)) else 5.0)),
        compressor_release_ms=float(getattr(args, "compressor_release_ms", 90.0) if getattr(args, "compressor_release_ms", None) is not None else (90.0 if bool(getattr(args, "polish", False)) else 80.0)),
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

    p.add_argument(
        "--candidates",
        type=int,
        default=1,
        help="Generate N candidates per output and pick the best using QA metrics (default 1). Manifest item can override with 'candidates'.",
    )

    # Concrete SFX presets (optional): engine + prompt + defaults + FX chain.
    p.add_argument(
        "--sfx-preset",
        default="off",
        help=(
            "Apply a concrete SFX preset (engine + prompt + defaults + FX chain). "
            "Searches library/sfx_presets.json then configs/sfx_presets_v1.example.json."
        ),
    )
    p.add_argument(
        "--sfx-preset-file",
        default=None,
        help="Path to a JSON preset library file containing a 'presets' list.",
    )

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

    # FX chains (v1): named chains + optional JSON definition.
    p.add_argument(
        "--fx-chain",
        choices=["off", *fx_chain_keys()],
        default="off",
        help="Named FX chain preset (shareable post chain). Only overrides values still at their defaults.",
    )
    p.add_argument(
        "--fx-chain-json",
        default=None,
        help="Load an FX chain JSON file. Supports either an args-patch JSON or an effect-list JSON.",
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
    p.add_argument("--formant-shift", type=float, default=1.0)

    # Pro Mode overrides (optional). If omitted, polish/conditioning selects conservative values.
    p.add_argument("--denoise-amount", type=float, default=None, help="Override denoise amount 0..1 (omit to use defaults).")
    p.add_argument("--transient-attack", type=float, default=None, help="Override transient attack emphasis -1..+1 (omit to use defaults).")
    p.add_argument("--transient-sustain", type=float, default=None, help="Override transient sustain emphasis -1..+1 (omit to use defaults).")
    p.add_argument("--exciter-amount", type=float, default=None, help="Override exciter amount 0..1 (omit to use defaults).")
    p.add_argument("--compressor-attack-ms", type=float, default=None, help="Override compressor attack (ms) for polish mode.")
    p.add_argument("--compressor-release-ms", type=float, default=None, help="Override compressor release (ms) for polish mode.")

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


def run_item(item: ManifestItem, *, args: argparse.Namespace, parser: argparse.ArgumentParser) -> list[Path]:
    pack_root = Path(args.pack_root)
    write_pack_mcmeta = args.mc_target == "resourcepack"

    # Per-item overrides (manifest) layered on top of CLI defaults.
    effective_args = argparse.Namespace(**vars(args))

    # Apply concrete SFX preset (manifest item wins over CLI).
    sfx_preset_obj = None
    sfx_preset_lib = None
    sfx_key = str(getattr(item, "sfx_preset", None) or getattr(args, "sfx_preset", "off") or "off").strip()
    sfx_file = str(getattr(item, "sfx_preset_file", None) or getattr(args, "sfx_preset_file", None) or "").strip() or None
    if sfx_key and sfx_key.lower() != "off":
        try:
            sfx_preset_obj, sfx_preset_lib = apply_sfx_preset(
                preset_key=sfx_key,
                preset_file=sfx_file,
                args=effective_args,
                parser=parser,
            )
        except ValueError as e:
            raise ValueError(f"Manifest item preset error (event={item.event}): {e}") from e
    if item.pro_preset:
        effective_args.pro_preset = str(item.pro_preset)
        apply_pro_preset(preset_key=str(item.pro_preset), args=effective_args, parser=parser)
    if item.polish_profile:
        effective_args.polish_profile = str(item.polish_profile)
        apply_polish_profile(profile_key=str(item.polish_profile), args=effective_args, parser=parser)

    if item.emotion:
        effective_args.emotion = str(item.emotion)
    if item.intensity is not None:
        effective_args.intensity = float(item.intensity)
    if item.variation is not None:
        effective_args.variation = float(item.variation)
    if item.pitch_contour:
        effective_args.pitch_contour = str(item.pitch_contour)
    if item.loop is not None:
        effective_args.loop = bool(item.loop)
    if item.loop_crossfade_ms is not None:
        effective_args.loop_crossfade_ms = int(item.loop_crossfade_ms)

    # Resolve prompt/engine/seconds with preset support.
    engine = str(getattr(item, "engine", None) or "").strip() or None
    prompt = str(getattr(item, "prompt", "") or "").strip()
    seconds = item.seconds
    candidates = item.candidates
    seed_base = item.seed

    if sfx_preset_obj is not None:
        # If engine/prompt/seconds omitted (or left as the manifest defaults), fill from preset.
        if not prompt:
            prompt = str(sfx_preset_obj.prompt)
        if (engine is None) or (engine == "rfxgen"):
            engine = str(sfx_preset_obj.engine)
        if seconds is None:
            seconds = sfx_preset_obj.seconds
        if seed_base is None:
            seed_base = sfx_preset_obj.seed

    preset_engine_params: dict = dict(getattr(sfx_preset_obj, "engine_params", None) or {}) if sfx_preset_obj is not None else {}

    if not prompt:
        raise ValueError(f"Manifest item missing prompt (event={item.event}).")
    if engine is None:
        engine = "rfxgen"
    if seconds is None:
        seconds = 3.0

    effective_candidates = int(candidates) if candidates is not None else int(getattr(args, "candidates", 1) or 1)
    if effective_candidates < 1:
        effective_candidates = 1

    # Apply FX chain (manifest item wins over preset wins over CLI).
    fx_chain_key = str(getattr(item, "fx_chain", None) or getattr(effective_args, "fx_chain", "off") or "off").strip()
    fx_chain_json = str(getattr(item, "fx_chain_json", None) or getattr(effective_args, "fx_chain_json", None) or "").strip() or None
    apply_fx_chain(chain_key=fx_chain_key, chain_json=fx_chain_json, args=effective_args, parser=parser)

    namespace = item.namespace
    event = (item.event or "").strip()
    if not event:
        # Automatic naming for modder workflows: derive an event id from the prompt.
        event = f"generated.batch.{_slug(prompt)}"

    base_sound_path = item.sound_path or _default_sound_path(namespace, event)
    out_files: list[Path] = []

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        default_zips = sorted(Path(".examples").joinpath("sound libraies").glob("*.zip"))
        zip_args = effective_args.library_zip if effective_args.library_zip else [str(p) for p in default_zips]

        for i in range(max(1, int(item.variants))):
            suffix = f"_{i+1:02d}" if item.variants > 1 else ""
            sound_path = f"{base_sound_path}{suffix}"

            tmp_wav = tmp_dir / f"{namespace}_{event.replace('.', '_')}{suffix}.wav"

            seed_i = (int(seed_base) + i) if seed_base is not None else None

            if engine == "samplelib" and not zip_args:
                raise FileNotFoundError(
                    "engine=samplelib but no --library-zip provided and no default zips found at .examples/sound libraies/*.zip"
                )

            pitch_contour = str(getattr(effective_args, "pitch_contour", "flat") or "flat")
            effective_prompt = prompt

            # Smart preset v2: render prompt template vars using the per-variant seed.
            if sfx_preset_obj is not None and getattr(sfx_preset_obj, "vars", None):
                try:
                    effective_prompt, chosen = render_prompt_template(
                        str(effective_prompt),
                        vars_def=dict(getattr(sfx_preset_obj, "vars") or {}),
                        seed=seed_i,
                    )
                    setattr(effective_args, "sfx_preset_vars_chosen", chosen)
                except Exception:
                    pass

            # Optional pro preset prompt augmentation (only for AI/sample selection engines).
            preset_key = str(getattr(effective_args, "pro_preset", "off") or "off").strip()
            preset_obj = PRO_PRESETS.get(preset_key) if preset_key.lower() != "off" else None
            if preset_obj is not None and preset_obj.prompt_suffix and engine in {"diffusers", "replicate", "samplelib", "layered", "stable_audio_open"}:
                suf = str(preset_obj.prompt_suffix).strip()
                if suf and suf.lower() not in effective_prompt.lower():
                    effective_prompt = f"{effective_prompt}, {suf}"

            if pitch_contour != "flat":
                effective_prompt = f"{effective_prompt}, pitch contour {pitch_contour}"

            def _pp(audio, sr):
                processed, _ = post_process_audio(
                    audio,
                    sr,
                    _pp_params(args=effective_args, post_seed=seed_i, prompt_hint=effective_prompt),
                )
                return processed, "post"

            generated = generate_wav(
                engine,
                prompt=effective_prompt,
                seconds=float(seconds),
                seed=seed_i,
                out_wav=tmp_wav,
                candidates=effective_candidates,
                postprocess_fn=(_pp if item.post else None),
                device=str(effective_args.device),
                model=str(effective_args.model),
                diffusers_multiband=bool(getattr(effective_args, "diffusers_multiband", False)),
                diffusers_multiband_mode=str(getattr(effective_args, "diffusers_mb_mode", "auto")),
                diffusers_multiband_low_hz=float(getattr(effective_args, "diffusers_mb_low_hz", 250.0)),
                diffusers_multiband_high_hz=float(getattr(effective_args, "diffusers_mb_high_hz", 3000.0)),
                preset=(
                    item.preset
                    or (str(preset_engine_params.get("rfxgen_preset")).strip() if preset_engine_params.get("rfxgen_preset") else None)
                    or (str(preset_engine_params.get("preset")).strip() if preset_engine_params.get("preset") else None)
                    or None
                ),
                layered_preset=(
                    item.preset
                    or (str(preset_engine_params.get("layered_preset")).strip() if preset_engine_params.get("layered_preset") else None)
                    or "auto"
                ),
                rfxgen_path=(Path(effective_args.rfxgen_path) if effective_args.rfxgen_path else None),
                library_zips=tuple(Path(p) for p in zip_args),
                library_pitch_min=float(effective_args.library_pitch_min),
                library_pitch_max=float(effective_args.library_pitch_max),
                stable_audio_model=(
                    str(item.stable_audio_model)
                    if item.stable_audio_model
                    else (str(preset_engine_params.get("stable_audio_model")).strip() if preset_engine_params.get("stable_audio_model") else None)
                ),
                stable_audio_negative_prompt=(
                    str(item.stable_audio_negative_prompt)
                    if item.stable_audio_negative_prompt
                    else (
                        str(getattr(sfx_preset_obj, "negative_prompt", None)).strip()
                        if (sfx_preset_obj is not None and getattr(sfx_preset_obj, "negative_prompt", None))
                        else (str(preset_engine_params.get("stable_audio_negative_prompt")).strip() if preset_engine_params.get("stable_audio_negative_prompt") else None)
                    )
                ),
                stable_audio_steps=(
                    int(item.stable_audio_steps)
                    if item.stable_audio_steps is not None
                    else (int(preset_engine_params["stable_audio_steps"]) if preset_engine_params.get("stable_audio_steps") is not None else 100)
                ),
                stable_audio_guidance_scale=(
                    float(item.stable_audio_guidance_scale)
                    if item.stable_audio_guidance_scale is not None
                    else (float(preset_engine_params["stable_audio_guidance_scale"]) if preset_engine_params.get("stable_audio_guidance_scale") is not None else 7.0)
                ),
                stable_audio_sampler=(
                    str(item.stable_audio_sampler)
                    if item.stable_audio_sampler
                    else (str(preset_engine_params.get("stable_audio_sampler")).strip() if preset_engine_params.get("stable_audio_sampler") else None)
                ),
                stable_audio_hf_token=(str(item.stable_audio_hf_token) if item.stable_audio_hf_token else None),
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
                ogg_quality=int(effective_args.ogg_quality),
                sample_rate=int(effective_args.mc_sample_rate),
                channels=int(effective_args.mc_channels),
                description="S-NDB-UND pack",
                write_pack_mcmeta=write_pack_mcmeta,
            )
            out_files.append(ogg_path)

            # Pack credits for all engines.
            credits: dict = {
                "engine": engine,
                "prompt": prompt,
                "candidates": effective_candidates,
                "emotion": str(getattr(effective_args, "emotion", "neutral") or "neutral"),
                "intensity": float(getattr(effective_args, "intensity", 0.0) or 0.0),
                "variation": float(getattr(effective_args, "variation", 0.0) or 0.0),
                "pitch_contour": pitch_contour,
                "pro_preset": str(getattr(effective_args, "pro_preset", "off")),
                "polish_profile": str(getattr(effective_args, "polish_profile", "off")),
                "fx_chain": fx_chain_key,
                "loop_clean": bool(getattr(effective_args, "loop", False)),
                "loop_crossfade_ms": int(getattr(effective_args, "loop_crossfade_ms", 100)),
            }
            if sfx_preset_obj is not None:
                credits["sfx_preset"] = str(sfx_preset_obj.name)
            if sfx_preset_lib is not None:
                credits["sfx_preset_library"] = str(sfx_preset_lib)
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
                Path(effective_args.catalog),
                make_record(
                    engine=engine,
                    prompt=prompt,
                    namespace=namespace,
                    event=event,
                    sound_path=sound_path,
                    output_file=ogg_path,
                    tags=item.tags,
                    sources=sources,
                    seconds=seconds,
                    seed=seed_base,
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
        outs = run_item(item, args=args, parser=parser)
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
