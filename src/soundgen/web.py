from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import gradio as gr
import numpy as np

from .credits import upsert_pack_credits, write_sidecar_credits
from .engine_registry import generate_wav
from .io_utils import convert_audio_with_ffmpeg, read_wav_mono, write_wav
from .rfxgen_backend import SUPPORTED_PRESETS
from .minecraft import export_wav_to_minecraft_pack
from .postprocess import PostProcessParams, post_process_audio
from .qa import compute_metrics, detect_long_tail
from .qa_viz import spectrogram_image, waveform_image
from .controls import map_prompt_to_controls
from .pro_presets import PRO_PRESETS, get_pro_preset, pro_preset_keys, pro_preset_recommended_profile
from .polish_profiles import POLISH_PROFILES, polish_profile_keys


def _generate(
    engine: str,
    prompt: str,
    seconds: float,
    seed: int | None,
    candidates: int,
    device: str,
    model: str,
    stable_audio_model: str,
    stable_audio_negative_prompt: str,
    stable_audio_hf_token: str,
    stable_audio_steps: int,
    stable_audio_guidance_scale: float,
    stable_audio_sampler: str,
    diffusers_multiband: bool,
    diffusers_mb_mode: str,
    diffusers_mb_low_hz: float,
    diffusers_mb_high_hz: float,
    preset: str,
    rfxgen_path: str,
    library_mix_count: int,
    synth_waveform: str,
    layered_preset: str,
    layered_curve: str,
    layered_duck: float,
    layered_family: bool,
    layered_source_lock: bool,
    layered_source_seed: float | None,
    layered_micro_variation: float,
    layered_granular_preset: str,
    layered_granular_amount: float,
    layered_granular_grain_ms: float,
    layered_granular_spray: float,
    layered_transient_sharpness: float,
    layered_tail_length_ms: int,
    layered_transient_tilt: float,
    layered_body_tilt: float,
    layered_tail_tilt: float,
    out_format: str,
    out_sample_rate: int | None,
    wav_subtype: str,
    mp3_bitrate: str,
    map_controls: bool,

    pro_preset: str,
    polish_profile: str,

    emotion: str,
    intensity: float,
    variation: float,
    pitch_contour: str,
    multiband: bool,
    mb_low_hz: float,
    mb_high_hz: float,
    mb_low_gain_db: float,
    mb_mid_gain_db: float,
    mb_high_gain_db: float,
    mb_comp_threshold_db: float | None,
    mb_comp_ratio: float,
    creature_size: float,
    formant_shift: float,
    texture_preset: str,
    texture_amount: float,
    texture_grain_ms: float,
    texture_spray: float,
    reverb_preset: str,
    reverb_mix: float,
    reverb_time: float,

    denoise_amount: float | None,
    transient_attack: float | None,
    transient_sustain: float | None,
    exciter_amount: float | None,
    compressor_attack_ms: float | None,
    compressor_release_ms: float | None,

    loop_clean: bool,

    export_minecraft: bool,
    mc_target: str,
    pack_root: str,
    namespace: str,
    event: str,
    sound_path: str,
    subtitle: str,
    variants: int,
    weight: int,
    volume: float,
    pitch: float,
    ogg_quality: int,

    post: bool,
    polish: bool,

    # Extra engine/settings knobs (kept at the end of the signature to avoid breaking existing UI wiring)
    replicate_model: str,
    replicate_token: str,
    replicate_input_json: str,

    samplelib_zips: object,
    library_pitch_min: float,
    library_pitch_max: float,
    library_index: str,

    synth_freq_hz: float,
    synth_attack_ms: float,
    synth_decay_ms: float,
    synth_sustain: float,
    synth_release_ms: float,
    synth_noise_mix: float,
    synth_drive: float,
    synth_pitch_min: float,
    synth_pitch_max: float,
    synth_lowpass_hz: float,
    synth_highpass_hz: float,

    loop_crossfade_ms: int,

    subtitle_key: str,
    mc_sample_rate: int,
    mc_channels: int,

    layered_transient_ms_adv: int | None,
    layered_tail_ms_adv: int | None,
    layered_transient_attack_ms_adv: float | None,
    layered_transient_hold_ms_adv: float | None,
    layered_transient_decay_ms_adv: float | None,
    layered_body_attack_ms_adv: float | None,
    layered_body_hold_ms_adv: float | None,
    layered_body_decay_ms_adv: float | None,
    layered_tail_attack_ms_adv: float | None,
    layered_tail_hold_ms_adv: float | None,
    layered_tail_decay_ms_adv: float | None,
    layered_duck_release_ms_adv: float | None,
) -> tuple[str, str, str, object, object]:
    def _maybe(s: str) -> str | None:
        t = str(s or "").strip()
        return t if t else None

    def _as_zip_paths(v: object) -> tuple[Path, ...]:
        if v is None:
            return ()
        items: list[object]
        if isinstance(v, (list, tuple)):
            items = list(v)
        else:
            items = [v]
        out: list[Path] = []
        for it in items:
            if it is None:
                continue
            if isinstance(it, (str, Path)):
                p = Path(str(it))
                if p.exists() and p.suffix.lower() == ".zip":
                    out.append(p)
                continue
            # gradio file objects
            name = getattr(it, "name", None)
            if name:
                p = Path(str(name))
                if p.exists() and p.suffix.lower() == ".zip":
                    out.append(p)
                continue
            if isinstance(it, dict) and it.get("name"):
                p = Path(str(it["name"]))
                if p.exists() and p.suffix.lower() == ".zip":
                    out.append(p)
        return tuple(out)

    def _infer_out_format() -> str:
        fmt = (out_format or "wav").strip().lower()
        return fmt if fmt in {"wav", "mp3", "ogg", "flac"} else "wav"

    fmt = _infer_out_format()
    out_path = Path("outputs") / f"web.{fmt}"

    # Apply pro preset conservatively (only if fields are at UI defaults).
    preset_key = str(pro_preset or "off").strip()
    if preset_key and preset_key.lower() != "off":
        preset_obj = get_pro_preset(preset_key)
        if preset_obj is not None:
            # Prompt augmentation for AI/sample selection engines.
            eng = str(engine or "").strip().lower()
            if preset_obj.prompt_suffix and eng in {"diffusers", "stable_audio_open", "replicate", "samplelib", "layered"}:
                suf = str(preset_obj.prompt_suffix).strip()
                if suf and suf.lower() not in str(prompt).lower():
                    prompt = str(prompt).rstrip() + ", " + suf

            # Map preset dest names (CLI) -> web variable names.
            dest_map = {
                "reverb": "reverb_preset",
            }

            # UI defaults (must match the values defined in main()).
            defaults = {
                "seconds": 3.0,
                "post": True,
                "polish": False,
                "emotion": "neutral",
                "intensity": 0.0,
                "variation": 0.0,
                "pitch_contour": "flat",
                "multiband": False,
                "mb_low_hz": 250.0,
                "mb_high_hz": 3000.0,
                "mb_low_gain_db": 0.0,
                "mb_mid_gain_db": 0.0,
                "mb_high_gain_db": 0.0,
                "mb_comp_threshold_db": None,
                "mb_comp_ratio": 2.0,
                "creature_size": 0.0,
                "formant_shift": 1.0,
                "texture_preset": "off",
                "texture_amount": 0.0,
                "texture_grain_ms": 22.0,
                "texture_spray": 0.55,
                "reverb_preset": "off",
                "reverb_mix": 0.0,
                "reverb_time": 1.2,
                "denoise_amount": None,
                "transient_attack": None,
                "transient_sustain": None,
                "exciter_amount": None,
                "compressor_attack_ms": None,
                "compressor_release_ms": None,
            }

            state = {
                "seconds": seconds,
                "post": post,
                "polish": polish,
                "emotion": emotion,
                "intensity": intensity,
                "variation": variation,
                "pitch_contour": pitch_contour,
                "multiband": multiband,
                "mb_low_hz": mb_low_hz,
                "mb_high_hz": mb_high_hz,
                "mb_low_gain_db": mb_low_gain_db,
                "mb_mid_gain_db": mb_mid_gain_db,
                "mb_high_gain_db": mb_high_gain_db,
                "mb_comp_threshold_db": mb_comp_threshold_db,
                "mb_comp_ratio": mb_comp_ratio,
                "creature_size": creature_size,
                "formant_shift": formant_shift,
                "texture_preset": texture_preset,
                "texture_amount": texture_amount,
                "texture_grain_ms": texture_grain_ms,
                "texture_spray": texture_spray,
                "reverb_preset": reverb_preset,
                "reverb_mix": reverb_mix,
                "reverb_time": reverb_time,
                "denoise_amount": denoise_amount,
                "transient_attack": transient_attack,
                "transient_sustain": transient_sustain,
                "exciter_amount": exciter_amount,
                "compressor_attack_ms": compressor_attack_ms,
                "compressor_release_ms": compressor_release_ms,
            }

            for dest, value in (preset_obj.args or {}).items():
                web_dest = dest_map.get(str(dest), str(dest))
                if web_dest in state and state[web_dest] == defaults.get(web_dest, object()):
                    state[web_dest] = value

            seconds = float(state["seconds"])
            post = bool(state["post"])
            polish = bool(state["polish"])
            emotion = str(state["emotion"])
            intensity = float(state["intensity"])
            variation = float(state["variation"])
            pitch_contour = str(state["pitch_contour"])
            multiband = bool(state["multiband"])
            mb_low_hz = float(state["mb_low_hz"])
            mb_high_hz = float(state["mb_high_hz"])
            mb_low_gain_db = float(state["mb_low_gain_db"])
            mb_mid_gain_db = float(state["mb_mid_gain_db"])
            mb_high_gain_db = float(state["mb_high_gain_db"])
            mb_comp_threshold_db = state["mb_comp_threshold_db"]
            mb_comp_ratio = float(state["mb_comp_ratio"])
            creature_size = float(state["creature_size"])
            formant_shift = float(state["formant_shift"])
            texture_preset = str(state["texture_preset"])
            texture_amount = float(state["texture_amount"])
            texture_grain_ms = float(state["texture_grain_ms"])
            texture_spray = float(state["texture_spray"])
            reverb_preset = str(state["reverb_preset"])
            reverb_mix = float(state["reverb_mix"])
            reverb_time = float(state["reverb_time"])
            denoise_amount = state["denoise_amount"]
            transient_attack = state["transient_attack"]
            transient_sustain = state["transient_sustain"]
            exciter_amount = state["exciter_amount"]
            compressor_attack_ms = state["compressor_attack_ms"]
            compressor_release_ms = state["compressor_release_ms"]

    # Apply named polish profile conservatively (only if fields are at UI defaults).
    profile_key = str(polish_profile or "off").strip()
    if profile_key and profile_key.lower() != "off":
        prof = POLISH_PROFILES.get(profile_key)
        if prof is not None:
            defaults = {
                "post": True,
                "polish": False,
                "multiband": False,
                "mb_low_hz": 250.0,
                "mb_high_hz": 3000.0,
                "mb_low_gain_db": 0.0,
                "mb_mid_gain_db": 0.0,
                "mb_high_gain_db": 0.0,
                "mb_comp_threshold_db": None,
                "mb_comp_ratio": 2.0,
                "creature_size": 0.0,
                "formant_shift": 1.0,
                "texture_preset": "off",
                "texture_amount": 0.0,
                "texture_grain_ms": 22.0,
                "texture_spray": 0.55,
                "reverb_preset": "off",
                "reverb_mix": 0.0,
                "reverb_time": 1.2,
                "denoise_amount": None,
                "transient_attack": None,
                "transient_sustain": None,
                "exciter_amount": None,
                "compressor_attack_ms": None,
                "compressor_release_ms": None,
            }

            state = {
                "post": post,
                "polish": polish,
                "multiband": multiband,
                "mb_low_hz": mb_low_hz,
                "mb_high_hz": mb_high_hz,
                "mb_low_gain_db": mb_low_gain_db,
                "mb_mid_gain_db": mb_mid_gain_db,
                "mb_high_gain_db": mb_high_gain_db,
                "mb_comp_threshold_db": mb_comp_threshold_db,
                "mb_comp_ratio": mb_comp_ratio,
                "creature_size": creature_size,
                "formant_shift": formant_shift,
                "texture_preset": texture_preset,
                "texture_amount": texture_amount,
                "texture_grain_ms": texture_grain_ms,
                "texture_spray": texture_spray,
                "reverb_preset": reverb_preset,
                "reverb_mix": reverb_mix,
                "reverb_time": reverb_time,
                "denoise_amount": denoise_amount,
                "transient_attack": transient_attack,
                "transient_sustain": transient_sustain,
                "exciter_amount": exciter_amount,
                "compressor_attack_ms": compressor_attack_ms,
                "compressor_release_ms": compressor_release_ms,
            }

            if bool(prof.enable_post) and state["post"] == defaults["post"]:
                state["post"] = True
            if bool(prof.enable_polish) and state["polish"] == defaults["polish"]:
                state["polish"] = True

            dest_map = {
                "reverb": "reverb_preset",
            }
            for dest, value in (prof.args or {}).items():
                web_dest = dest_map.get(str(dest), str(dest))
                if web_dest in state and state[web_dest] == defaults.get(web_dest, object()):
                    state[web_dest] = value

            post = bool(state["post"])
            polish = bool(state["polish"])
            multiband = bool(state["multiband"])
            mb_low_hz = float(state["mb_low_hz"])
            mb_high_hz = float(state["mb_high_hz"])
            mb_low_gain_db = float(state["mb_low_gain_db"])
            mb_mid_gain_db = float(state["mb_mid_gain_db"])
            mb_high_gain_db = float(state["mb_high_gain_db"])
            mb_comp_threshold_db = state["mb_comp_threshold_db"]
            mb_comp_ratio = float(state["mb_comp_ratio"])
            creature_size = float(state["creature_size"])
            formant_shift = float(state["formant_shift"])
            texture_preset = str(state["texture_preset"])
            texture_amount = float(state["texture_amount"])
            texture_grain_ms = float(state["texture_grain_ms"])
            texture_spray = float(state["texture_spray"])
            reverb_preset = str(state["reverb_preset"])
            reverb_mix = float(state["reverb_mix"])
            reverb_time = float(state["reverb_time"])
            denoise_amount = state["denoise_amount"]
            transient_attack = state["transient_attack"]
            transient_sustain = state["transient_sustain"]
            exciter_amount = state["exciter_amount"]
            compressor_attack_ms = state["compressor_attack_ms"]
            compressor_release_ms = state["compressor_release_ms"]

    def _pp_params() -> PostProcessParams:
        # Conditioning
        inten = float(np.clip(float(intensity), 0.0, 1.0))
        var = float(np.clip(float(variation), 0.0, 1.0))
        emo = str(emotion or "neutral").strip().lower()

        denoise = 0.0
        trans = 0.0
        comp_thr = None
        comp_makeup = 0.0
        limiter = None

        if polish:
            denoise = 0.25
            trans = 0.25
            comp_thr = -18.0
            comp_makeup = 3.0
            limiter = -1.0

        trans_sus = 0.0
        exc = 0.0

        if inten > 0.0:
            trans = float(np.clip(trans + 0.35 * inten, -1.0, 1.0))
            comp_thr = float(-18.0 - 8.0 * inten)
            comp_makeup = float(comp_makeup + 2.0 * inten)
            denoise = float(np.clip(denoise + 0.10 * inten, 0.0, 1.0))
            if emo == "aggressive":
                trans = float(np.clip(trans + 0.20, -1.0, 1.0))
            elif emo == "calm":
                trans = float(np.clip(trans - 0.15, -1.0, 1.0))
            elif emo == "scared":
                trans = float(np.clip(trans + 0.10, -1.0, 1.0))

        # Optional Pro Mode overrides (win over conditioning/polish defaults).
        if denoise_amount is not None:
            denoise = float(np.clip(float(denoise_amount), 0.0, 1.0))
        if transient_attack is not None:
            trans = float(np.clip(float(transient_attack), -1.0, 1.0))
        if transient_sustain is not None:
            trans_sus = float(np.clip(float(transient_sustain), -1.0, 1.0))
        if exciter_amount is not None:
            exc = float(np.clip(float(exciter_amount), 0.0, 1.0))

        tex_p = str(texture_preset or "off")
        tex_a = float(texture_amount)
        if inten > 0.0 and tex_p == "off" and tex_a <= 0.0 and emo in {"scared", "aggressive"}:
            tex_p = "auto"
            tex_a = float(np.clip(0.18 + 0.25 * inten + 0.10 * var, 0.0, 1.0))

        rev_p = str(reverb_preset or "off")
        rev_m = float(reverb_mix)
        rev_t = float(reverb_time)
        if inten > 0.0 and rev_p == "off" and rev_m <= 0.0 and emo == "calm":
            rev_p = "room"
            rev_m = float(np.clip(0.06 + 0.10 * inten, 0.0, 0.35))

        params = PostProcessParams(
            denoise_strength=float(denoise),
            transient_amount=float(trans),
            transient_sustain=float(trans_sus),
            multiband=bool(multiband or (polish and inten > 0.0)),
            multiband_low_hz=float(mb_low_hz),
            multiband_high_hz=float(mb_high_hz),
            multiband_low_gain_db=float(mb_low_gain_db),
            multiband_mid_gain_db=float(mb_mid_gain_db),
            multiband_high_gain_db=float(mb_high_gain_db),
            multiband_comp_threshold_db=(float(mb_comp_threshold_db) if mb_comp_threshold_db is not None else (-24.0 if (polish and inten > 0.0) else None)),
            multiband_comp_ratio=float(mb_comp_ratio),
            formant_shift=float(formant_shift),
            creature_size=float(creature_size),
            texture_preset=str(tex_p),
            texture_amount=float(tex_a),
            texture_grain_ms=float(texture_grain_ms),
            texture_spray=float(texture_spray),
            reverb_preset=str(rev_p),
            reverb_mix=float(rev_m),
            reverb_time_s=float(rev_t),
            prompt_hint=str(prompt),
            loop_clean=bool(loop_clean),
            loop_crossfade_ms=int(loop_crossfade_ms) if loop_crossfade_ms is not None else 100,
            exciter_amount=float(exc),
        )
        if hints is not None:
            # Apply only the hints we support for post-processing.
            if hints.loudness_rms_db is not None:
                params = replace(params, normalize_rms_db=float(hints.loudness_rms_db))
            if hints.highpass_hz is not None:
                params = replace(params, highpass_hz=float(hints.highpass_hz))
            if hints.lowpass_hz is not None:
                params = replace(params, lowpass_hz=float(hints.lowpass_hz))

        # Polish mode DSP (conservative defaults)
        if polish:
            atk = float(compressor_attack_ms) if compressor_attack_ms is not None else 5.0
            rel = float(compressor_release_ms) if compressor_release_ms is not None else 90.0
            params = replace(
                params,
                compressor_threshold_db=-18.0,
                compressor_ratio=4.0,
                compressor_attack_ms=float(max(0.5, atk)),
                compressor_release_ms=float(max(5.0, rel)),
                compressor_makeup_db=max(float(params.compressor_makeup_db), 3.0),
                limiter_ceiling_db=-1.0,
            )
        return params
    def _qa_info(audio: np.ndarray, sr: int) -> str:
        m = compute_metrics(audio, sr)
        flags: list[str] = []
        if m.clipped:
            flags.append("CLIPPING")
        if detect_long_tail(audio, sr):
            flags.append("LONG_TAIL")
        flag_s = (" " + " ".join(flags)) if flags else ""
        return f"qa: {m.seconds:.2f}s @ {m.sample_rate}Hz peak={m.peak:.3f} rms={m.rms:.3f}{flag_s}".strip()

    def _postprocess_fn(audio: np.ndarray, sr: int) -> tuple[np.ndarray, str]:
        if not (post or polish):
            return audio, _qa_info(audio, sr)
        pp = _pp_params()
        # Tie stochastic DSP to the current variant seed.
        pp = replace(pp, random_seed=int(seed_i) if seed_i is not None else None)
        processed, rep = post_process_audio(audio, sr, pp)
        info = f"post: trimmed={rep.trimmed} {_qa_info(processed, sr)}".strip()
        return processed, info

    def _export_non_minecraft(wav_path: Path, target_path: Path) -> Path:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if fmt == "wav":
            a, sr = read_wav_mono(wav_path)
            sr_out = int(out_sample_rate) if out_sample_rate else sr
            if sr_out != sr:
                # Resample using scipy if available via soundgen.generate path; for web, prefer ffmpeg.
                convert_audio_with_ffmpeg(wav_path, target_path, sample_rate=sr_out, channels=1, out_format="wav")
                # Rewrite subtype using soundfile.
                a2, sr2 = read_wav_mono(target_path)
                write_wav(target_path, a2, sr2, subtype=str(wav_subtype))
            else:
                write_wav(target_path, a, sr, subtype=str(wav_subtype))
            return target_path

        convert_audio_with_ffmpeg(
            wav_path,
            target_path,
            sample_rate=(int(out_sample_rate) if out_sample_rate else None),
            channels=1,
            out_format=fmt,
            ogg_quality=int(ogg_quality),
            mp3_bitrate=str(mp3_bitrate or "192k"),
        )
        return target_path

    def _minecraft_export(wav_path: Path, sp: str) -> tuple[str, str]:
        ogg_path = export_wav_to_minecraft_pack(
            wav_path,
            pack_root=Path(pack_root or "resourcepack"),
            namespace=(namespace or "soundgen"),
            event=(event or "generated.web"),
            sound_path=(sound_path or sp),
            subtitle=(subtitle or None),
            subtitle_key=_maybe(subtitle_key),
            ogg_quality=int(ogg_quality),
            sample_rate=int(mc_sample_rate) if mc_sample_rate else 44100,
            channels=int(mc_channels) if mc_channels else 1,
            weight=max(1, int(weight)),
            volume=float(volume),
            pitch=float(pitch),
            write_pack_mcmeta=(mc_target == "resourcepack"),
        )
        playsound = f"/playsound {(namespace or 'soundgen')}:{(event or 'generated.web')} master @s"
        return str(ogg_path), playsound

    hints = map_prompt_to_controls(prompt) if map_controls else None

    inten = float(np.clip(float(intensity), 0.0, 1.0))
    var = float(np.clip(float(variation), 0.0, 1.0))
    emo = str(emotion or "neutral").strip().lower()
    contour = str(pitch_contour or "flat").strip().lower()

    prompt2 = str(prompt)
    if inten > 0.0 and contour != "flat":
        prompt2 = f"{prompt2} pitch {contour}".strip()

    v = max(1, int(variants)) if export_minecraft else 1
    base_seed = int(seed) if seed is not None else 1337

    last_file: Path | None = None
    last_download: str = ""
    playsound: str = ""
    info: str = ""
    wav_img = None
    spec_img = None

    default_zips = tuple(Path(".examples").joinpath("sound libraies").glob("*.zip"))
    user_zips = _as_zip_paths(samplelib_zips)
    active_zips = user_zips if user_zips else default_zips
    index_path = None if str(library_index or "").strip() == "" else Path(str(library_index))

    def _lerp(a: float, b: float, t: float) -> float:
        tt = float(np.clip(t, 0.0, 1.0))
        return float(a + (b - a) * tt)

    sharp = float(np.clip(float(layered_transient_sharpness), 0.0, 1.0))
    transient_attack_ms = _lerp(3.0, 0.3, sharp)
    transient_decay_ms = _lerp(140.0, 45.0, sharp)
    tail_len_ms = int(np.clip(int(layered_tail_length_ms), 80, 2000))
    tail_decay_ms = max(30.0, float(tail_len_ms) - 40.0)
    synth_attack = float(hints.attack_ms) if hints and hints.attack_ms is not None else float(synth_attack_ms)
    synth_release = float(hints.release_ms) if hints and hints.release_ms is not None else float(synth_release_ms)
    synth_pitch_min = float(hints.pitch_min) if hints and hints.pitch_min is not None else float(synth_pitch_min)
    synth_pitch_max = float(hints.pitch_max) if hints and hints.pitch_max is not None else float(synth_pitch_max)
    synth_lp = float(hints.lowpass_hz) if hints and hints.lowpass_hz is not None else float(synth_lowpass_hz)
    synth_hp = float(hints.highpass_hz) if hints and hints.highpass_hz is not None else float(synth_highpass_hz)
    synth_drive = float(hints.drive) if hints and hints.drive is not None else float(synth_drive)

    # Layered advanced overrides (optional)
    transient_ms_final = int(layered_transient_ms_adv) if layered_transient_ms_adv is not None else 110
    tail_ms_final = int(layered_tail_ms_adv) if layered_tail_ms_adv is not None else int(tail_len_ms)
    transient_attack_final = float(layered_transient_attack_ms_adv) if layered_transient_attack_ms_adv is not None else float(transient_attack_ms)
    transient_hold_final = float(layered_transient_hold_ms_adv) if layered_transient_hold_ms_adv is not None else 10.0
    transient_decay_final = float(layered_transient_decay_ms_adv) if layered_transient_decay_ms_adv is not None else float(transient_decay_ms)

    body_attack_final = float(layered_body_attack_ms_adv) if layered_body_attack_ms_adv is not None else 5.0
    body_hold_final = float(layered_body_hold_ms_adv) if layered_body_hold_ms_adv is not None else 0.0
    body_decay_final = float(layered_body_decay_ms_adv) if layered_body_decay_ms_adv is not None else 80.0

    tail_attack_final = float(layered_tail_attack_ms_adv) if layered_tail_attack_ms_adv is not None else 15.0
    tail_hold_final = float(layered_tail_hold_ms_adv) if layered_tail_hold_ms_adv is not None else 0.0
    tail_decay_final = float(layered_tail_decay_ms_adv) if layered_tail_decay_ms_adv is not None else float(tail_decay_ms)

    duck_release_final = float(layered_duck_release_ms_adv) if layered_duck_release_ms_adv is not None else 90.0

    if inten > 0.0:
        synth_drive = float(np.clip(synth_drive + 0.55 * inten, 0.0, 1.0))
        if emo == "aggressive":
            synth_drive = float(np.clip(synth_drive + 0.20, 0.0, 1.0))
        if emo == "calm":
            synth_lp = float(max(2000.0, synth_lp - 3500.0 * inten))

    v_pitch = 0.08 * var
    synth_pitch_min = float(np.clip(synth_pitch_min - v_pitch, 0.60, 1.20))
    synth_pitch_max = float(np.clip(synth_pitch_max + v_pitch, 0.80, 1.60))
    if inten > 0.0 and contour in {"rise", "updown"}:
        synth_pitch_max = float(np.clip(synth_pitch_max + 0.10, 0.80, 1.80))
    if inten > 0.0 and contour in {"fall", "downup"}:
        synth_pitch_min = float(np.clip(synth_pitch_min - 0.10, 0.50, 1.20))

    for i in range(v):
        suffix = f"_{i+1:02d}" if v > 1 else ""
        wav_path = Path("outputs") / f"web_{engine}{suffix}.wav"
        seed_i = base_seed if (engine == "layered" and layered_family) else (base_seed + i)
        if engine == "layered" and layered_source_lock:
            if layered_source_seed is None:
                source_seed_i = base_seed
            else:
                source_seed_i = int(layered_source_seed)
        else:
            source_seed_i = None

        try:
            generated = generate_wav(
                engine,
                prompt=prompt2,
                seconds=float(seconds),
                seed=seed_i,
                out_wav=wav_path,
                candidates=max(1, int(candidates or 1)),
                postprocess_fn=_postprocess_fn,
                device=device,
                model=model,
                stable_audio_model=str(stable_audio_model or "stabilityai/stable-audio-open-1.0"),
                stable_audio_negative_prompt=(stable_audio_negative_prompt or None),
                stable_audio_hf_token=(stable_audio_hf_token or None),
                stable_audio_steps=int(stable_audio_steps),
                stable_audio_guidance_scale=float(stable_audio_guidance_scale),
                stable_audio_sampler=(None if str(stable_audio_sampler or "auto").strip().lower() in {"", "auto", "default"} else str(stable_audio_sampler)),
                diffusers_multiband=bool(diffusers_multiband),
                diffusers_multiband_mode=str(diffusers_mb_mode or "auto"),
                diffusers_multiband_low_hz=float(diffusers_mb_low_hz),
                diffusers_multiband_high_hz=float(diffusers_mb_high_hz),
                preset=(preset or None),
                rfxgen_path=(Path(rfxgen_path) if rfxgen_path else None),
                replicate_model=_maybe(replicate_model),
                replicate_token=_maybe(replicate_token),
                replicate_input_json=_maybe(replicate_input_json),

                library_zips=active_zips,
                library_mix_count=max(1, int(library_mix_count)),
                library_pitch_min=float(library_pitch_min),
                library_pitch_max=float(library_pitch_max),
                library_index_path=index_path,
                sample_rate=44100,
                synth_waveform=str(synth_waveform),
                synth_freq_hz=float(synth_freq_hz),
                synth_attack_ms=synth_attack,
                synth_decay_ms=float(synth_decay_ms),
                synth_sustain_level=float(synth_sustain),
                synth_release_ms=synth_release,
                synth_noise_mix=float(synth_noise_mix),
                synth_pitch_min=synth_pitch_min,
                synth_pitch_max=synth_pitch_max,
                synth_lowpass_hz=synth_lp,
                synth_highpass_hz=synth_hp,
                synth_drive=synth_drive,

                layered_preset=str(layered_preset or "auto"),
                layered_env_curve_shape=str(layered_curve or "linear"),
                layered_preset_lock=True,
                layered_variant_index=int(i),
                layered_micro_variation=float(layered_micro_variation),
                layered_transient_tilt=float(layered_transient_tilt),
                layered_body_tilt=float(layered_body_tilt),
                layered_tail_tilt=float(layered_tail_tilt),
                layered_source_lock=bool(layered_source_lock),
                layered_source_seed=(int(source_seed_i) if source_seed_i is not None else None),
                layered_granular_preset=str(layered_granular_preset),
                layered_granular_amount=float(layered_granular_amount),
                layered_granular_grain_ms=float(layered_granular_grain_ms),
                layered_granular_spray=float(layered_granular_spray),
                layered_duck_amount=float(layered_duck),
                layered_transient_ms=int(transient_ms_final),
                layered_tail_ms=int(tail_ms_final),
                layered_transient_attack_ms=float(transient_attack_final),
                layered_transient_hold_ms=float(transient_hold_final),
                layered_transient_decay_ms=float(transient_decay_final),
                layered_body_attack_ms=float(body_attack_final),
                layered_body_hold_ms=float(body_hold_final),
                layered_body_decay_ms=float(body_decay_final),
                layered_tail_attack_ms=float(tail_attack_final),
                layered_tail_hold_ms=float(tail_hold_final),
                layered_tail_decay_ms=float(tail_decay_final),
                layered_duck_release_ms=float(duck_release_final),
            )
        except Exception as e:
            msg = str(e).strip() or e.__class__.__name__
            eng = str(engine or "").strip().lower()
            if eng == "stable_audio_open":
                msg = (
                    f"ERROR: {msg}\n\n"
                    "Stable Audio Open is often gated on Hugging Face. "
                    "Accept the model terms on the HF model page and set HUGGINGFACE_HUB_TOKEN "
                    "(or run `huggingface-cli login`)."
                )
            else:
                msg = f"ERROR: {msg}"
            return "", "", msg, None, None
        last_file = Path(generated.wav_path)

        a, sr = read_wav_mono(last_file)
        info = str(generated.post_info) if generated.post_info else _qa_info(a, sr)
        wav_img = waveform_image(a, sr)
        spec_img = spectrogram_image(a, sr)

        sp = (sound_path or f"generated/web{suffix}") if export_minecraft else (sound_path or "generated/web")

        credits = {
            "engine": str(engine),
            "prompt": str(prompt),
            "emotion": str(emotion),
            "intensity": float(intensity),
            "variation": float(variation),
            "pitch_contour": str(pitch_contour),
            "multiband": bool(multiband),
            "creature_size": float(creature_size),
            "formant_shift": float(formant_shift),
            "texture_preset": str(texture_preset),
            "texture_amount": float(texture_amount),
            "reverb_preset": str(reverb_preset),
            "reverb_mix": float(reverb_mix),
            "sound_path": str(sp),
            **{k: v for k, v in generated.credits_extra.items() if v is not None},
        }
        credits["pro_preset"] = str(pro_preset or "off")
        credits["polish_profile"] = str(polish_profile or "off")
        credits["loop_clean"] = bool(loop_clean)
        credits["loop_crossfade_ms"] = int(loop_crossfade_ms) if loop_crossfade_ms is not None else 100
        if generated.sources:
            credits["sources"] = list(generated.sources)

        if export_minecraft:
            last_download, playsound = _minecraft_export(last_file, sp)
            upsert_pack_credits(
                pack_root=Path(pack_root or "resourcepack"),
                namespace=(namespace or "soundgen"),
                event=(event or "generated.web"),
                sound_path=(sound_path or sp),
                credits=credits,
            )
        else:
            # Convert to selected output format for download.
            out_file = Path("outputs") / f"web_{engine}{suffix}.{fmt}"
            written = _export_non_minecraft(last_file, out_file)
            last_download = str(written)
            playsound = ""
            write_sidecar_credits(written, credits)

    return last_download, playsound, info, wav_img, spec_img


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="S-NDB-UND") as demo:
        gr.Markdown(
            "# S-NDB-UND — Prompt → Sound Effect\n"
            "Start with: Engine + Prompt + Seconds. Expand the accordions only if you need more control.\n"
            "Tip: turn on **Pro preset** or a **Polish profile** for quick wins."
        )
        with gr.Accordion("Engine & preset", open=True):
            engine = gr.Radio(["diffusers", "stable_audio_open", "rfxgen", "replicate", "samplelib", "synth", "layered"], value="diffusers", label="Engine")
            with gr.Row():
                device = gr.Dropdown(["cpu", "cuda"], value="cpu", label="Device")
                model = gr.Dropdown(
                    ["cvssp/audioldm2"],
                    value="cvssp/audioldm2",
                    label="Model",
                )

            with gr.Accordion("Stable Audio Open (engine settings)", open=False, visible=False) as stable_audio_acc:
                stable_audio_model = gr.Dropdown(
                    ["stabilityai/stable-audio-open-1.0"],
                    value="stabilityai/stable-audio-open-1.0",
                    label="Stable Audio model",
                )
                stable_audio_negative_prompt = gr.Textbox(value="", label="Negative prompt (optional)")
                stable_audio_hf_token = gr.Textbox(value="", label="HF token (optional; not saved)")
                with gr.Row():
                    stable_audio_steps = gr.Slider(10, 200, value=100, step=1, label="Steps")
                    stable_audio_guidance_scale = gr.Slider(1.0, 12.0, value=7.0, step=0.5, label="Guidance (CFG)")
                stable_audio_sampler = gr.Dropdown(
                    ["auto", "ddim", "deis", "dpmpp", "dpmpp_2m", "euler", "euler_a"],
                    value="auto",
                    label="Sampler (scheduler)",
                )

            with gr.Accordion("Diffusers multi-band (model-side)", open=False, visible=True) as diffusers_mb_acc:
                diffusers_multiband = gr.Checkbox(value=False, label="Enable multi-band diffusers (slower, cleaner bands)")
                diffusers_mb_mode = gr.Dropdown(["auto", "2band", "3band"], value="auto", label="Bands")
                with gr.Row():
                    diffusers_mb_low_hz = gr.Slider(80.0, 800.0, value=250.0, step=10.0, label="Low crossover (Hz)")
                    diffusers_mb_high_hz = gr.Slider(800.0, 8000.0, value=3000.0, step=50.0, label="High crossover (Hz)")

            with gr.Row():
                preset = gr.Dropdown(list(SUPPORTED_PRESETS), value="blip", label="rfxgen preset", visible=False)
                rfxgen_path = gr.Textbox(
                    value="",
                    label="rfxgen path (optional)",
                    placeholder="e.g. tools/rfxgen/rfxgen.exe",
                    visible=False,
                )

            with gr.Accordion("Replicate (paid API engine)", open=False, visible=False) as replicate_acc:
                replicate_model = gr.Textbox(value="", label="Replicate model (optional)", placeholder="e.g. owner/model")
                replicate_token = gr.Textbox(value="", label="Replicate token (optional; not saved)")
                replicate_input_json = gr.Textbox(value="", label="Extra input JSON (optional)", placeholder='{ "cfg": 7.0 }')

            with gr.Accordion("Sample library (engine=samplelib)", open=False, visible=False) as samplelib_acc:
                samplelib_zips = gr.File(
                    file_count="multiple",
                    file_types=[".zip"],
                    label="Library ZIP(s) (optional; defaults to .examples/sound libraies/*.zip)",
                )
                with gr.Row():
                    library_pitch_min = gr.Slider(0.50, 1.20, value=0.85, step=0.01, label="Pitch min")
                    library_pitch_max = gr.Slider(0.80, 2.00, value=1.20, step=0.01, label="Pitch max")
                    library_index = gr.Textbox(value="library/samplelib_index.json", label="Index cache path (blank disables)")

            with gr.Row():
                library_mix_count = gr.Slider(1, 2, value=1, step=1, label="samplelib mix count", visible=False)

            with gr.Accordion("Synth (engine=synth)", open=False, visible=False) as synth_acc:
                with gr.Row():
                    synth_waveform = gr.Dropdown(["sine", "square", "saw", "triangle", "noise"], value="sine", label="Waveform")
                    synth_freq_hz = gr.Slider(40.0, 4000.0, value=440.0, step=1.0, label="Base frequency (Hz)")
                with gr.Row():
                    synth_attack_ms = gr.Slider(0.5, 200.0, value=5.0, step=0.5, label="Attack (ms)")
                    synth_decay_ms = gr.Slider(1.0, 800.0, value=80.0, step=1.0, label="Decay (ms)")
                    synth_sustain = gr.Slider(0.0, 1.0, value=0.35, step=0.01, label="Sustain")
                    synth_release_ms = gr.Slider(1.0, 1200.0, value=120.0, step=1.0, label="Release (ms)")
                with gr.Row():
                    synth_noise_mix = gr.Slider(0.0, 1.0, value=0.05, step=0.01, label="Noise mix")
                    synth_drive = gr.Slider(0.0, 1.0, value=0.0, step=0.01, label="Drive")
                with gr.Row():
                    synth_pitch_min = gr.Slider(0.50, 1.20, value=0.90, step=0.01, label="Pitch min")
                    synth_pitch_max = gr.Slider(0.80, 2.00, value=1.10, step=0.01, label="Pitch max")
                with gr.Row():
                    synth_lowpass_hz = gr.Slider(200.0, 20000.0, value=16000.0, step=50.0, label="Lowpass (Hz)")
                    synth_highpass_hz = gr.Slider(0.0, 2000.0, value=30.0, step=10.0, label="Highpass (Hz)")

            with gr.Column(visible=False) as layered_col:
                with gr.Row():
                    layered_preset = gr.Dropdown(
                        ["auto", "ui", "impact", "whoosh", "creature"],
                        value="auto",
                        label="layered preset",
                    )
                    layered_curve = gr.Dropdown(["linear", "exponential"], value="linear", label="layered curve")
                    layered_duck = gr.Slider(0.0, 1.0, value=0.35, step=0.05, label="layered duck (transient→body)")
                    layered_family = gr.Checkbox(value=True, label="layered family mode")
                    layered_source_lock = gr.Checkbox(value=True, label="layered source lock")
                    layered_source_seed = gr.Number(value=None, precision=0, label="layered source seed (optional)", visible=False)
                    layered_micro_variation = gr.Slider(0.0, 1.0, value=0.25, step=0.05, label="layered micro-variation")
                    layered_granular_preset = gr.Dropdown(
                        ["off", "auto", "chitter", "rasp", "buzz", "screech"],
                        value="off",
                        label="layered granular preset",
                    )
                    layered_granular_amount = gr.Slider(
                        0.0,
                        1.0,
                        value=0.0,
                        step=0.05,
                        label="layered granular amount",
                        visible=False,
                    )
                    layered_granular_grain_ms = gr.Slider(
                        6.0,
                        120.0,
                        value=28.0,
                        step=1.0,
                        label="layered granular grain (ms)",
                        visible=False,
                    )
                    layered_granular_spray = gr.Slider(
                        0.0,
                        1.0,
                        value=0.35,
                        step=0.05,
                        label="layered granular spray",
                        visible=False,
                    )
                    layered_transient_sharpness = gr.Slider(0.0, 1.0, value=0.7, step=0.05, label="layered transient sharpness")
                    layered_tail_length_ms = gr.Slider(80, 1200, value=350, step=10, label="layered tail length (ms)")

                with gr.Accordion("Layered (advanced timing/envelopes)", open=False, visible=False) as layered_adv_acc:
                    gr.Markdown("Leave blank to use the simple knobs / defaults.")
                    with gr.Row():
                        layered_transient_ms_adv = gr.Number(value=None, precision=0, label="transient window (ms)")
                        layered_tail_ms_adv = gr.Number(value=None, precision=0, label="tail window (ms)")
                    with gr.Row():
                        layered_transient_attack_ms_adv = gr.Number(value=None, precision=1, label="transient attack (ms)")
                        layered_transient_hold_ms_adv = gr.Number(value=None, precision=1, label="transient hold (ms)")
                        layered_transient_decay_ms_adv = gr.Number(value=None, precision=1, label="transient decay (ms)")
                    with gr.Row():
                        layered_body_attack_ms_adv = gr.Number(value=None, precision=1, label="body attack (ms)")
                        layered_body_hold_ms_adv = gr.Number(value=None, precision=1, label="body hold (ms)")
                        layered_body_decay_ms_adv = gr.Number(value=None, precision=1, label="body decay (ms)")
                    with gr.Row():
                        layered_tail_attack_ms_adv = gr.Number(value=None, precision=1, label="tail attack (ms)")
                        layered_tail_hold_ms_adv = gr.Number(value=None, precision=1, label="tail hold (ms)")
                        layered_tail_decay_ms_adv = gr.Number(value=None, precision=1, label="tail decay (ms)")
                    layered_duck_release_ms_adv = gr.Number(value=None, precision=1, label="duck release (ms)")

            layered_source_lock.change(
                fn=lambda v: gr.update(visible=bool(v)),
                inputs=[layered_source_lock],
                outputs=[layered_source_seed],
            )

            layered_granular_preset.change(
                fn=lambda v: (
                    gr.update(visible=str(v).strip().lower() != "off"),
                    gr.update(visible=str(v).strip().lower() != "off"),
                    gr.update(visible=str(v).strip().lower() != "off"),
                ),
                inputs=[layered_granular_preset],
                outputs=[layered_granular_amount, layered_granular_grain_ms, layered_granular_spray],
            )

            with gr.Row(visible=False) as layered_tilt_row:
                layered_transient_tilt = gr.Slider(-1.0, 1.0, value=0.0, step=0.05, label="layered transient tilt")
                layered_body_tilt = gr.Slider(-1.0, 1.0, value=0.0, step=0.05, label="layered body tilt")
                layered_tail_tilt = gr.Slider(-1.0, 1.0, value=0.0, step=0.05, label="layered tail tilt")

            def _update_engine_sections(eng: str):
                e = str(eng or "").strip().lower()
                is_diffusers = e == "diffusers"
                is_stable = e == "stable_audio_open"
                is_rfx = e == "rfxgen"
                is_rep = e == "replicate"
                is_sample = e == "samplelib"
                is_synth = e == "synth"
                is_layered = e == "layered"

                return (
                    gr.update(visible=is_diffusers),
                    gr.update(visible=is_stable, open=is_stable),
                    gr.update(visible=is_diffusers),
                    gr.update(visible=is_rfx),
                    gr.update(visible=is_rfx),
                    gr.update(visible=is_rep, open=is_rep),
                    gr.update(visible=(is_sample or is_layered), open=False),
                    gr.update(visible=(is_sample or is_layered)),
                    gr.update(visible=(is_synth or is_layered), open=False),
                    gr.update(visible=is_layered),
                    gr.update(visible=is_layered),
                    gr.update(visible=is_layered, open=False),
                )

            engine.change(
                fn=_update_engine_sections,
                inputs=[engine],
                outputs=[
                    model,
                    stable_audio_acc,
                    diffusers_mb_acc,
                    preset,
                    rfxgen_path,
                    replicate_acc,
                    samplelib_acc,
                    library_mix_count,
                    synth_acc,
                    layered_col,
                    layered_tilt_row,
                    layered_adv_acc,
                ],
            )

        with gr.Accordion("Prompt & duration", open=True):
            with gr.Row():
                prompt = gr.Textbox(label="Prompt", placeholder="e.g. laser zap, sci-fi blaster, short")
            with gr.Row():
                seconds = gr.Slider(0.5, 10.0, value=3.0, step=0.5, label="Seconds")
                seed = gr.Number(value=None, precision=0, label="Seed (optional)")
                candidates = gr.Slider(1, 8, value=1, step=1, label="Candidates (best-of-N)")
            map_controls = gr.Checkbox(value=False, label="Map prompt → control hints")

        gr.Markdown("## Export")
        with gr.Row():
            out_format = gr.Dropdown(["wav", "mp3", "ogg", "flac"], value="wav", label="Output format")
            out_sample_rate = gr.Number(value=None, precision=0, label="Sample rate (optional)")
        with gr.Row():
            wav_subtype = gr.Dropdown(["PCM_16", "PCM_24", "FLOAT"], value="PCM_16", label="WAV subtype")
            mp3_bitrate = gr.Textbox(value="192k", label="MP3 bitrate")

        with gr.Accordion("Pro preset", open=False):
            pro_preset = gr.Dropdown(["off", *pro_preset_keys()], value="off", label="pro preset")
            pro_preset_info = gr.Markdown("", visible=False)
            with gr.Row():
                emotion = gr.Dropdown(["neutral", "aggressive", "calm", "scared"], value="neutral", label="emotion")
                intensity = gr.Slider(0.0, 1.0, value=0.0, step=0.05, label="intensity")
                variation = gr.Slider(0.0, 1.0, value=0.0, step=0.05, label="variation")
                pitch_contour = gr.Dropdown(["flat", "rise", "fall", "updown", "downup"], value="flat", label="pitch contour")

        with gr.Accordion("Advanced / Pro Mode", open=False):
            multiband = gr.Checkbox(value=False, label="multi-band cleanup")
            with gr.Row():
                mb_low_hz = gr.Slider(80.0, 600.0, value=250.0, step=10.0, label="mb low crossover (Hz)")
                mb_high_hz = gr.Slider(1200.0, 8000.0, value=3000.0, step=50.0, label="mb high crossover (Hz)")
            with gr.Row():
                mb_low_gain_db = gr.Slider(-6.0, 6.0, value=0.0, step=0.25, label="mb low gain (dB)")
                mb_mid_gain_db = gr.Slider(-6.0, 6.0, value=0.0, step=0.25, label="mb mid gain (dB)")
                mb_high_gain_db = gr.Slider(-6.0, 6.0, value=0.0, step=0.25, label="mb high gain (dB)")
            with gr.Row():
                mb_comp_threshold_db = gr.Number(value=None, precision=1, label="mb comp threshold (dB) optional")
                mb_comp_ratio = gr.Slider(1.0, 6.0, value=2.0, step=0.25, label="mb comp ratio")

            with gr.Row():
                creature_size = gr.Slider(-1.0, 1.0, value=0.0, step=0.05, label="creature size (-1 small .. +1 large)")
                formant_shift = gr.Slider(0.5, 2.0, value=1.0, step=0.02, label="formant shift (factor)")

            texture_preset = gr.Dropdown(["off", "auto", "chitter", "rasp", "buzz", "screech"], value="off", label="texture preset")
            with gr.Row():
                texture_amount = gr.Slider(0.0, 1.0, value=0.0, step=0.05, label="texture amount", visible=False)
                texture_grain_ms = gr.Slider(4.0, 80.0, value=22.0, step=1.0, label="texture grain (ms)", visible=False)
                texture_spray = gr.Slider(0.0, 1.0, value=0.55, step=0.05, label="texture spray", visible=False)

            texture_preset.change(
                fn=lambda v: (
                    gr.update(visible=str(v).strip().lower() != "off"),
                    gr.update(visible=str(v).strip().lower() != "off"),
                    gr.update(visible=str(v).strip().lower() != "off"),
                ),
                inputs=[texture_preset],
                outputs=[texture_amount, texture_grain_ms, texture_spray],
            )

            reverb_preset = gr.Dropdown(["off", "room", "cave", "forest", "nether"], value="off", label="reverb preset")
            with gr.Row():
                reverb_mix = gr.Slider(0.0, 1.0, value=0.0, step=0.02, label="reverb mix", visible=False)
                reverb_time = gr.Slider(0.1, 6.0, value=1.2, step=0.1, label="reverb time (s)", visible=False)

            reverb_preset.change(
                fn=lambda v: (
                    gr.update(visible=str(v).strip().lower() != "off"),
                    gr.update(visible=str(v).strip().lower() != "off"),
                ),
                inputs=[reverb_preset],
                outputs=[reverb_mix, reverb_time],
            )

            gr.Markdown("### Pro Mode overrides (optional)")
            with gr.Row():
                denoise_amount = gr.Number(value=None, precision=2, label="denoise amount (0..1, blank=auto)")
                exciter_amount = gr.Number(value=None, precision=2, label="exciter amount (0..1, blank=auto)")
            with gr.Row():
                transient_attack = gr.Number(value=None, precision=2, label="transient attack (-1..+1, blank=auto)")
                transient_sustain = gr.Number(value=None, precision=2, label="transient sustain (-1..+1, blank=auto)")
            with gr.Row():
                compressor_attack_ms = gr.Number(value=None, precision=1, label="compressor attack (ms, blank=default)")
                compressor_release_ms = gr.Number(value=None, precision=1, label="compressor release (ms, blank=default)")

        with gr.Accordion("Polish profile", open=False):
            polish_profile = gr.Dropdown(["off", *polish_profile_keys()], value="off", label="polish profile")
            polish_profile_info = gr.Markdown("", visible=False)
            post = gr.Checkbox(value=True, label="Post-process (trim/fade/normalize/EQ)")
            polish = gr.Checkbox(value=False, label="Polish mode (denoise/transients/compress/limit)")
            loop_clean = gr.Checkbox(value=False, label="Loop-clean ambience (100ms seam crossfade)")
            loop_crossfade_ms = gr.Slider(10, 400, value=100, step=10, label="Loop crossfade (ms)")

        def _preset_info_md(k: str) -> gr.Update:
            key = str(k or "off").strip()
            if not key or key.lower() == "off":
                return gr.update(value="", visible=False)
            obj = PRO_PRESETS.get(key)
            if obj is None:
                return gr.update(value="", visible=False)
            rec = pro_preset_recommended_profile(key)
            rec_line = f"\n\nRecommended polish profile: `{rec}`" if rec else ""
            return gr.update(value=f"**{obj.title}** — {obj.description}{rec_line}", visible=True)

        def _profile_info_md(k: str) -> gr.Update:
            key = str(k or "off").strip()
            if not key or key.lower() == "off":
                return gr.update(value="", visible=False)
            obj = POLISH_PROFILES.get(key)
            if obj is None:
                return gr.update(value="", visible=False)
            return gr.update(value=f"**{obj.title}** — {obj.description}", visible=True)

        pro_preset.change(fn=_preset_info_md, inputs=[pro_preset], outputs=[pro_preset_info])
        polish_profile.change(fn=_profile_info_md, inputs=[polish_profile], outputs=[polish_profile_info])

        gr.Markdown("## Minecraft export (1.20.1)")
        export_minecraft = gr.Checkbox(value=False, label="Export to Minecraft (.ogg + sounds.json)")
        with gr.Row():
            mc_target = gr.Dropdown(["resourcepack", "forge"], value="resourcepack", label="Target")
            pack_root = gr.Textbox(value="resourcepack", label="Pack/Resources root")
        with gr.Row():
            namespace = gr.Textbox(value="soundgen", label="Namespace (modid)")
            event = gr.Textbox(value="generated.web", label="Event (sounds.json key)")
        with gr.Row():
            sound_path = gr.Textbox(value="generated/web", label="Sound path (under sounds/, no extension)")
            subtitle = gr.Textbox(value="", label="Subtitle (optional)")
            subtitle_key = gr.Textbox(value="", label="Subtitle key (optional)")
        with gr.Row():
            variants = gr.Slider(1, 10, value=1, step=1, label="Variants")
            weight = gr.Slider(1, 20, value=1, step=1, label="Weight")
            volume = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="Volume")
            pitch = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="Pitch")
            ogg_quality = gr.Slider(0, 10, value=5, step=1, label="OGG quality")

        with gr.Accordion("Minecraft advanced", open=False):
            with gr.Row():
                mc_sample_rate = gr.Dropdown([22050, 32000, 44100, 48000], value=44100, label="Sample rate")
                mc_channels = gr.Dropdown([1, 2], value=1, label="Channels")

        btn = gr.Button("Generate")
        out_file = gr.File(label="Generated file")
        playsound_cmd = gr.Textbox(label="Minecraft playsound", interactive=False)
        info = gr.Textbox(label="QA / post-process", interactive=False)
        with gr.Row():
            wave = gr.Image(label="Waveform", type="pil")
            spec = gr.Image(label="Spectrogram", type="pil")

        btn.click(
            fn=_generate,
            inputs=[
                engine,
                prompt,
                seconds,
                seed,
                candidates,
                device,
                model,
                stable_audio_model,
                stable_audio_negative_prompt,
                stable_audio_hf_token,
                stable_audio_steps,
                stable_audio_guidance_scale,
                stable_audio_sampler,
                diffusers_multiband,
                diffusers_mb_mode,
                diffusers_mb_low_hz,
                diffusers_mb_high_hz,
                preset,
                rfxgen_path,
                library_mix_count,
                synth_waveform,
                layered_preset,
                layered_curve,
                layered_duck,
                layered_family,
                layered_source_lock,
                layered_source_seed,
                layered_micro_variation,
                layered_granular_preset,
                layered_granular_amount,
                layered_granular_grain_ms,
                layered_granular_spray,
                layered_transient_sharpness,
                layered_tail_length_ms,
                layered_transient_tilt,
                layered_body_tilt,
                layered_tail_tilt,
                out_format,
                out_sample_rate,
                wav_subtype,
                mp3_bitrate,
                map_controls,

                pro_preset,
                polish_profile,

                emotion,
                intensity,
                variation,
                pitch_contour,
                multiband,
                mb_low_hz,
                mb_high_hz,
                mb_low_gain_db,
                mb_mid_gain_db,
                mb_high_gain_db,
                mb_comp_threshold_db,
                mb_comp_ratio,
                creature_size,
                formant_shift,
                texture_preset,
                texture_amount,
                texture_grain_ms,
                texture_spray,
                reverb_preset,
                reverb_mix,
                reverb_time,

                denoise_amount,
                transient_attack,
                transient_sustain,
                exciter_amount,
                compressor_attack_ms,
                compressor_release_ms,

                loop_clean,

                export_minecraft,
                mc_target,
                pack_root,
                namespace,
                event,
                sound_path,
                subtitle,
                variants,
                weight,
                volume,
                pitch,
                ogg_quality,
                post,
                polish,

                replicate_model,
                replicate_token,
                replicate_input_json,

                samplelib_zips,
                library_pitch_min,
                library_pitch_max,
                library_index,

                synth_freq_hz,
                synth_attack_ms,
                synth_decay_ms,
                synth_sustain,
                synth_release_ms,
                synth_noise_mix,
                synth_drive,
                synth_pitch_min,
                synth_pitch_max,
                synth_lowpass_hz,
                synth_highpass_hz,

                loop_crossfade_ms,

                subtitle_key,
                mc_sample_rate,
                mc_channels,

                layered_transient_ms_adv,
                layered_tail_ms_adv,
                layered_transient_attack_ms_adv,
                layered_transient_hold_ms_adv,
                layered_transient_decay_ms_adv,
                layered_body_attack_ms_adv,
                layered_body_hold_ms_adv,
                layered_body_decay_ms_adv,
                layered_tail_attack_ms_adv,
                layered_tail_hold_ms_adv,
                layered_tail_decay_ms_adv,
                layered_duck_release_ms_adv,
            ],
            outputs=[out_file, playsound_cmd, info, wave, spec],
        )

    return demo


def main() -> None:
    demo = build_demo()
    demo.launch()


if __name__ == "__main__":
    main()
