from __future__ import annotations

import json
import math
import os
import uuid
import zipfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np

from .engine_registry import available_engines, generate_wav
from .io_utils import convert_audio_with_ffmpeg, read_wav_mono, write_wav
from .postprocess import PostProcessParams, post_process_audio
from .qa import compute_metrics
from .qa_viz import spectrogram_image, waveform_image
from .rfxgen_backend import SUPPORTED_PRESETS


def _rms_dbfs(rms: float) -> float:
    r = float(rms)
    if r <= 0.0:
        return float("-inf")
    return float(20.0 * math.log10(r))


def _fmt_db(x: float) -> str:
    if not math.isfinite(float(x)):
        return "-inf"
    return f"{float(x):.1f}"


def _variant_id(i: int) -> str:
    return f"v{i + 1:02d}"


def _rows_from_variants(variants: list[dict[str, Any]]) -> list[list[object]]:
    rows: list[list[object]] = []
    for v in variants:
        rows.append(
            [
                bool(v.get("select", False)),
                str(v.get("id", "")),
                float(v.get("seconds", 0.0)),
                _fmt_db(float(v.get("rms_dbfs", float("-inf")))),
                int(v.get("seed")) if v.get("seed") is not None else None,
                bool(v.get("locked", False)),
            ]
        )
    return rows


def _variants_from_df(rows: list[list[object]] | None, prev: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return prev
    by_id = {str(v.get("id")): v for v in (prev or []) if isinstance(v, dict) and v.get("id") is not None}
    out: list[dict[str, Any]] = []
    for r in rows:
        if not isinstance(r, list) or len(r) < 6:
            continue
        vid = str(r[1] or "").strip()
        if not vid:
            continue
        base = dict(by_id.get(vid) or {"id": vid})
        base["select"] = bool(r[0])
        base["locked"] = bool(r[5])
        out.append(base)
    return out


def _safe_item_key(item_id: str) -> str:
    t = str(item_id or "").strip().replace("\\", "/")
    t = t.replace("/", "_").replace(".", "_")
    t = "".join((c if (c.isalnum() or c in {"_", "-"}) else "_") for c in t)
    t = "_".join([p for p in t.split("_") if p])
    return t or "item"


def _ui_project_load(project_root: str) -> tuple[dict[str, Any] | None, list[list[object]]]:
    root = Path(str(project_root or "").strip() or ".")
    try:
        from .project import load_project

        proj = load_project(root)
    except Exception as e:
        return {"error": str(e), "root": str(root)}, []

    rows: list[list[object]] = []
    for it in (proj.get("items") or []):
        if not isinstance(it, dict):
            continue
        rows.append(
            [
                it.get("id"),
                it.get("category"),
                it.get("engine"),
                it.get("event"),
                it.get("sound_path"),
                it.get("variants"),
                it.get("active_version"),
            ]
        )

    return proj, rows


def _run_generate_variants(
    engine: str,
    prompt: str,
    seconds: float,
    base_seed: int,
    variant_count: int,
    seed_mode: str,
    device: str,
    model: str,
    rfxgen_preset: str,
    rfxgen_path: str,
    post: bool,
    polish: bool,
) -> list[dict[str, Any]]:
    eng = str(engine or "").strip()
    pr = str(prompt or "").strip()
    sec = float(seconds)
    n = max(1, int(variant_count or 1))

    seed_mode_s = str(seed_mode or "lock").strip().lower()
    if seed_mode_s == "random":
        base_seed = int.from_bytes(os.urandom(4), "big")
    elif seed_mode_s == "step":
        base_seed = int(base_seed) + 1

    out: list[dict[str, Any]] = []
    for i in range(n):
        vid = _variant_id(i)
        seed_i = int(base_seed) + i
        out_wav = Path("outputs") / f"cp_{uuid.uuid4().hex}_{vid}.wav"

        def _pp(audio: np.ndarray, sr: int) -> tuple[np.ndarray, str]:
            if not (bool(post) or bool(polish)):
                return audio, ""
            pp = PostProcessParams(trim_silence=True, fade_ms=8)
            pp = replace(pp, random_seed=seed_i)  # keep deterministic DSP
            y, rep = post_process_audio(audio, sr, pp)
            return y, f"post: trimmed={rep.trimmed}"

        res = generate_wav(
            eng,
            prompt=pr,
            seconds=sec,
            seed=seed_i,
            out_wav=out_wav,
            candidates=1,
            postprocess_fn=_pp,
            device=str(device or "cpu"),
            model=str(model or "cvssp/audioldm2"),
            preset=str(rfxgen_preset or "blip"),
            rfxgen_path=(Path(str(rfxgen_path)) if str(rfxgen_path or "").strip() else None),
        )

        a, sr = read_wav_mono(Path(res.wav_path))
        m = compute_metrics(a, int(sr))
        out.append(
            {
                "id": vid,
                "seed": seed_i,
                "seconds": float(m.seconds),
                "rms_dbfs": _rms_dbfs(float(m.rms)),
                "locked": False,
                "select": False,
                "wav_path": str(res.wav_path),
                "edited_wav_path": None,
            }
        )

    return out


def _ui_generate_variants(
    engine: str,
    prompt: str,
    seconds: float,
    base_seed: int | None,
    seed_mode: str,
    variant_count: int,
    device: str,
    model: str,
    rfxgen_preset: str,
    rfxgen_path: str,
    post: bool,
    polish: bool,
) -> tuple[list[dict[str, Any]], list[list[object]], dict, str, int]:
    try:
        bs = int(base_seed) if base_seed is not None else 1337
        variants = _run_generate_variants(
            engine,
            prompt,
            float(seconds),
            bs,
            int(variant_count),
            str(seed_mode),
            str(device),
            str(model),
            str(rfxgen_preset),
            str(rfxgen_path),
            bool(post),
            bool(polish),
        )
        dd = gr.update(choices=[v["id"] for v in variants], value=(variants[0]["id"] if variants else None))
        next_seed = bs
        if str(seed_mode or "").strip().lower() == "random":
            next_seed = variants[0]["seed"] if variants else bs
        elif str(seed_mode or "").strip().lower() == "step":
            next_seed = bs + 1
        return variants, _rows_from_variants(variants), dd, f"Generated {len(variants)} variant(s).", int(next_seed)
    except Exception as e:
        return [], [], gr.update(choices=[], value=None), f"Generate failed: {e}", int(base_seed or 1337)


def _ui_regen_unlocked(
    variants: list[dict[str, Any]],
    engine: str,
    prompt: str,
    seconds: float,
    base_seed: int | None,
    seed_mode: str,
    device: str,
    model: str,
    rfxgen_preset: str,
    rfxgen_path: str,
    post: bool,
    polish: bool,
) -> tuple[list[dict[str, Any]], list[list[object]], dict, str, int]:
    if not variants:
        return [], [], gr.update(choices=[], value=None), "No variants to regenerate.", int(base_seed or 1337)

    try:
        bs = int(base_seed) if base_seed is not None else 1337
        seed_mode_s = str(seed_mode or "lock").strip().lower()
        if seed_mode_s == "random":
            bs = int.from_bytes(os.urandom(4), "big")
        elif seed_mode_s == "step":
            bs = bs + 1

        out: list[dict[str, Any]] = []
        regen = 0
        for i, v in enumerate(variants):
            if not isinstance(v, dict):
                continue
            if bool(v.get("locked", False)):
                out.append(v)
                continue

            vid = str(v.get("id") or _variant_id(i))
            seed_i = int(bs) + i
            out_wav = Path("outputs") / f"cp_{uuid.uuid4().hex}_{vid}.wav"

            def _pp(audio: np.ndarray, sr: int) -> tuple[np.ndarray, str]:
                if not (bool(post) or bool(polish)):
                    return audio, ""
                pp = PostProcessParams(trim_silence=True, fade_ms=8)
                pp = replace(pp, random_seed=seed_i)
                y, rep = post_process_audio(audio, sr, pp)
                return y, f"post: trimmed={rep.trimmed}"

            res = generate_wav(
                str(engine),
                prompt=str(prompt),
                seconds=float(seconds),
                seed=seed_i,
                out_wav=out_wav,
                candidates=1,
                postprocess_fn=_pp,
                device=str(device or "cpu"),
                model=str(model or "cvssp/audioldm2"),
                preset=str(rfxgen_preset or "blip"),
                rfxgen_path=(Path(str(rfxgen_path)) if str(rfxgen_path or "").strip() else None),
            )

            a, sr = read_wav_mono(Path(res.wav_path))
            m = compute_metrics(a, int(sr))
            v2 = dict(v)
            v2["seed"] = seed_i
            v2["seconds"] = float(m.seconds)
            v2["rms_dbfs"] = _rms_dbfs(float(m.rms))
            v2["wav_path"] = str(res.wav_path)
            v2["edited_wav_path"] = None
            out.append(v2)
            regen += 1

        dd = gr.update(choices=[v["id"] for v in out], value=(out[0]["id"] if out else None))
        return out, _rows_from_variants(out), dd, f"Regenerated {regen} unlocked variant(s).", int(bs)
    except Exception as e:
        return variants, _rows_from_variants(variants), gr.update(), f"Regenerate failed: {e}", int(base_seed or 1337)


def _ui_variant_audio(variants: list[dict[str, Any]], current_variant: str) -> tuple[tuple[int, np.ndarray] | None, object, object, str, str]:
    vid = str(current_variant or "").strip()
    for v in variants or []:
        if isinstance(v, dict) and str(v.get("id")) == vid:
            p = Path(str(v.get("edited_wav_path") or v.get("wav_path") or ""))
            if not p.exists():
                return None, None, None, "Missing WAV.", ""
            a, sr = read_wav_mono(p)
            m = compute_metrics(a, int(sr))
            analysis = f"seconds={m.seconds:.2f} sr={m.sample_rate} peak={m.peak:.3f} rms_dbfs={_fmt_db(_rms_dbfs(float(m.rms)))}"
            return (int(sr), a.astype(np.float32, copy=False)), waveform_image(a, sr), spectrogram_image(a, sr), "", analysis
    return None, None, None, "Select a variant.", "Select a variant."


def _ui_waveform_apply_edits(
    variants: list[dict[str, Any]],
    current_variant: str,
    start_s: float,
    end_s: float,
    fade_in_ms: int,
    fade_out_ms: int,
) -> tuple[list[dict[str, Any]], tuple[int, np.ndarray] | None, object, object, str, str]:
    vid = str(current_variant or "").strip()
    if not vid:
        return variants, None, None, None, "Select a variant first.", ""

    out: list[dict[str, Any]] = []
    for v in variants or []:
        if not isinstance(v, dict) or str(v.get("id")) != vid:
            out.append(v)
            continue

        p = Path(str(v.get("wav_path") or ""))
        if not p.exists():
            out.append(v)
            return out, None, None, None, f"Missing WAV for {vid}.", ""

        a, sr = read_wav_mono(p)
        n = int(a.size)
        ss = max(0.0, float(start_s))
        ee = float(end_s)
        if not math.isfinite(ee) or ee <= 0.0:
            ee = float(n) / float(sr)
        ee = max(ss, min(ee, float(n) / float(sr)))

        i0 = max(0, min(int(ss * sr), n))
        i1 = max(i0, min(int(ee * sr), n))
        y = a[i0:i1].astype(np.float32, copy=True)

        fi = max(0, int(fade_in_ms or 0))
        fo = max(0, int(fade_out_ms or 0))
        if fi > 0:
            fade_len = min(int(sr * (fi / 1000.0)), max(1, y.size // 2))
            if fade_len > 1:
                y[:fade_len] *= np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
        if fo > 0:
            fade_len = min(int(sr * (fo / 1000.0)), max(1, y.size // 2))
            if fade_len > 1:
                y[-fade_len:] *= np.linspace(1.0, 0.0, fade_len, dtype=np.float32)

        out_path = Path("outputs") / f"cp_{uuid.uuid4().hex}_{vid}_edited.wav"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_wav(out_path, y.astype(np.float32, copy=False), int(sr), subtype="PCM_16")

        v2 = dict(v)
        v2["edited_wav_path"] = str(out_path)
        out.append(v2)

        m = compute_metrics(y, int(sr))
        analysis = f"seconds={m.seconds:.2f} sr={m.sample_rate} peak={m.peak:.3f} rms_dbfs={_fmt_db(_rms_dbfs(float(m.rms)))}"
        return out, (int(sr), y), waveform_image(y, int(sr)), spectrogram_image(y, int(sr)), f"Applied edits to {vid}.", analysis

    return variants, None, None, None, "Variant not found.", ""


def _ui_fx_slots_apply(
    variants: list[dict[str, Any]],
    current_variant: str,
    slots_df: list[list[object]] | None,
) -> tuple[tuple[int, np.ndarray] | None, str]:
    from .fx_chain_v2 import FxChainV2, FxStepV2, apply_fx_chain_v2

    vid = str(current_variant or "").strip()
    if not vid:
        return None, "Select a variant first."

    wav_path = None
    for v in variants or []:
        if isinstance(v, dict) and str(v.get("id")) == vid:
            wav_path = v.get("edited_wav_path") or v.get("wav_path")
            break
    if not wav_path:
        return None, "Missing WAV for selected variant."

    p = Path(str(wav_path))
    if not p.exists() or p.suffix.lower() != ".wav":
        return None, "FX apply requires a .wav input."

    steps: list[FxStepV2] = []
    for r in (slots_df or []):
        if not isinstance(r, list) or len(r) < 3:
            continue
        enabled = bool(r[2])
        mid = str(r[1] or "").strip().lower()
        if not enabled or not mid:
            continue
        params_s = str(r[3] or "{}").strip() if len(r) >= 4 else "{}"
        try:
            params = json.loads(params_s) if params_s else {}
            if not isinstance(params, dict):
                params = {}
        except Exception:
            params = {}
        steps.append(FxStepV2(module_id=mid, params=dict(params)))

    chain = FxChainV2(name="control_panel", steps=tuple(steps), description=None)
    a, sr = read_wav_mono(p)
    y = apply_fx_chain_v2(a, int(sr), chain)
    return (int(sr), y.astype(np.float32, copy=False)), "Applied FX slots."


def _ui_export_bundle(
    variants: list[dict[str, Any]],
    mode: str,
    out_format: str,
    out_sample_rate: int | None,
    wav_subtype: str,
    mp3_bitrate: str,
    ogg_quality: int,
    filename_template: str,
) -> tuple[str, str]:
    want = str(mode or "selected").strip().lower()
    fmt = str(out_format or "wav").strip().lower()
    if fmt not in {"wav", "mp3", "ogg", "flac"}:
        fmt = "wav"

    picks: list[dict[str, Any]] = []
    for v in variants or []:
        if not isinstance(v, dict):
            continue
        if want == "locked" and not bool(v.get("locked", False)):
            continue
        if want == "selected" and not bool(v.get("select", False)):
            continue
        picks.append(v)

    if not picks:
        return "", "No variants matched export mode."

    out_dir = Path("outputs") / "control_panel_exports"
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = Path("outputs") / "control_panel_export_bundle.zip"

    templ = str(filename_template or "{variant}_{seed}").strip() or "{variant}_{seed}"

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for v in picks:
            vid = str(v.get("id") or "variant")
            seed = v.get("seed")
            src = Path(str(v.get("edited_wav_path") or v.get("wav_path") or ""))
            if not src.exists():
                continue

            name = templ
            name = name.replace("{variant}", vid)
            name = name.replace("{seed}", str(seed if seed is not None else ""))
            name = name.replace("{project}", "")
            name = name.replace("{preset}", "")
            name = "_".join([p for p in name.replace("/", "_").replace("\\", "_").split("_") if p])
            if not name:
                name = vid

            wav_in = src
            if src.suffix.lower() != ".wav":
                tmp = out_dir / f"_{vid}_tmp.wav"
                convert_audio_with_ffmpeg(src, tmp, sample_rate=None, channels=1, out_format="wav")
                wav_in = tmp

            out_file = out_dir / f"{name}.{fmt}"
            if fmt == "wav":
                a, sr = read_wav_mono(wav_in)
                sr_out = int(out_sample_rate) if out_sample_rate else int(sr)
                if sr_out != int(sr):
                    convert_audio_with_ffmpeg(wav_in, out_file, sample_rate=sr_out, channels=1, out_format="wav")
                    a2, sr2 = read_wav_mono(out_file)
                    write_wav(out_file, a2.astype(np.float32, copy=False), int(sr2), subtype=str(wav_subtype))
                else:
                    write_wav(out_file, a.astype(np.float32, copy=False), int(sr), subtype=str(wav_subtype))
            else:
                convert_audio_with_ffmpeg(
                    wav_in,
                    out_file,
                    sample_rate=(int(out_sample_rate) if out_sample_rate else None),
                    channels=1,
                    out_format=fmt,
                    ogg_quality=int(ogg_quality),
                    mp3_bitrate=str(mp3_bitrate or "192k"),
                )

            z.write(out_file, arcname=out_file.name)

    return str(zip_path), f"Exported {len(picks)} variant(s) to bundle."


def build_demo_control_panel() -> gr.Blocks:
    css = """
    .gradio-container { background: #0b0d10; }
    .gradio-container * { color-scheme: dark; }
    """

    with gr.Blocks(title="S-NDB-UND", css=css) as demo:
        gr.Markdown(
            "# S-NDB-UND — Control Panel UI (Gradio-native)\n"
            "Discrete actions, explicit state, zero continuous interaction."
        )

        variants_state = gr.State([])

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## Project Browser")
                pr_root = gr.Textbox(value=".", label="Project root")
                pr_load_btn = gr.Button("Load project")
                pr_json = gr.JSON(label="Project")
                pr_items = gr.Dataframe(
                    headers=["id", "category", "engine", "event", "sound_path", "variants", "active_version"],
                    datatype=["str", "str", "str", "str", "str", "number", "number"],
                    row_count=(0, "dynamic"),
                    col_count=(7, "fixed"),
                    interactive=False,
                    label="Items",
                )
                pr_status = gr.Textbox(label="Status", interactive=False)
                pr_load_btn.click(fn=_ui_project_load, inputs=[pr_root], outputs=[pr_json, pr_items])

            with gr.Column(scale=1):
                gr.Markdown("## Preset Browser")
                pb_search = gr.Textbox(value="", label="Search")
                pb_keys = gr.Radio(choices=list(SUPPORTED_PRESETS), value="blip", label="rfxgen presets")
                with gr.Row():
                    pb_preview_btn = gr.Button("Preview")
                    pb_load_btn = gr.Button("Load into generator")
                pb_preview_audio = gr.Audio(label="Preview", type="numpy")
                pb_status = gr.Textbox(label="Status", interactive=False)

                def _pb_update(q: str):
                    query = str(q or "").strip().lower()
                    keys = list(SUPPORTED_PRESETS)
                    if query:
                        keys = [k for k in keys if query in str(k).lower()]
                    return gr.update(choices=keys[:250], value=(keys[0] if keys else None))

                pb_search.change(fn=_pb_update, inputs=[pb_search], outputs=[pb_keys])

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## Generator Controls")
                engine = gr.Dropdown(available_engines(), value="diffusers", label="Engine")
                prompt = gr.Textbox(label="Prompt", placeholder="e.g. coin pickup")
                seconds = gr.Slider(0.5, 10.0, value=3.0, step=0.5, label="Seconds")
                with gr.Row():
                    seed = gr.Number(value=1337, precision=0, label="Base seed")
                    seed_mode = gr.Dropdown(["lock", "random", "step"], value="lock", label="Seed mode")
                variant_count = gr.Slider(1, 16, value=6, step=1, label="Variant count")
                randomness = gr.Slider(0.0, 1.0, value=0.0, step=0.05, label="Randomness (reserved)")
                with gr.Row():
                    post = gr.Checkbox(value=True, label="Post-process")
                    polish = gr.Checkbox(value=False, label="Polish mode")

                with gr.Row():
                    device = gr.Dropdown(["cpu", "cuda"], value="cpu", label="Device")
                    model = gr.Dropdown(["cvssp/audioldm2"], value="cvssp/audioldm2", label="Model")

                with gr.Row():
                    rfxgen_preset = gr.Dropdown(list(SUPPORTED_PRESETS), value="blip", label="rfxgen preset")
                    rfxgen_path = gr.Textbox(value="", label="rfxgen path (optional)")

                with gr.Row():
                    generate_btn = gr.Button("Generate variants")
                    regen_btn = gr.Button("Regenerate unlocked")
                gen_status = gr.Textbox(label="Status", interactive=False)

            with gr.Column(scale=1):
                gr.Markdown("## Variant Table")
                variants_df = gr.Dataframe(
                    headers=["select", "id", "seconds", "rms_dbfs", "seed", "locked"],
                    datatype=["bool", "str", "number", "str", "number", "bool"],
                    row_count=(0, "dynamic"),
                    col_count=(6, "fixed"),
                    interactive=True,
                    label="Variants",
                )
                current_variant = gr.Dropdown([], value=None, label="Current variant")
                load_btn = gr.Button("Load into viewer")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## Waveform Viewer")
                viewer_audio = gr.Audio(label="Playback", type="numpy")
                with gr.Row():
                    viewer_wave = gr.Image(label="Waveform", type="pil")
                    viewer_spec = gr.Image(label="Spectrogram", type="pil")
                with gr.Row():
                    start_s = gr.Number(value=0.0, precision=3, label="Start (s)")
                    end_s = gr.Number(value=0.0, precision=3, label="End (s, 0=full)")
                with gr.Row():
                    fade_in_ms = gr.Slider(0, 250, value=0, step=5, label="Fade in (ms)")
                    fade_out_ms = gr.Slider(0, 250, value=0, step=5, label="Fade out (ms)")
                apply_edits_btn = gr.Button("Apply edits (non-destructive)")
                viewer_status = gr.Textbox(label="Status", interactive=False)

            with gr.Column(scale=1):
                gr.Markdown("## FX Chain Controls (slot-based)")
                fx_slots = gr.Dataframe(
                    headers=["slot", "fx_type", "enabled", "params_json"],
                    datatype=["number", "str", "bool", "str"],
                    row_count=(6, "fixed"),
                    col_count=(4, "fixed"),
                    interactive=True,
                    value=[[i + 1, "", True, "{}"] for i in range(6)],
                    label="FX slots",
                )
                fx_apply_btn = gr.Button("Apply FX to current (audition)")
                fx_audio_out = gr.Audio(label="FX result", type="numpy")
                fx_status = gr.Textbox(label="Status", interactive=False)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## Export Settings")
                with gr.Row():
                    ex_fmt = gr.Dropdown(["wav", "mp3", "ogg", "flac"], value="wav", label="Format")
                    ex_sr = gr.Number(value=None, precision=0, label="Sample rate (optional)")
                with gr.Row():
                    ex_wav_subtype = gr.Dropdown(["PCM_16", "PCM_24", "FLOAT"], value="PCM_16", label="WAV subtype")
                    ex_mp3_bitrate = gr.Textbox(value="192k", label="MP3 bitrate")
                    ex_ogg_quality = gr.Slider(0, 10, value=5, step=1, label="OGG quality")
                ex_mode = gr.Dropdown(["selected", "locked"], value="selected", label="Export mode")
                filename_template = gr.Textbox(value="{variant}_{seed}", label="Filename template")

            with gr.Column(scale=1):
                gr.Markdown("## Export + Analysis")
                export_btn = gr.Button("Export bundle (.zip)")
                export_out = gr.File(label="Export bundle")
                export_status = gr.Textbox(label="Status", interactive=False)
                with gr.Tabs():
                    with gr.Tab("Spectrum"):
                        analysis_spec = gr.Image(label="Spectrogram", type="pil")
                    with gr.Tab("Loudness"):
                        analysis_txt = gr.Textbox(label="RMS dBFS / Peak", interactive=False)

        generate_btn.click(
            fn=_ui_generate_variants,
            inputs=[
                engine,
                prompt,
                seconds,
                seed,
                seed_mode,
                variant_count,
                device,
                model,
                rfxgen_preset,
                rfxgen_path,
                post,
                polish,
            ],
            outputs=[variants_state, variants_df, current_variant, gen_status, seed],
        )

        regen_btn.click(
            fn=_ui_regen_unlocked,
            inputs=[
                variants_state,
                engine,
                prompt,
                seconds,
                seed,
                seed_mode,
                device,
                model,
                rfxgen_preset,
                rfxgen_path,
                post,
                polish,
            ],
            outputs=[variants_state, variants_df, current_variant, gen_status, seed],
        )

        variants_df.change(fn=_variants_from_df, inputs=[variants_df, variants_state], outputs=[variants_state])

        load_btn.click(
            fn=_ui_variant_audio,
            inputs=[variants_state, current_variant],
            outputs=[viewer_audio, viewer_wave, viewer_spec, viewer_status, analysis_txt],
        )
        load_btn.click(
            fn=lambda v, c: _ui_variant_audio(v, c)[2],
            inputs=[variants_state, current_variant],
            outputs=[analysis_spec],
        )

        apply_edits_btn.click(
            fn=_ui_waveform_apply_edits,
            inputs=[variants_state, current_variant, start_s, end_s, fade_in_ms, fade_out_ms],
            outputs=[variants_state, viewer_audio, viewer_wave, viewer_spec, viewer_status, analysis_txt],
        )

        fx_apply_btn.click(fn=_ui_fx_slots_apply, inputs=[variants_state, current_variant, fx_slots], outputs=[fx_audio_out, fx_status])

        export_btn.click(
            fn=_ui_export_bundle,
            inputs=[variants_state, ex_mode, ex_fmt, ex_sr, ex_wav_subtype, ex_mp3_bitrate, ex_ogg_quality, filename_template],
            outputs=[export_out, export_status],
        )

        # Preset browser wiring: load & preview rfxgen presets.
        pb_load_btn.click(fn=lambda k: gr.update(value=str(k)), inputs=[pb_keys], outputs=[rfxgen_preset])

        def _pb_preview(
            preset_key: str,
            engine_v: str,
            prompt_v: str,
            seconds_v: float,
            base_seed: int,
            device_v: str,
            model_v: str,
            rfx_path_v: str,
            post_v: bool,
            polish_v: bool,
        ) -> tuple[tuple[int, np.ndarray] | None, str]:
            try:
                variants = _run_generate_variants(
                    engine=str(engine_v),
                    prompt=str(prompt_v),
                    seconds=float(seconds_v),
                    base_seed=int(base_seed),
                    variant_count=1,
                    seed_mode="lock",
                    device=str(device_v),
                    model=str(model_v),
                    rfxgen_preset=str(preset_key),
                    rfxgen_path=str(rfx_path_v),
                    post=bool(post_v),
                    polish=bool(polish_v),
                )
                if not variants:
                    return None, "Preview produced no output."
                p = Path(str(variants[0].get("wav_path") or ""))
                if not p.exists():
                    return None, "Preview WAV missing."
                a, sr = read_wav_mono(p)
                m = compute_metrics(a, int(sr))
                return (int(sr), a.astype(np.float32, copy=False)), f"Preview ok: seconds={m.seconds:.2f} rms_dbfs={_fmt_db(_rms_dbfs(float(m.rms)))}"
            except Exception as e:
                return None, f"Preview failed: {e}"

        pb_preview_btn.click(
            fn=_pb_preview,
            inputs=[pb_keys, engine, prompt, seconds, seed, device, model, rfxgen_path, post, polish],
            outputs=[pb_preview_audio, pb_status],
        )

    return demo
