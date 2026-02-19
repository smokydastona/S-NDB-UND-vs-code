from __future__ import annotations

import base64
import json
import math
import os
import subprocess
import sys
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
from .ui_models import Variant, normalize_variants_state


AudioCache = dict[str, tuple[int, np.ndarray]]


def _pct_status(pct: int, msg: str) -> str:
    p = max(0, min(100, int(pct)))
    m = str(msg or "").strip()
    if not m:
        return f"{p}%"
    return f"{p}% - {m}"


def _ui_icon_data_uri() -> str | None:
    """Return a data: URI for the UI icon, if available.

    Used only for optional UI styling (background watermark).
    """

    candidates: list[Path] = []
    # Dev / repo layout.
    candidates.append(Path(".examples") / "icon.png")
    try:
        candidates.append(Path(__file__).resolve().parents[2] / ".examples" / "icon.png")
    except Exception:
        pass

    # PyInstaller / packaged layouts.
    try:
        exe_dir = Path(sys.executable).resolve().parent
        candidates.append(exe_dir / ".examples" / "icon.png")
        candidates.append(exe_dir / "icon.png")
    except Exception:
        pass

    for p in candidates:
        try:
            if p.exists() and p.is_file():
                b = p.read_bytes()
                enc = base64.b64encode(b).decode("ascii")
                return f"data:image/png;base64,{enc}"
        except Exception:
            continue
    return None


def _ui_background_data_uri() -> str | None:
    """Return a data: URI for the UI background image, if available."""

    candidates: list[Path] = []
    # Dev / repo layout.
    candidates.append(Path(".examples") / "background.png")
    try:
        candidates.append(Path(__file__).resolve().parents[2] / ".examples" / "background.png")
    except Exception:
        pass

    # PyInstaller / packaged layouts.
    try:
        exe_dir = Path(sys.executable).resolve().parent
        candidates.append(exe_dir / ".examples" / "background.png")
        candidates.append(exe_dir / "background.png")
    except Exception:
        pass

    for p in candidates:
        try:
            if p.exists() and p.is_file():
                b = p.read_bytes()
                enc = base64.b64encode(b).decode("ascii")
                return f"data:image/png;base64,{enc}"
        except Exception:
            continue
    return None


def _keep_control_panel_wavs() -> bool:
    v = (
        os.environ.get("SOUNDGEN_CONTROL_PANEL_KEEP_WAVS")
        or os.environ.get("SOUNDGEN_CP_KEEP_WAVS")
        or ""
    )
    return str(v).strip().lower() in {"1", "true", "yes", "on"}


def _cp_temp_wav_path(variant_id: str) -> Path:
    d = Path("outputs") / "_tmp"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"cp_{uuid.uuid4().hex}_{str(variant_id)}.wav"


def _best_effort_unlink(p: Path) -> None:
    try:
        if p.exists() and p.is_file():
            p.unlink()
    except Exception:
        pass


def _cache_key(variant_id: str, kind: str) -> str:
    return f"{str(variant_id)}::{str(kind)}"


def _variant_active_audio_key(v: dict[str, Any]) -> str | None:
    k = str(v.get("edited_audio_key") or "").strip()
    if k:
        return k
    k = str(v.get("audio_key") or "").strip()
    return k or None


def _audio_from_variant(v: dict[str, Any], cache: AudioCache) -> tuple[int, np.ndarray] | None:
    k = _variant_active_audio_key(v)
    if k and k in cache:
        return cache[k]

    # Fallback for older state: load from disk.
    p = Path(str(v.get("edited_wav_path") or v.get("wav_path") or ""))
    if p.exists() and p.suffix.lower() == ".wav":
        a, sr = read_wav_mono(p)
        return int(sr), a.astype(np.float32, copy=False)
    return None


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
    for v in normalize_variants_state(variants):
        vv = Variant.from_dict(v)
        rows.append(
            [
                bool(vv.select),
                str(vv.id),
                float(vv.seconds),
                _fmt_db(float(vv.rms_dbfs)),
                int(vv.seed) if vv.seed is not None else None,
                bool(vv.locked),
            ]
        )
    return rows


def _variants_from_df(rows: list[list[object]] | None, prev: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return prev
    prev_norm = normalize_variants_state(prev)
    by_id = {str(v.get("id")): v for v in prev_norm if isinstance(v, dict) and v.get("id") is not None}
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
    return normalize_variants_state(out)


def _safe_item_key(item_id: str) -> str:
    t = str(item_id or "").strip().replace("\\", "/")
    t = t.replace("/", "_").replace(".", "_")
    t = "".join((c if (c.isalnum() or c in {"_", "-"}) else "_") for c in t)
    t = "_".join([p for p in t.split("_") if p])
    return t or "item"


def _ui_project_rows(proj: dict[str, Any]) -> tuple[list[list[object]], list[str]]:
    rows: list[list[object]] = []
    ids: list[str] = []
    for it in (proj.get("items") or []):
        if not isinstance(it, dict):
            continue
        iid = str(it.get("id") or "").strip()
        if iid:
            ids.append(iid)
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
    return rows, ids


def _run_subprocess(argv: list[str]) -> str:
    completed = subprocess.run(argv, capture_output=True, text=True)
    out = (completed.stdout or "") + ("\n" if completed.stdout and completed.stderr else "") + (completed.stderr or "")
    out = out.strip()
    if completed.returncode != 0:
        return (out + f"\n(exit {completed.returncode})").strip()
    return out or "(ok)"


def _ui_project_load(project_root: str):
    root = Path(str(project_root or "").strip() or ".")
    try:
        from .project import load_project

        proj = load_project(root)
    except Exception as e:
        return {"error": str(e), "root": str(root)}, [], gr.update(choices=[], value=None), f"Load failed: {e}"

    rows, ids = _ui_project_rows(proj)
    dd = gr.update(choices=ids, value=(ids[0] if ids else None))
    return proj, rows, dd, f"Loaded project with {len(ids)} item(s)."


def _ui_project_rename(project_root: str, old_id: str, new_id: str):
    root = Path(str(project_root or "").strip() or ".")
    oid = str(old_id or "").strip()
    nid = str(new_id or "").strip()
    if not oid or not nid:
        proj, rows, dd, _ = _ui_project_load(str(root))
        return proj, rows, dd, "Set both old id and new id."

    try:
        from .project import load_project, save_project

        proj = load_project(root)
        items = proj.get("items")
        if not isinstance(items, list):
            items = []
            proj["items"] = items

        if any(isinstance(it, dict) and str(it.get("id") or "").strip() == nid for it in items):
            rows, ids = _ui_project_rows(proj)
            return proj, rows, gr.update(choices=ids, value=(nid if nid in ids else (ids[0] if ids else None))), (
                f"Rename failed: id already exists: {nid}"
            )

        found = False
        for it in items:
            if isinstance(it, dict) and str(it.get("id") or "").strip() == oid:
                it["id"] = nid
                found = True
                break
        if not found:
            rows, ids = _ui_project_rows(proj)
            return proj, rows, gr.update(choices=ids, value=(ids[0] if ids else None)), f"Rename failed: unknown id: {oid}"

        save_project(root, proj)
        rows, ids = _ui_project_rows(proj)
        return proj, rows, gr.update(choices=ids, value=(nid if nid in ids else (ids[0] if ids else None))), (
            f"Renamed item: {oid} -> {nid} (existing versions keep their original file paths)."
        )
    except Exception as e:
        return {"error": str(e), "root": str(root)}, [], gr.update(choices=[], value=None), f"Rename failed: {e}"


def _ui_project_duplicate(project_root: str, src_id: str, new_id: str):
    root = Path(str(project_root or "").strip() or ".")
    sid = str(src_id or "").strip()
    nid = str(new_id or "").strip()
    if not sid:
        proj, rows, dd, _ = _ui_project_load(str(root))
        return proj, rows, dd, "Pick an item id to duplicate."
    if not nid:
        nid = f"{sid}_copy"

    try:
        from .project import load_project, save_project

        proj = load_project(root)
        items = proj.get("items")
        if not isinstance(items, list):
            items = []
            proj["items"] = items

        if any(isinstance(it, dict) and str(it.get("id") or "").strip() == nid for it in items):
            rows, ids = _ui_project_rows(proj)
            return proj, rows, gr.update(choices=ids, value=(nid if nid in ids else (ids[0] if ids else None))), (
                f"Duplicate failed: id already exists: {nid}"
            )

        src_item: dict[str, Any] | None = None
        for it in items:
            if isinstance(it, dict) and str(it.get("id") or "").strip() == sid:
                src_item = it
                break
        if src_item is None:
            rows, ids = _ui_project_rows(proj)
            return proj, rows, gr.update(choices=ids, value=(ids[0] if ids else None)), f"Duplicate failed: unknown id: {sid}"

        new_item = json.loads(json.dumps(src_item))
        if not isinstance(new_item, dict):
            raise ValueError("Duplicate failed: item is not a dict")

        new_item["id"] = nid
        new_item["active_version"] = None
        new_item["versions"] = []
        items.append(new_item)
        save_project(root, proj)

        rows, ids = _ui_project_rows(proj)
        return proj, rows, gr.update(choices=ids, value=(nid if nid in ids else (ids[0] if ids else None))), (
            f"Duplicated item: {sid} -> {nid}. Note: event/sound_path were copied; update them to avoid pack collisions."
        )
    except Exception as e:
        return {"error": str(e), "root": str(root)}, [], gr.update(choices=[], value=None), f"Duplicate failed: {e}"


def _ui_project_delete(project_root: str, item_id: str, confirm: bool):
    root = Path(str(project_root or "").strip() or ".")
    iid = str(item_id or "").strip()
    if not iid:
        proj, rows, dd, _ = _ui_project_load(str(root))
        return proj, rows, dd, "Pick an item id to delete."
    if not bool(confirm):
        proj, rows, dd, _ = _ui_project_load(str(root))
        return proj, rows, dd, "Delete not confirmed. Check 'Confirm delete' first."

    try:
        from .project import load_project, save_project

        proj = load_project(root)
        items = proj.get("items")
        if not isinstance(items, list):
            items = []
            proj["items"] = items

        before = len(items)
        proj["items"] = [it for it in items if not (isinstance(it, dict) and str(it.get("id") or "").strip() == iid)]
        after = len(proj["items"])
        if after == before:
            rows, ids = _ui_project_rows(proj)
            return proj, rows, gr.update(choices=ids, value=(ids[0] if ids else None)), f"Delete failed: unknown id: {iid}"

        save_project(root, proj)
        rows, ids = _ui_project_rows(proj)
        return proj, rows, gr.update(choices=ids, value=(ids[0] if ids else None)), (
            f"Deleted item: {iid}. Note: existing files under project_audio/ are not removed automatically."
        )
    except Exception as e:
        return {"error": str(e), "root": str(root)}, [], gr.update(choices=[], value=None), f"Delete failed: {e}"


def _ui_project_build_and_reload(project_root: str, item_id: str):
    root = str(project_root or "").strip()
    if not root:
        return None, [], gr.update(choices=[], value=None), "Set Project root first."

    args = [sys.executable, "-m", "soundgen.app", "project", "build", "--root", root]
    if str(item_id or "").strip():
        args += ["--id", str(item_id).strip()]
    out = _run_subprocess(args)

    # Reload project to show updated active_version / versions.
    proj, rows, dd, _ = _ui_project_load(root)
    return proj, rows, dd, out


def _ui_project_edit(project_root: str, item_id: str) -> str:
    root = str(project_root or "").strip()
    iid = str(item_id or "").strip()
    if not root or not iid:
        return "Set Project root and Item id first."

    try:
        subprocess.Popen([sys.executable, "-m", "soundgen.app", "project", "edit", "--root", root, "--id", iid])
        return f"Launched editor for item: {iid}"
    except Exception as e:
        return f"Failed to launch: {e}"


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
) -> tuple[list[dict[str, Any]], AudioCache]:
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
    cache: AudioCache = {}
    for i in range(n):
        vid = _variant_id(i)
        seed_i = int(base_seed) + i
        out_wav = _cp_temp_wav_path(vid)

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

        wav_p = Path(res.wav_path)
        a, sr = read_wav_mono(wav_p)
        base_key = _cache_key(vid, "base")
        cache[base_key] = (int(sr), a.astype(np.float32, copy=False))

        if not _keep_control_panel_wavs():
            _best_effort_unlink(out_wav)
            if wav_p.resolve() != out_wav.resolve():
                _best_effort_unlink(wav_p)
        m = compute_metrics(a, int(sr))
        out.append(
            Variant(
                id=vid,
                seed=seed_i,
                seconds=float(m.seconds),
                rms_dbfs=_rms_dbfs(float(m.rms)),
                locked=False,
                select=False,
                audio_key=base_key,
                edited_audio_key=None,
                wav_path=(str(wav_p) if _keep_control_panel_wavs() else ""),
                edited_wav_path=None,
                meta={},
            ).to_dict()
        )

    return normalize_variants_state(out), cache


def _iter_generate_variants(
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
):
    """Yield (variants_so_far, cache_so_far, done, total)."""

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
    cache: AudioCache = {}
    for i in range(n):
        vid = _variant_id(i)
        seed_i = int(base_seed) + i
        out_wav = _cp_temp_wav_path(vid)

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

        wav_p = Path(res.wav_path)
        a, sr = read_wav_mono(wav_p)
        base_key = _cache_key(vid, "base")
        cache[base_key] = (int(sr), a.astype(np.float32, copy=False))

        if not _keep_control_panel_wavs():
            _best_effort_unlink(out_wav)
            if wav_p.resolve() != out_wav.resolve():
                _best_effort_unlink(wav_p)

        m = compute_metrics(a, int(sr))
        out.append(
            Variant(
                id=vid,
                seed=seed_i,
                seconds=float(m.seconds),
                rms_dbfs=_rms_dbfs(float(m.rms)),
                locked=False,
                select=False,
                audio_key=base_key,
                edited_audio_key=None,
                wav_path=(str(wav_p) if _keep_control_panel_wavs() else ""),
                edited_wav_path=None,
                meta={},
            ).to_dict()
        )

        yield normalize_variants_state(out), cache, i + 1, n


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
) -> tuple[list[dict[str, Any]], AudioCache, list[list[object]], dict, str, int]:
    # Stream progress so the UI shows explicit percentages while processing.
    try:
        bs = int(base_seed) if base_seed is not None else 1337
        seed_mode_s = str(seed_mode or "").strip().lower()
        yield [], {}, [], gr.update(choices=[], value=None), _pct_status(0, "starting"), int(bs)

        variants: list[dict[str, Any]] = []
        cache: AudioCache = {}
        for v_so_far, c_so_far, done, total in _iter_generate_variants(
            engine=str(engine),
            prompt=str(prompt),
            seconds=float(seconds),
            base_seed=bs,
            variant_count=int(variant_count),
            seed_mode=str(seed_mode),
            device=str(device),
            model=str(model),
            rfxgen_preset=str(rfxgen_preset),
            rfxgen_path=str(rfxgen_path),
            post=bool(post),
            polish=bool(polish),
        ):
            variants = v_so_far
            cache = c_so_far
            pct = int(round((done / max(1, total)) * 95.0))
            dd = gr.update(choices=[vv["id"] for vv in variants], value=(variants[0]["id"] if variants else None))
            yield variants, cache, _rows_from_variants(variants), dd, _pct_status(pct, f"generating {done}/{total}"), int(bs)

        next_seed = bs
        if seed_mode_s == "random":
            next_seed = int(variants[0]["seed"]) if variants else bs
        elif seed_mode_s == "step":
            next_seed = int(bs) + 1

        dd = gr.update(choices=[vv["id"] for vv in variants], value=(variants[0]["id"] if variants else None))
        yield variants, cache, _rows_from_variants(variants), dd, _pct_status(100, f"Generated {len(variants)} variant(s)."), int(next_seed)
    except Exception as e:
        yield [], {}, [], gr.update(choices=[], value=None), _pct_status(100, f"Generate failed: {e}"), int(base_seed or 1337)


def _ui_regen_unlocked(
    variants: list[dict[str, Any]],
    audio_cache: AudioCache,
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
) -> tuple[list[dict[str, Any]], AudioCache, list[list[object]], dict, str, int]:
    if not variants:
        yield [], audio_cache or {}, [], gr.update(choices=[], value=None), _pct_status(100, "No variants to regenerate."), int(base_seed or 1337)
        return

    try:
        bs = int(base_seed) if base_seed is not None else 1337
        seed_mode_s = str(seed_mode or "lock").strip().lower()
        if seed_mode_s == "random":
            bs = int.from_bytes(os.urandom(4), "big")
        elif seed_mode_s == "step":
            bs = bs + 1

        unlocked_total = sum(1 for v in (variants or []) if isinstance(v, dict) and not bool(v.get("locked", False)))
        if unlocked_total <= 0:
            vnorm = normalize_variants_state(variants)
            dd0 = gr.update(choices=[v["id"] for v in vnorm], value=(vnorm[0]["id"] if vnorm else None))
            yield vnorm, (audio_cache or {}), _rows_from_variants(vnorm), dd0, _pct_status(100, "All variants are locked."), int(bs)
            return

        yield normalize_variants_state(variants), (audio_cache or {}), _rows_from_variants(normalize_variants_state(variants)), gr.update(), _pct_status(0, "starting"), int(bs)

        out: list[dict[str, Any]] = []
        cache: AudioCache = dict(audio_cache or {})
        regen = 0
        for i, v in enumerate(variants):
            if not isinstance(v, dict):
                continue
            if bool(v.get("locked", False)):
                out.append(v)
                continue

            vid = str(v.get("id") or _variant_id(i))
            seed_i = int(bs) + i
            out_wav = _cp_temp_wav_path(vid)

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

            wav_p = Path(res.wav_path)
            a, sr = read_wav_mono(wav_p)
            base_key = _cache_key(vid, "base")
            cache[base_key] = (int(sr), a.astype(np.float32, copy=False))

            if not _keep_control_panel_wavs():
                _best_effort_unlink(out_wav)
                if wav_p.resolve() != out_wav.resolve():
                    _best_effort_unlink(wav_p)
            m = compute_metrics(a, int(sr))
            v2 = dict(v)
            v2["seed"] = seed_i
            v2["seconds"] = float(m.seconds)
            v2["rms_dbfs"] = _rms_dbfs(float(m.rms))
            v2["audio_key"] = base_key
            v2["edited_audio_key"] = None
            v2["wav_path"] = (str(wav_p) if _keep_control_panel_wavs() else "")
            v2["edited_wav_path"] = None
            out.append(v2)
            regen += 1

            out_norm = normalize_variants_state(out + [vv for vv in (variants[i + 1 :] or []) if isinstance(vv, dict)])
            dd = gr.update(choices=[vv["id"] for vv in out_norm], value=(out_norm[0]["id"] if out_norm else None))
            pct = int(round((regen / max(1, unlocked_total)) * 95.0))
            yield out_norm, cache, _rows_from_variants(out_norm), dd, _pct_status(pct, f"regenerating {regen}/{unlocked_total}"), int(bs)

        dd = gr.update(choices=[v["id"] for v in out], value=(out[0]["id"] if out else None))
        out_norm = normalize_variants_state(out)
        yield out_norm, cache, _rows_from_variants(out_norm), dd, _pct_status(100, f"Regenerated {regen} unlocked variant(s)."), int(bs)
    except Exception as e:
        vnorm = normalize_variants_state(variants)
        yield vnorm, (audio_cache or {}), _rows_from_variants(vnorm), gr.update(), _pct_status(100, f"Regenerate failed: {e}"), int(base_seed or 1337)


def _ui_variant_audio(
    variants: list[dict[str, Any]],
    audio_cache: AudioCache,
    current_variant: str,
) -> tuple[tuple[int, np.ndarray] | None, object, object, str, str]:
    vid = str(current_variant or "").strip()
    for v in variants or []:
        if isinstance(v, dict) and str(v.get("id")) == vid:
            got = _audio_from_variant(v, audio_cache or {})
            if got is None:
                return None, None, None, "Missing audio.", ""
            sr, a = got
            m = compute_metrics(a, int(sr))
            analysis = f"seconds={m.seconds:.2f} sr={m.sample_rate} peak={m.peak:.3f} rms_dbfs={_fmt_db(_rms_dbfs(float(m.rms)))}"
            return (int(sr), a.astype(np.float32, copy=False)), waveform_image(a, sr), spectrogram_image(a, sr), "", analysis
    return None, None, None, "Select a variant.", "Select a variant."


def _ui_waveform_apply_edits(
    variants: list[dict[str, Any]],
    audio_cache: AudioCache,
    current_variant: str,
    start_s: float,
    end_s: float,
    fade_in_ms: int,
    fade_out_ms: int,
) -> tuple[list[dict[str, Any]], AudioCache, tuple[int, np.ndarray] | None, object, object, str, str]:
    vid = str(current_variant or "").strip()
    if not vid:
        return variants, (audio_cache or {}), None, None, None, "Select a variant first.", ""

    out: list[dict[str, Any]] = []
    cache: AudioCache = dict(audio_cache or {})
    for v in variants or []:
        if not isinstance(v, dict) or str(v.get("id")) != vid:
            out.append(v)
            continue

        got = _audio_from_variant(v, cache)
        if got is None:
            out.append(v)
            return out, cache, None, None, None, f"Missing audio for {vid}.", ""

        sr, a = got
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

        edit_key = _cache_key(vid, "edit")
        cache[edit_key] = (int(sr), y.astype(np.float32, copy=False))

        v2 = dict(v)
        v2["edited_audio_key"] = edit_key
        v2["edited_wav_path"] = None
        out.append(v2)

        m = compute_metrics(y, int(sr))
        analysis = f"seconds={m.seconds:.2f} sr={m.sample_rate} peak={m.peak:.3f} rms_dbfs={_fmt_db(_rms_dbfs(float(m.rms)))}"
        out_norm = normalize_variants_state(out)
        return out_norm, cache, (int(sr), y), waveform_image(y, int(sr)), spectrogram_image(y, int(sr)), f"Applied edits to {vid}.", analysis

    return normalize_variants_state(variants), cache, None, None, None, "Variant not found.", ""


def _ui_fx_slots_apply(
    variants: list[dict[str, Any]],
    audio_cache: AudioCache,
    current_variant: str,
    slots_df: list[list[object]] | None,
) -> tuple[tuple[int, np.ndarray] | None, str]:
    from .fx_chain_v2 import FxChainV2, FxStepV2, apply_fx_chain_v2

    vid = str(current_variant or "").strip()
    if not vid:
        return None, "Select a variant first."

    for v in variants or []:
        if isinstance(v, dict) and str(v.get("id")) == vid:
            got = _audio_from_variant(v, audio_cache or {})
            if got is None:
                return None, "Missing audio for selected variant."
            sr, a = got
            break
    else:
        return None, "Missing audio for selected variant."

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
    y = apply_fx_chain_v2(a, int(sr), chain)
    return (int(sr), y.astype(np.float32, copy=False)), "Applied FX slots."


def _ui_export_bundle(
    variants: list[dict[str, Any]],
    audio_cache: AudioCache,
    mode: str,
    out_format: str,
    out_sample_rate: int | None,
    wav_subtype: str,
    mp3_bitrate: str,
    ogg_quality: int,
    filename_template: str,
    project_json: dict[str, Any] | None,
    rfxgen_preset: str,
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
        yield "", _pct_status(100, "No variants matched export mode.")
        return

    out_dir = Path("outputs") / "control_panel_exports"
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = Path("outputs") / "control_panel_export_bundle.zip"

    templ = str(filename_template or "{variant}_{seed}").strip() or "{variant}_{seed}"

    proj_tok = ""
    if isinstance(project_json, dict):
        # Prefer stable id, fall back to title.
        proj_tok = str(project_json.get("project_id") or project_json.get("title") or "").strip()
    proj_tok = _safe_item_key(proj_tok) if proj_tok else ""

    preset_tok = _safe_item_key(str(rfxgen_preset or "").strip()) if str(rfxgen_preset or "").strip() else ""

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        total = len(picks)
        yield "", _pct_status(0, f"exporting {total} file(s)")
        for idx, v in enumerate(picks):
            vid = str(v.get("id") or "variant")
            seed = v.get("seed")

            got = _audio_from_variant(v, audio_cache or {})
            if got is None:
                continue
            sr_in, a_in = got

            name = templ
            name = name.replace("{variant}", vid)
            name = name.replace("{seed}", str(seed if seed is not None else ""))
            name = name.replace("{project}", proj_tok)
            name = name.replace("{preset}", preset_tok)
            name = "_".join([p for p in name.replace("/", "_").replace("\\", "_").split("_") if p])
            if not name:
                name = vid

            out_file = out_dir / f"{name}.{fmt}"
            if fmt == "wav":
                sr_out = int(out_sample_rate) if out_sample_rate else int(sr_in)
                if sr_out != int(sr_in):
                    tmp = out_dir / f"_{vid}_tmp_resample.wav"
                    write_wav(tmp, a_in.astype(np.float32, copy=False), int(sr_in), subtype="PCM_16")
                    convert_audio_with_ffmpeg(tmp, out_file, sample_rate=sr_out, channels=1, out_format="wav")
                    a2, sr2 = read_wav_mono(out_file)
                    write_wav(out_file, a2.astype(np.float32, copy=False), int(sr2), subtype=str(wav_subtype))
                else:
                    write_wav(out_file, a_in.astype(np.float32, copy=False), int(sr_in), subtype=str(wav_subtype))
            else:
                tmp = out_dir / f"_{vid}_tmp_src.wav"
                write_wav(tmp, a_in.astype(np.float32, copy=False), int(sr_in), subtype="PCM_16")
                convert_audio_with_ffmpeg(
                    tmp,
                    out_file,
                    sample_rate=(int(out_sample_rate) if out_sample_rate else None),
                    channels=1,
                    out_format=fmt,
                    ogg_quality=int(ogg_quality),
                    mp3_bitrate=str(mp3_bitrate or "192k"),
                )

            z.write(out_file, arcname=out_file.name)

            done = idx + 1
            pct = int(round((done / max(1, total)) * 95.0))
            yield "", _pct_status(pct, f"exporting {done}/{total}")

    yield str(zip_path), _pct_status(100, f"Exported {len(picks)} variant(s) to bundle.")


def build_demo_control_panel() -> gr.Blocks:
    icon_uri = _ui_icon_data_uri()
    bg_uri = _ui_background_data_uri()
    watermark_css = ""
    if icon_uri:
        watermark_css = f"""
    .gradio-container {{ position: relative; }}
    .gradio-container::before {{
        content: "";
        position: fixed;
        inset: 0;
        background-image: url(\"{icon_uri}\");
        background-repeat: no-repeat;
        background-position: center;
        background-size: min(60vw, 520px);
        opacity: 0.06;
        pointer-events: none;
        z-index: 0;
    }}
    .gradio-container > * {{ position: relative; z-index: 1; }}
    """

    bg_css = "    .gradio-container { background: #0b0d10; }"
    if bg_uri:
        bg_css = f"""
    .gradio-container {{
        background: #0b0d10;
        background-image: url(\"{bg_uri}\");
        background-repeat: no-repeat;
        background-position: center;
        background-size: cover;
        background-attachment: fixed;
    }}
    """.strip("\n")

    css = f"""
    {bg_css}
    .gradio-container * {{ color-scheme: dark; }}
    {watermark_css}
    """

    with gr.Blocks(title="SÖNDBÖUND", css=css) as demo:
        gr.Markdown(
            "# SÖNDBÖUND — Control Panel UI (Gradio-native)\n"
            "Discrete actions, explicit state, zero continuous interaction."
        )

        variants_state = gr.State([])
        audio_cache_state = gr.State({})

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## Project Browser")
                pr_root = gr.Textbox(value=".", label="Project root")
                pr_load_btn = gr.Button("Load project")
                pr_json = gr.JSON(label="Project")
                pr_item_id = gr.Dropdown(choices=[], value=None, label="Item id")
                pr_new_id = gr.Textbox(value="", label="New id (rename/duplicate)")
                pr_confirm_delete = gr.Checkbox(value=False, label="Confirm delete")
                with gr.Row():
                    pr_rename_btn = gr.Button("Rename")
                    pr_dup_btn = gr.Button("Duplicate")
                    pr_del_btn = gr.Button("Delete")
                with gr.Row():
                    pr_build_btn = gr.Button("Build")
                    pr_edit_btn = gr.Button("Edit")
                pr_items = gr.Dataframe(
                    headers=["id", "category", "engine", "event", "sound_path", "variants", "active_version"],
                    datatype=["str", "str", "str", "str", "str", "number", "number"],
                    row_count=(0, "dynamic"),
                    col_count=(7, "fixed"),
                    interactive=False,
                    label="Items",
                )
                pr_status = gr.Textbox(label="Status", interactive=False)
                pr_load_btn.click(fn=_ui_project_load, inputs=[pr_root], outputs=[pr_json, pr_items, pr_item_id, pr_status])
                pr_rename_btn.click(
                    fn=_ui_project_rename,
                    inputs=[pr_root, pr_item_id, pr_new_id],
                    outputs=[pr_json, pr_items, pr_item_id, pr_status],
                )
                pr_dup_btn.click(
                    fn=_ui_project_duplicate,
                    inputs=[pr_root, pr_item_id, pr_new_id],
                    outputs=[pr_json, pr_items, pr_item_id, pr_status],
                )
                pr_del_btn.click(
                    fn=_ui_project_delete,
                    inputs=[pr_root, pr_item_id, pr_confirm_delete],
                    outputs=[pr_json, pr_items, pr_item_id, pr_status],
                )
                pr_build_btn.click(
                    fn=_ui_project_build_and_reload,
                    inputs=[pr_root, pr_item_id],
                    outputs=[pr_json, pr_items, pr_item_id, pr_status],
                )
                pr_edit_btn.click(
                    fn=_ui_project_edit,
                    inputs=[pr_root, pr_item_id],
                    outputs=[pr_status],
                )

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
            outputs=[variants_state, audio_cache_state, variants_df, current_variant, gen_status, seed],
        )

        regen_btn.click(
            fn=_ui_regen_unlocked,
            inputs=[
                variants_state,
                audio_cache_state,
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
            outputs=[variants_state, audio_cache_state, variants_df, current_variant, gen_status, seed],
        )

        variants_df.change(fn=_variants_from_df, inputs=[variants_df, variants_state], outputs=[variants_state])

        load_btn.click(
            fn=_ui_variant_audio,
            inputs=[variants_state, audio_cache_state, current_variant],
            outputs=[viewer_audio, viewer_wave, viewer_spec, viewer_status, analysis_txt],
        )
        load_btn.click(
            fn=lambda v, ac, c: _ui_variant_audio(v, ac, c)[2],
            inputs=[variants_state, audio_cache_state, current_variant],
            outputs=[analysis_spec],
        )

        apply_edits_btn.click(
            fn=_ui_waveform_apply_edits,
            inputs=[variants_state, audio_cache_state, current_variant, start_s, end_s, fade_in_ms, fade_out_ms],
            outputs=[variants_state, audio_cache_state, viewer_audio, viewer_wave, viewer_spec, viewer_status, analysis_txt],
        )

        fx_apply_btn.click(
            fn=_ui_fx_slots_apply,
            inputs=[variants_state, audio_cache_state, current_variant, fx_slots],
            outputs=[fx_audio_out, fx_status],
        )

        export_btn.click(
            fn=_ui_export_bundle,
            inputs=[
                variants_state,
                audio_cache_state,
                ex_mode,
                ex_fmt,
                ex_sr,
                ex_wav_subtype,
                ex_mp3_bitrate,
                ex_ogg_quality,
                filename_template,
                pr_json,
                rfxgen_preset,
            ],
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
                variants, cache = _run_generate_variants(
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
                got = _audio_from_variant(variants[0], cache)
                if got is None:
                    return None, "Preview audio missing."
                sr, a = got
                m = compute_metrics(a, int(sr))
                return (int(sr), a.astype(np.float32, copy=False)), f"Preview ok: seconds={m.seconds:.2f} rms_dbfs={_fmt_db(_rms_dbfs(float(m.rms)))}"
            except Exception as e:
                return None, f"Preview failed: {e}"

        pb_preview_btn.click(
            fn=_pb_preview,
            inputs=[pb_keys, engine, prompt, seconds, seed, device, model, rfxgen_path, post, polish],
            outputs=[pb_preview_audio, pb_status],
        )

    # Enable queueing so generator-based progress updates stream to the UI.
    try:
        demo.queue()
    except Exception:
        pass
    return demo
