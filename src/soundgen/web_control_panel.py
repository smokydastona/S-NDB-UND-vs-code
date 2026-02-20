from __future__ import annotations

import base64
import json
import math
import os
import subprocess
import shutil
import sys
import uuid
import zipfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np

from .ai_assistant import (
    AIChatError,
    chat_once,
    ollama_list_models,
    ollama_pull_model,
    ollama_reachable,
)
from .ai_context import build_app_context
from .ai_index import build_or_update_index
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


def _variants_from_df(rows: Any, prev: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Gradio's Dataframe component may yield:
    # - list[list[...]] (older)
    # - list[dict] (some configs)
    # - pandas.DataFrame (Gradio 5/6 common)
    if rows is None:
        return prev

    # pandas.DataFrame (duck-typed to avoid hard dependency at import time)
    if hasattr(rows, "to_dict") and hasattr(rows, "empty"):
        try:
            if bool(getattr(rows, "empty")):
                return prev
        except Exception:
            # If anything goes sideways, fall back to previous state.
            return prev

        try:
            records = rows.to_dict(orient="records")  # type: ignore[call-arg]
        except TypeError:
            records = rows.to_dict("records")  # type: ignore[call-arg]

        normalized_rows: list[list[object]] = []
        for rec in (records or []):
            if not isinstance(rec, dict):
                continue
            normalized_rows.append(
                [
                    rec.get("select", rec.get(0)),
                    rec.get("id", rec.get(1)),
                    rec.get("seconds", rec.get(2)),
                    rec.get("rms_dbfs", rec.get(3)),
                    rec.get("seed", rec.get(4)),
                    rec.get("locked", rec.get(5)),
                ]
            )
        rows = normalized_rows

    # list-of-dicts
    if isinstance(rows, list) and rows and isinstance(rows[0], dict):
        normalized_rows = []
        for rec in rows:
            if not isinstance(rec, dict):
                continue
            normalized_rows.append(
                [
                    rec.get("select", rec.get(0)),
                    rec.get("id", rec.get(1)),
                    rec.get("seconds", rec.get(2)),
                    rec.get("rms_dbfs", rec.get(3)),
                    rec.get("seed", rec.get(4)),
                    rec.get("locked", rec.get(5)),
                ]
            )
        rows = normalized_rows

    if not isinstance(rows, list) or len(rows) == 0:
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
    except FileNotFoundError:
        # Auto-init: create a basic project so the UI doesn't error on first use.
        try:
            from .minecraft import sanitize_id
            from .project import create_project, load_project

            root.mkdir(parents=True, exist_ok=True)
            ns = sanitize_id(str(root.resolve().name or "soundgen"), kind="namespace")
            create_project(
                project_root=root,
                kind="soundpack",
                project_id="project",
                title="SÖNDBÖUND Project",
                namespace=ns,
                pack_root="resourcepack",
            )
            proj = load_project(root)
        except Exception as e:
            expected = root / "sndbund_project.json"
            hint = (
                "No project found in this folder and auto-create failed.\n\n"
                "Create one with:\n"
                f"  SÖNDBÖUND.exe project create --root \"{root}\" --id myproj --namespace mymod\n\n"
                f"Expected file: {expected}\n\n"
                f"Auto-create error: {e}"
            )
            stub = {
                "error": "Project not initialized",
                "root": str(root),
                "expected": str(expected),
                "hint": hint,
                "items": [],
            }
            return stub, [], gr.update(choices=[], value=None), hint
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


def _ui_mode_visibility(mode_label: str) -> tuple[dict, dict, dict, dict]:
    """Compute visibility updates for major UI sections.

    Returns updates for: advanced controls, pro-only tabs (projects/copilot), and edit tools.
    """

    m = str(mode_label or "Basic").strip().lower()
    is_basic = m.startswith("basic")
    is_advanced = m.startswith("advanced")
    is_pro = m.startswith("pro")

    show_advanced = bool(is_advanced or is_pro)
    show_edit = bool(is_advanced or is_pro)
    show_projects = bool(is_pro)
    show_copilot = bool(is_pro)

    return (
        gr.update(visible=show_advanced),
        gr.update(visible=show_edit),
        gr.update(visible=show_projects),
        gr.update(visible=show_copilot),
    )


def _ui_mode_defaults(mode_label: str):
    m = str(mode_label or "Basic").strip().lower()
    if m.startswith("basic"):
        return (
            gr.update(value=1),  # variant_count
            gr.update(value="random"),  # seed_mode
        )
    # Advanced/Pro
    return (
        gr.update(value=6),
        gr.update(value="lock"),
    )


def _ui_hint_text(engine: str, variant_count: int, post: bool, polish: bool) -> str:
    e = str(engine or "").strip().lower()
    bs = int(variant_count) if variant_count is not None else 1
    parts: list[str] = []

    if e in {"diffusers", "replicate"}:
        parts.append("AI engine: slower, higher quality")
    elif e in {"rfxgen", "layered"}:
        parts.append("Procedural engine: fast, snappy transients")
    else:
        parts.append(f"Engine: {engine}")

    if bs > 1:
        parts.append(f"Batch enabled: {bs} variants will be generated")

    if bool(polish):
        parts.append("Polish mode: extra cleanup + finishing")
    elif bool(post):
        parts.append("Post-process: trims + small fades")

    return " • ".join(parts)


_WORKFLOW_PRESETS: dict[str, dict[str, object]] = {
    "(none)": {},
    "Creature SFX": {"engine": "diffusers", "seconds": 2.0, "variant_count": 6, "post": True, "polish": True},
    "UI/Tech Sounds": {"engine": "rfxgen", "seconds": 0.4, "variant_count": 8, "post": True, "polish": False},
    "Foley": {"engine": "layered", "seconds": 2.5, "variant_count": 6, "post": True, "polish": True},
}


def _ui_apply_workflow_preset(preset_label: str):
    p = _WORKFLOW_PRESETS.get(str(preset_label or "(none)"), {})
    if not p:
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
        )
    return (
        gr.update(value=p.get("engine")),
        gr.update(value=p.get("seconds")),
        gr.update(value=p.get("variant_count")),
        gr.update(value=p.get("post")),
        gr.update(value=p.get("polish")),
    )


def _ui_smart_defaults(engine: str, enabled: bool):
    if not bool(enabled):
        return gr.update(), gr.update(), gr.update(), gr.update()
    e = str(engine or "").strip().lower()
    if e == "rfxgen":
        return gr.update(value=0.4), gr.update(value=8), gr.update(value=True), gr.update(value=False)
    if e == "diffusers":
        return gr.update(value=2.0), gr.update(value=6), gr.update(value=True), gr.update(value=True)
    if e == "layered":
        return gr.update(value=2.5), gr.update(value=6), gr.update(value=True), gr.update(value=True)
    return gr.update(), gr.update(), gr.update(), gr.update()


def _history_append(
    history: list[dict[str, Any]] | None,
    *,
    engine: str,
    prompt: str,
    seconds: float,
    variant_count: int,
    post: bool,
    polish: bool,
    seed: int,
) -> list[dict[str, Any]]:
    h = list(history or [])
    h.append(
        {
            "engine": str(engine),
            "prompt": str(prompt),
            "seconds": float(seconds),
            "variants": int(variant_count),
            "post": bool(post),
            "polish": bool(polish),
            "seed": int(seed),
        }
    )
    return h[-10:]


def _history_rows(history: list[dict[str, Any]] | None) -> list[list[object]]:
    rows: list[list[object]] = []
    for i, it in enumerate(list(history or [])[::-1], start=1):
        if not isinstance(it, dict):
            continue
        rows.append(
            [
                i,
                it.get("engine"),
                it.get("prompt"),
                it.get("seconds"),
                it.get("variants"),
                it.get("seed"),
            ]
        )
    return rows


def _history_choices(history: list[dict[str, Any]] | None) -> list[str]:
    out: list[str] = []
    for idx, it in enumerate(list(history or [])[::-1]):
        if not isinstance(it, dict):
            continue
        prompt = str(it.get("prompt") or "").strip()
        engine = str(it.get("engine") or "").strip()
        out.append(f"{idx + 1}. {engine} — {prompt[:60]}")
    return out


def _history_get(history: list[dict[str, Any]] | None, choice_label: str) -> dict[str, Any] | None:
    labels = _history_choices(history)
    try:
        pos = labels.index(str(choice_label))
    except ValueError:
        return None
    items = list(history or [])[::-1]
    if pos < 0 or pos >= len(items):
        return None
    return items[pos] if isinstance(items[pos], dict) else None


def _ui_history_apply(history: list[dict[str, Any]] | None, choice_label: str):
    it = _history_get(history, choice_label)
    if not it:
        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), "Pick a history entry first."
    return (
        gr.update(value=it.get("engine")),
        gr.update(value=it.get("prompt")),
        gr.update(value=it.get("seconds")),
        gr.update(value=it.get("variants")),
        gr.update(value=it.get("seed")),
        "Applied history entry.",
    )


def _ui_generate_variants_with_history(
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
    history: list[dict[str, Any]] | None,
):
    h = list(history or [])
    for variants, cache, rows, dd, status, next_seed in _ui_generate_variants(
        engine,
        prompt,
        seconds,
        base_seed,
        seed_mode,
        variant_count,
        device,
        model,
        rfxgen_preset,
        rfxgen_path,
        post,
        polish,
    ):
        h_out = h
        if isinstance(status, str) and status.startswith("100%") and "Generated" in status:
            h_out = _history_append(
                h,
                engine=str(engine),
                prompt=str(prompt),
                seconds=float(seconds),
                variant_count=int(variant_count),
                post=bool(post),
                polish=bool(polish),
                seed=int(base_seed or 1337),
            )
            h = h_out
        yield (
            variants,
            cache,
            rows,
            dd,
            status,
            next_seed,
            h,
            _history_rows(h),
            gr.update(choices=_history_choices(h), value=(_history_choices(h)[0] if _history_choices(h) else None)),
        )


def _ui_regen_unlocked_with_history(
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
    history: list[dict[str, Any]] | None,
):
    h = list(history or [])
    for v2, cache2, rows2, dd2, status2, next_seed2 in _ui_regen_unlocked(
        variants,
        audio_cache,
        engine,
        prompt,
        seconds,
        base_seed,
        seed_mode,
        device,
        model,
        rfxgen_preset,
        rfxgen_path,
        post,
        polish,
    ):
        h_out = h
        if isinstance(status2, str) and status2.startswith("100%") and "Regenerated" in status2:
            h_out = _history_append(
                h,
                engine=str(engine),
                prompt=str(prompt),
                seconds=float(seconds),
                variant_count=int(len(v2 or [])),
                post=bool(post),
                polish=bool(polish),
                seed=int(base_seed or 1337),
            )
            h = h_out
        yield (
            v2,
            cache2,
            rows2,
            dd2,
            status2,
            next_seed2,
            h,
            _history_rows(h),
            gr.update(choices=_history_choices(h), value=(_history_choices(h)[0] if _history_choices(h) else None)),
        )


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
            from .json_utils import loads_json_object_lenient

            params = loads_json_object_lenient(params_s, context="FX slot params JSON") if params_s else {}
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

    js = r"""
() => {
  // Keyboard shortcuts (best-effort): Ctrl+Enter generates.
  document.addEventListener('keydown', (e) => {
    try {
      const isCtrlEnter = (e.ctrlKey || e.metaKey) && (e.key === 'Enter');
      if (isCtrlEnter) {
        const btn = document.querySelector('#sndb_gen_btn');
        if (btn) {
          btn.click();
          e.preventDefault();
        }
      }
    } catch (_) {}
  }, { capture: true });
}
"""

    with gr.Blocks(title="SÖNDBÖUND", css=css, js=js) as demo:
        gr.Markdown(
            "# SÖNDBÖUND — Control Panel UI (Gradio-native)\n"
            "Discrete actions, explicit state, zero continuous interaction."
        )

        variants_state = gr.State([])
        audio_cache_state = gr.State({})
        history_state = gr.State([])

        with gr.Row():
            ui_mode = gr.Radio(
                ["Basic", "Advanced", "Pro"],
                value="Basic",
                label="Mode",
                info="Basic reduces cognitive load; Advanced reveals editing/export; Pro reveals projects + Copilot.",
            )
            workflow_preset = gr.Dropdown(
                list(_WORKFLOW_PRESETS.keys()),
                value="(none)",
                label="Workflow preset",
                info="Quickly applies recommended settings for common workflows.",
            )

        hint_md = gr.Markdown("")

        # Sections are organized into tabs so the UI feels lighter.
        with gr.Tabs():
            with gr.Tab("Generate"):
                gr.Markdown("## Generate")
                with gr.Accordion("Generation", open=True):
                    engine = gr.Dropdown(
                        available_engines(),
                        value="diffusers",
                        label="Engine",
                        info="Choose how audio is generated (AI vs procedural).",
                    )
                    prompt = gr.Textbox(label="Prompt", placeholder="e.g. coin pickup")
                    # Many SFX are shorter than 0.5s; allow sub-half-second values.
                    seconds = gr.Slider(0.1, 10.0, value=3.0, step=0.1, label="Seconds")
                    with gr.Row():
                        generate_btn = gr.Button("Generate", elem_id="sndb_gen_btn")
                        regen_btn = gr.Button("Regenerate unlocked")
                    gen_status = gr.Textbox(label="Status", interactive=False)

                advanced_box = gr.Group(visible=False)
                with advanced_box:
                    with gr.Accordion("Advanced", open=False):
                        with gr.Row():
                            seed = gr.Number(value=1337, precision=0, label="Base seed", info="Lock for repeatable results.")
                            seed_mode = gr.Dropdown(
                                ["lock", "random", "step"],
                                value="lock",
                                label="Seed mode",
                                info="lock=reproducible, random=new seed per run, step=increment seed.",
                            )
                        variant_count = gr.Slider(1, 16, value=6, step=1, label="Variant count")
                        randomness = gr.Slider(0.0, 1.0, value=0.0, step=0.05, label="Randomness (reserved)")
                        with gr.Row():
                            post = gr.Checkbox(value=True, label="Post-process", info="Trims silence + applies small fades.")
                            polish = gr.Checkbox(value=False, label="Polish mode", info="Extra finishing; typically slower.")
                        smart_defaults = gr.Checkbox(value=True, label="Smart defaults", info="Auto-fill recommended values per engine.")

                with gr.Accordion("Results", open=True):
                    variants_df = gr.Dataframe(
                        headers=["select", "id", "seconds", "rms_dbfs", "seed", "locked"],
                        datatype=["bool", "str", "number", "str", "number", "bool"],
                        row_count=(0, "dynamic"),
                        col_count=(6, "fixed"),
                        interactive=True,
                        label="Variants",
                    )
                    current_variant = gr.Dropdown([], value=None, label="Current variant")
                    load_btn = gr.Button("Load preview")

                with gr.Accordion("History", open=False):
                    history_df = gr.Dataframe(
                        headers=["#", "engine", "prompt", "seconds", "variants", "seed"],
                        datatype=["number", "str", "str", "number", "number", "number"],
                        row_count=(0, "dynamic"),
                        col_count=(6, "fixed"),
                        interactive=False,
                        label="Last 10 runs",
                    )
                    history_pick = gr.Dropdown(choices=[], value=None, label="Pick")
                    history_apply_btn = gr.Button("Apply to controls")
                    history_status = gr.Textbox(label="Status", interactive=False)

                with gr.Accordion("Export", open=False):
                    with gr.Row():
                        ex_fmt = gr.Dropdown(["wav", "mp3", "ogg", "flac"], value="wav", label="Format")
                        ex_sr = gr.Number(value=None, precision=0, label="Sample rate (optional)")
                    with gr.Row():
                        ex_wav_subtype = gr.Dropdown(["PCM_16", "PCM_24", "FLOAT"], value="PCM_16", label="WAV subtype")
                        ex_mp3_bitrate = gr.Textbox(value="192k", label="MP3 bitrate")
                        ex_ogg_quality = gr.Slider(0, 10, value=5, step=1, label="OGG quality")
                    ex_mode = gr.Dropdown(["selected", "locked"], value="selected", label="Export mode")
                    filename_template = gr.Textbox(value="{variant}_{seed}", label="Filename template")
                    export_btn = gr.Button("Export bundle (.zip)")
                    export_out = gr.File(label="Export bundle")
                    export_status = gr.Textbox(label="Status", interactive=False)

            edit_tab = gr.Tab("Edit", visible=False)
            with edit_tab:
                gr.Markdown("## Edit")
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Waveform Viewer")
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
                        gr.Markdown("### FX Chain Controls (slot-based)")
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

                with gr.Accordion("Analysis", open=True):
                    with gr.Tabs():
                        with gr.Tab("Spectrum"):
                            analysis_spec = gr.Image(label="Spectrogram", type="pil")
                        with gr.Tab("Loudness"):
                            analysis_txt = gr.Textbox(label="RMS dBFS / Peak", interactive=False)

            projects_tab = gr.Tab("Projects", visible=False)
            with projects_tab:
                gr.Markdown("## Projects")
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Project Browser")
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

                    with gr.Column(scale=1):
                        gr.Markdown("### Preset Browser")
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

            settings_tab = gr.Tab("Settings")
            with settings_tab:
                gr.Markdown("## Settings")
                gr.Markdown("These settings affect generation behavior.")
                with gr.Row():
                    device = gr.Dropdown(["cpu", "cuda"], value="cpu", label="Device", info="cpu is safest; cuda requires a compatible GPU/driver.")
                    model = gr.Dropdown(["cvssp/audioldm2"], value="cvssp/audioldm2", label="Model")
                with gr.Row():
                    rfxgen_preset = gr.Dropdown(list(SUPPORTED_PRESETS), value="blip", label="rfxgen preset")
                    rfxgen_path = gr.Textbox(value="", label="rfxgen path (optional)")

            copilot_tab = gr.Tab("Copilot", visible=False)
            with copilot_tab:
                gr.Markdown("## Copilot (optional)")
                gr.Markdown(
                    "Local-first helper for prompt writing, naming, manifests, subtitles, and error explanations. "
                    "No API keys are saved."
                )

                ai_provider = gr.Dropdown(
                    ["Local (Ollama)", "Cloud (OpenAI-compatible)", "Cloud (Azure OpenAI)"],
                    value="Local (Ollama)",
                    label="Provider",
                )
                ai_include_context = gr.Checkbox(value=True, label="Include app context (recommended)")
                with gr.Accordion("Settings", open=False):
                    ai_endpoint = gr.Textbox(value="http://localhost:11434", label="Endpoint")
                    ai_model = gr.Textbox(value="llama3.2", label="Model (or Azure deployment)")
                    ai_api_key = gr.Textbox(value="", label="API key (cloud only)", type="password")
                    ai_api_version = gr.Textbox(value="2024-02-15-preview", label="Azure API version (Azure only)")
                    ai_temperature = gr.Slider(0.0, 1.0, value=0.2, step=0.05, label="Temperature")
                    ai_setup_btn = gr.Button("Setup local Copilot (Ollama)")
                    ai_reindex_btn = gr.Button("Rebuild Copilot index")

                ai_chat = gr.Chatbot(label="Chat", height=360)
                ai_msg = gr.Textbox(value="", label="Message")
                with gr.Row():
                    ai_send_btn = gr.Button("Send")
                    ai_clear_btn = gr.Button("Clear")
                ai_status = gr.Textbox(label="Status", interactive=False)

        # Mode + preset wiring
        ui_mode.change(
            fn=_ui_mode_visibility,
            inputs=[ui_mode],
            outputs=[advanced_box, edit_tab, projects_tab, copilot_tab],
        )
        ui_mode.change(
            fn=_ui_mode_defaults,
            inputs=[ui_mode],
            outputs=[variant_count, seed_mode],
        )

        workflow_preset.change(
            fn=_ui_apply_workflow_preset,
            inputs=[workflow_preset],
            outputs=[engine, seconds, variant_count, post, polish],
        )

        engine.change(
            fn=_ui_smart_defaults,
            inputs=[engine, smart_defaults],
            outputs=[seconds, variant_count, post, polish],
        )

        # Live hint text
        engine.change(fn=_ui_hint_text, inputs=[engine, variant_count, post, polish], outputs=[hint_md])
        variant_count.change(fn=_ui_hint_text, inputs=[engine, variant_count, post, polish], outputs=[hint_md])
        post.change(fn=_ui_hint_text, inputs=[engine, variant_count, post, polish], outputs=[hint_md])
        polish.change(fn=_ui_hint_text, inputs=[engine, variant_count, post, polish], outputs=[hint_md])

        generate_btn.click(
            fn=_ui_generate_variants_with_history,
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
                history_state,
            ],
            outputs=[
                variants_state,
                audio_cache_state,
                variants_df,
                current_variant,
                gen_status,
                seed,
                history_state,
                history_df,
                history_pick,
            ],
        )

        regen_btn.click(
            fn=_ui_regen_unlocked_with_history,
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
                history_state,
            ],
            outputs=[
                variants_state,
                audio_cache_state,
                variants_df,
                current_variant,
                gen_status,
                seed,
                history_state,
                history_df,
                history_pick,
            ],
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

        history_apply_btn.click(
            fn=_ui_history_apply,
            inputs=[history_state, history_pick],
            outputs=[engine, prompt, seconds, variant_count, seed, history_status],
        )

        # Projects wiring (pro mode)
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

        def _ai_defaults(provider_label: str):
            p = str(provider_label or "").strip().lower()
            if "azure" in p:
                return (
                    gr.update(value="https://YOUR-RESOURCE-NAME.openai.azure.com"),
                    gr.update(value="YOUR-DEPLOYMENT"),
                    gr.update(value="2024-02-15-preview"),
                )
            if "openai" in p:
                return (
                    gr.update(value="https://api.openai.com/v1"),
                    gr.update(value="gpt-4o-mini"),
                    gr.update(value="2024-02-15-preview"),
                )
            # Local (Ollama)
            return (
                gr.update(value="http://localhost:11434"),
                gr.update(value="llama3.2"),
                gr.update(value="2024-02-15-preview"),
            )

        def _ai_send(
            provider_label: str,
            endpoint: str,
            model_or_deployment: str,
            api_key: str,
            api_version: str,
            temperature: float,
            include_context: bool,
            history: list[tuple[str, str]] | None,
            msg: str,
        ):
            text = str(msg or "").strip()
            if not text:
                return history or [], ""

            p = str(provider_label or "").strip().lower()
            if "azure" in p:
                kind = "cloud-azure"
            elif "openai" in p:
                kind = "cloud-openai"
            else:
                kind = "local-ollama"

            try:
                ctx = ""
                if bool(include_context):
                    ctx = build_app_context(query=text)
                out = chat_once(
                    provider=kind,  # type: ignore[arg-type]
                    user_text=text,
                    history=history or [],
                    app_context=ctx,
                    endpoint=str(endpoint or "").strip(),
                    model_or_deployment=str(model_or_deployment or "").strip(),
                    api_key=str(api_key or "").strip(),
                    api_version=str(api_version or "").strip(),
                    temperature=float(temperature),
                )
                new_hist = list(history or []) + [(text, out)]
                return new_hist, "OK"
            except AIChatError as e:
                err = f"Error: {e}"
                new_hist = list(history or []) + [(text, err)]
                return new_hist, err
            except Exception as e:
                err = f"Error: {e}"
                new_hist = list(history or []) + [(text, err)]
                return new_hist, err

        def _ai_clear():
            return [], ""

        def _run_winget_install_from_ui(package_id: str) -> bool:
            if sys.platform != "win32":
                return False
            if not bool(shutil.which("winget") or shutil.which("winget.exe")):
                return False
            cmd = [
                "winget",
                "install",
                "-e",
                "--id",
                str(package_id),
                "--accept-source-agreements",
                "--accept-package-agreements",
            ]
            try:
                creationflags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
                subprocess.Popen(cmd, creationflags=creationflags)
                return True
            except Exception:
                return False

        def _ai_setup_local(endpoint: str, model_or_deployment: str):
            base = str(endpoint or "").strip() or "http://localhost:11434"
            model_name = str(model_or_deployment or "").strip() or "llama3.2"

            yield _pct_status(0, "checking Ollama")
            if not ollama_reachable(base_url=base, timeout_s=2.5):
                # Best-effort: offer a winget install on Windows.
                if sys.platform == "win32":
                    started = _run_winget_install_from_ui("Ollama.Ollama")
                    if started:
                        yield _pct_status(
                            5,
                            "started Ollama installer via winget — finish install, then relaunch and click setup again",
                        )
                        return
                yield _pct_status(
                    5,
                    "Ollama not reachable. Install Ollama and ensure it's running (tip: open Ollama once), then retry.",
                )
                return

            yield _pct_status(10, "listing local models")
            try:
                models = ollama_list_models(base_url=base, timeout_s=10.0)
            except Exception as e:
                yield _pct_status(10, f"failed to query models: {e}")
                return

            if model_name in models:
                yield _pct_status(100, f"ready: {model_name} is installed")
                return

            yield _pct_status(15, f"downloading model: {model_name}")
            try:
                for pct, status in ollama_pull_model(base_url=base, model=model_name):
                    # Map pull progress into 15..95 so we can finish with a clean 100%.
                    mapped = 15 + int(round((max(0, min(100, int(pct))) / 100.0) * 80.0))
                    yield _pct_status(mapped, status)
            except Exception as e:
                yield _pct_status(15, f"pull failed: {e}")
                return

            yield _pct_status(98, "verifying install")
            try:
                models2 = ollama_list_models(base_url=base, timeout_s=10.0)
                if model_name not in models2:
                    yield _pct_status(98, f"model not found after pull: {model_name}")
                    return
            except Exception as e:
                yield _pct_status(98, f"verify failed: {e}")
                return
            yield _pct_status(100, f"ready: installed {model_name}")

        ai_provider.change(fn=_ai_defaults, inputs=[ai_provider], outputs=[ai_endpoint, ai_model, ai_api_version])

        ai_setup_btn.click(
            fn=_ai_setup_local,
            inputs=[ai_endpoint, ai_model],
            outputs=[ai_status],
        )

        def _ai_reindex():
            try:
                _ = build_or_update_index(force=True)
                return "OK - index rebuilt"
            except Exception as e:
                return f"Error: {e}"

        ai_reindex_btn.click(fn=_ai_reindex, inputs=[], outputs=[ai_status])

        ai_send_btn.click(
            fn=_ai_send,
            inputs=[
                ai_provider,
                ai_endpoint,
                ai_model,
                ai_api_key,
                ai_api_version,
                ai_temperature,
                ai_include_context,
                ai_chat,
                ai_msg,
            ],
            outputs=[ai_chat, ai_status],
        )
        ai_msg.submit(
            fn=_ai_send,
            inputs=[
                ai_provider,
                ai_endpoint,
                ai_model,
                ai_api_key,
                ai_api_version,
                ai_temperature,
                ai_include_context,
                ai_chat,
                ai_msg,
            ],
            outputs=[ai_chat, ai_status],
        )
        ai_clear_btn.click(fn=_ai_clear, inputs=[], outputs=[ai_chat, ai_status])

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
