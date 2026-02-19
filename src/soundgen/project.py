from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .credits import upsert_pack_credits
from .minecraft import export_wav_to_minecraft_pack, sanitize_id


PROJECT_FILENAME = "sndbund_project.json"
PROJECT_SCHEMA_VERSION = 1


def _now_utc_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _read_json(path: Path) -> dict[str, Any]:
    from .json_utils import JsonParseError, loads_json_lenient

    try:
        data = loads_json_lenient(path.read_text(encoding="utf-8"), context=f"project JSON ({path})")
    except JsonParseError as e:
        raise ValueError(str(e)) from e
    return data if isinstance(data, dict) else {}


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _project_path(project_root: Path) -> Path:
    return Path(project_root) / PROJECT_FILENAME


def load_project(project_root: Path) -> dict[str, Any]:
    path = _project_path(project_root)
    if not path.exists():
        raise FileNotFoundError(f"Project file not found: {path}")
    proj = _read_json(path)
    if int(proj.get("schema_version", 0)) != PROJECT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported project schema_version={proj.get('schema_version')!r}; expected {PROJECT_SCHEMA_VERSION}"
        )
    if not isinstance(proj.get("items"), list):
        proj["items"] = []
    return proj


def save_project(project_root: Path, project: dict[str, Any]) -> Path:
    path = _project_path(project_root)
    _write_json(path, project)
    return path


def _safe_item_key(item_id: str) -> str:
    # For filesystem paths, not Minecraft IDs.
    t = str(item_id or "").strip().replace("\\", "/")
    t = t.replace("/", "_").replace(".", "_")
    t = "".join((c if (c.isalnum() or c in {"_", "-"}) else "_") for c in t)
    t = "_".join([p for p in t.split("_") if p])
    return t or "item"


def _find_item(project: dict[str, Any], item_id: str) -> dict[str, Any]:
    want = str(item_id).strip()
    for it in project.get("items", []):
        if isinstance(it, dict) and str(it.get("id", "")) == want:
            return it
    raise KeyError(f"Unknown item id: {item_id}")


def _next_item_version(item: dict[str, Any]) -> int:
    versions = item.get("versions")
    if not isinstance(versions, list) or not versions:
        return 1
    mx = 0
    for v in versions:
        if isinstance(v, dict):
            try:
                mx = max(mx, int(v.get("version", 0)))
            except Exception:
                pass
    return int(mx) + 1


@dataclass(frozen=True)
class _BuildTarget:
    namespace: str
    pack_root: Path


def _repo_root() -> Path:
    # src/soundgen/project.py -> repo root is 3 levels up
    return Path(__file__).resolve().parents[2]


def _run_generate(*, argv: list[str]) -> None:
    cmd = [sys.executable, "-m", "soundgen.generate", *argv]
    completed = subprocess.run(cmd, cwd=str(_repo_root()), capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "soundgen.generate failed:\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout: {completed.stdout}\n"
            f"stderr: {completed.stderr}"
        )


def _rel(project_root: Path, path: Path) -> str:
    return str(Path(path).resolve().relative_to(Path(project_root).resolve())).replace("\\", "/")


def _maybe_read_credits(wav_path: Path) -> dict[str, Any] | None:
    from .json_utils import load_json_file_lenient

    sidecar = Path(str(wav_path) + ".credits.json")
    if not sidecar.exists():
        return None
    try:
        data = load_json_file_lenient(sidecar, context=f"Credits sidecar JSON file: {sidecar}")
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _write_credits_sidecar(wav_path: Path, credits: dict[str, Any]) -> None:
    sidecar = Path(str(wav_path) + ".credits.json")
    sidecar.write_text(json.dumps(credits, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def create_project(
    *,
    project_root: Path,
    kind: str,
    project_id: str,
    title: str,
    namespace: str,
    pack_root: str,
    mob: str | None = None,
    event_prefix: str | None = None,
    style: str | None = None,
    subtitle_base: str | None = None,
) -> Path:
    project_root = Path(project_root)
    project_root.mkdir(parents=True, exist_ok=True)

    kind_s = str(kind).strip().lower()
    if kind_s not in {"soundpack", "mob"}:
        raise ValueError("--kind must be 'soundpack' or 'mob'")

    proj: dict[str, Any] = {
        "schema_version": PROJECT_SCHEMA_VERSION,
        "project_id": str(project_id).strip() or "project",
        "kind": kind_s,
        "title": str(title).strip() or str(project_id).strip() or "SÖNDBÖUND Project",
        "created_utc": _now_utc_iso(),
        "namespace": sanitize_id(str(namespace).strip() or "soundgen", kind="namespace"),
        "pack_root": str(pack_root).strip() or "resourcepack",
        "items": [],
    }

    if kind_s == "mob":
        mob_s = str(mob or "").strip()
        if not mob_s:
            raise ValueError("--mob is required for --kind mob")
        mob_slug = sanitize_id(mob_s, kind="mob")
        evp = str(event_prefix or "").strip() or f"entity.{mob_slug}"
        proj["mob"] = {
            "mob": mob_slug,
            "event_prefix": sanitize_id(evp, kind="event"),
            "style": str(style or "").strip(),
            "subtitle_base": str(subtitle_base or "").strip(),
            "types": ["hurt", "death", "ambient", "step"],
        }

    return save_project(project_root, proj)


def add_item_soundpack(
    *,
    project_root: Path,
    item_id: str,
    category: str,
    engine: str,
    prompt: str,
    seconds: float,
    event: str,
    sound_path: str,
    subtitle: str | None,
    variants: int,
    seed: int | None,
    generate_args: list[str],
) -> Path:
    project_root = Path(project_root)
    proj = load_project(project_root)
    if str(proj.get("kind")) != "soundpack":
        raise ValueError("This project is not a soundpack project")

    item_id_s = str(item_id).strip()
    if not item_id_s:
        raise ValueError("--id is required")

    for it in proj.get("items", []):
        if isinstance(it, dict) and str(it.get("id", "")) == item_id_s:
            raise ValueError(f"Item already exists: {item_id_s}")

    ns = sanitize_id(str(proj.get("namespace", "soundgen")), kind="namespace")
    event_s = sanitize_id(str(event).strip(), kind="event")
    sound_path_s = sanitize_id(str(sound_path).strip(), kind="sound_path")

    it: dict[str, Any] = {
        "id": item_id_s,
        "category": str(category or "").strip(),
        "engine": str(engine).strip() or "rfxgen",
        "prompt": str(prompt).strip(),
        "seconds": float(seconds),
        "seed": (int(seed) if seed is not None else None),
        "namespace": ns,
        "event": event_s,
        "sound_path": sound_path_s,
        "subtitle": (str(subtitle).strip() if subtitle else None),
        "variants": max(1, int(variants)),
        "generate_args": list(generate_args or []),
        "active_version": None,
        "versions": [],
    }

    proj["items"].append(it)
    return save_project(project_root, proj)


def _build_target(project_root: Path, proj: dict[str, Any]) -> _BuildTarget:
    namespace = sanitize_id(str(proj.get("namespace", "soundgen")), kind="namespace")
    pack_root = Path(project_root) / str(proj.get("pack_root", "resourcepack"))
    return _BuildTarget(namespace=namespace, pack_root=pack_root)


def build_item(*, project_root: Path, item: dict[str, Any]) -> dict[str, Any]:
    project_root = Path(project_root)
    proj = load_project(project_root)
    tgt = _build_target(project_root, proj)

    item_id = str(item.get("id", "")).strip() or "item"
    item_key = _safe_item_key(item_id)
    version = _next_item_version(item)
    version_dir = project_root / "project_audio" / item_key / f"v{version:04d}"
    version_dir.mkdir(parents=True, exist_ok=True)

    engine = str(item.get("engine", "rfxgen") or "rfxgen")
    prompt = str(item.get("prompt", "")).strip()
    seconds = float(item.get("seconds", 1.0))
    event = sanitize_id(str(item.get("event", "generated.project")), kind="event")
    base_sound_path = sanitize_id(str(item.get("sound_path", f"generated/{item_key}")), kind="sound_path")
    variants = max(1, int(item.get("variants", 1)))
    subtitle = str(item.get("subtitle") or "").strip() or None
    seed = item.get("seed")

    extra_args = item.get("generate_args")
    extra_args_list = [str(x) for x in (extra_args if isinstance(extra_args, list) else [])]

    # Generate + export to pack in one call; keep WAVs in version_dir.
    argv = [
        "--engine",
        engine,
        "--prompt",
        prompt,
        "--seconds",
        str(seconds),
        "--minecraft",
        "--pack-root",
        str(tgt.pack_root),
        "--namespace",
        tgt.namespace,
        "--event",
        event,
        "--sound-path",
        base_sound_path,
        "--variants",
        str(variants),
        "--mc-wav-dir",
        str(version_dir),
        "--out",
        str(version_dir / "_unused.wav"),
    ]
    if seed is not None:
        argv += ["--seed", str(int(seed))]
    if subtitle:
        argv += ["--subtitle", subtitle]

    argv += extra_args_list

    _run_generate(argv=argv)

    wavs = sorted(version_dir.glob("*.wav"))
    if not wavs:
        raise FileNotFoundError(f"No WAVs written to {version_dir}")

    version_rec: dict[str, Any] = {
        "version": int(version),
        "created_utc": _now_utc_iso(),
        "type": "generate",
        "wav_files": [_rel(project_root, p) for p in wavs],
        "ogg_files": [],
        "credits_files": [],
    }

    # Update per-wav credits to include project and pack metadata.
    for variant_index, wav_path in enumerate(wavs):
        credits = _maybe_read_credits(wav_path) or {}
        credits["project_id"] = str(proj.get("project_id"))
        credits["project_kind"] = str(proj.get("kind"))
        credits["project_item_id"] = item_id
        credits["item_version"] = int(version)
        credits["namespace"] = tgt.namespace

        # Match generate.py Minecraft naming convention: suffixes in sound_path.
        suffix = f"_{variant_index+1:02d}" if variants > 1 else ""
        sp_full = f"{base_sound_path}{suffix}"

        credits["event"] = event
        credits["sound_path"] = sp_full

        _write_credits_sidecar(wav_path, credits)
        version_rec["credits_files"].append(_rel(project_root, Path(str(wav_path) + ".credits.json")))

        # Pack credits lives inside assets/<ns>/soundgen_credits.json
        upsert_pack_credits(
            pack_root=tgt.pack_root,
            namespace=tgt.namespace,
            event=event,
            sound_path=sp_full,
            credits=credits,
        )

        ogg_path = tgt.pack_root / "assets" / tgt.namespace / "sounds" / f"{sp_full}.ogg"
        if ogg_path.exists():
            version_rec["ogg_files"].append(_rel(project_root, ogg_path))

    versions = item.get("versions")
    if not isinstance(versions, list):
        versions = []
    versions.append(version_rec)
    item["versions"] = versions
    item["active_version"] = int(version)

    # Persist back to project file.
    for idx, it in enumerate(proj.get("items", [])):
        if isinstance(it, dict) and str(it.get("id")) == item_id:
            proj["items"][idx] = item
            break

    save_project(project_root, proj)
    return version_rec


def import_edit(
    *,
    project_root: Path,
    item_id: str,
    wav_path: Path,
    notes: str | None,
) -> dict[str, Any]:
    project_root = Path(project_root)
    proj = load_project(project_root)
    item = _find_item(proj, item_id)
    tgt = _build_target(project_root, proj)

    version = _next_item_version(item)
    item_key = _safe_item_key(item_id)
    version_dir = project_root / "project_audio" / item_key / f"v{version:04d}"
    version_dir.mkdir(parents=True, exist_ok=True)

    src = Path(wav_path)
    if not src.exists():
        raise FileNotFoundError(str(src))

    dst = version_dir / f"edit.wav"
    shutil.copy2(src, dst)

    # Carry credits forward if present.
    src_credits = _maybe_read_credits(src) or {}
    src_credits["project_id"] = str(proj.get("project_id"))
    src_credits["project_kind"] = str(proj.get("kind"))
    src_credits["project_item_id"] = str(item_id)
    src_credits["item_version"] = int(version)

    event = sanitize_id(str(item.get("event", "generated.project")), kind="event")
    base_sound_path = sanitize_id(str(item.get("sound_path", f"generated/{item_key}")), kind="sound_path")

    src_credits["namespace"] = tgt.namespace
    src_credits["event"] = event
    src_credits["sound_path"] = base_sound_path
    _write_credits_sidecar(dst, src_credits)

    export_wav_to_minecraft_pack(
        dst,
        pack_root=tgt.pack_root,
        namespace=tgt.namespace,
        event=event,
        sound_path=base_sound_path,
        subtitle=(str(item.get("subtitle")).strip() if item.get("subtitle") else None),
        subtitle_key=(str(item.get("subtitle_key")).strip() if item.get("subtitle_key") else None),
        ogg_quality=int(item.get("ogg_quality", 5) or 5),
        sample_rate=int(item.get("mc_sample_rate", 44100) or 44100),
        channels=int(item.get("mc_channels", 1) or 1),
        weight=int(item.get("weight", 1) or 1),
        volume=float(item.get("volume", 1.0) or 1.0),
        pitch=float(item.get("pitch", 1.0) or 1.0),
        description=str(proj.get("title", "SÖNDBÖUND project")),
        write_pack_mcmeta=True,
    )

    upsert_pack_credits(
        pack_root=tgt.pack_root,
        namespace=tgt.namespace,
        event=event,
        sound_path=base_sound_path,
        credits=src_credits,
    )

    version_rec: dict[str, Any] = {
        "version": int(version),
        "created_utc": _now_utc_iso(),
        "type": "edit",
        "notes": (str(notes).strip() if notes else None),
        "wav_files": [_rel(project_root, dst)],
        "ogg_files": [_rel(project_root, tgt.pack_root / "assets" / tgt.namespace / "sounds" / f"{base_sound_path}.ogg")],
        "credits_files": [_rel(project_root, Path(str(dst) + ".credits.json"))],
    }

    versions = item.get("versions")
    if not isinstance(versions, list):
        versions = []
    versions.append(version_rec)
    item["versions"] = versions
    item["active_version"] = int(version)

    for idx, it in enumerate(proj.get("items", [])):
        if isinstance(it, dict) and str(it.get("id")) == str(item_id):
            proj["items"][idx] = item
            break

    save_project(project_root, proj)
    return version_rec


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "SÖNDBÖUND Project System (v2.4)\n\n"
            "A project tracks sounds, versions (generated + edited), and supports Minecraft-ready export."
        )
    )

    sub = p.add_subparsers(dest="cmd", required=True)

    create = sub.add_parser("create", help="Create a new project folder")
    create.add_argument("--root", required=True, help="Project root folder")
    create.add_argument("--kind", choices=["soundpack", "mob"], default="soundpack")
    create.add_argument("--id", required=True, help="Project id")
    create.add_argument("--title", default="", help="Project title")
    create.add_argument("--namespace", default="soundgen", help="Minecraft namespace (modid)")
    create.add_argument("--pack-root", default="resourcepack", help="Pack root folder (relative to project root)")

    # Mob project fields
    create.add_argument("--mob", default="", help="For --kind mob: mob name")
    create.add_argument("--event-prefix", default="", help="For --kind mob: sounds.json event prefix")
    create.add_argument("--style", default="", help="For --kind mob: optional style appended to prompts")
    create.add_argument("--subtitle-base", default="", help="For --kind mob: subtitle base (optional)")

    add = sub.add_parser("add", help="Add a sound item to a soundpack project")
    add.add_argument("--root", required=True, help="Project root folder")
    add.add_argument("--id", required=True, help="Item id (project-local identifier)")
    add.add_argument("--category", default="", help="Optional category (ui/ambience/creature/etc)")
    add.add_argument("--engine", default="rfxgen", help="Engine (default rfxgen)")
    add.add_argument("--prompt", required=True, help="Prompt")
    add.add_argument("--seconds", type=float, default=1.0, help="Duration seconds")
    add.add_argument("--seed", type=int, default=None, help="Base seed (optional)")
    add.add_argument("--event", required=True, help="Minecraft sounds.json event key")
    add.add_argument("--sound-path", required=True, help="Path under sounds/ without extension")
    add.add_argument("--subtitle", default="", help="Subtitle (optional)")
    add.add_argument("--variants", type=int, default=1, help="Variants per item")
    add.add_argument(
        "--generate-arg",
        action="append",
        default=[],
        help="Extra args forwarded to soundgen.generate (repeatable, e.g. --generate-arg --post)",
    )

    ls = sub.add_parser("list", help="List items")
    ls.add_argument("--root", required=True)

    build = sub.add_parser("build", help="Generate (or regenerate) items and export into the project pack")
    build.add_argument("--root", required=True)
    build.add_argument("--id", default="", help="If set, only build this item")

    edit = sub.add_parser("edit", help="Open the editor on the active version of an item")
    edit.add_argument("--root", required=True)
    edit.add_argument("--id", required=True)

    imp = sub.add_parser("import-edit", help="Import an externally edited WAV as a new version and export it")
    imp.add_argument("--root", required=True)
    imp.add_argument("--id", required=True)
    imp.add_argument("--wav", required=True, help="Path to edited wav")
    imp.add_argument("--notes", default="", help="Optional notes")

    return p


def _cmd_create(args: argparse.Namespace) -> int:
    out = create_project(
        project_root=Path(args.root),
        kind=str(args.kind),
        project_id=str(args.id),
        title=str(args.title),
        namespace=str(args.namespace),
        pack_root=str(args.pack_root),
        mob=(str(args.mob) if str(args.mob).strip() else None),
        event_prefix=(str(args.event_prefix) if str(args.event_prefix).strip() else None),
        style=(str(args.style) if str(args.style).strip() else None),
        subtitle_base=(str(args.subtitle_base) if str(args.subtitle_base).strip() else None),
    )
    print(f"Wrote {out}")
    return 0


def _cmd_add(args: argparse.Namespace) -> int:
    out = add_item_soundpack(
        project_root=Path(args.root),
        item_id=str(args.id),
        category=str(args.category),
        engine=str(args.engine),
        prompt=str(args.prompt),
        seconds=float(args.seconds),
        event=str(args.event),
        sound_path=str(args.sound_path),
        subtitle=(str(args.subtitle).strip() if str(args.subtitle).strip() else None),
        variants=int(args.variants),
        seed=(int(args.seed) if args.seed is not None else None),
        generate_args=[str(x) for x in (args.generate_arg or [])],
    )
    print(f"Updated {out}")
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    proj = load_project(Path(args.root))
    print(f"Project: {proj.get('project_id')} ({proj.get('kind')})")
    for it in proj.get("items", []):
        if not isinstance(it, dict):
            continue
        vid = it.get("active_version")
        print(f"- {it.get('id')}  engine={it.get('engine')}  event={it.get('event')}  active_v={vid}")
    return 0


def _cmd_build(args: argparse.Namespace) -> int:
    project_root = Path(args.root)
    proj = load_project(project_root)

    if str(proj.get("kind")) == "mob":
        mob_cfg = proj.get("mob") if isinstance(proj.get("mob"), dict) else {}
        mob = str(mob_cfg.get("mob", ""))
        if not mob:
            raise ValueError("Mob project missing mob config")

        from .mob_soundset import _prompt_for  # intentional reuse of canonical prompts

        event_prefix = str(mob_cfg.get("event_prefix") or f"entity.{mob}")
        style = str(mob_cfg.get("style") or "")
        subtitle_base = str(mob_cfg.get("subtitle_base") or "").strip()
        kinds = mob_cfg.get("types") if isinstance(mob_cfg.get("types"), list) else ["hurt", "death", "ambient", "step"]

        # Ensure items exist.
        existing: set[str] = set()
        for it in proj.get("items", []):
            if isinstance(it, dict) and isinstance(it.get("id"), str):
                existing.add(it["id"])

        for kind in kinds:
            kind_s = str(kind).strip().lower()
            if not kind_s or kind_s in existing:
                continue
            prompt, seconds = _prompt_for(mob, kind_s, style)
            proj["items"].append(
                {
                    "id": kind_s,
                    "category": "mob",
                    "engine": str(mob_cfg.get("engine") or "layered"),
                    "prompt": prompt,
                    "seconds": float(seconds),
                    "seed": int(mob_cfg.get("seed", 1337)) if mob_cfg.get("seed") is not None else 1337,
                    "namespace": str(proj.get("namespace", "soundgen")),
                    "event": f"{event_prefix}.{kind_s}",
                    "sound_path": f"entity/{mob}/{kind_s}",
                    "subtitle": (f"{subtitle_base} {kind_s}".strip() if subtitle_base else None),
                    "variants": int(mob_cfg.get("variants", 4) or 4),
                    "generate_args": list(mob_cfg.get("generate_args", [])) if isinstance(mob_cfg.get("generate_args"), list) else [],
                    "active_version": None,
                    "versions": [],
                }
            )

        save_project(project_root, proj)

    if str(args.id).strip():
        item = _find_item(proj, str(args.id).strip())
        rec = build_item(project_root=project_root, item=item)
        print(f"Built {args.id}: v{rec['version']:04d}")
        return 0

    built = 0
    for it in list(proj.get("items", [])):
        if not isinstance(it, dict) or not str(it.get("id", "")).strip():
            continue
        rec = build_item(project_root=project_root, item=it)
        print(f"Built {it.get('id')}: v{rec['version']:04d}")
        built += 1

    print(f"Done. Built {built} item(s).")
    return 0


def _cmd_edit(args: argparse.Namespace) -> int:
    project_root = Path(args.root)
    proj = load_project(project_root)
    item = _find_item(proj, str(args.id))

    active = item.get("active_version")
    versions = item.get("versions") if isinstance(item.get("versions"), list) else []
    if active is None:
        raise ValueError("Item has no active_version. Run 'build' first.")

    best: dict[str, Any] | None = None
    for v in versions:
        if isinstance(v, dict) and int(v.get("version", -1)) == int(active):
            best = v
            break
    if not best:
        raise ValueError("Active version record not found")

    wavs = best.get("wav_files") if isinstance(best.get("wav_files"), list) else []
    if not wavs:
        raise ValueError("Active version has no wav_files")

    wav_path = project_root / str(wavs[0])
    if not wav_path.exists():
        raise FileNotFoundError(str(wav_path))

    from .editor import launch_editor

    launch_editor(wav_path)
    return 0


def _cmd_import_edit(args: argparse.Namespace) -> int:
    rec = import_edit(
        project_root=Path(args.root),
        item_id=str(args.id),
        wav_path=Path(args.wav),
        notes=(str(args.notes) if str(args.notes).strip() else None),
    )
    print(f"Imported edit: v{rec['version']:04d}")
    return 0


def run_project(argv: list[str] | None = None) -> int:
    p = build_parser()
    args = p.parse_args(argv)

    if args.cmd == "create":
        return _cmd_create(args)
    if args.cmd == "add":
        return _cmd_add(args)
    if args.cmd == "list":
        return _cmd_list(args)
    if args.cmd == "build":
        return _cmd_build(args)
    if args.cmd == "edit":
        return _cmd_edit(args)
    if args.cmd == "import-edit":
        return _cmd_import_edit(args)

    raise SystemExit(2)


def main() -> None:
    raise SystemExit(run_project())


if __name__ == "__main__":
    main()
