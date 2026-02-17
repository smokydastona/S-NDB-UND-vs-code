from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path
from types import SimpleNamespace

from .manifest import ManifestItem
from .doc_reader import UnsupportedDocumentError, extract_sound_prompts, read_document_text, to_prompt
from .batch import run_item


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Read documents from a folder and use their text as prompts to generate Minecraft-ready sounds."
    )

    p.add_argument(
        "--in",
        dest="in_dir",
        default="pre_gen_sound",
        help="Input folder containing .docx/.txt/.md files (default: pre_gen_sound)",
    )
    p.add_argument(
        "--glob",
        default="**/*",
        help="Glob pattern inside the folder (default: **/*)",
    )

    p.add_argument("--engine", choices=["rfxgen", "diffusers"], default="rfxgen")
    p.add_argument(
        "--prefer-doc-manifest",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If true (default), when a doc entry includes a generated 'Manifest' line, use its engine/seconds/"
            "sound_path/variants/etc. CLI flags act as defaults/fallbacks."
        ),
    )
    p.add_argument("--namespace", default="soundgen")
    p.add_argument(
        "--event-prefix",
        default="generated.docs",
        help="Event prefix. Final event becomes <prefix>.<file_stem>",
    )
    p.add_argument(
        "--sound-path-prefix",
        default="generated/docs",
        help="Sound path prefix. Final sound_path becomes <prefix>/<file_stem>",
    )

    p.add_argument("--variants", type=int, default=1)
    p.add_argument("--weight", type=int, default=1)
    p.add_argument("--volume", type=float, default=1.0)
    p.add_argument("--pitch", type=float, default=1.0)

    p.add_argument("--subtitle", default=None, help="Subtitle text (optional). If omitted, uses file stem.")
    p.add_argument("--post", action="store_true", help="Enable post-processing chain (recommended).")

    # Pro controls (passed through to batch.run_item)
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

    # Minecraft export settings
    p.add_argument("--pack-root", default="resourcepack")
    p.add_argument("--mc-target", choices=["resourcepack", "forge"], default="resourcepack")
    p.add_argument(
        "--pack-per-doc",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If true (default), each input document is exported into its own fresh pack folder under --pack-root. "
            "Example: --pack-root resourcepack => resourcepack/<doc_stem>/."
        ),
    )
    p.add_argument("--ogg-quality", type=int, default=5)
    p.add_argument("--mc-sample-rate", type=int, default=44100)
    p.add_argument("--mc-channels", type=int, default=1)

    # Engine-specific
    p.add_argument("--seconds", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--preset", default=None, help="rfxgen preset override")
    p.add_argument("--rfxgen-path", default=None)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--model", default="cvssp/audioldm2")

    # Catalog
    p.add_argument("--catalog", default="library/catalog.jsonl")

    return p


def _slug(s: str) -> str:
    import re

    s = s.strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_\-.]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:64] or "doc"


def _stable_seed(*parts: str) -> int:
    """Deterministic seed from text parts (keeps families unique and repeatable)."""

    h = hashlib.sha256("|".join(p for p in parts if p).encode("utf-8", errors="ignore")).hexdigest()
    return int(h[:8], 16)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    in_dir = Path(args.in_dir)
    if not in_dir.exists():
        raise SystemExit(f"Input folder not found: {in_dir}")

    files = [p for p in in_dir.glob(args.glob) if p.is_file()]
    files = [p for p in files if p.suffix.lower() in {".docx", ".txt", ".md"}]

    if not files:
        print(f"No supported docs found in {in_dir} (glob={args.glob})")
        return 0

    # Build the base args object expected by batch.run_item
    base_pack_root = Path(args.pack_root)
    run_args = SimpleNamespace(
        pack_root=str(base_pack_root),
        mc_target=args.mc_target,
        ogg_quality=args.ogg_quality,
        mc_sample_rate=args.mc_sample_rate,
        mc_channels=args.mc_channels,
        catalog=args.catalog,
        zip=None,
        device=args.device,
        model=args.model,
        rfxgen_path=args.rfxgen_path,
        # Pro controls
        polish=bool(args.polish),
        emotion=str(args.emotion),
        intensity=float(args.intensity),
        variation=float(args.variation),
        pitch_contour=str(args.pitch_contour),
        multiband=bool(args.multiband),
        mb_low_hz=float(args.mb_low_hz),
        mb_high_hz=float(args.mb_high_hz),
        mb_low_gain_db=float(args.mb_low_gain_db),
        mb_mid_gain_db=float(args.mb_mid_gain_db),
        mb_high_gain_db=float(args.mb_high_gain_db),
        mb_comp_threshold_db=(float(args.mb_comp_threshold_db) if args.mb_comp_threshold_db is not None else None),
        mb_comp_ratio=float(args.mb_comp_ratio),
        creature_size=float(args.creature_size),
        formant_shift=float(args.formant_shift),
        texture_preset=str(args.texture_preset),
        texture_amount=float(args.texture_amount),
        texture_grain_ms=float(args.texture_grain_ms),
        texture_spray=float(args.texture_spray),
        reverb=str(args.reverb),
        reverb_mix=float(args.reverb_mix),
        reverb_time=float(args.reverb_time),
    )

    exported = 0
    for doc in sorted(files):
        # Optionally create a fresh pack root per document.
        if bool(args.pack_per_doc):
            doc_pack_root = base_pack_root / _slug(doc.stem)
            if doc_pack_root.exists():
                shutil.rmtree(doc_pack_root, ignore_errors=True)
            doc_pack_root.mkdir(parents=True, exist_ok=True)
            run_args.pack_root = str(doc_pack_root)
            print(f"Pack root (doc): {doc_pack_root}")
        else:
            run_args.pack_root = str(base_pack_root)

        try:
            raw = read_document_text(doc)
        except UnsupportedDocumentError as e:
            print(f"Skip {doc.name}: {e}")
            continue

        # Multi-entry docs: generate an entire family of sounds.
        entries = extract_sound_prompts(raw)
        if entries:
            print(f"Doc entries: {doc.name} -> {len(entries)} sounds")
            for e in entries:
                # If the doc provides a namespace and the user didn't override the default, adopt it.
                namespace = args.namespace
                if namespace == "soundgen" and e.get("namespace"):
                    namespace = str(e["namespace"]).strip()

                event = str(e.get("event") or "").strip()
                if not event:
                    continue

                # Prefer doc-provided manifest settings when available.
                prefer_doc = bool(args.prefer_doc_manifest)
                engine = (str(e.get("engine") or "").strip() if prefer_doc else "") or args.engine
                seconds_s = (str(e.get("seconds") or "").strip() if prefer_doc else "")
                variants_s = (str(e.get("variants") or "").strip() if prefer_doc else "")
                sound_path_from_doc = (str(e.get("sound_path") or "").strip() if prefer_doc else "")

                # sound_path: doc manifest wins; else computed from event.
                if sound_path_from_doc:
                    sound_path = sound_path_from_doc
                else:
                    event_for_path = event.replace(":", ".")
                    sound_path_prefix = (args.sound_path_prefix or "generated/docs")
                    sound_path = f"{sound_path_prefix}/{event_for_path.replace('.', '/')}"

                # Subtitles: prefer explicit doc subtitle text/key if present.
                subtitle_key = (str(e.get("subtitle_key") or "").strip() if prefer_doc else "") or None
                subtitle = args.subtitle or (str(e.get("subtitle") or "").strip() if prefer_doc else "")
                subtitle = subtitle or (str(e.get("title") or "").strip() if prefer_doc else "") or doc.stem

                # Seeds: if user didn't pass --seed, derive a stable unique seed per (namespace,event).
                seed = args.seed
                if seed is None:
                    seed_s = (str(e.get("seed") or "").strip() if prefer_doc else "")
                    if seed_s:
                        try:
                            seed = int(seed_s)
                        except ValueError:
                            seed = None
                if seed is None:
                    seed = _stable_seed(namespace, event)

                seconds = float(args.seconds)
                if seconds_s:
                    try:
                        seconds = float(seconds_s)
                    except ValueError:
                        pass

                variants = max(1, int(args.variants))
                if variants_s:
                    try:
                        variants = max(1, int(float(variants_s)))
                    except ValueError:
                        pass

                # Per-entry overrides (optional).
                preset = args.preset
                if prefer_doc and e.get("preset"):
                    preset = str(e.get("preset") or "").strip() or preset

                item = ManifestItem(
                    prompt=str(e.get("prompt") or "").strip(),
                    engine=engine,
                    namespace=namespace,
                    event=event,
                    sound_path=sound_path,
                    seconds=float(seconds),
                    seed=int(seed) if seed is not None else None,
                    preset=preset,
                    variants=int(variants),
                    weight=max(1, int(args.weight)),
                    volume=float(args.volume),
                    pitch=float(args.pitch),
                    subtitle=str(subtitle) if subtitle else None,
                    subtitle_key=subtitle_key,
                    post=bool(args.post or args.polish),
                    tags=("from_doc", doc.suffix.lower().lstrip(".")),
                )

                if not item.prompt:
                    continue

                outs = run_item(item, args=run_args)
                exported += len(outs)
                for o in outs:
                    print(f"Wrote {o}")
                print(f"Playsound: /playsound {namespace}:{event} master @s")
            continue

        # Fallback: treat the full doc as one prompt.
        prompt = to_prompt(raw)
        if not prompt:
            print(f"Skip {doc.name}: empty prompt")
            continue

        stem = _slug(doc.stem)
        event = f"{args.event_prefix}.{stem}" if args.event_prefix else stem
        sound_path = f"{args.sound_path_prefix}/{stem}" if args.sound_path_prefix else stem
        subtitle = args.subtitle or doc.stem

        item = ManifestItem(
            prompt=prompt,
            engine=args.engine,
            namespace=args.namespace,
            event=event,
            sound_path=sound_path,
            seconds=float(args.seconds),
            seed=args.seed,
            preset=args.preset,
            variants=max(1, int(args.variants)),
            weight=max(1, int(args.weight)),
            volume=float(args.volume),
            pitch=float(args.pitch),
            subtitle=subtitle,
            post=bool(args.post or args.polish),
            tags=("from_doc", doc.suffix.lower().lstrip(".")),
        )

        outs = run_item(item, args=run_args)
        exported += len(outs)
        for o in outs:
            print(f"Wrote {o}")
        print(f"Playsound: /playsound {args.namespace}:{event} master @s")

    print(f"Done. Exported {exported} file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
