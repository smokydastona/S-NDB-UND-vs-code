from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

from .manifest import ManifestItem
from .doc_reader import UnsupportedDocumentError, read_document_text, to_prompt
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

    # Minecraft export settings
    p.add_argument("--pack-root", default="resourcepack")
    p.add_argument("--mc-target", choices=["resourcepack", "forge"], default="resourcepack")
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

    # Build the args object expected by batch.run_item
    run_args = SimpleNamespace(
        pack_root=args.pack_root,
        mc_target=args.mc_target,
        ogg_quality=args.ogg_quality,
        mc_sample_rate=args.mc_sample_rate,
        mc_channels=args.mc_channels,
        catalog=args.catalog,
        zip=None,
        device=args.device,
        model=args.model,
        rfxgen_path=args.rfxgen_path,
    )

    exported = 0
    for doc in sorted(files):
        try:
            raw = read_document_text(doc)
            prompt = to_prompt(raw)
        except UnsupportedDocumentError as e:
            print(f"Skip {doc.name}: {e}")
            continue

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
            post=bool(args.post),
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
