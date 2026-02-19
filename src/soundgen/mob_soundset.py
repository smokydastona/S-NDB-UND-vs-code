from __future__ import annotations

import argparse
import json
from pathlib import Path

from .engine_registry import available_engines
from .json_utils import load_json_file_lenient


def _slug(s: str) -> str:
    import re

    t = str(s or "").strip().lower()
    t = re.sub(r"\s+", "_", t)
    t = re.sub(r"[^a-z0-9_\-.]+", "_", t)
    t = re.sub(r"_+", "_", t).strip("_")
    return t or "mob"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Generate a complete Minecraft mob soundset (hurt/death/ambient/step) into a resource pack.\n"
            "This writes .ogg files + updates sounds.json via the existing Minecraft exporter."
        )
    )

    p.add_argument("--mob", required=True, help="Mob name (e.g. zombie, skeleton, slime).")
    p.add_argument(
        "--engine",
        choices=available_engines(),
        default="layered",
        help="Engine to use for all generated sounds (default layered).",
    )

    p.add_argument("--pack-root", default="resourcepack", help="Pack root folder to write into.")
    p.add_argument(
        "--mc-target",
        choices=["resourcepack", "forge"],
        default="resourcepack",
        help="Export target layout.",
    )
    p.add_argument("--namespace", default="soundgen", help="Minecraft namespace (modid).")
    p.add_argument(
        "--event-prefix",
        default="",
        help=(
            "Event prefix for sounds.json keys. Default: entity.<mob> (vanilla-style). "
            "Example: entity.zombie"
        ),
    )

    p.add_argument("--variants", type=int, default=4, help="Variants per sound type (default 4).")
    p.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Base seed (used to derive per-type seeds; default 1337).",
    )
    p.add_argument(
        "--types",
        default="hurt,death,ambient,step",
        help="Comma list of types to generate (hurt,death,ambient,step).",
    )
    p.add_argument(
        "--style",
        default="",
        help="Optional style text appended to prompts (e.g. 'pixel art, retro, crunchy').",
    )
    p.add_argument(
        "--subtitle-base",
        default="",
        help="Subtitle base text (optional). If set, writes per-type subtitles.",
    )

    p.add_argument(
        "--show-snippet",
        action="store_true",
        help="Print a JSON snippet for the generated sounds.json entries.",
    )

    # Pass-through args to soundgen.generate (everything after --)
    p.add_argument(
        "generate_args",
        nargs=argparse.REMAINDER,
        help="Extra args passed to 'soundgen.generate' after a '--' separator.",
    )
    return p


def _prompt_for(mob: str, kind: str, style: str) -> tuple[str, float]:
    mob_s = str(mob).strip()
    style_s = str(style or "").strip()
    suf = f", {style_s}" if style_s else ""

    if kind == "hurt":
        return f"{mob_s} hurt sound, pain grunt, short, clean transient{suf}", 1.2
    if kind == "death":
        return f"{mob_s} death sound, dying groan, impactful, no music{suf}", 2.6
    if kind == "ambient":
        return f"{mob_s} ambient idle sound, breathing, subtle texture, loopable{suf}", 6.0
    if kind == "step":
        return f"{mob_s} footsteps, short, dry, game-ready{suf}", 0.6
    return f"{mob_s} sound{suf}", 2.0


def run_mob_soundset(argv: list[str] | None = None) -> int:
    p = build_parser()
    args = p.parse_args(argv)

    mob = _slug(args.mob)
    ns = str(args.namespace).strip() or "soundgen"
    pack_root = str(args.pack_root).strip() or "resourcepack"
    mc_target = str(args.mc_target).strip() or "resourcepack"
    engine = str(args.engine).strip() or "layered"

    event_prefix = str(args.event_prefix).strip()
    if not event_prefix:
        event_prefix = f"entity.{mob}"

    wanted = [t.strip().lower() for t in str(args.types).split(",") if t.strip()]
    allowed = {"hurt", "death", "ambient", "step"}
    kinds = [t for t in wanted if t in allowed]
    if not kinds:
        raise SystemExit("No valid --types provided. Use a comma list of: hurt,death,ambient,step")

    variants = max(1, int(args.variants))
    seed_base = int(args.seed)
    style = str(args.style or "")
    subtitle_base = str(args.subtitle_base or "").strip()

    # Normalize pass-through args: argparse.REMAINDER includes the leading "--" if present.
    forward: list[str] = [str(x) for x in (args.generate_args or [])]
    if forward and forward[0] == "--":
        forward = forward[1:]

    from .generate import main as generate_main

    # Use large per-kind offsets so variant seeds won't collide across kinds.
    offsets = {"hurt": 10000, "death": 20000, "ambient": 30000, "step": 40000}

    for kind in kinds:
        prompt, seconds = _prompt_for(mob, kind, style)
        ev = f"{event_prefix}.{kind}"
        sub = f"{subtitle_base} {kind}".strip() if subtitle_base else None
        seed_kind = seed_base + int(offsets.get(kind, 0))

        gen_args = list(forward)
        gen_args += [
            "--engine",
            engine,
            "--prompt",
            prompt,
            "--seconds",
            str(seconds),
            "--minecraft",
            "--mc-target",
            mc_target,
            "--pack-root",
            pack_root,
            "--namespace",
            ns,
            "--event",
            ev,
            "--variants",
            str(variants),
            "--seed",
            str(seed_kind),
        ]
        if sub:
            gen_args += ["--subtitle", sub]

        print(f"Generating {kind}: {ns}:{ev} ({variants} variants)")
        generate_main(gen_args)

    if args.show_snippet:
        sounds_json = Path(pack_root) / "assets" / ns / "sounds.json"
        if sounds_json.exists():
            try:
                data = load_json_file_lenient(sounds_json, context=f"Minecraft sounds.json file: {sounds_json}")
            except Exception:
                data = {}
            if isinstance(data, dict):
                snippet = {k: v for k, v in data.items() if str(k).startswith(f"{event_prefix}.")}
                print("\n--- sounds.json snippet ---")
                print(json.dumps(snippet, indent=2, ensure_ascii=False))
        else:
            print(f"(No sounds.json found at {sounds_json})")

    print("Done.")
    return 0


def main() -> None:
    raise SystemExit(run_mob_soundset())


if __name__ == "__main__":
    main()
