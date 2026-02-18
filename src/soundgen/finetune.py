from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .model_versions import ModelVersion


@dataclass(frozen=True)
class TrainArgs:
    dataset: Path
    out_dir: Path
    base_model: str | None
    trainer: str
    extra: list[str]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Fine-tuning support (optional).\n\n"
            "This repo intentionally does NOT bundle a full training framework. Instead it provides:\n"
            "- Dataset export: python -m soundgen.creature_finetune prepare ...\n"
            "- Named model versions (LoRA references): configs/model_versions.json\n"
            "- A thin wrapper to launch an external trainer (optional).\n\n"
            "Use docs/creature_family_training_windows.md for a practical Windows/WSL recipe."
        )
    )

    sub = p.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train", help="Run an external LoRA trainer (wrapper)")
    tr.add_argument("--dataset", required=True, help="Dataset folder containing audio/ and metadata.jsonl")
    tr.add_argument("--out", required=True, help="Output folder for the trainer")
    tr.add_argument(
        "--trainer",
        default="external",
        choices=["external"],
        help="Trainer type (this repo only provides an external wrapper).",
    )
    tr.add_argument(
        "--base-model",
        default=None,
        help="Optional base model id/path for the trainer (depends on your trainer stack).",
    )
    tr.add_argument(
        "--",
        dest="extra",
        nargs=argparse.REMAINDER,
        help="Additional arguments passed to the external trainer command.",
    )

    val = sub.add_parser("validate", help="Generate a preview grid using a model version")
    val.add_argument("--model-version", required=True, help="Key from model_versions.json")
    val.add_argument("--prompt", required=True, help="Prompt for validation")
    val.add_argument("--seconds", type=float, default=1.6, help="Seconds per sample")
    val.add_argument("--variants", type=int, default=6, help="Number of variants")
    val.add_argument("--seed", type=int, default=123, help="Base seed")
    val.add_argument("--out", required=True, help="Output folder")
    val.add_argument("--post", action="store_true", help="Apply post-processing")
    val.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    val.add_argument(
        "--engine",
        default="stable_audio_open",
        choices=["stable_audio_open", "hybrid"],
        help="Engine for validation (stable_audio_open or hybrid base stable_audio_open).",
    )

    return p


def _ensure_dataset_ok(dataset: Path) -> None:
    if not dataset.exists():
        raise FileNotFoundError(f"Dataset folder not found: {dataset}")
    meta = dataset / "metadata.jsonl"
    audio = dataset / "audio"
    if not meta.exists():
        raise FileNotFoundError(f"Dataset missing metadata.jsonl: {meta}")
    if not audio.exists():
        raise FileNotFoundError(f"Dataset missing audio/ folder: {audio}")


def _parse_train_args(args: argparse.Namespace) -> TrainArgs:
    dataset = Path(str(args.dataset))
    out_dir = Path(str(args.out))
    _ensure_dataset_ok(dataset)
    out_dir.mkdir(parents=True, exist_ok=True)

    extra = list(args.extra or [])
    # Strip leading "--" if argparse preserved it.
    if extra and extra[0] == "--":
        extra = extra[1:]

    return TrainArgs(
        dataset=dataset,
        out_dir=out_dir,
        base_model=(str(args.base_model).strip() if args.base_model else None),
        trainer=str(args.trainer),
        extra=extra,
    )


def run_train(args: argparse.Namespace) -> int:
    ta = _parse_train_args(args)

    # This wrapper is intentionally minimal. Users can point it at any command.
    cmd = os.environ.get("SOUNDGEN_TRAIN_CMD", "")
    cmd = str(cmd).strip()
    if not cmd:
        raise RuntimeError(
            "SOUNDGEN_TRAIN_CMD is not set.\n\n"
            "Set it to a trainer command (e.g. a python script) and re-run. Example:\n"
            "  $env:SOUNDGEN_TRAIN_CMD = 'python train_lora.py'\n"
            "  python -m soundgen.finetune train --dataset datasets/ghoul --out runs/ghoul -- --help\n"
        )

    # Expand a basic tokenization (we avoid shell=True).
    base = cmd.split()
    full = base + [
        "--dataset",
        str(ta.dataset),
        "--out",
        str(ta.out_dir),
    ]
    if ta.base_model:
        full += ["--base-model", ta.base_model]
    full += ta.extra

    print("Running trainer:")
    print("  " + " ".join(json.dumps(x) for x in full))

    proc = subprocess.run(full, check=False)
    return int(proc.returncode)


def _apply_model_version_to_engine(*, mv: ModelVersion, engine: str) -> dict[str, str | float | None]:
    eng = str(engine).strip().lower()
    if eng not in {"stable_audio_open", "hybrid"}:
        raise ValueError(f"Unsupported engine for validate: {engine}")
    if str(mv.engine).strip().lower() != "stable_audio_open":
        raise ValueError(f"Model version '{mv.key}' engine={mv.engine!r} not supported")

    # The generate CLI understands --model-version directly, so we pass it through.
    return {
        "model_version": mv.key,
    }


def run_validate(args: argparse.Namespace) -> int:
    from .model_versions import resolve_model_version
    from .generate import main as generate_main

    mv = resolve_model_version(str(args.model_version))
    out_dir = Path(str(args.out))
    out_dir.mkdir(parents=True, exist_ok=True)

    engine = str(args.engine)
    common = [
        "--engine",
        engine,
        "--model-version",
        mv.key,
        "--prompt",
        str(args.prompt),
        "--seconds",
        str(float(args.seconds)),
        "--seed",
        str(int(args.seed)),
        "--variants",
        str(int(args.variants)),
        "--device",
        str(args.device),
        "--out",
        str(out_dir / "validate.wav"),
    ]

    if engine == "hybrid":
        # validate hybrid with stable_audio base; layered params default.
        common += ["--hybrid-base-engine", "stable_audio_open"]

    if bool(args.post):
        common += ["--post"]

    # We want individual variants, so use minecraft kept-wav mode without creating a full resource pack.
    # The generator already supports writing multiple WAVs in --mc-wav-dir mode.
    cmd = [
        "--minecraft",
        "--namespace",
        "validate",
        "--event",
        "finetune.preview",
        "--sound-path",
        "generated/validate/finetune_preview",
        "--mc-wav-dir",
        str(out_dir),
    ] + common

    return int(generate_main(cmd))


def main(argv: list[str] | None = None) -> None:
    p = build_parser()
    args = p.parse_args(argv)

    if args.cmd == "train":
        raise SystemExit(run_train(args))
    if args.cmd == "validate":
        raise SystemExit(run_validate(args))

    raise SystemExit(2)


if __name__ == "__main__":
    main()
