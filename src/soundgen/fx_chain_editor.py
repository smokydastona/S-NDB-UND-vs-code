from __future__ import annotations

import argparse
import json
import shlex
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .fx_chain_v2 import (
    FxChainV2,
    FxStepV2,
    apply_fx_chain_v2,
    dump_fx_chain_v2_json,
    fx_module_v2_ids,
    get_fx_module_v2,
    is_fx_chain_v2_json,
    load_fx_chain_v2_json,
)
from .io_utils import read_wav_mono, write_wav


def _parse_value(s: str) -> Any:
    t = s.strip()
    if t == "":
        return ""
    try:
        from .json_utils import loads_json_lenient

        return loads_json_lenient(t, context="FX chain editor value")
    except Exception:
        pass
    # fall back to raw string
    return t


def _print_help() -> None:
    print(
        "Commands:\n"
        "  help\n"
        "  modules                       # list available v2 modules\n"
        "  show                          # show current chain steps\n"
        "  add <module_id> [json_params] # append step (params as JSON object)\n"
        "  del <index>                   # delete step\n"
        "  up <index> | down <index>     # reorder\n"
        "  set <index> <param> <value>   # set step param (value parsed as JSON if possible)\n"
        "  wav <path>                    # set audition WAV path\n"
        "  audition                      # apply chain and play (Windows best-effort)\n"
        "  render <out.wav>              # apply chain and write WAV\n"
        "  save [path]                   # save chain JSON\n"
        "  quit\n"
    )


def _play_wav_best_effort(path: Path) -> None:
    try:
        import winsound

        winsound.PlaySound(str(path), winsound.SND_FILENAME)
    except Exception:
        print("Playback is only implemented on Windows (winsound).")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="FX chain editor (v2 modular JSON)")
    ap.add_argument("--chain", default="configs/fx_chain_v2.example.json", help="Path to a v2 FX chain json.")
    ap.add_argument("--wav", default=None, help="Optional WAV path for audition/render.")
    args = ap.parse_args(argv)

    chain_path = Path(args.chain)
    wav_path: Path | None = (Path(args.wav) if args.wav else None)

    if chain_path.exists() and is_fx_chain_v2_json(chain_path):
        chain = load_fx_chain_v2_json(chain_path)
    else:
        chain = FxChainV2(name=chain_path.stem, steps=(), description=None)

    print(f"FX chain v2 editor — {chain.name}")
    _print_help()

    while True:
        try:
            line = input("fx> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("")
            break

        if not line:
            continue
        parts = shlex.split(line)
        if not parts:
            continue

        cmd = parts[0].strip().lower()
        rest = parts[1:]

        if cmd in {"quit", "exit", "q"}:
            break

        if cmd in {"help", "h", "?"}:
            _print_help()
            continue

        if cmd == "modules":
            for mid in fx_module_v2_ids():
                m = get_fx_module_v2(mid)
                if m is None:
                    continue
                print(f"- {m.module_id}: {m.title} — {m.description}")
            continue

        if cmd == "show":
            if not chain.steps:
                print("(empty)")
                continue
            for i, s in enumerate(chain.steps):
                print(f"[{i}] {s.module_id} {json.dumps(s.params, ensure_ascii=False)}")
            continue

        if cmd == "add":
            if not rest:
                print("Usage: add <module_id> [json_params]")
                continue
            mid = str(rest[0]).strip().lower()
            if get_fx_module_v2(mid) is None:
                print(f"Unknown module_id: {mid}")
                continue
            params: dict[str, Any] = {}
            if len(rest) >= 2:
                raw = " ".join(rest[1:]).strip()
                v = _parse_value(raw)
                if isinstance(v, dict):
                    params = dict(v)
                else:
                    print("Params must be a JSON object; ignoring.")
            chain = FxChainV2(name=chain.name, steps=tuple([*chain.steps, FxStepV2(module_id=mid, params=params)]), description=chain.description)
            print(f"Added {mid}.")
            continue

        if cmd == "del":
            if len(rest) != 1:
                print("Usage: del <index>")
                continue
            idx = int(rest[0])
            if idx < 0 or idx >= len(chain.steps):
                print("Index out of range")
                continue
            steps = list(chain.steps)
            removed = steps.pop(idx)
            chain = FxChainV2(name=chain.name, steps=tuple(steps), description=chain.description)
            print(f"Deleted [{idx}] {removed.module_id}.")
            continue

        if cmd in {"up", "down"}:
            if len(rest) != 1:
                print(f"Usage: {cmd} <index>")
                continue
            idx = int(rest[0])
            if idx < 0 or idx >= len(chain.steps):
                print("Index out of range")
                continue
            j = idx - 1 if cmd == "up" else idx + 1
            if j < 0 or j >= len(chain.steps):
                continue
            steps = list(chain.steps)
            steps[idx], steps[j] = steps[j], steps[idx]
            chain = FxChainV2(name=chain.name, steps=tuple(steps), description=chain.description)
            continue

        if cmd == "set":
            if len(rest) < 3:
                print("Usage: set <index> <param> <value>")
                continue
            idx = int(rest[0])
            if idx < 0 or idx >= len(chain.steps):
                print("Index out of range")
                continue
            param = str(rest[1]).strip()
            value_raw = " ".join(rest[2:])
            value = _parse_value(value_raw)

            steps = list(chain.steps)
            step = steps[idx]
            params2 = dict(step.params or {})
            params2[param] = value
            steps[idx] = FxStepV2(module_id=step.module_id, params=params2)
            chain = FxChainV2(name=chain.name, steps=tuple(steps), description=chain.description)
            continue

        if cmd == "wav":
            if len(rest) != 1:
                print("Usage: wav <path>")
                continue
            wav_path = Path(rest[0])
            print(f"WAV set to: {wav_path}")
            continue

        if cmd == "audition":
            if wav_path is None:
                print("Set a WAV first: wav <path>")
                continue
            if not wav_path.exists():
                print(f"Not found: {wav_path}")
                continue
            audio, sr = read_wav_mono(wav_path)
            y = apply_fx_chain_v2(audio, int(sr), chain)

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp = Path(f.name)
            write_wav(tmp, y.astype(np.float32, copy=False), int(sr))
            _play_wav_best_effort(tmp)
            continue

        if cmd == "render":
            if len(rest) != 1:
                print("Usage: render <out.wav>")
                continue
            if wav_path is None:
                print("Set a WAV first: wav <path>")
                continue
            if not wav_path.exists():
                print(f"Not found: {wav_path}")
                continue

            out_path = Path(rest[0])
            audio, sr = read_wav_mono(wav_path)
            y = apply_fx_chain_v2(audio, int(sr), chain)
            write_wav(out_path, y.astype(np.float32, copy=False), int(sr))
            print(f"Wrote: {out_path}")
            continue

        if cmd == "save":
            out = chain_path
            if rest:
                out = Path(rest[0])
            out.parent.mkdir(parents=True, exist_ok=True)
            obj = dump_fx_chain_v2_json(chain)
            out.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"Saved: {out}")
            continue

        print(f"Unknown command: {cmd}")

    # Auto-save back to --chain on exit (safe default for interactive editing)
    chain_path.parent.mkdir(parents=True, exist_ok=True)
    obj = dump_fx_chain_v2_json(chain)
    chain_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved: {chain_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
