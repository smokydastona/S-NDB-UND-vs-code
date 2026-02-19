from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .runtime_config import configure_runtime


_DEFAULT_MODELS: tuple[str, ...] = (
    # Diffusers AudioLDM2 (engine=diffusers)
    "cvssp/audioldm2",
    # Stable Audio Open (engine=stable_audio_open)
    "stabilityai/stable-audio-open-1.0",
)


def default_model_ids() -> tuple[str, ...]:
    return _DEFAULT_MODELS


def _is_model_cached(repo_id: str, *, token: str | None = None, revision: str | None = None) -> bool:
    configure_runtime()

    # Lazy import so `--help` stays fast.
    from huggingface_hub import snapshot_download

    kwargs: dict[str, object] = {"local_files_only": True}
    if token and token.strip():
        kwargs["token"] = token.strip()
    if revision and revision.strip():
        kwargs["revision"] = revision.strip()

    try:
        snapshot_download(repo_id=str(repo_id), **kwargs)
        return True
    except Exception:
        return False


def missing_default_models(*, token: str | None = None) -> list[str]:
    missing: list[str] = []
    for rid in _DEFAULT_MODELS:
        if not _is_model_cached(rid, token=token, revision=None):
            missing.append(str(rid))
    return missing


def _print_cache_locations() -> None:
    configure_runtime()
    keys = [
        "SOUNDGEN_DATA_DIR",
        "SOUNDGEN_LOG_DIR",
        "HF_HOME",
        "HUGGINGFACE_HUB_CACHE",
        "TRANSFORMERS_CACHE",
        "DIFFUSERS_CACHE",
        "TORCH_HOME",
    ]
    for k in keys:
        v = str(os.environ.get(k) or "").strip()
        if v:
            print(f"{k}={v}")


def _resolve_hf_cache_dir(*, runtime_dir: Path) -> Path:
    """Return a safe Hugging Face hub cache directory.

    Prefer the env vars (which `configure_runtime()` sets) and fall back to a
    reasonable subfolder under the app runtime directory.
    """

    hub_cache = str(os.environ.get("HUGGINGFACE_HUB_CACHE") or "").strip()
    if hub_cache:
        return Path(hub_cache).expanduser()

    hf_home = str(os.environ.get("HF_HOME") or "").strip()
    if hf_home:
        return Path(hf_home).expanduser() / "hub"

    return Path(runtime_dir) / "cache" / "hf" / "hub"


def _download_model(
    repo_id: str,
    *,
    token: str | None,
    revision: str | None,
    force: bool,
    allow_patterns: list[str] | None,
    ignore_patterns: list[str] | None,
) -> str:
    runtime_dir = configure_runtime()

    # Lazy import so `--help` stays fast.
    from huggingface_hub import snapshot_download

    kwargs: dict[str, object] = {}
    if token and token.strip():
        kwargs["token"] = token.strip()
    if revision and revision.strip():
        kwargs["revision"] = revision.strip()

    if allow_patterns:
        kwargs["allow_patterns"] = list(allow_patterns)
    if ignore_patterns:
        kwargs["ignore_patterns"] = list(ignore_patterns)

    kwargs["cache_dir"] = str(_resolve_hf_cache_dir(runtime_dir=runtime_dir))

    if force:
        kwargs["force_download"] = True

    path = snapshot_download(repo_id=str(repo_id), **kwargs)
    return str(path)


def run_models(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Model utilities (download/check cache locations).")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub_where = sub.add_parser("where", help="Print model/cache locations used by SONDBOUND.")
    sub_where.set_defaults(_cmd="where")

    sub_defaults = sub.add_parser("defaults", help="List default model ids used by built-in engines.")
    sub_defaults.set_defaults(_cmd="defaults")

    sub_missing = sub.add_parser("missing", help="List default models that are not yet cached.")
    sub_missing.add_argument(
        "--token",
        default=None,
        help="HF token (optional). You can also set HF_TOKEN/HUGGINGFACE_HUB_TOKEN.",
    )
    sub_missing.add_argument(
        "--json",
        action="store_true",
        help="Print a single JSON line (machine-readable) of the missing model ids.",
    )
    sub_missing.set_defaults(_cmd="missing")

    sub_dl = sub.add_parser("download", help="Download a Hugging Face model snapshot into the app cache.")
    sub_dl.add_argument("repo_id", help="Hugging Face repo id, e.g. cvssp/audioldm2")
    sub_dl.add_argument(
        "--token",
        default=None,
        help="HF token (optional). You can also set HF_TOKEN/HUGGINGFACE_HUB_TOKEN.",
    )
    sub_dl.add_argument("--revision", default=None, help="Optional model revision (branch/tag/commit).")
    sub_dl.add_argument("--force", action="store_true", help="Force redownload.")
    sub_dl.add_argument(
        "--allow",
        action="append",
        default=None,
        help="Allow only matching patterns (can be repeated), e.g. --allow '*.safetensors'",
    )
    sub_dl.add_argument(
        "--ignore",
        action="append",
        default=None,
        help="Ignore matching patterns (can be repeated), e.g. --ignore '*.bin'",
    )
    sub_dl.set_defaults(_cmd="download")

    args = p.parse_args(argv)

    cmd = str(getattr(args, "_cmd", ""))
    if cmd == "where":
        _print_cache_locations()
        return 0

    if cmd == "defaults":
        for m in _DEFAULT_MODELS:
            print(m)
        return 0

    if cmd == "missing":
        token = args.token
        if not (token and str(token).strip()):
            token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
        missing = missing_default_models(token=(str(token) if token else None))
        if bool(getattr(args, "json", False)):
            print(json.dumps(missing, ensure_ascii=True), flush=True)
        else:
            for rid in missing:
                print(rid)
        return 0

    if cmd == "download":
        token = args.token
        if not (token and str(token).strip()):
            token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
        out = _download_model(
            str(args.repo_id),
            token=(str(token) if token else None),
            revision=(str(args.revision) if args.revision else None),
            force=bool(args.force),
            allow_patterns=(list(args.allow) if args.allow else None),
            ignore_patterns=(list(args.ignore) if args.ignore else None),
        )
        print(out)
        return 0

    p.print_help()
    return 2


def main() -> None:
    raise SystemExit(run_models())


if __name__ == "__main__":
    main()
