from __future__ import annotations

import os
from pathlib import Path

from .app_dirs import app_data_dir, ensure_dir


def _setdefault_env_path(key: str, value: Path) -> None:
    if str(os.environ.get(key) or "").strip():
        return
    os.environ[key] = str(value)


def configure_runtime(*, data_dir: Path | None = None) -> Path:
    """Configure a predictable runtime layout (caches/settings/logs).

    Returns:
        The resolved `data_dir`.

    This is intentionally idempotent and safe to call for any subcommand.
    """

    base = Path(data_dir) if data_dir is not None else app_data_dir()
    ensure_dir(base)

    # If the caller explicitly selected a runtime directory, propagate it so any
    # subprocesses and helper modules can resolve the same location.
    if data_dir is not None:
        _setdefault_env_path("SOUNDGEN_DATA_DIR", base)

    # Keep common subfolders stable.
    cache_dir = ensure_dir(base / "cache")
    logs_dir = ensure_dir(base / "logs")
    ensure_dir(base / "models")

    # Hugging Face + diffusers + transformers caches
    hf_home = ensure_dir(cache_dir / "hf")
    _setdefault_env_path("HF_HOME", hf_home)
    _setdefault_env_path("HUGGINGFACE_HUB_CACHE", hf_home / "hub")

    # transformers historically uses TRANSFORMERS_CACHE; diffusers often uses HF caches.
    _setdefault_env_path("TRANSFORMERS_CACHE", hf_home / "transformers")
    _setdefault_env_path("DIFFUSERS_CACHE", hf_home / "diffusers")

    # Torch cache (model weights, etc.)
    _setdefault_env_path("TORCH_HOME", ensure_dir(cache_dir / "torch"))

    # Optional: encourage libraries to place temporary logs under our app directory.
    _setdefault_env_path("SOUNDGEN_LOG_DIR", logs_dir)

    return base


def user_configs_dir(*, data_dir: Path | None = None) -> Path:
    """Return a per-user writable configs directory.

    This is distinct from the bundled app's read-only `configs/` resources.
    """

    base = configure_runtime(data_dir=data_dir)
    return ensure_dir(Path(base) / "configs")
