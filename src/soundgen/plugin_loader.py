from __future__ import annotations

from dataclasses import dataclass
import importlib
import inspect
import sys
from pathlib import Path
from typing import Any, Callable, Iterable


@dataclass(frozen=True)
class PluginLoadReport:
    loaded_modules: tuple[str, ...]
    errors: tuple[str, ...]


def _iter_local_plugin_module_names(search_roots: Iterable[Path]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()

    for root in search_roots:
        try:
            base = Path(root) / "soundgen_plugins"
            if not base.exists() or not base.is_dir():
                continue

            # Make sure `soundgen_plugins` is importable as a (namespace) package.
            # We add the parent folder so `import soundgen_plugins.xyz` works.
            parent = base.parent
            parent_str = str(parent)
            if parent_str not in sys.path:
                sys.path.insert(0, parent_str)

            for p in base.iterdir():
                if p.name.startswith("_"):
                    continue
                if p.is_file() and p.suffix == ".py":
                    mod = f"soundgen_plugins.{p.stem}"
                    if mod not in seen:
                        names.append(mod)
                        seen.add(mod)
                elif p.is_dir():
                    # Allow package-style plugins: soundgen_plugins/<name>/__init__.py
                    mod = f"soundgen_plugins.{p.name}"
                    if mod not in seen:
                        names.append(mod)
                        seen.add(mod)
        except Exception:
            # Local plugin discovery is best-effort.
            continue

    return names


def _call_register_hook(obj: Any, *, register_engine: Callable[[str, Any], None]) -> None:
    # Supported shapes:
    # - module-level function: register(register_engine)
    # - module-level function: register_soundgen(register_engine)
    # - object with .register(register_engine)
    for attr in ("register_soundgen", "register"):
        fn = getattr(obj, attr, None)
        if fn is None:
            continue
        if callable(fn):
            try:
                sig = inspect.signature(fn)
                if len(sig.parameters) == 1:
                    fn(register_engine)
                else:
                    fn(register_engine=register_engine)
            except TypeError:
                fn(register_engine)
            return

    # If the entrypoint directly provides a dict mapping engine->callable.
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str) and callable(v):
                register_engine(k, v)
        return

    # If the entrypoint returns a callable, we assume it is itself a register hook.
    if callable(obj):
        try:
            sig = inspect.signature(obj)
            if len(sig.parameters) == 1:
                obj(register_engine)
            else:
                obj(register_engine=register_engine)
        except TypeError:
            obj(register_engine)


def _iter_entrypoint_objects(group: str) -> list[Any]:
    try:
        from importlib.metadata import entry_points
    except Exception:
        return []

    try:
        eps = entry_points()
        # Python 3.10+ has .select
        selected = eps.select(group=group) if hasattr(eps, "select") else eps.get(group, [])
    except Exception:
        return []

    objs: list[Any] = []
    for ep in list(selected):
        try:
            objs.append(ep.load())
        except Exception:
            continue
    return objs


def load_engine_plugins(*, register_engine: Callable[[str, Any], None]) -> PluginLoadReport:
    """Best-effort plugin discovery.

    Supports:
    - Local dev folder: `./soundgen_plugins/` (modules under `soundgen_plugins.*`)
    - Python entry points: group `soundgen.engines`

    Plugins should expose one of:
    - `register_soundgen(register_engine)`
    - `register(register_engine)`
    - a dict `{engine_name: engine_fn}`

    The `engine_fn` is validated/typed by the caller (engine_registry).
    """

    loaded: list[str] = []
    errors: list[str] = []

    # Search roots: cwd + repo root (best-effort).
    here = Path(__file__).resolve()
    repo_root = here.parents[2]  # .../src/soundgen/plugin_loader.py -> repo root
    module_names = _iter_local_plugin_module_names([Path.cwd(), repo_root])

    for mod_name in module_names:
        try:
            mod = importlib.import_module(mod_name)
            _call_register_hook(mod, register_engine=register_engine)
            loaded.append(mod_name)
        except Exception as e:
            errors.append(f"{mod_name}: {e}")

    for obj in _iter_entrypoint_objects("soundgen.engines"):
        try:
            _call_register_hook(obj, register_engine=register_engine)
            loaded.append(getattr(obj, "__name__", obj.__class__.__name__))
        except Exception as e:
            errors.append(f"entrypoint: {e}")

    return PluginLoadReport(loaded_modules=tuple(loaded), errors=tuple(errors))
