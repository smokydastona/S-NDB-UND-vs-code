from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


_TOKEN_RE = re.compile(r"[a-z0-9_\-\.]{3,}")


@dataclass(frozen=True)
class IndexedFile:
    rel_path: str
    mtime_ns: int
    size: int
    # token -> count (bounded)
    tokens: dict[str, int]


def _appdata_dir() -> Path:
    # Keep consistent with prereqs_windows.py but avoid importing it.
    if sys.platform == "win32":
        la = str(os.environ.get("LOCALAPPDATA") or "").strip()
        base = (Path(la) if la else (Path.home() / "AppData" / "Local")) / "SÖNDBÖUND"
    else:
        base = Path.home() / ".söndböund"
    return base


def _index_path() -> Path:
    return _appdata_dir() / "ai_index.json"


def _repo_root_candidates() -> list[Path]:
    roots: list[Path] = []
    try:
        roots.append(Path.cwd())
    except Exception:
        pass
    try:
        roots.append(Path(__file__).resolve().parents[2])
    except Exception:
        pass
    try:
        roots.append(Path(sys.executable).resolve().parent)
    except Exception:
        pass

    out: list[Path] = []
    seen: set[str] = set()
    for r in roots:
        try:
            key = str(r.resolve())
        except Exception:
            continue
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def _safe_rel(root: Path, p: Path) -> str | None:
    try:
        rel = p.relative_to(root)
        return str(rel).replace("\\", "/")
    except Exception:
        return None


def _read_text(p: Path, max_chars: int = 250_000) -> str | None:
    try:
        if not p.exists() or not p.is_file():
            return None
        b = p.read_bytes()
        try:
            t = b.decode("utf-8")
        except Exception:
            t = b.decode("utf-8", errors="replace")
        t = t.replace("\r\n", "\n")
        if len(t) > max_chars:
            return t[:max_chars] + "\n…(truncated)…\n"
        return t
    except Exception:
        return None


def _token_counts(text: str, max_tokens: int = 2048) -> dict[str, int]:
    counts: dict[str, int] = {}
    for tok in _TOKEN_RE.findall(text.lower()):
        counts[tok] = counts.get(tok, 0) + 1

    # Bound per-file token map to reduce index size.
    if len(counts) <= max_tokens:
        return counts

    # Keep the most common tokens.
    items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:max_tokens]
    return {k: int(v) for k, v in items}


def _iter_candidate_files(root: Path) -> list[Path]:
    # Keep this set focused and predictable: docs + key configs + key code.
    paths: list[Path] = []

    # Docs.
    try:
        paths.extend(list((root / "docs").glob("**/*.md")))
    except Exception:
        pass

    # Example/config/schema files.
    for rel in [
        "README.md",
        "CHANGELOG.md",
        "example_manifest.json",
    ]:
        p = root / rel
        if p.exists() and p.is_file():
            paths.append(p)

    try:
        paths.extend(list((root / "configs").glob("*.json")))
    except Exception:
        pass

    # Core code modules.
    code_globs = [
        root / "src" / "soundgen" / "*.py",
    ]
    for g in code_globs:
        try:
            paths.extend(list(g.parent.glob(g.name)))
        except Exception:
            pass

    # Deduplicate.
    out: list[Path] = []
    seen: set[str] = set()
    for p in paths:
        try:
            key = str(p.resolve())
        except Exception:
            continue
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def _should_index_rel(rel_path: str) -> bool:
    rp = str(rel_path).replace("\\", "/").lstrip("/")
    bad_prefixes = (
        "outputs/",
        "resourcepack/",
        "pre_gen_sound/",
        "tools/",
        "library/",
        "soundpack_v1/",
        ".git/",
        "electron/",
    )
    if rp.startswith(bad_prefixes):
        return False
    # Keep only text-like files.
    if not (
        rp.endswith(".md")
        or rp.endswith(".py")
        or rp.endswith(".json")
        or rp.endswith(".toml")
        or rp.endswith(".txt")
    ):
        return False
    return True


def load_index() -> dict[str, IndexedFile] | None:
    p = _index_path()
    try:
        if not p.exists() or not p.is_file():
            return None
        from .json_utils import load_json_file_lenient

        data = load_json_file_lenient(p, context=f"AI index JSON file: {p}")
        files = {}
        for rel, v in (data.get("files") or {}).items():
            if not isinstance(v, dict):
                continue
            files[str(rel)] = IndexedFile(
                rel_path=str(rel),
                mtime_ns=int(v.get("mtime_ns") or 0),
                size=int(v.get("size") or 0),
                tokens={str(k): int(c) for k, c in (v.get("tokens") or {}).items()},
            )
        return files
    except Exception:
        return None


def save_index(files: dict[str, IndexedFile]) -> None:
    p = _index_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": 1,
            "created": int(time.time()),
            "files": {
                rel: {
                    "mtime_ns": int(f.mtime_ns),
                    "size": int(f.size),
                    "tokens": f.tokens,
                }
                for rel, f in files.items()
            },
        }
        p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    except Exception:
        return


def build_or_update_index(*, force: bool = False) -> dict[str, IndexedFile]:
    if str(os.environ.get("SOUNDGEN_AI_INDEX", "1")).strip().lower() in {"0", "false", "off", "no"}:
        return {}

    roots = _repo_root_candidates()
    root = None
    for r in roots:
        # Prefer dev checkout root.
        if (r / "src" / "soundgen").exists() or (r / "docs").exists():
            root = r
            break
    if root is None:
        return {}

    existing = load_index() or {}
    out: dict[str, IndexedFile] = {} if force else dict(existing)

    for p in _iter_candidate_files(root):
        rel = _safe_rel(root, p)
        if not rel or not _should_index_rel(rel):
            continue
        try:
            st = p.stat()
        except Exception:
            continue
        mtime_ns = int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)))
        size = int(st.st_size)

        prev = existing.get(rel)
        if (not force) and prev and prev.mtime_ns == mtime_ns and prev.size == size:
            out[rel] = prev
            continue

        txt = _read_text(p)
        if not txt:
            continue
        out[rel] = IndexedFile(rel_path=rel, mtime_ns=mtime_ns, size=size, tokens=_token_counts(txt))

    # Drop removed files.
    if not force:
        try:
            existing_keys = set(out.keys())
            # no-op
        except Exception:
            existing_keys = set(out.keys())
    save_index(out)
    return out


def _query_terms(query: str, max_terms: int = 16) -> list[str]:
    raw = str(query or "").strip().lower()
    terms = _TOKEN_RE.findall(raw)
    out: list[str] = []
    seen: set[str] = set()
    for t in terms:
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
        if len(out) >= max_terms:
            break
    return out


def score_indexed_file(f: IndexedFile, terms: Iterable[str]) -> float:
    score = 0.0
    for t in terms:
        score += float(f.tokens.get(t, 0))
    # Gentle bias: prefer docs over code when tied.
    if f.rel_path.endswith(".md"):
        score *= 1.08
    return score


def top_matching_files(
    *,
    query: str,
    files: dict[str, IndexedFile] | None = None,
    k: int = 10,
) -> list[str]:
    idx = files if files is not None else (load_index() or {})
    if not idx:
        idx = build_or_update_index(force=False)
    if not idx:
        return []

    terms = _query_terms(query)
    if not terms:
        return []

    scored: list[tuple[float, str]] = []
    for rel, f in idx.items():
        s = score_indexed_file(f, terms)
        if s > 0.0:
            scored.append((s, rel))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [rel for _, rel in scored[: max(1, int(k))]]
