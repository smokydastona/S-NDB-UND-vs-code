from __future__ import annotations

import re
from pathlib import Path


_MD_ENTRY_HEADING_RE = re.compile(r"^###\s+(.+?)\s*$", re.MULTILINE)
_MD_PROMPT_RE = re.compile(r"^\s*-\s*\*\*Prompt\*\*:\s*(.+?)\s*$", re.MULTILINE)
_MD_SOUNDEVENT_RE = re.compile(r"^\s*-\s*\*\*SoundEvent\*\*:\s*`([^`]+)`\s*$", re.MULTILINE)
_MD_MANIFEST_RE = re.compile(r"^\s*-\s*\*\*Manifest\*\*:\s*(.+?)\s*$", re.MULTILINE)
_MD_SUBTITLE_KEY_RE = re.compile(
    r"^\s*-\s*\*\*(Suggested|Existing) subtitle key\*\*:\s*`([^`]+)`\s*(?:\((.+?)\))?\s*$",
    re.MULTILINE,
)


class UnsupportedDocumentError(RuntimeError):
    pass


def read_document_text(path: Path) -> str:
    """Read a document into plain text.

    Supported:
      - .txt, .md
      - .docx (requires python-docx)

    Notes:
      - Legacy .doc is not supported (binary format).
    """

    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="replace")

    if suffix == ".docx":
        try:
            import docx  # type: ignore
        except Exception as e:  # pragma: no cover
            raise UnsupportedDocumentError(
                "Reading .docx requires the 'python-docx' package. Install it with: pip install python-docx"
            ) from e

        d = docx.Document(str(path))
        parts: list[str] = []
        for p in d.paragraphs:
            t = (p.text or "").strip()
            if t:
                parts.append(t)
        return "\n".join(parts)

    if suffix == ".doc":
        raise UnsupportedDocumentError(
            "Legacy .doc is not supported. Save/export as .docx or .txt and try again."
        )

    raise UnsupportedDocumentError(f"Unsupported document type: {suffix}")


def to_prompt(text: str, *, max_chars: int = 2000) -> str:
    """Convert a doc's text into a prompt.

    Keeps it simple: collapse whitespace and cap length.
    """

    t = " ".join(text.split())
    if not t:
        return ""
    if len(t) > max_chars:
        t = t[: max_chars - 1].rstrip() + "…"
    return t


def extract_sound_prompts(text: str) -> list[dict[str, str]]:
    """Extract multiple sound prompts from a doc.

    Supports an "auto" heuristic focused on Markdown docs like `pregen_sound_bible.md`:
    - Each entry starts with a `### <id>` heading
    - Contains a `- **Prompt**: "..."` line
    - May contain `- **SoundEvent**: `<namespace>:<event>`

        Returns a list of dicts like:
            {
                "title": ..., "prompt": ..., "namespace": ..., "event": ...,
                "engine": ..., "seconds": ..., "variants": ..., "sound_path": ...,
                "subtitle_key": ..., "subtitle": ...,
            }

    If no multi-prompt structure is detected, returns an empty list.
    """

    if not text.strip():
        return []

    headings = list(_MD_ENTRY_HEADING_RE.finditer(text))
    if not headings:
        return []

    entries: list[dict[str, str]] = []
    for idx, h in enumerate(headings):
        title = (h.group(1) or "").strip()
        start = h.end()
        end = headings[idx + 1].start() if idx + 1 < len(headings) else len(text)
        block = text[start:end]

        pm = _MD_PROMPT_RE.search(block)
        if not pm:
            continue

        raw_prompt = (pm.group(1) or "").strip()
        # prompt is usually quoted: "..."; accept either.
        if (raw_prompt.startswith('"') and raw_prompt.endswith('"')) or (
            raw_prompt.startswith("'") and raw_prompt.endswith("'")
        ):
            raw_prompt = raw_prompt[1:-1].strip()

        if not raw_prompt:
            continue

        namespace = ""
        event = ""
        sem = _MD_SOUNDEVENT_RE.search(block)
        if sem:
            se = (sem.group(1) or "").strip()
            if ":" in se:
                ns, ev = se.split(":", 1)
                namespace = ns.strip()
                event = ev.strip()
            else:
                event = se

        if not event:
            # In pregen docs, the `###` heading is typically the desired sounds.json key.
            event = title

        parsed: dict[str, str] = {
            "title": title,
            "prompt": raw_prompt,
            "namespace": namespace,
            "event": event,
        }

        # Optional: parse the generated manifest summary line.
        mm = _MD_MANIFEST_RE.search(block)
        if mm:
            manifest_text = (mm.group(1) or "").strip()
            for part in [p.strip() for p in manifest_text.split(",") if p.strip()]:
                if "=" not in part:
                    continue
                k, v = part.split("=", 1)
                k = k.strip().lower()
                v = v.strip()
                if v.startswith("`") and v.endswith("`"):
                    v = v[1:-1].strip()
                if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
                    v = v[1:-1].strip()
                if not v:
                    continue
                # Allow a compact, human-editable manifest summary line in docs.
                # Keep this list explicit to avoid accidentally treating arbitrary text as config.
                if k == "hf_token":
                    k = "stable_audio_hf_token"

                allowed = {
                    "engine",
                    "seconds",
                    "candidates",
                    "variants",
                    "sound_path",
                    "seed",
                    "preset",
                    "weight",
                    "volume",
                    "pitch",
                    # Stable Audio Open controls
                    "stable_audio_model",
                    "stable_audio_negative_prompt",
                    "stable_audio_steps",
                    "stable_audio_guidance_scale",
                    "stable_audio_sampler",
                    "stable_audio_hf_token",
                }
                if k in allowed:
                    parsed[k] = v

        # Optional: parse subtitle key + suggested English text.
        sm = _MD_SUBTITLE_KEY_RE.search(block)
        if sm:
            parsed["subtitle_key"] = (sm.group(2) or "").strip()
            subtitle = (sm.group(3) or "").strip()
            # Captured text often includes quotes.
            if (subtitle.startswith('"') and subtitle.endswith('"')) or (
                subtitle.startswith("'") and subtitle.endswith("'")
            ):
                subtitle = subtitle[1:-1].strip()
            if subtitle:
                parsed["subtitle"] = subtitle

        entries.append(parsed)

    # Only treat as multi-prompt if we found 2+ entries.
    return entries if len(entries) >= 2 else []
