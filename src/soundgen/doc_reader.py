from __future__ import annotations

from pathlib import Path


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
