from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class JsonParseError(ValueError):
    context: str
    original: str
    message: str

    def __str__(self) -> str:  # pragma: no cover
        return f"{self.context}: {self.message}"


_VALID_ESCAPES = {'"', "\\", "/", "b", "f", "n", "r", "t", "u"}


def _escape_invalid_backslashes_in_json_strings(raw: str) -> str:
    """Make a best-effort fix for Windows paths pasted into JSON.

    Example user input (invalid JSON):
        {"path": "C:\\Users\\me\\file.wav"}  ✅
        {"path": "C:\\Users\\me\\file.wav"}      ❌ if the JSON contains single backslashes (common when pasting a Windows path)

    This routine only touches backslashes inside JSON strings, and only when the
    backslash is not starting a valid JSON escape sequence.
    """

    s = str(raw)
    out: list[str] = []
    in_string = False
    escaped = False

    n = len(s)
    for i, ch in enumerate(s):
        if not in_string:
            out.append(ch)
            if ch == '"':
                in_string = True
            continue

        # Inside a JSON string
        if escaped:
            out.append(ch)
            escaped = False
            continue

        if ch == "\\":
            nxt = s[i + 1] if (i + 1) < n else ""
            if nxt in _VALID_ESCAPES:
                out.append(ch)
                escaped = True
            else:
                # Invalid escape (common for Windows paths like \U) => double the backslash.
                out.append("\\\\")
            continue

        out.append(ch)
        if ch == '"':
            in_string = False

    return "".join(out)


def loads_json_object_lenient(raw: str, *, context: str) -> dict[str, Any]:
    """Parse a JSON object from a user-provided string.

    - On success: returns a dict.
    - On failure: raises JsonParseError with a message tailored to common Windows issues.

    This intentionally does *not* attempt to support JSON5/YAML; it only performs a
    targeted fixup for invalid backslashes inside quoted strings.
    """

    text = str(raw or "").strip()
    if not text:
        return {}

    obj = loads_json_lenient(text, context=context)

    if not isinstance(obj, dict):
        raise JsonParseError(
            context=context,
            original=text,
            message="JSON must be an object (example: { \"cfg\": 7.0 }).",
        )

    return dict(obj)


def loads_json_lenient(raw: str, *, context: str) -> Any:
    """Parse JSON from a user-provided string with a targeted Windows-path fix."""

    text = str(raw or "").strip()
    if not text:
        return None

    try:
        return json.loads(text)
    except json.JSONDecodeError as e1:
        msg = str(e1)
        if "Bad Unicode escape" in msg or "Invalid \\escape" in msg:
            fixed = _escape_invalid_backslashes_in_json_strings(text)
            try:
                return json.loads(fixed)
            except json.JSONDecodeError as e2:
                raise JsonParseError(
                    context=context,
                    original=text,
                    message=(
                        f"Invalid JSON ({e2}). If you pasted a Windows path, escape backslashes: "
                        f'"C:\\\\Users\\\\name\\\\file.wav" (or use forward slashes: "C:/Users/name/file.wav").'
                    ),
                ) from e1
        raise JsonParseError(
            context=context,
            original=text,
            message=f"Invalid JSON ({e1}).",
        ) from e1


def load_json_file_lenient(path: str | Path, *, context: str) -> Any:
    p = Path(path)
    try:
        raw = p.read_text(encoding="utf-8")
    except Exception as e:
        raise JsonParseError(context=context, original=str(p), message=f"Failed to read file: {e}") from e
    return loads_json_lenient(raw, context=context)
