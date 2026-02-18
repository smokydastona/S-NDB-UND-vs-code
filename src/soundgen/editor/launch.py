from __future__ import annotations

from pathlib import Path


def launch_editor(wav_path: str | Path) -> None:
    """Launch the built-in editor UI.

    This is a stub/scaffold so the repo has a stable import path while the
    full editor UI is implemented.
    """

    wav_path = Path(wav_path)
    if not wav_path.exists():
        raise FileNotFoundError(str(wav_path))

    try:
        import PySide6  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Editor UI dependencies are not installed. Install PySide6 to use the built-in editor."
        ) from e

    raise NotImplementedError(
        "Editor UI scaffold is present, but the UI is not implemented yet. "
        "See docs/editor_v1_ui_layout.md for the v1 layout spec."
    )
