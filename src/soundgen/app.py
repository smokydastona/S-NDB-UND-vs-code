from __future__ import annotations

import datetime as _dt
import traceback
from pathlib import Path
import sys


def _hide_console_window_windows() -> None:
    """Hide the console window when launching GUI modes on Windows.

    We keep the EXE as a console subsystem build so CLI usage works.
    When the user double-clicks the EXE (defaulting to desktop mode),
    this hides the spawned console for a more app-like experience.
    """

    if sys.platform != "win32":
        return

    try:
        import ctypes

        hwnd = ctypes.windll.kernel32.GetConsoleWindow()
        if hwnd:
            ctypes.windll.user32.ShowWindow(hwnd, 0)  # SW_HIDE
    except Exception:
        # Best-effort only; never fail app startup because of this.
        return


def _print_help() -> None:
    print(
        "S-NDB-UND (single app)\n\n"
        "Usage:\n"
        "  S-NDB-UND.exe                  (opens the desktop UI)\n"
        "  S-NDB-UND.exe generate <args>  (CLI generator; same flags as python -m soundgen.generate)\n"
        "  S-NDB-UND.exe finetune <args>  (Fine-tuning helpers: train wrapper + validation preview)\n"
        "  S-NDB-UND.exe web <args>       (Gradio UI in your browser)\n"
        "  S-NDB-UND.exe desktop <args>   (UI in an embedded desktop window)\n"
        "  S-NDB-UND.exe project <args>   (Project system: track versions + export packs)\n"
        "  S-NDB-UND.exe mobset <args>    (Minecraft mob soundset generator)\n"
        "  S-NDB-UND.exe edit <wav>       (Built-in destructive editor)\n"
        "  S-NDB-UND.exe regions <args>   (Editor region tools: export regions non-interactively)\n"
        "  S-NDB-UND.exe loop <args>      (Loop suite: auto loop points, tail trim, noise bed helpers)\n\n"
        "Help:\n"
        "  S-NDB-UND.exe mobset --help\n"
        "  S-NDB-UND.exe project --help\n"
        "  S-NDB-UND.exe generate --help\n"
        "  S-NDB-UND.exe finetune --help\n"
        "  S-NDB-UND.exe edit --help\n"
        "  S-NDB-UND.exe regions --help\n"
        "  S-NDB-UND.exe loop --help\n"
    )


def _startup_log_path() -> Path:
    if sys.platform == "win32":
        base = Path.home() / "AppData" / "Local" / "S-NDB-UND"
    else:
        base = Path.home() / ".sndbund"
    return base / "startup.log"


def _write_startup_log(text: str) -> None:
    try:
        p = _startup_log_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        stamp = _dt.datetime.now().isoformat(timespec="seconds")
        with p.open("a", encoding="utf-8") as f:
            f.write(f"[{stamp}] {text}\n")
    except Exception:
        return


def _show_error_dialog(title: str, message: str) -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.user32.MessageBoxW(0, str(message), str(title), 0x10)  # MB_ICONERROR
    except Exception:
        return


def _run_gui_mode(fn, *, mode_name: str, argv: list[str]) -> int:
    """Run a GUI-ish mode safely (console may be hidden on Windows)."""

    _hide_console_window_windows()
    _write_startup_log(f"mode={mode_name} argv={argv!r}")
    try:
        return int(fn())
    except SystemExit as e:
        # Preserve exit codes, but capture the message for users.
        msg = str(e).strip() or repr(e)
        _write_startup_log(f"SystemExit in {mode_name}: {msg}\n{traceback.format_exc()}")
        _show_error_dialog(
            "S-NDB-UND failed to start",
            f"{mode_name} failed to start.\n\n{msg}\n\n"
            f"Tip: you can run `S-NDB-UND.exe web` to open in your browser.\n"
            f"Log: {_startup_log_path()}",
        )
        raise
    except Exception as e:
        msg = str(e).strip() or e.__class__.__name__
        _write_startup_log(f"Exception in {mode_name}: {msg}\n{traceback.format_exc()}")
        _show_error_dialog(
            "S-NDB-UND failed to start",
            f"{mode_name} crashed on startup.\n\n{msg}\n\n"
            f"Tip: you can run `S-NDB-UND.exe web` to open in your browser.\n"
            f"Log: {_startup_log_path()}",
        )
        return 1


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # Default: desktop UI.
    if not argv:
        from .desktop import run_desktop

        return _run_gui_mode(lambda: run_desktop([]), mode_name="Desktop UI", argv=[])

    cmd = str(argv[0]).strip().lower()
    rest = [str(x) for x in argv[1:]]

    if cmd in {"-h", "--help", "help"}:
        _print_help()
        return 0

    if cmd == "generate":
        from .generate import main as generate_main

        return int(generate_main(rest))

    if cmd == "finetune":
        from .finetune import main as finetune_main

        finetune_main(rest)
        return 0

    if cmd == "web":
        from .web import main as web_main

        return _run_gui_mode(lambda: (web_main() or 0), mode_name="Web UI", argv=rest)

    if cmd == "desktop":
        from .desktop import run_desktop

        return _run_gui_mode(lambda: run_desktop(rest), mode_name="Desktop UI", argv=rest)

    if cmd == "mobset":
        from .mob_soundset import run_mob_soundset

        return int(run_mob_soundset(rest))

    if cmd == "project":
        from .project import run_project

        return int(run_project(rest))

    if cmd == "edit":
        if not rest or rest[0] in {"-h", "--help", "help"}:
            print("Usage: S-NDB-UND.exe edit <path-to-wav>")
            return 0
        from .editor import launch_editor

        launch_editor(rest[0])
        return 0

    if cmd == "regions":
        from .editor_regions import main as regions_main

        return int(regions_main(rest))

    if cmd == "loop":
        from .loop_suite import run_loop_suite

        return int(run_loop_suite(rest))

    # Unknown subcommand.
    _print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
