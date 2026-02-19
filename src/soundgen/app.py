from __future__ import annotations

import datetime as _dt
import os
import traceback
from pathlib import Path
import sys


def _launched_from_interactive_terminal() -> bool:
    try:
        return bool(getattr(sys.stdout, "isatty", lambda: False)()) or bool(getattr(sys.stderr, "isatty", lambda: False)())
    except Exception:
        return False


def _should_hide_console_window_windows() -> bool:
    if sys.platform != "win32":
        return False
    # If launched from an interactive terminal, do not hide the console.
    return not _launched_from_interactive_terminal()


def _hide_console_window_windows() -> None:
    """Hide the console window when launching GUI modes on Windows.

    We keep the EXE as a console subsystem build so CLI usage works.
    When the user double-clicks the EXE (defaulting to desktop mode),
    this hides the spawned console for a more app-like experience.
    """

    if sys.platform != "win32":
        return

    if not _should_hide_console_window_windows():
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
        "SÖNDBÖUND (single app)\n\n"
        "Usage:\n"
        "  SÖNDBÖUND.exe                  (opens the desktop UI)\n"
        "  SÖNDBÖUND.exe generate <args>  (CLI generator; same flags as python -m soundgen.generate)\n"
        "  SÖNDBÖUND.exe finetune <args>  (Fine-tuning helpers: train wrapper + validation preview)\n"
        "  SÖNDBÖUND.exe web <args>       (Gradio UI in your browser)\n"
        "  SÖNDBÖUND.exe desktop <args>   (UI in an embedded desktop window)\n"
        "  SÖNDBÖUND.exe project <args>   (Project system: track versions + export packs)\n"
        "  SÖNDBÖUND.exe mobset <args>    (Minecraft mob soundset generator)\n"
        "  SÖNDBÖUND.exe edit <wav>       (Built-in destructive editor)\n"
        "  SÖNDBÖUND.exe regions <args>   (Editor region tools: export regions non-interactively)\n"
        "  SÖNDBÖUND.exe loop <args>      (Loop suite: auto loop points, tail trim, noise bed helpers)\n\n"
        "Help:\n"
        "  SÖNDBÖUND.exe mobset --help\n"
        "  SÖNDBÖUND.exe project --help\n"
        "  SÖNDBÖUND.exe generate --help\n"
        "  SÖNDBÖUND.exe finetune --help\n"
        "  SÖNDBÖUND.exe edit --help\n"
        "  SÖNDBÖUND.exe regions --help\n"
        "  SÖNDBÖUND.exe loop --help\n"
    )


def _startup_log_path() -> Path:
    if sys.platform == "win32":
        la = str(os.environ.get("LOCALAPPDATA") or "").strip()
        base = (Path(la) if la else (Path.home() / "AppData" / "Local")) / "SÖNDBÖUND"
    else:
        base = Path.home() / ".söndböund"
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

    _write_startup_log(f"mode={mode_name} argv={argv!r}")

    # Desktop UI: on Windows, try a best-effort first-run prereq bootstrap.
    if mode_name == "Desktop UI":
        try:
            from .prereqs_windows import ensure_desktop_prereqs_first_run

            ensure_desktop_prereqs_first_run()
        except Exception:
            # Best-effort only; never prevent startup.
            pass

    # Desktop UI: hide the console only after the embedded window is actually up.
    # The desktop launcher will act on this env var when the GUI backend is ready.
    if mode_name == "Desktop UI" and _should_hide_console_window_windows():
        os.environ["SOUNDGEN_HIDE_CONSOLE_ON_READY"] = "1"
    else:
        os.environ.pop("SOUNDGEN_HIDE_CONSOLE_ON_READY", None)
        _hide_console_window_windows()
    try:
        return int(fn())
    except SystemExit as e:
        # Preserve exit codes, but capture the message for users.
        msg = str(e).strip() or repr(e)
        _write_startup_log(f"SystemExit in {mode_name}: {msg}\n{traceback.format_exc()}")
        _show_error_dialog(
            "SÖNDBÖUND failed to start",
            f"{mode_name} failed to start.\n\n{msg}\n\n"
            f"Tip: you can run `SÖNDBÖUND.exe web` to open in your browser.\n"
            f"Log: {_startup_log_path()}",
        )
        raise
    except Exception as e:
        msg = str(e).strip() or e.__class__.__name__
        _write_startup_log(f"Exception in {mode_name}: {msg}\n{traceback.format_exc()}")
        _show_error_dialog(
            "SÖNDBÖUND failed to start",
            f"{mode_name} crashed on startup.\n\n{msg}\n\n"
            f"Tip: you can run `SÖNDBÖUND.exe web` to open in your browser.\n"
            f"Log: {_startup_log_path()}",
        )
        return 1


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    _write_startup_log(f"invoke argv={argv!r}")
    # Default: desktop UI.
    if not argv:
        return _run_gui_mode(
            lambda: __import__("soundgen.desktop", fromlist=["run_desktop"]).run_desktop([]),
            mode_name="Desktop UI",
            argv=[],
        )

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
        return _run_gui_mode(
            lambda: (__import__("soundgen.web", fromlist=["main"]).main() or 0),
            mode_name="Web UI",
            argv=rest,
        )

    if cmd == "desktop":
        return _run_gui_mode(
            lambda: __import__("soundgen.desktop", fromlist=["run_desktop"]).run_desktop(rest),
            mode_name="Desktop UI",
            argv=rest,
        )

    if cmd == "mobset":
        from .mob_soundset import run_mob_soundset

        return int(run_mob_soundset(rest))

    if cmd == "project":
        from .project import run_project

        return int(run_project(rest))

    if cmd == "edit":
        if not rest or rest[0] in {"-h", "--help", "help"}:
            print("Usage: SÖNDBÖUND.exe edit <path-to-wav>")
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
