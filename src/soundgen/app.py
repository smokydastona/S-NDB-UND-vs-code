from __future__ import annotations

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
        "Sound Generator (single app)\n\n"
        "Usage:\n"
        "  soundgen.exe                  (opens the desktop UI)\n"
        "  soundgen.exe generate <args>  (CLI generator; same flags as python -m soundgen.generate)\n"
        "  soundgen.exe web <args>       (Gradio UI in your browser)\n"
        "  soundgen.exe desktop <args>   (UI in an embedded desktop window)\n"
        "  soundgen.exe mobset <args>    (Minecraft mob soundset generator)\n\n"
        "Help:\n"
        "  soundgen.exe mobset --help\n"
        "  soundgen.exe generate --help\n"
    )


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # Default: desktop UI.
    if not argv:
        _hide_console_window_windows()
        from .desktop import run_desktop

        return int(run_desktop([]))

    cmd = str(argv[0]).strip().lower()
    rest = [str(x) for x in argv[1:]]

    if cmd in {"-h", "--help", "help"}:
        _print_help()
        return 0

    if cmd == "generate":
        from .generate import main as generate_main

        return int(generate_main(rest))

    if cmd == "web":
        _hide_console_window_windows()
        from .web import main as web_main

        web_main()
        return 0

    if cmd == "desktop":
        _hide_console_window_windows()
        from .desktop import run_desktop

        return int(run_desktop(rest))

    if cmd == "mobset":
        from .mob_soundset import run_mob_soundset

        return int(run_mob_soundset(rest))

    # Unknown subcommand.
    _print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
