from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def _appdata_dir() -> Path:
    if sys.platform == "win32":
        la = str(os.environ.get("LOCALAPPDATA") or "").strip()
        base = (Path(la) if la else (Path.home() / "AppData" / "Local")) / "SÖNDBÖUND"
    else:
        base = Path.home() / ".söndböund"
    return base


def _write_log(text: str) -> None:
    try:
        p = _appdata_dir() / "prereqs.log"
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(text.rstrip() + "\n")
    except Exception:
        return


def _message_box_yes_no(title: str, message: str) -> bool:
    if sys.platform != "win32":
        return False
    try:
        import ctypes

        MB_YESNO = 0x00000004
        MB_ICONQUESTION = 0x00000020
        res = ctypes.windll.user32.MessageBoxW(0, str(message), str(title), MB_YESNO | MB_ICONQUESTION)
        return int(res) == 6  # IDYES
    except Exception:
        return False


def _message_box_info(title: str, message: str) -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes

        MB_OK = 0x00000000
        MB_ICONINFORMATION = 0x00000040
        ctypes.windll.user32.MessageBoxW(0, str(message), str(title), MB_OK | MB_ICONINFORMATION)
    except Exception:
        return


def _winget_available() -> bool:
    return bool(shutil.which("winget") or shutil.which("winget.exe"))


def _run_winget_install(package_id: str) -> bool:
    if not _winget_available():
        return False

    cmd = [
        "winget",
        "install",
        "-e",
        "--id",
        str(package_id),
        "--accept-source-agreements",
        "--accept-package-agreements",
    ]

    try:
        creationflags = 0
        if sys.platform == "win32":
            creationflags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
        subprocess.Popen(cmd, creationflags=creationflags)
        _write_log(f"winget started: {cmd!r}")
        return True
    except Exception as e:
        _write_log(f"winget failed to start ({package_id}): {e}")
        return False


def _ffmpeg_missing() -> bool:
    try:
        from .io_utils import find_ffmpeg

        _ = find_ffmpeg()
        return False
    except Exception:
        return True


def _webview2_installed() -> bool:
    if sys.platform != "win32":
        return True

    # Official WebView2 Runtime client GUID used by EdgeUpdate.
    # We check both 64-bit and WOW6432Node for safety.
    key_paths = [
        r"SOFTWARE\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}",
        r"SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}",
    ]

    try:
        import winreg

        for kp in key_paths:
            try:
                with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, kp) as k:
                    pv, _ = winreg.QueryValueEx(k, "pv")
                    if str(pv).strip():
                        return True
            except FileNotFoundError:
                continue
            except OSError:
                continue
    except Exception:
        # If registry checks fail, don't block startup.
        return True

    return False


def ensure_desktop_prereqs_first_run() -> None:
    """Best-effort first-run dependency setup for the packaged desktop app.

    This runs only on Windows, and only when NOT launched from an interactive terminal.
    It does not hard-fail the app: it either launches installers or provides guidance.
    """

    if sys.platform != "win32":
        return

    try:
        is_tty = bool(getattr(sys.stdout, "isatty", lambda: False)()) or bool(
            getattr(sys.stderr, "isatty", lambda: False)()
        )
    except Exception:
        is_tty = False

    # If a user is running from a terminal, they likely don't want prompts.
    if is_tty:
        return

    marker = _appdata_dir() / "first_run_prereqs_done.txt"
    try:
        if marker.exists():
            return
    except Exception:
        pass

    missing_ffmpeg = _ffmpeg_missing()
    missing_webview2 = not _webview2_installed()

    _write_log(f"first_run: ffmpeg_missing={missing_ffmpeg} webview2_missing={missing_webview2}")

    # WebView2 is recommended for best desktop experience (edgechromium backend).
    if missing_webview2:
        if _winget_available() and _message_box_yes_no(
            "SÖNDBÖUND setup",
            "Microsoft WebView2 Runtime is not detected.\n\n"
            "Install it now (recommended for the desktop window)?\n\n"
            "This will open a Windows installer in a new console window.",
        ):
            started = _run_winget_install("Microsoft.EdgeWebView2Runtime")
            if started:
                _message_box_info(
                    "SÖNDBÖUND setup",
                    "WebView2 install has been started.\n\n"
                    "If the desktop window fails to open, finish the install and relaunch SÖNDBÖUND.",
                )
        else:
            _message_box_info(
                "SÖNDBÖUND setup",
                "Microsoft WebView2 Runtime is not detected.\n\n"
                "If the desktop window fails to open, install WebView2 Runtime and relaunch.\n"
                "Tip (Windows): winget install Microsoft.EdgeWebView2Runtime",
            )

    # ffmpeg is required for .ogg/.mp3 exports and some library decoding.
    if missing_ffmpeg:
        if _winget_available() and _message_box_yes_no(
            "SÖNDBÖUND setup",
            "ffmpeg is not detected.\n\n"
            "Install it now (required for .ogg/.mp3 export and some conversions)?\n\n"
            "This will open a Windows installer in a new console window.",
        ):
            started = _run_winget_install("Gyan.FFmpeg")
            if started:
                _message_box_info(
                    "SÖNDBÖUND setup",
                    "ffmpeg install has been started.\n\n"
                    "Finish the install, then relaunch SÖNDBÖUND for exports to work.",
                )
        else:
            _message_box_info(
                "SÖNDBÖUND setup",
                "ffmpeg is not detected.\n\n"
                "Exports that require conversion (like .ogg) will not work until ffmpeg is installed.\n"
                "Tip (Windows): winget install Gyan.FFmpeg",
            )

    try:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("ok\n", encoding="utf-8")
    except Exception:
        pass
