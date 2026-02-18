from __future__ import annotations

import argparse
import socket

from .web import build_demo


def _pick_free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])


def run_desktop(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run the SÖNDBÖUND UI in a native desktop window (embedded webview).")
    p.add_argument("--host", default="127.0.0.1", help="Bind host for the local Gradio server.")
    p.add_argument(
        "--port",
        type=int,
        default=0,
        help="Bind port for the local Gradio server (0 = auto).",
    )
    args = p.parse_args(argv)

    try:
        import webview  # pyright: ignore[reportMissingImports]
    except Exception as e:
        msg = str(e).strip() or e.__class__.__name__
        raise SystemExit(
            "pywebview is required for the desktop window. "
            "Install it with: pip install pywebview\n\n"
            f"Import error: {msg}"
        )

    host = str(args.host).strip() or "127.0.0.1"
    port = int(args.port)
    if port <= 0:
        port = _pick_free_port(host)

    demo = build_demo()

    launch_kwargs = {
        "server_name": host,
        "server_port": port,
        "inbrowser": False,
        "prevent_thread_lock": True,
    }

    local_url: str | None = None
    try:
        res = demo.launch(**launch_kwargs, quiet=True)
    except TypeError:
        # Older Gradio versions might not support `quiet`.
        res = demo.launch(**launch_kwargs)

    if isinstance(res, tuple) and len(res) >= 2:
        # Gradio commonly returns (app, local_url, share_url)
        try:
            local_url = str(res[1])
        except Exception:
            local_url = None
    elif isinstance(res, str):
        local_url = res

    if not local_url:
        local_url = f"http://{host}:{port}"

    webview.create_window("SÖNDBÖUND", local_url, width=1200, height=800)
    webview.start()
    return 0


def main() -> None:
    raise SystemExit(run_desktop())


if __name__ == "__main__":
    main()
