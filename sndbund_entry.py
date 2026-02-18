from __future__ import annotations

import sys


def main() -> int:
    from soundgen.app import main as app_main

    return int(app_main(sys.argv[1:]))


if __name__ == "__main__":
    raise SystemExit(main())
