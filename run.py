#!/usr/bin/env python
"""
Launch the Xubb Agents Simulator web sandbox.

    python run.py                 # http://127.0.0.1:8000
    python run.py --port 9000
    XUBB_AGENTS_PATH=/path/to/xubb_agents python run.py   # explicit framework path

Mock mode needs no API key. For real-LLM mode, set OPENAI_API_KEY (or paste a key
in the UI) and flip the mode toggle.
"""

import argparse
import os
import sys

# Make `sim` importable when run from anywhere.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sim.framework import ensure_framework_importable


def main() -> None:
    parser = argparse.ArgumentParser(description="Xubb Agents Simulator")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="uvicorn auto-reload (dev)")
    args = parser.parse_args()

    try:
        path = ensure_framework_importable()
    except ImportError as e:
        print(f"[xubb-sim] {e}")
        sys.exit(1)

    print(f"[xubb-sim] using framework at: {path}")
    print(f"[xubb-sim] open http://{args.host}:{args.port}")

    # Under --reload, uvicorn spawns a child that re-imports "sim.server:app" in a
    # fresh process which never runs this file's sys.path bootstrap. Export the
    # project root on PYTHONPATH so the reloader child can import `sim`.
    here = os.path.dirname(os.path.abspath(__file__))
    if args.reload:
        existing = os.environ.get("PYTHONPATH", "")
        parts = [here] + ([existing] if existing else [])
        os.environ["PYTHONPATH"] = os.pathsep.join(parts)

    import uvicorn

    try:
        uvicorn.run(
            "sim.server:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            reload_dirs=[here] if args.reload else None,
        )
    except OSError as e:
        print(f"[xubb-sim] could not bind {args.host}:{args.port} — {e}")
        print("[xubb-sim] try a different port: python run.py --port 8001")
        sys.exit(1)


if __name__ == "__main__":
    main()
