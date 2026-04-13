"""CLI entrypoint: python -m maestro or maestro (via pyproject console_scripts).

Requires the [server] extra: pip install maestro[server]
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Maestro LLM platform server")
    parser.add_argument("--config", default="config.yaml", help="configuration file path")
    parser.add_argument("--host", default="0.0.0.0", help="bind address")
    parser.add_argument("--port", type=int, default=8080, help="server port")

    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError:
        raise SystemExit("uvicorn not found — install with: pip install maestro[server]")

    from maestro.config import load
    from maestro.server import create_app

    cfg = load(args.config)
    app = create_app(config=cfg)

    # Parse port from config address (Go-style ":8080") if not overridden
    port = args.port
    if cfg.address:
        try:
            port = int(cfg.address.rsplit(":", 1)[-1])
        except (ValueError, IndexError):
            pass

    uvicorn.run(app, host=args.host, port=port)


if __name__ == "__main__":
    main()
