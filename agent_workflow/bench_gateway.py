#!/usr/bin/env python3
"""
Bench-only PrivAiTe gateway launcher.

Runs the exact same app object as `python -m privaite --config ...` (same
factory, same config path mechanism, same logging setup, single worker) and
adds one read-only route the harness needs and the product deliberately does
not expose: /bench/cachestats, the detection cache's hit/miss/entry counters.
Counters only, never keys, never spans, never text.

This file lives in the benchmark so measuring the cache never requires
touching PrivAiTe source.
"""

from __future__ import annotations

import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to config YAML file")
    args = parser.parse_args()
    os.environ["PRIVAITE_CONFIG_PATH"] = args.config

    import uvicorn

    from privaite.app import create_app

    app = create_app()

    @app.get("/bench/cachestats")
    async def cachestats() -> dict:
        engine = getattr(app.state, "pii_engine", None)
        cache = getattr(engine, "_cache", None) if engine is not None else None
        if cache is None:
            return {"enabled": False}
        return {
            "enabled": True,
            "hits": cache.hits,
            "misses": cache.misses,
            "entries": len(cache),
        }

    cfg = app.state.config
    uvicorn.run(app, host=cfg.server.host, port=cfg.server.port, log_level=cfg.server.log_level)


if __name__ == "__main__":
    main()
