#!/usr/bin/env python3
"""
Main entry point for the Tool Healing Proxy.

This module provides the CLI interface for running the proxy. It acts as a
bridge between an OpenAI-compatible client (like llama-switch) and an
upstream server (like beellama-server), implementing Unsloth-style tool healing.

The proxy:
  - Detects tool calls emitted as XML in content.
  - Strips tool XML from streamed content before forwarding.
  - Converts XML tool calls into structured tool_calls deltas.

Usage:
    python __main__.py --upstream http://127.0.0.1:8080 --port 8081
"""

import argparse
import logging

# Configure logging at the entry point
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("tool_healing_proxy")

# Default upstream kept local to the top-level entry file for discoverability
_UPSTREAM_URL_DEFAULT = "http://127.0.0.1:8080"
from studio.backend.core.tool_healing import (
    strip_tool_call_markup as _strip_tool_markup,
    parse_tool_calls_from_text as _parse_tool_calls_from_text,
)
from tool_healing.streaming import (
    heal_chat_completion_stream as _heal_chat_completion_stream,
)
from tool_healing.api import build_app as _build_app
from tool_healing.streaming import DEFAULT_TOOL_SIGNALS as _DEFAULT_TOOL_SIGNALS

# ── Upstream configuration ───────────────────────────────────
# The default upstream server URL (e.g. beellama-server)
UPSTREAM_URL = _UPSTREAM_URL_DEFAULT


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="Tool Healing Proxy")
    parser.add_argument("--upstream", default=_UPSTREAM_URL_DEFAULT,
                        help="beellama-server base URL")
    parser.add_argument("--timeout", type=float, default=600.0,
                        help="Upstream read timeout in seconds")
    parser.add_argument("--port", type=int, default=8081,
                        help="Port to listen on")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Host to bind to")
    parser.add_argument("--signals", nargs="+", default=_DEFAULT_TOOL_SIGNALS,
                        help="XML signals that indicate a tool call (space separated)")
    args = parser.parse_args()

    UPSTREAM_URL = args.upstream
    signals = tuple(args.signals)
    # Build app with the provided upstream URL, timeout and signals
    app = _build_app(UPSTREAM_URL, default_timeout=args.timeout, signals=signals)
    logger.info(f"Tool Healing Proxy: {args.host}:{args.port} -> {UPSTREAM_URL} (timeout={args.timeout}s, signals={signals})")
    uvicorn.run(app, host=args.host, port=args.port)