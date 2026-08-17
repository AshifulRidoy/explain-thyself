"""SSE framing — hand-rolled, no sse-starlette dependency.

Exactly one terminal event (done|error) per stream, then close.
"""

from __future__ import annotations


def format_sse(event: str | None, data: str) -> str:
    """Format one SSE frame. Multi-line data is split across data: lines."""
    lines = data.splitlines() or [""]
    body = "".join(f"data: {line}\n" for line in lines)
    prefix = f"event: {event}\n" if event else ""
    return f"{prefix}{body}\n"


def heartbeat() -> str:
    """: ping — a comment frame that keeps intermediaries from idling out."""
    return ": ping\n\n"
