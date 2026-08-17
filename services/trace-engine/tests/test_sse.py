"""SSE framing symmetry."""

from __future__ import annotations

from app.engine.sse import format_sse, heartbeat


def test_single_line_frame() -> None:
    assert format_sse("done", '{"ok":true}') == 'event: done\ndata: {"ok":true}\n\n'


def test_multi_line_data() -> None:
    frame = format_sse("trace_event", "line1\nline2")
    assert frame == "event: trace_event\ndata: line1\ndata: line2\n\n"


def test_empty_event_name_omitted() -> None:
    assert format_sse(None, "x") == "data: x\n\n"


def test_heartbeat_is_comment() -> None:
    assert heartbeat() == ": ping\n\n"
