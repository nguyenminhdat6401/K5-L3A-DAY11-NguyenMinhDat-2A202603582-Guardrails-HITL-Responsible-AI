"""
Assignment 11 — Audit Log.

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from guardrails.output_guardrails import content_filter


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, tuple[str, str, float]] = {}

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Capture a redacted input and start the request timer."""
        key = request_id or user_id
        safe_text = content_filter(text)["redacted"]
        self._open[key] = (safe_text, utc_now_iso(), time.monotonic())

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Record the response and close its pending audit entry."""
        key = request_id or user_id
        pending = self._open.pop(key, None)
        input_text, timestamp, start = pending or ("", utc_now_iso(), time.monotonic())
        self.logs.append({
            "request_id": key,
            "user_id": user_id,
            "timestamp": timestamp,
            "input": input_text,
            "output": content_filter(text)["redacted"],
            "blocked": blocked,
            "layer": layer,
            "latency_ms": round((time.monotonic() - start) * 1000, 3),
        })

    def export_json(self, filepath: str | None = None):
        """Write the audit log under repo-root outputs by default."""
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.logs, indent=2, ensure_ascii=False), encoding="utf-8")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
