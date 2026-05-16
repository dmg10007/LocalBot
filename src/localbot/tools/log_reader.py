"""Audit-log reader for self-diagnostics.

Exposes a single `read_logs` function that filters the append-only JSONL
audit log and returns a JSON string suitable for the LLM to reason over.

Security model
--------------
* By default every call is scoped to the requesting user's ``user_id`` so
  the LLM can never surface another user's conversation data even if it
  tries.
* A ``BOT_OWNER_ID`` env var (optional) designates a single trusted user
  who may pass ``user_id=None`` to query the full log (useful for
  diagnosing global errors like scheduler crashes or OOM kills).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from localbot.config import cfg

# Hard cap — prevents the LLM from injecting so many log entries that it
# exhausts the context window.  Even at 200 entries the JSON is ~40 KB
# which is comfortably within typical ctx sizes.
_MAX_ENTRIES = 200

# Optional: the Discord user ID of the bot owner.  When set, that user may
# call read_logs without a user_id filter to see the full audit log.
_OWNER_ID: str | None = os.environ.get("BOT_OWNER_ID", "").strip() or None

_VALID_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR"})


def read_logs(
    requesting_user_id: str,
    level: str | None = None,
    limit: int = 50,
    user_id: str | None = None,
) -> str:
    """Return recent audit-log entries as a JSON string.

    Args:
        requesting_user_id: The Discord user ID making the request.  Used
            to enforce the per-user scope unless the requester is the
            designated bot owner.
        level: Optional filter — one of DEBUG / INFO / WARNING / ERROR.
            Matches the ``event`` field for audit entries that carry a
            severity concept, and the log-level prefix for raw lines.
        limit: Maximum number of entries to return.  Clamped to
            ``_MAX_ENTRIES`` (200) regardless of what the LLM passes.
        user_id: Explicit user scope.  Non-owner callers may only pass
            their own ``requesting_user_id`` here; any other value is
            silently overridden.
    """
    limit = max(1, min(limit, _MAX_ENTRIES))

    # Scope enforcement: non-owners always see only their own entries.
    is_owner = _OWNER_ID is not None and requesting_user_id == _OWNER_ID
    if not is_owner:
        user_id = requesting_user_id

    if level is not None:
        level = level.upper()
        if level not in _VALID_LEVELS:
            return f"Invalid level {level!r}. Choose from: {', '.join(sorted(_VALID_LEVELS))}."

    path = Path(cfg.audit_log_path)
    if not path.exists():
        return "No audit log found."

    raw_lines = path.read_text(errors="replace").splitlines()
    results: list[dict[str, Any]] = []

    for raw in reversed(raw_lines):  # newest-first
        raw = raw.strip()
        if not raw:
            continue
        try:
            entry = json.loads(raw)
        except json.JSONDecodeError:
            continue

        # user_id filter
        if user_id and entry.get("user_id") != user_id:
            continue

        # level filter — audit entries don't have a "level" key, so we
        # match WARNING/ERROR by event type heuristics instead.
        if level:
            event = entry.get("event", "").lower()
            entry_level = _infer_level(event)
            if entry_level != level:
                continue

        results.append(entry)
        if len(results) >= limit:
            break

    if not results:
        return "No matching log entries found."

    return json.dumps(results, indent=2)


def _infer_level(event: str) -> str:
    """Map an audit event name to a notional log level for filtering."""
    if event in ("assistant_reply", "user_message", "tool_call", "tool_result"):
        return "INFO"
    if "error" in event or "fail" in event or "crash" in event:
        return "ERROR"
    if "warn" in event or "timeout" in event or "missed" in event:
        return "WARNING"
    return "INFO"
