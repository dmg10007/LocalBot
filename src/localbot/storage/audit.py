"""Append-only JSONL audit log.

Write path is guarded by a threading.Lock so concurrent asyncio tasks
that call log_event() from different threads (e.g. APScheduler callbacks)
do not interleave partial JSONL records.
"""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Any

from localbot.config import cfg

_lock = threading.Lock()
_log_dir_ensured = False


def _ensure_log_dir() -> None:
    global _log_dir_ensured
    if _log_dir_ensured:
        return
    log_dir = os.path.dirname(cfg.audit_log_path) or "logs"
    os.makedirs(log_dir, exist_ok=True)
    _log_dir_ensured = True


def log_event(event_type: str, **kwargs: Any) -> None:
    """Append one JSONL record to the audit log.

    Thread-safe: multiple asyncio tasks or APScheduler threads may call
    this concurrently without corrupting the file.
    """
    _ensure_log_dir()
    record = {"ts": time.time(), "event": event_type, **kwargs}
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with _lock:
        with open(cfg.audit_log_path, "a", encoding="utf-8") as fh:
            fh.write(line)
