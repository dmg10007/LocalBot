"""Append-only JSONL audit log."""
from __future__ import annotations

import json
import os
import time
from typing import Any

from localbot.config import cfg

# Fix #8: guard against repeated os.makedirs calls on every log_event
# invocation. init_db() creates the directory at startup; this flag avoids
# redundant syscalls on every subsequent write while still providing a
# safe fallback if audit.py is used before init_db().
_log_dir_ensured = False


def log_event(event_type: str, **kwargs: Any) -> None:
    global _log_dir_ensured
    if not _log_dir_ensured:
        log_dir = os.path.dirname(cfg.audit_log_path) or "logs"
        os.makedirs(log_dir, exist_ok=True)
        _log_dir_ensured = True

    record = {"ts": time.time(), "event": event_type, **kwargs}
    with open(cfg.audit_log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
