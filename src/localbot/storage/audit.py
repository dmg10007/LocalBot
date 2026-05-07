"""Append-only JSONL audit log."""
from __future__ import annotations

import json
import os
import time
from typing import Any

from localbot.config import cfg


def log_event(event_type: str, **kwargs: Any) -> None:
    # Ensure the log directory exists. This is a no-op after the first call
    # but guards against missing directories if init_db() was not called first.
    log_dir = os.path.dirname(cfg.audit_log_path) or "logs"
    os.makedirs(log_dir, exist_ok=True)

    record = {"ts": time.time(), "event": event_type, **kwargs}
    with open(cfg.audit_log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
