"""Append-only JSONL audit log."""
from __future__ import annotations

import json
import time
from typing import Any

from localbot.config import cfg


def log_event(event_type: str, **kwargs: Any) -> None:
    record = {"ts": time.time(), "event": event_type, **kwargs}
    with open(cfg.audit_log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
