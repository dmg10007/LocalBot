"""Append-only JSONL audit log.

Records are enqueued without blocking and flushed by a single background
daemon thread, so callers on the asyncio event loop never perform disk I/O.
A threading.Lock still guards the writer so records never interleave.
"""
from __future__ import annotations

import atexit
import json
import os
import queue
import threading
import time
from typing import Any

from localbot.config import cfg

_lock = threading.Lock()
_log_dir_ensured = False
_queue: "queue.Queue[str]" = queue.Queue(maxsize=10_000)
_writer_started = False
_writer_lock = threading.Lock()


def _ensure_log_dir() -> None:
    global _log_dir_ensured
    if _log_dir_ensured:
        return
    log_dir = os.path.dirname(cfg.audit_log_path) or "logs"
    os.makedirs(log_dir, exist_ok=True)
    _log_dir_ensured = True


def _drain() -> None:
    while True:
        line = _queue.get()
        try:
            _ensure_log_dir()
            with _lock:
                with open(cfg.audit_log_path, "a", encoding="utf-8") as fh:
                    fh.write(line)
        except Exception:
            pass  # never let logging crash the writer thread
        finally:
            _queue.task_done()


def _ensure_writer() -> None:
    global _writer_started
    if _writer_started:
        return
    with _writer_lock:
        if _writer_started:
            return
        threading.Thread(target=_drain, name="audit-writer", daemon=True).start()
        atexit.register(lambda: _queue.join())
        _writer_started = True


def log_event(event_type: str, **kwargs: Any) -> None:
    """Enqueue one JSONL record for the background writer.

    Non-blocking: safe to call from the asyncio event loop. If the queue is
    full (writer stalled), the record is dropped rather than blocking callers.
    """
    _ensure_writer()
    record = {"ts": time.time(), "event": event_type, **kwargs}
    line = json.dumps(record, ensure_ascii=False) + "\n"
    try:
        _queue.put_nowait(line)
    except queue.Full:
        pass  # drop under sustained overload to protect latency
