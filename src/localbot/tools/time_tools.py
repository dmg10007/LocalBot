"""Current time / timezone helpers."""
from __future__ import annotations

import datetime
import zoneinfo


def get_current_time(timezone: str = "UTC") -> str:
    try:
        tz = zoneinfo.ZoneInfo(timezone)
    except Exception:
        return f"Unknown timezone: {timezone}"
    now = datetime.datetime.now(tz)
    return now.strftime(f"%A, %B %d %Y %I:%M %p %Z")
