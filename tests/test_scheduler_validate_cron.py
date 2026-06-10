"""Tests for scheduler.service._validate_cron."""
import pytest
from localbot.scheduler.service import _validate_cron


@pytest.mark.parametrize("expr", [
    "* * * * *",
    "0 8 * * 1",
    "*/15 * * * *",
    "0 9-17 * * 1-5",
    "30 6 1 1 0",
    "59 23 31 12 6",
])
def test_valid_expressions(expr):
    assert _validate_cron(expr) is None


@pytest.mark.parametrize("expr,reason", [
    ("* * * *",         "only 4 fields"),
    ("* * * * * *",     "6 fields"),
    ("60 * * * *",      "minute 60 out of range"),
    ("* 24 * * *",      "hour 24 out of range"),
    ("* * 0 * *",       "day 0 out of range"),
    ("* * 32 * *",      "day 32 out of range"),
    ("* * * 0 *",       "month 0 out of range"),
    ("* * * 13 *",      "month 13 out of range"),
    ("* * * * 7",       "day_of_week 7 out of range"),
    ("abc * * * *",     "non-numeric minute"),
    ("",               "empty string"),
])
def test_invalid_expressions(expr, reason):
    result = _validate_cron(expr)
    assert result is not None, f"Expected error for {expr!r} ({reason})"
