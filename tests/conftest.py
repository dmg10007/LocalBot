"""
pytest conftest — stubs out localbot.config before any localbot module loads,
so tests can run without real env vars or a running llama-server.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys
import types
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# 1. Register the real 'localbot' package so sub-module imports resolve to
#    the source tree instead of a bare stub.
# ---------------------------------------------------------------------------
_HERE = pathlib.Path(__file__).parent.parent  # repo root
_candidates = [
    _HERE / "src" / "localbot",   # standard src-layout
    _HERE / "localbot",           # flat dev layout
]
_PKG_PATH = next((str(p) for p in _candidates if p.exists()), None)

if _PKG_PATH and "localbot" not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        "localbot",
        _PKG_PATH + "/__init__.py",
        submodule_search_locations=[_PKG_PATH],
    )
    pkg = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    pkg.__path__ = [_PKG_PATH]  # type: ignore[attr-defined]
    sys.modules["localbot"] = pkg

# ---------------------------------------------------------------------------
# 2. Stub localbot.config *before* any localbot import runs.
# ---------------------------------------------------------------------------
stub_cfg = MagicMock()
stub_cfg.sandbox_root = ""
stub_cfg.github_token = ""
stub_cfg.tool_timeout_seconds = 10
stub_cfg.max_tool_result_chars = 4_000
stub_cfg.history_max_messages = 50
stub_cfg.request_deadline_seconds = 30
stub_cfg.max_tool_iterations = 8

stub_config_mod = types.ModuleType("localbot.config")
stub_config_mod.cfg = stub_cfg  # type: ignore[attr-defined]
sys.modules["localbot.config"] = stub_config_mod

# ---------------------------------------------------------------------------
# 3. Stub heavy optional runtime dependencies so collection never fails in
#    environments where they are not installed (CI without Discord / APScheduler).
# ---------------------------------------------------------------------------
for _mod_name in [
    "apscheduler",
    "apscheduler.schedulers",
    "apscheduler.schedulers.asyncio",
    "apscheduler.triggers",
    "apscheduler.triggers.cron",
    "discord",
    "discord.ext",
    "discord.ext.commands",
]:
    sys.modules.setdefault(_mod_name, types.ModuleType(_mod_name))
