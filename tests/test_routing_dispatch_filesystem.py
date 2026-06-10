"""
Async tests for agent routing, tool dispatch, and filesystem sandbox.

Test groups
-----------
TestSelectSlot          — _select_slot()  intent → model slot
TestDetectWorkspaceMode — _detect_workspace_mode()  intent → workspace mode
TestNeedsTools          — _needs_tools()  should tools be available this turn?
TestDispatch            — registry.dispatch()  routes to correct tool functions
TestFilesystemSandbox   — sandbox path guard, binary block, all tool functions
"""
from __future__ import annotations

import asyncio
import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from localbot.agent import _detect_workspace_mode, _needs_tools, _select_slot
from localbot.tools import filesystem, registry


# ===========================================================================
# _select_slot
# ===========================================================================

class TestSelectSlot:
    def test_reasoning_beats_coding(self):
        assert _select_slot("Can you review this code and suggest the best design approach?") == "reasoning"

    def test_pure_coding(self):
        assert _select_slot("Write a function that reverses a list in Python.") == "coding"

    def test_coding_via_pr_keyword(self):
        assert _select_slot("Open a pull request with these changes.") == "coding"

    def test_general_fallback(self):
        assert _select_slot("What time is it in Tokyo?") == "general"

    def test_general_for_casual(self):
        assert _select_slot("hey, how are you?") == "general"

    def test_reasoning_on_compare(self):
        assert _select_slot("Compare asyncio vs threading for this use case.") == "reasoning"

    def test_reasoning_on_analyze(self):
        assert _select_slot("Analyze the trade-offs of this architecture.") == "reasoning"

    def test_coding_on_pytest(self):
        assert _select_slot("Add pytest tests for the new module.") == "coding"


# ===========================================================================
# _detect_workspace_mode
# ===========================================================================

class TestDetectWorkspaceMode:
    def test_local_keywords(self):
        assert _detect_workspace_mode("Read this local file for me.") == "local"

    def test_sandbox_implies_local(self):
        assert _detect_workspace_mode("Look at the file in my sandbox.") == "local"

    def test_github_commit_keyword(self):
        assert _detect_workspace_mode("Commit these changes to the repo.") == "github"

    def test_github_explicit(self):
        assert _detect_workspace_mode("Commit this to the repo on GitHub.") == "github"

    def test_push_to_implies_github(self):
        assert _detect_workspace_mode("Push to the GitHub repo.") == "github"

    def test_pr_implies_github(self):
        assert _detect_workspace_mode("Open a PR with my changes.") == "github"

    def test_both_when_mixed(self):
        msg = "Read the local file and commit it to the GitHub repo."
        assert _detect_workspace_mode(msg) == "both"

    def test_none_for_unrelated(self):
        assert _detect_workspace_mode("What is the weather like?") is None

    def test_tilde_implies_local(self):
        assert _detect_workspace_mode("Read ~/config.py please.") == "local"


# ===========================================================================
# _needs_tools
# ===========================================================================

class TestNeedsTools:
    def test_search_intent_needs_tools(self):
        assert _needs_tools("Search for the latest AI news.", []) is True

    def test_lookup_intent_needs_tools(self):
        assert _needs_tools("Look up the current Bitcoin price.", []) is True

    def test_schedule_intent_needs_tools_when_scheduler_present(self):
        assert _needs_tools("Remind me every morning at 8am.", [], has_scheduler=True) is True

    def test_schedule_without_scheduler_returns_bool(self):
        result = _needs_tools("Remind me every morning at 8am.", [], has_scheduler=False)
        assert isinstance(result, bool)

    def test_cancel_intent_needs_tools(self):
        assert _needs_tools("Cancel job abc123.", [], has_scheduler=True) is True

    def test_diagnostic_intent_needs_tools(self):
        assert _needs_tools("Check the logs for errors.", []) is True

    def test_workspace_mode_forces_tools(self):
        assert _needs_tools("Read main.py in my sandbox.", [], workspace_mode="local") is True

    def test_casual_greeting_no_tools(self):
        assert _needs_tools("thanks", []) is False

    def test_casual_ok_no_tools(self):
        assert _needs_tools("ok", []) is False


# ===========================================================================
# registry.dispatch()
# ===========================================================================

@pytest.mark.asyncio
class TestDispatch:
    async def test_web_search_routes_correctly(self):
        with patch.object(registry.search, "web_search", new=AsyncMock(return_value="search result")) as m:
            result = await registry.dispatch("web_search", {"query": "pytest async"})
        m.assert_awaited_once_with("pytest async")
        assert result == "search result"

    async def test_reddit_search_routes_correctly(self):
        with patch.object(registry.reddit, "reddit_search", new=AsyncMock(return_value="reddit result")) as m:
            result = await registry.dispatch("reddit_search", {"query": "asyncio", "subreddit": "python"})
        m.assert_awaited_once_with("asyncio", "python")
        assert result == "reddit result"

    async def test_reddit_search_no_subreddit(self):
        with patch.object(registry.reddit, "reddit_search", new=AsyncMock(return_value="r")) as m:
            await registry.dispatch("reddit_search", {"query": "news"})
        m.assert_awaited_once_with("news", None)

    async def test_get_current_time_routes_correctly(self):
        with patch.object(registry.time_tools, "get_current_time", return_value="12:00 UTC") as m:
            result = await registry.dispatch("get_current_time", {"timezone": "UTC"})
        m.assert_called_once_with("UTC")
        assert result == "12:00 UTC"

    async def test_get_current_time_defaults_to_utc(self):
        with patch.object(registry.time_tools, "get_current_time", return_value="now") as m:
            await registry.dispatch("get_current_time", {})
        m.assert_called_once_with("UTC")

    async def test_unknown_tool_returns_error_string(self):
        result = await registry.dispatch("does_not_exist", {})
        assert "Unknown tool" in result

    async def test_tool_exception_returns_error_string(self):
        with patch.object(registry.search, "web_search", new=AsyncMock(side_effect=RuntimeError("boom"))):
            result = await registry.dispatch("web_search", {"query": "x"})
        assert "error" in result.lower()

    async def test_tool_timeout_returns_timeout_message(self):
        async def _slow(*_a, **_kw):
            await asyncio.sleep(999)

        original = registry.cfg.tool_timeout_seconds
        registry.cfg.tool_timeout_seconds = 0.01
        try:
            with patch.object(registry.search, "web_search", new=_slow):
                result = await registry.dispatch("web_search", {"query": "x"})
        finally:
            registry.cfg.tool_timeout_seconds = original
        assert "timed out" in result.lower()

    async def test_read_file_dispatched_via_registry(self):
        with patch("localbot.tools.registry.read_file", return_value="file content") as m:
            result = await registry.dispatch("read_file", {"path": "hello.txt"})
        m.assert_called_once_with("hello.txt")
        assert result == "file content"

    async def test_write_file_dispatched_via_registry(self):
        with patch("localbot.tools.registry.write_file", return_value="OK: wrote 5 chars") as m:
            result = await registry.dispatch("write_file", {"path": "out.txt", "content": "hello"})
        m.assert_called_once_with("out.txt", "hello")
        assert "OK" in result

    async def test_list_directory_dispatched_via_registry(self):
        with patch("localbot.tools.registry.list_directory", return_value="[D] src/") as m:
            result = await registry.dispatch("list_directory", {"path": "."})
        m.assert_called_once_with(".")
        assert "[D]" in result

    async def test_apply_patch_dispatched_via_registry(self):
        with patch("localbot.tools.registry.apply_patch", return_value="OK: patch applied") as m:
            result = await registry.dispatch("apply_patch", {"path": "f.py", "patch": "@@ ... @@"})
        m.assert_called_once_with("f.py", "@@ ... @@")
        assert "OK" in result

    async def test_search_in_files_dispatched_via_registry(self):
        with patch("localbot.tools.registry.search_in_files", return_value="f.py:1: match") as m:
            result = await registry.dispatch("search_in_files", {"pattern": "foo", "path": ".", "file_glob": "*.py"})
        m.assert_called_once_with("foo", path=".", file_glob="*.py")
        assert "match" in result


# ===========================================================================
# Filesystem sandbox guard and tool functions
# ===========================================================================

class TestFilesystemSandbox:
    """All tests operate against a real temp directory — no filesystem mocking."""

    @pytest.fixture(autouse=True)
    def tmp_sandbox(self, tmp_path, monkeypatch):
        monkeypatch.setattr(filesystem.cfg, "sandbox_root", str(tmp_path))
        self.root = tmp_path

    # --- path traversal guard ---

    def test_traversal_attack_blocked(self):
        with pytest.raises(PermissionError, match="escapes the sandbox"):
            filesystem._safe_resolve("../../etc/passwd")

    def test_deep_traversal_blocked(self):
        with pytest.raises(PermissionError, match="escapes the sandbox"):
            filesystem._safe_resolve("a/b/../../../etc/passwd")

    def test_absolute_path_jailed_to_sandbox(self):
        resolved = filesystem._safe_resolve("/etc/passwd")
        assert resolved.is_relative_to(self.root)

    def test_normal_relative_path_ok(self):
        resolved = filesystem._safe_resolve("subdir/file.py")
        assert resolved.is_relative_to(self.root)

    # --- binary extension guard ---

    def test_binary_extension_blocked_on_read(self):
        with pytest.raises(ValueError, match="binary file type"):
            filesystem._check_binary(Path("image.png"))

    def test_binary_extension_blocked_on_write(self):
        with pytest.raises(ValueError, match="binary file type"):
            filesystem.write_file("archive.zip", "data")

    def test_pdf_extension_blocked(self):
        with pytest.raises(ValueError, match="binary file type"):
            filesystem._check_binary(Path("report.pdf"))

    def test_text_extensions_allowed(self):
        for name in ("main.py", "README.md", "config.toml", "data.json"):
            filesystem._check_binary(Path(name))

    # --- read_file ---

    def test_read_existing_file(self):
        (self.root / "hello.txt").write_text("world", encoding="utf-8")
        assert filesystem.read_file("hello.txt") == "world"

    def test_read_missing_file_returns_error(self):
        result = filesystem.read_file("nope.txt")
        assert result.startswith("ERROR:")

    def test_read_directory_returns_error(self):
        (self.root / "adir").mkdir()
        result = filesystem.read_file("adir")
        assert result.startswith("ERROR:")

    def test_read_truncates_large_file(self):
        (self.root / "big.txt").write_text("x" * 40_000, encoding="utf-8")
        result = filesystem.read_file("big.txt")
        assert "truncated" in result

    def test_read_nested_path(self):
        (self.root / "pkg").mkdir()
        (self.root / "pkg" / "mod.py").write_text("pass", encoding="utf-8")
        assert filesystem.read_file("pkg/mod.py") == "pass"

    # --- write_file ---

    def test_write_creates_file(self):
        filesystem.write_file("new.py", "print('hi')")
        assert (self.root / "new.py").read_text() == "print('hi')"

    def test_write_creates_parent_dirs(self):
        filesystem.write_file("sub/dir/file.py", "pass")
        assert (self.root / "sub" / "dir" / "file.py").exists()

    def test_write_returns_ok_confirmation(self):
        result = filesystem.write_file("out.txt", "hello")
        assert result.startswith("OK:")

    def test_write_overwrites_existing(self):
        (self.root / "f.txt").write_text("old")
        filesystem.write_file("f.txt", "new")
        assert (self.root / "f.txt").read_text() == "new"

    # --- list_directory ---

    def test_list_shows_files_and_dirs(self):
        (self.root / "a.py").write_text("x")
        (self.root / "subdir").mkdir()
        result = filesystem.list_directory(".")
        assert "[F] a.py" in result
        assert "[D] subdir/" in result

    def test_list_missing_path_returns_error(self):
        result = filesystem.list_directory("nonexistent")
        assert result.startswith("ERROR:")

    def test_list_file_path_returns_error(self):
        (self.root / "f.py").write_text("x")
        result = filesystem.list_directory("f.py")
        assert result.startswith("ERROR:")

    def test_list_empty_directory(self):
        (self.root / "empty").mkdir()
        result = filesystem.list_directory("empty")
        assert "empty directory" in result

    def test_list_shows_file_sizes(self):
        (self.root / "sized.py").write_text("hello")
        result = filesystem.list_directory(".")
        assert "bytes" in result

    # --- apply_patch ---

    def test_patch_adds_line(self):
        (self.root / "file.py").write_text("line1\nline2\nline3\n")
        patch = textwrap.dedent("""\
            @@ -1,3 +1,4 @@
             line1
             line2
            +line2b
             line3
        """)
        result = filesystem.apply_patch("file.py", patch)
        assert result.startswith("OK:")
        assert "line2b" in (self.root / "file.py").read_text()

    def test_patch_removes_line(self):
        (self.root / "f.py").write_text("a\nb\nc\n")
        patch = textwrap.dedent("""\
            @@ -1,3 +1,2 @@
             a
            -b
             c
        """)
        filesystem.apply_patch("f.py", patch)
        assert "b\n" not in (self.root / "f.py").read_text()

    def test_patch_invalid_returns_error(self):
        (self.root / "x.py").write_text("hi\n")
        result = filesystem.apply_patch("x.py", "not a real patch")
        assert result.startswith("ERROR:")

    def test_patch_missing_file_returns_error(self):
        result = filesystem.apply_patch("ghost.py", "@@ -1 +1 @@\n-x\n+y\n")
        assert result.startswith("ERROR:")

    def test_patch_ok_message_includes_line_counts(self):
        (self.root / "g.py").write_text("a\nb\n")
        patch = textwrap.dedent("""\
            @@ -1,2 +1,2 @@
            -a
            +alpha
             b
        """)
        result = filesystem.apply_patch("g.py", patch)
        assert "+" in result and "\u2212" in result

    # --- search_in_files ---

    def test_search_finds_match(self):
        (self.root / "code.py").write_text("def my_function():\n    pass\n")
        result = filesystem.search_in_files("my_function")
        assert "code.py" in result and "my_function" in result

    def test_search_case_insensitive(self):
        (self.root / "code.py").write_text("HELLO world\n")
        result = filesystem.search_in_files("hello")
        assert "code.py" in result

    def test_search_no_match_returns_message(self):
        (self.root / "empty.py").write_text("")
        result = filesystem.search_in_files("zzznomatch")
        assert "No matches" in result

    def test_search_glob_filter(self):
        (self.root / "code.py").write_text("needle")
        (self.root / "code.md").write_text("needle")
        result = filesystem.search_in_files("needle", file_glob="*.py")
        assert "code.py" in result
        assert "code.md" not in result

    def test_search_skips_binary_extensions(self):
        (self.root / "image.png").write_bytes(b"\x89PNG\r\n")
        (self.root / "source.py").write_text("needle")
        result = filesystem.search_in_files("needle")
        assert "image.png" not in result
        assert "source.py" in result

    def test_search_includes_line_numbers(self):
        (self.root / "f.py").write_text("line1\nfind_me\nline3\n")
        result = filesystem.search_in_files("find_me")
        assert ":2:" in result
