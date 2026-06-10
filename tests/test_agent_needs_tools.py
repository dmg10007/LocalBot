"""Tests for agent._needs_tools."""
import pytest
from localbot.agent import _needs_tools


def test_search_intent_requires_tools():
    assert _needs_tools("search for latest python news", [], has_scheduler=False) is True


def test_diagnostic_intent_requires_tools():
    assert _needs_tools("check the logs for errors", [], has_scheduler=False) is True


def test_schedule_intent_with_scheduler_requires_tools():
    assert _needs_tools("remind me every morning at 8am", [], has_scheduler=True) is True


def test_schedule_intent_without_scheduler_does_not_require_tools():
    # No scheduler configured — scheduling intent should not trigger tools
    # (there are none to call).
    assert _needs_tools("remind me every morning at 8am", [], has_scheduler=False) is False


def test_plain_conversational_does_not_require_tools():
    assert _needs_tools("hi", [], has_scheduler=False) is False
    assert _needs_tools("thanks", [], has_scheduler=False) is False
    assert _needs_tools("ok", [], has_scheduler=False) is False


def test_workspace_mode_requires_tools():
    assert _needs_tools("fix the bug in this repo", [], workspace_mode="github") is True


def test_non_conversational_non_search_requires_tools():
    """A message that is not obviously conversational defaults to tools=True."""
    assert _needs_tools("what is the capital of France?", [], has_scheduler=False) is True


def test_history_search_context_requires_tools():
    """If the recent assistant turn contained search results, keep tools on."""
    history = [
        {"role": "user",      "content": "find me the latest news"},
        {"role": "assistant", "content": "Here are the search results..."},
    ]
    assert _needs_tools("tell me more", history, has_scheduler=False) is True
