"""Intent classification helpers.

Previously these lived inside agent.py, making that module impossible to
unit-test without importing the full agent stack.  Extracted so they can
be tested in isolation.
"""
from __future__ import annotations

import re
from typing import Any, Literal

WorkspaceMode = Literal["local", "github", "both"] | None

# ---------------------------------------------------------------------------
# Regexes
# ---------------------------------------------------------------------------

_SYSTEM_ECHO_MARKERS = (
    "you are a helpful, concise assistant",
    "you are an expert software engineer",
    "rules:",
    "1. when the user asks to search",
    "call the tool immediately",
)

_CONVERSATIONAL = re.compile(
    r"(hi+|hello+|hey+|howdy|sup|what'?s up|how are you|how r u|"
    r"good (morning|evening|afternoon|night)|thanks?( you)?|thank you|"
    r"ok(ay)?|sure|cool|great|nice|awesome|sounds good|got it|"
    r"bye|goodbye|see ya|later|lol|haha|:?\.?[)|(|D])\.?\!?",
    re.IGNORECASE,
)

_SEARCH_INTENT = re.compile(
    r"\b(search|look up|lookup|find|news|latest|current|today|trending|top stories)",
    re.IGNORECASE,
)

_SCHEDULE_INTENT = re.compile(
    r"\b(remind|reminder|schedule|recurring|"
    r"every (day|week|hour|morning|evening|night|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|weekday|weekend|"
    r"\d+ (minute|hour|day)s?)|at \d+(am|pm|:\d+))",
    re.IGNORECASE,
)

_CANCEL_INTENT = re.compile(
    r"\b(cancel|remove|delete|stop|unschedule)\b.*\b(job|reminder|schedule)",
    re.IGNORECASE,
)

_DIAGNOSTIC_INTENT = re.compile(
    r"\b(log|logs|error|errors|crash|crashed|fail|failed|broke|broken|"
    r"why did|debug|diagnos|troubleshoot|something wrong|not working|"
    r"what happened|check the logs)",
    re.IGNORECASE,
)

_CODING_INTENT = re.compile(
    r"\b(write|implement|code|fix|refactor|debug|patch|edit|create|generate|"
    r"add (a |the )?function|add (a |the )?class|add (a |the )?method|"
    r"pull request|PR|commit|branch|push|diff|test|unittest|pytest|"
    r"in (this|the|my|our) (repo|repository|codebase|project|file|folder|directory))",
    re.IGNORECASE,
)

_REASONING_INTENT = re.compile(
    r"\b(design|architect|plan|compare|analyse|analyze|explain|why|how does|"
    r"trade.?off|pros and cons|best approach|best way|should I|evaluate|"
    r"what is the difference|review this|code review|suggest)",
    re.IGNORECASE,
)

_LOCAL_WORKSPACE = re.compile(
    r"(local folder|local file|this file|in ~|sandbox|"
    r"in (my |the )?(folder|directory|project)|on (my |this )?(machine|laptop|computer))",
    re.IGNORECASE,
)

_GITHUB_WORKSPACE = re.compile(
    r"(in (the |my |our )?(repo|repository|github)|on github|"
    r"pull request|PR|commit (to|on)|push to|open a PR|create a branch)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def select_slot(message: str) -> Literal["general", "coding", "reasoning"]:
    """Choose a model slot based on message intent.

    Reasoning wins over coding so the model plans before it implements.
    """
    if _REASONING_INTENT.search(message):
        return "reasoning"
    if _CODING_INTENT.search(message):
        return "coding"
    return "general"


def detect_workspace_mode(message: str) -> WorkspaceMode:
    local = bool(_LOCAL_WORKSPACE.search(message))
    remote = bool(_GITHUB_WORKSPACE.search(message))
    if local and remote:
        return "both"
    if local:
        return "local"
    if remote:
        return "github"
    return None


def needs_tools(
    message: str,
    history: list[dict[str, Any]],
    has_scheduler: bool = False,
    workspace_mode: WorkspaceMode = None,
) -> bool:
    """Return True if this turn should receive tool schemas."""
    stripped = message.strip()
    if _SEARCH_INTENT.search(stripped):
        return True
    if _DIAGNOSTIC_INTENT.search(stripped):
        return True
    if workspace_mode is not None:
        return True
    if has_scheduler and (
        _SCHEDULE_INTENT.search(stripped) or _CANCEL_INTENT.search(stripped)
    ):
        return True
    # Carry-over: if the most recent assistant turn referenced search/scheduling,
    # assume the follow-up still needs tools.
    for turn in reversed(history[-4:]):
        if turn.get("role") == "assistant":
            content = turn.get("content") or ""
            if _SEARCH_INTENT.search(content):
                return True
            if has_scheduler and (
                _SCHEDULE_INTENT.search(content) or _CANCEL_INTENT.search(content)
            ):
                return True
            break
    # Conversational one-liners don't need tools.
    return not bool(_CONVERSATIONAL.fullmatch(stripped))


def is_system_echo(content: str) -> bool:
    """Return True when the model has regurgitated its own system prompt."""
    lower = content.lower().strip()
    return any(marker in lower[:300] for marker in _SYSTEM_ECHO_MARKERS)


def is_coding_with_lookup(message: str) -> bool:
    """True when the message needs both code generation AND external lookup.

    Triggers the two-phase dispatch: general model fetches context,
    coding model synthesises the implementation.
    """
    return bool(_CODING_INTENT.search(message)) and bool(
        _SEARCH_INTENT.search(message)
        or re.search(r"\b(api|docs?|documentation|how to use|example)", message, re.IGNORECASE)
    )
