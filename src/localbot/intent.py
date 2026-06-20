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
# Tool name sets — used to decide which queries can be routed to Groq
# ---------------------------------------------------------------------------

# Tools that are safe to call from Groq (no private user data).
PUBLIC_TOOL_NAMES: frozenset[str] = frozenset({
    "web_search",
    "reddit_search",
    "get_current_time",
})

# Tools that must stay on the local model (private data or side-effects).
PRIVATE_TOOL_NAMES: frozenset[str] = frozenset({
    "read_logs",
    "schedule_job",
    "cancel_job",
    "list_jobs",
    "read_file",
    "write_file",
    "list_directory",
    "apply_patch",
    "search_in_files",
    "github_read_file",
    "github_list_directory",
    "github_create_branch",
    "github_commit_files",
    "github_create_pull_request",
    "github_list_pull_requests",
})


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


def needs_private_tools(
    message: str,
    history: list[dict[str, Any]],
    has_scheduler: bool = False,
    workspace_mode: WorkspaceMode = None,
) -> bool:
    """Return True when this turn requires tools that access private user data.

    Private tools (filesystem, scheduler, logs, GitHub) must always run on
    the local model.  Queries that return True here are never eligible for
    the Groq fast path.
    """
    stripped = message.strip()
    if _DIAGNOSTIC_INTENT.search(stripped):
        return True
    if workspace_mode is not None:
        return True
    if has_scheduler and (
        _SCHEDULE_INTENT.search(stripped) or _CANCEL_INTENT.search(stripped)
    ):
        return True
    # Carry-over: if the most recent assistant turn referenced private tools,
    # assume the follow-up still needs them.
    for turn in reversed(history[-4:]):
        if turn.get("role") == "assistant":
            content = turn.get("content") or ""
            if has_scheduler and (
                _SCHEDULE_INTENT.search(content) or _CANCEL_INTENT.search(content)
            ):
                return True
            break
    return False


def needs_public_tools(
    message: str,
    history: list[dict[str, Any]],
) -> bool:
    """Return True when this turn needs only public (Groq-safe) tools.

    Public tools — web_search, reddit_search, get_current_time — carry no
    private user data and can be executed via Groq function calling.
    """
    stripped = message.strip()
    if _SEARCH_INTENT.search(stripped):
        return True
    # Carry-over: if the most recent assistant turn called a search tool,
    # assume the follow-up still needs it.
    for turn in reversed(history[-4:]):
        if turn.get("role") == "assistant":
            content = turn.get("content") or ""
            if _SEARCH_INTENT.search(content):
                return True
            break
    return False


def needs_tools(
    message: str,
    history: list[dict[str, Any]],
    has_scheduler: bool = False,
    workspace_mode: WorkspaceMode = None,
) -> bool:
    """Return True if this turn should receive any tool schemas.

    Thin wrapper kept for backward compatibility.  New code should call
    needs_private_tools() / needs_public_tools() directly.
    """
    if needs_private_tools(message, history, has_scheduler, workspace_mode):
        return True
    if needs_public_tools(message, history):
        return True
    # Conversational one-liners don't need tools.
    return not bool(_CONVERSATIONAL.fullmatch(message.strip()))


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


def is_groq_eligible(
    message: str,
    workspace_mode: WorkspaceMode,
    has_private_tools: bool = False,
) -> bool:
    """Return True when a query is safe to route to Groq.

    Routing policy — never send to Groq when:
    - Filesystem or GitHub workspace context is involved (private files/repos)
    - Scheduler intent detected (user schedule data)
    - Diagnostic intent detected (private audit log context)
    - The caller has already determined private tools are needed

    Everything else — general chat, search, reasoning without personal
    context, public tool calls — is eligible for the Groq fast path.
    """
    if has_private_tools:
        return False
    if workspace_mode is not None:
        return False
    if _SCHEDULE_INTENT.search(message) or _CANCEL_INTENT.search(message):
        return False
    if _DIAGNOSTIC_INTENT.search(message):
        return False
    return True
