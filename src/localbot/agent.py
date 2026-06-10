"""Core request/tool loop with multi-model slot routing."""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Literal, TYPE_CHECKING

from localbot.adapters.model_registry import ModelRegistry, SlotName
from localbot.config import cfg
from localbot.storage.audit import log_event
from localbot.storage.history import append_message, get_history
from localbot.tools.registry import WorkspaceMode, build_tool_schemas, dispatch
from localbot.tools.scheduler_tools import SchedulerTools

if TYPE_CHECKING:
    from localbot.scheduler.service import SchedulerService

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a helpful, concise assistant. You have the following tools:

SEARCH & TIME:
- web_search(query) — search the web for current information
- reddit_search(query, subreddit?) — search Reddit posts
- get_current_time(timezone?) — get the current date and time

DIAGNOSTICS:
- read_logs(level?, limit?) — read recent audit log entries to diagnose
  errors, failed jobs, timeouts, or unexpected behaviour

SCHEDULING:
- schedule_job(prompt, cron_expr) — create a recurring scheduled message
- cancel_job(job_id) — cancel a scheduled job by its ID
- list_jobs() — list all active scheduled jobs for the user

RULES:
1. When the user asks to search, look something up, or find current news —
   call the tool immediately. Do NOT describe what you will do.
2. When the user asks to be reminded or wants a recurring message —
   call schedule_job immediately. Convert their natural-language schedule
   into a 5-field cron expression (minute hour day month day_of_week).
3. NEVER confirm a job is scheduled unless schedule_job returned successfully.
   NEVER invent a job ID. Always relay the ID returned by the tool.
4. After receiving tool results, write a clear, concise summary.
5. For casual conversation or simple questions — respond directly.
6. Never call the same tool with the same arguments twice in one turn.
7. Keep responses concise and friendly.
8. When the user asks to check logs or troubleshoot — call read_logs immediately.
9. After any web_search or reddit_search, you MUST cite your sources.
   Reference each source inline with [1], [2], [3] … and end your reply
   with a "Sources:" section listing each as a clickable markdown link.
"""

CODING_SYSTEM_PROMPT = """\
You are an expert software engineer. You write clean, correct, production-quality code.

WORKSPACE TOOLS — LOCAL:
- read_file(path) — read a file from the local sandbox
- write_file(path, content) — create or overwrite a file
- list_directory(path?) — list sandbox directory contents
- apply_patch(path, patch) — apply a unified-diff patch to a file
- search_in_files(pattern, path?, file_glob?) — grep-style search

WORKSPACE TOOLS — GITHUB:
- github_read_file(owner, repo, path, ref?) — read a file from a GitHub repo
- github_list_directory(owner, repo, path?, ref?) — list a GitHub directory
- github_create_branch(owner, repo, branch, from_branch?) — create a branch
- github_commit_files(owner, repo, branch, message, files) — commit files
- github_create_pull_request(owner, repo, title, head, base?, body?) — open a PR
- github_list_pull_requests(owner, repo, state?) — list PRs

RULES:
1. Always read a file before editing it — never overwrite blindly.
2. Prefer apply_patch / github_commit_files for surgical edits over rewriting
   entire files.
3. Create a new branch before committing changes to a GitHub repo.
4. Verify paths exist before writing to them.
5. Never expose secrets (tokens, passwords, keys) in committed content.
6. Produce complete, runnable code — no placeholders, no TODOs unless asked.
7. When you finish, summarise exactly what was changed and where.
"""

REASONING_SYSTEM_PROMPT = """\
You are a senior software architect and technical reasoner.
You analyse problems deeply, identify trade-offs, and produce structured plans.

RULES:
1. Think step by step. Show your reasoning before your conclusion.
2. When comparing approaches, use a structured format (pros/cons or a table).
3. Produce a concrete, actionable recommendation — not just observations.
4. If the question requires code, produce a complete, correct implementation.
5. Keep your answer focused. Omit preamble.
"""

# ---------------------------------------------------------------------------
# Intent detection regexes
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
    r"\b(remind|reminder|schedule|recurring|every (day|week|hour|morning|evening|night|"
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

_TOOL_RESULT_MAX_CHARS = 4000


# ---------------------------------------------------------------------------
# Intent helpers
# ---------------------------------------------------------------------------

def _select_slot(message: str) -> SlotName:
    """Choose a model slot based on the message intent.

    Priority: reasoning > coding > general.
    When both coding and reasoning intent fire, reasoning wins so the
    model can plan first; the follow-up implementation message routes to
    coding.
    """
    has_coding = bool(_CODING_INTENT.search(message))
    has_reasoning = bool(_REASONING_INTENT.search(message))
    if has_reasoning:
        return "reasoning"
    if has_coding:
        return "coding"
    return "general"


def _detect_workspace_mode(message: str) -> WorkspaceMode:
    """Return the workspace mode implied by the message."""
    local = bool(_LOCAL_WORKSPACE.search(message))
    remote = bool(_GITHUB_WORKSPACE.search(message))
    if local and remote:
        return "both"
    if local:
        return "local"
    if remote:
        return "github"
    return None


def _needs_tools(
    message: str,
    history: list[dict[str, Any]],
    has_scheduler: bool = False,
    workspace_mode: WorkspaceMode = None,
) -> bool:
    """Return True if this turn should have tools available."""
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
    return not bool(_CONVERSATIONAL.fullmatch(stripped))


def _is_system_echo(content: str) -> bool:
    lower = content.lower().strip()
    return any(marker in lower[:300] for marker in _SYSTEM_ECHO_MARKERS)


def _is_coding_with_lookup(message: str) -> bool:
    """True when the message has both coding intent and a lookup/search need.

    Used to trigger two-phase dispatch: general model fetches context,
    coding model synthesises the implementation.
    """
    return bool(_CODING_INTENT.search(message)) and bool(
        _SEARCH_INTENT.search(message)
        or re.search(r"\b(api|docs?|documentation|how to use|example)", message, re.IGNORECASE)
    )


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class Agent:
    def __init__(
        self,
        registry: ModelRegistry,
        scheduler: "SchedulerService | None" = None,
    ) -> None:
        self._registry = registry
        self._scheduler = scheduler

    async def handle(self, user_id: str, user_message: str) -> str:
        history = get_history(user_id)
        log_event("user_message", user_id=user_id, content=user_message)

        sched_tools = (
            SchedulerTools(self._scheduler, user_id)
            if self._scheduler is not None
            else None
        )
        has_scheduler = sched_tools is not None
        workspace_mode = _detect_workspace_mode(user_message)
        slot = _select_slot(user_message)

        timeout_sentinel = "Sorry, your request took too long and was cancelled."
        reply = timeout_sentinel
        try:
            async with asyncio.timeout(cfg.request_deadline_seconds):
                if _is_coding_with_lookup(user_message):
                    reply = await self._run_two_phase(
                        user_id, user_message, history,
                        workspace_mode, sched_tools,
                    )
                else:
                    client = await self._registry.acquire(slot)
                    system = _system_prompt_for_slot(slot)
                    messages: list[dict[str, Any]] = [
                        {"role": "system", "content": system},
                        *history,
                        {"role": "user", "content": user_message},
                    ]
                    if _needs_tools(user_message, history, has_scheduler, workspace_mode):
                        tools = build_tool_schemas(
                            include_scheduler=has_scheduler,
                            workspace_mode=workspace_mode,
                        )
                    else:
                        tools = None
                    reply = await self._run_loop(
                        client, messages, tools, sched_tools,
                        requesting_user_id=user_id,
                    )
        except asyncio.TimeoutError:
            pass

        log_event("assistant_reply", user_id=user_id, content=reply)
        if reply and reply != timeout_sentinel:
            append_message(user_id, "user", user_message)
            append_message(user_id, "assistant", reply)

        return reply

    async def _run_two_phase(
        self,
        user_id: str,
        user_message: str,
        history: list[dict[str, Any]],
        workspace_mode: WorkspaceMode,
        sched_tools: Any,
    ) -> str:
        """Phase 1: general model fetches context via tools.
           Phase 2: coding model synthesises the implementation.
        """
        # Phase 1 — use general model for tool calls (web search / lookup).
        general_client = await self._registry.acquire("general")
        phase1_messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *history,
            {"role": "user", "content": user_message},
        ]
        search_tools = build_tool_schemas(include_scheduler=False, workspace_mode=None)
        context = await self._run_loop(
            general_client, phase1_messages, search_tools, sched_tools,
            requesting_user_id=user_id,
        )
        log.debug("[agent] two-phase: phase 1 context (%d chars)", len(context))

        # Phase 2 — swap to coding model with enriched context.
        coding_client = await self._registry.acquire("coding")
        enriched_request = (
            f"{user_message}\n\n"
            f"--- Context from research ---\n{context}\n"
            f"--- End context ---\n\n"
            f"Now implement the solution."
        )
        phase2_messages: list[dict[str, Any]] = [
            {"role": "system", "content": CODING_SYSTEM_PROMPT},
            *history,
            {"role": "user", "content": enriched_request},
        ]
        coding_tools = build_tool_schemas(
            include_scheduler=False,
            workspace_mode=workspace_mode,
        )
        return await self._run_loop(
            coding_client, phase2_messages, coding_tools, None,
            requesting_user_id=user_id,
        )

    async def _run_loop(
        self,
        client: Any,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        sched_tools: Any = None,
        requesting_user_id: str = "",
    ) -> str:
        called: set[tuple[str, str]] = set()

        for iteration in range(cfg.max_tool_iterations + 1):
            response = await client.chat(messages, tools=tools)
            choice = response["choices"][0]
            msg = choice["message"]
            content = msg.get("content") or ""

            if not msg.get("tool_calls"):
                if _is_system_echo(content):
                    log.warning("Model echoed system prompt — forcing synthesis.")
                    synth = await client.chat(
                        messages + [{"role": "user", "content": "Please answer the question above directly and concisely."}],
                        tools=None,
                    )
                    return synth["choices"][0]["message"].get("content") or ""
                return content

            if iteration == cfg.max_tool_iterations:
                messages.append({"role": "assistant", "content": None, "tool_calls": msg["tool_calls"]})
                messages.append({
                    "role": "user",
                    "content": "Please summarise all the information you have gathered and give me a final answer. Remember to cite your sources with [1], [2] … and list them at the end.",
                })
                final = await client.chat(messages, tools=None)
                return final["choices"][0]["message"].get("content") or ""

            messages.append(msg)
            any_new = False

            for tc in msg["tool_calls"]:
                fn = tc["function"]
                name = fn["name"]
                try:
                    args = json.loads(fn.get("arguments", "{}"))
                except json.JSONDecodeError:
                    args = {}

                dedup_key = (name, json.dumps(args, sort_keys=True))
                if dedup_key in called:
                    log.warning("Duplicate tool call blocked: %s(%s)", name, args)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": "[already called — use the result from the previous call]",
                    })
                    continue

                called.add(dedup_key)
                any_new = True

                log.info("Tool call: %s(%s)", name, args)
                log_event("tool_call", tool=name, args=args)
                result = await dispatch(
                    name,
                    args,
                    scheduler_tools=sched_tools,
                    requesting_user_id=requesting_user_id,
                )
                log_event("tool_result", tool=name, result=result[:500])

                if len(result) > _TOOL_RESULT_MAX_CHARS:
                    log.debug(
                        "Tool result from %s truncated from %d to %d chars",
                        name, len(result), _TOOL_RESULT_MAX_CHARS,
                    )
                    result = result[:_TOOL_RESULT_MAX_CHARS] + "\n\n[...truncated]"

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result,
                })

            if not any_new:
                messages.append({
                    "role": "user",
                    "content": "You have all the information needed. Please give your final answer now.",
                })
                final = await client.chat(messages, tools=None)
                return final["choices"][0]["message"].get("content") or ""

        return "I was unable to complete your request."


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _system_prompt_for_slot(slot: SlotName) -> str:
    if slot == "coding":
        return CODING_SYSTEM_PROMPT
    if slot == "reasoning":
        return REASONING_SYSTEM_PROMPT
    return SYSTEM_PROMPT
