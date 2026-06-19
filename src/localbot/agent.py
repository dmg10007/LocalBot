"""Core agentic request / tool loop with multi-model slot routing.

Design goals after refactor
---------------------------
* Single responsibility: Agent.handle() coordinates; intent classification,
  prompt selection, and tool dispatch are separate concerns.
* No regex intent logic inside the loop itself — moved to intent.py.
* _run_loop is pure: it receives everything it needs via arguments and
  returns a string.  No hidden global state.
* Explicit error types instead of bare Exception catch-alls.
* on_token callback: when provided, every model call streams live tokens.
  Tool-call chunk deltas are accumulated inside LlamaCppClient and never
  forwarded to on_token, so the caller only sees user-visible content.
  This means we no longer need to gate on_token behind is_final_call —
  the client layer already suppresses tool-call tokens.
* Groq fast path: when GROQ_API_KEY is set and the query is eligible
  (no filesystem/scheduler/diagnostics context), the request is routed
  to Groq for sub-100 ms TTFT.  Falls back to the local model on error.
* ModelSwappedError retry: if the idle timer swaps the active slot while
  a request is in-flight, _run_local and _run_two_phase catch
  ModelSwappedError, re-acquire a fresh client, and retry once.  A second
  ModelSwappedError propagates normally to avoid infinite loops.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Awaitable, Callable, TYPE_CHECKING

from localbot.adapters.groq_client import GroqClient
from localbot.adapters.llamacpp_client import ModelSwappedError
from localbot.adapters.model_registry import ModelRegistry, SlotName
from localbot.config import cfg
from localbot.intent import (
    WorkspaceMode,
    detect_workspace_mode,
    is_coding_with_lookup,
    is_groq_eligible,
    is_system_echo,
    needs_tools,
    select_slot,
)
from localbot.prompts import (
    CODING_SYSTEM_PROMPT,
    REASONING_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    system_prompt_for_slot,
)
from localbot.storage.audit import log_event
from localbot.storage.history import append_message, get_history
from localbot.tools.registry import build_tool_schemas, dispatch
from localbot.tools.scheduler_tools import SchedulerTools

if TYPE_CHECKING:
    from localbot.scheduler.service import SchedulerService

log = logging.getLogger(__name__)

_TOOL_RESULT_MAX_CHARS = 4_000
_TIMEOUT_REPLY = "Sorry, your request took too long and was cancelled."

# Type alias matching LlamaCppClient.TokenCallback
TokenCallback = Callable[[str], Awaitable[None]]


class Agent:
    """Stateless (per-request) coordinator that routes to the right model slot."""

    def __init__(
        self,
        registry: ModelRegistry,
        scheduler: "SchedulerService | None" = None,
    ) -> None:
        self._registry = registry
        self._scheduler = scheduler
        # Groq client is instantiated once and reused; None when key not set.
        self._groq: GroqClient | None = (
            GroqClient(cfg.groq_api_key, model=cfg.groq_model)
            if cfg.groq_api_key
            else None
        )

    async def handle(
        self,
        user_id: str,
        user_message: str,
        on_token: TokenCallback | None = None,
    ) -> str:
        """Entry point for all chat requests.  Returns the assistant reply.

        *on_token* is an optional async callback invoked with each streamed
        token during the final model call.  Pass it from the webui SSE path;
        omit it (or pass None) for the Discord / non-streaming path.
        """
        history = await asyncio.to_thread(get_history, user_id)
        log_event("user_message", user_id=user_id, content=user_message)

        sched_tools = (
            SchedulerTools(self._scheduler, user_id)
            if self._scheduler is not None
            else None
        )
        workspace_mode = detect_workspace_mode(user_message)
        slot = select_slot(user_message)

        reply = _TIMEOUT_REPLY
        try:
            async with asyncio.timeout(cfg.request_deadline_seconds):
                if is_coding_with_lookup(user_message):
                    reply = await self._run_two_phase(
                        user_id, user_message, history, workspace_mode, sched_tools,
                        on_token=on_token,
                    )
                elif (
                    self._groq is not None
                    and not needs_tools(user_message, history, sched_tools is not None, workspace_mode)
                    and is_groq_eligible(user_message, workspace_mode)
                ):
                    # Groq fast path: eligible queries with no tool calls go here.
                    # Falls back to local model on any error.
                    try:
                        messages_groq: list[dict[str, Any]] = [
                            {"role": "system", "content": system_prompt_for_slot(slot)},
                            *history,
                            {"role": "user", "content": user_message},
                        ]
                        log.debug("[agent] routing to Groq (%s)", self._groq.model)
                        reply = await self._groq.chat(
                            messages_groq,
                            on_token=on_token,
                        )
                        log_event("groq_reply", user_id=user_id, model=self._groq.model)
                    except Exception as groq_exc:
                        log.warning("Groq fast path failed (%s) — falling back to local model", groq_exc)
                        # Fall through to local path below
                        reply = await self._run_local(
                            user_id, user_message, history, slot, workspace_mode,
                            sched_tools, on_token,
                        )
                else:
                    reply = await self._run_local(
                        user_id, user_message, history, slot, workspace_mode,
                        sched_tools, on_token,
                    )
        except asyncio.TimeoutError:
            log.warning("Request for user %s timed out after %ds", user_id, cfg.request_deadline_seconds)

        log_event("assistant_reply", user_id=user_id, content=reply)
        if reply and reply != _TIMEOUT_REPLY:
            await asyncio.to_thread(append_message, user_id, "user", user_message)
            await asyncio.to_thread(append_message, user_id, "assistant", reply)

        return reply

    # ------------------------------------------------------------------
    # Local model dispatch (single-slot path)
    # ------------------------------------------------------------------

    async def _run_local(
        self,
        user_id: str,
        user_message: str,
        history: list[dict[str, Any]],
        slot: SlotName,
        workspace_mode: WorkspaceMode,
        sched_tools: Any,
        on_token: TokenCallback | None = None,
    ) -> str:
        """Acquire the appropriate local model slot and run the agent loop.

        If the idle timer swaps the slot while the request is in-flight,
        ModelSwappedError is caught, a fresh client is acquired, and the
        loop retries once.  A second ModelSwappedError propagates normally.
        """
        client = await self._registry.acquire(slot)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt_for_slot(slot)},
            *history,
            {"role": "user", "content": user_message},
        ]
        tools = (
            build_tool_schemas(
                include_scheduler=sched_tools is not None,
                workspace_mode=workspace_mode,
            )
            if needs_tools(user_message, history, sched_tools is not None, workspace_mode)
            else None
        )
        try:
            return await self._run_loop(
                client, messages, tools, sched_tools,
                requesting_user_id=user_id,
                on_token=on_token,
            )
        except ModelSwappedError:
            log.warning(
                "[agent] model was swapped mid-request for slot '%s' — "
                "re-acquiring and retrying once", slot
            )
            client = await self._registry.acquire(slot)
            return await self._run_loop(
                client, messages, tools, sched_tools,
                requesting_user_id=user_id,
                on_token=on_token,
            )

    # ------------------------------------------------------------------
    # Two-phase: general model fetches context, coding model implements
    # ------------------------------------------------------------------

    async def _run_two_phase(
        self,
        user_id: str,
        user_message: str,
        history: list[dict[str, Any]],
        workspace_mode: WorkspaceMode,
        sched_tools: Any,
        on_token: TokenCallback | None = None,
    ) -> str:
        general_client = await self._registry.acquire("general")
        phase1_messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *history,
            {"role": "user", "content": user_message},
        ]
        search_tools = build_tool_schemas(include_scheduler=False, workspace_mode=None)
        # Phase 1 (context gathering) is always non-streaming — tool calls
        # require complete JSON responses.
        try:
            context = await self._run_loop(
                general_client, phase1_messages, search_tools, sched_tools,
                requesting_user_id=user_id,
                on_token=None,
            )
        except ModelSwappedError:
            log.warning(
                "[agent] model swapped during two-phase phase-1 — "
                "re-acquiring general slot and retrying once"
            )
            general_client = await self._registry.acquire("general")
            context = await self._run_loop(
                general_client, phase1_messages, search_tools, sched_tools,
                requesting_user_id=user_id,
                on_token=None,
            )
        log.debug("[agent] two-phase phase-1 context: %d chars", len(context))

        coding_client = await self._registry.acquire("coding")
        enriched = (
            f"{user_message}\n\n"
            f"--- Context from research ---\n{context}\n"
            f"--- End context ---\n\nNow implement the solution."
        )
        phase2_messages: list[dict[str, Any]] = [
            {"role": "system", "content": CODING_SYSTEM_PROMPT},
            *history,
            {"role": "user", "content": enriched},
        ]
        coding_tools = build_tool_schemas(include_scheduler=False, workspace_mode=workspace_mode)
        # Phase 2 (implementation) streams to the caller.
        try:
            return await self._run_loop(
                coding_client, phase2_messages, coding_tools, None,
                requesting_user_id=user_id,
                on_token=on_token,
            )
        except ModelSwappedError:
            log.warning(
                "[agent] model swapped during two-phase phase-2 — "
                "re-acquiring coding slot and retrying once"
            )
            coding_client = await self._registry.acquire("coding")
            return await self._run_loop(
                coding_client, phase2_messages, coding_tools, None,
                requesting_user_id=user_id,
                on_token=on_token,
            )

    # ------------------------------------------------------------------
    # Core agentic loop
    # ------------------------------------------------------------------

    async def _run_loop(
        self,
        client: Any,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        sched_tools: Any = None,
        requesting_user_id: str = "",
        on_token: TokenCallback | None = None,
    ) -> str:
        """Repeatedly call the model and execute tool calls until done.

        on_token is passed to every client.chat() call.  LlamaCppClient
        suppresses tool-call delta tokens internally — only user-visible
        content tokens reach the callback.  This means we no longer need
        the is_final_call gate; streaming works correctly on the very first
        pass even when tools are available.

        Deduplication: (name, canonical-args-json) pairs prevent the model
        from calling the same tool with the same arguments twice in one turn.
        Iteration cap: when max_tool_iterations is reached, we force a final
        synthesis message so the user always gets a response.

        ModelSwappedError is NOT caught here — it propagates up to
        _run_local / _run_two_phase which hold the retry logic.
        """
        called: set[tuple[str, str]] = set()

        for iteration in range(cfg.max_tool_iterations + 1):
            response = await client.chat(
                messages,
                tools=tools,
                on_token=on_token,
            )
            choice = response["choices"][0]
            msg = choice["message"]
            content: str = msg.get("content") or ""

            # No tool calls → terminal reply.
            if not msg.get("tool_calls"):
                if is_system_echo(content):
                    log.warning("Model echoed system prompt — forcing re-synthesis.")
                    synth = await client.chat(
                        messages + [{
                            "role": "user",
                            "content": "Please answer the question above directly and concisely.",
                        }],
                        tools=None,
                        on_token=on_token,
                    )
                    return synth["choices"][0]["message"].get("content") or ""
                return content

            # Hard iteration cap: force a final summary pass.
            if iteration == cfg.max_tool_iterations:
                messages.append({"role": "assistant", "content": None, "tool_calls": msg["tool_calls"]})
                messages.append({
                    "role": "user",
                    "content": (
                        "Please summarise all information you have gathered "
                        "and give a final answer. Cite sources with [1], [2] … "
                        "and list them at the end."
                    ),
                })
                final = await client.chat(messages, tools=None, on_token=on_token)
                return final["choices"][0]["message"].get("content") or ""

            messages.append(msg)
            any_new = False

            for tc in msg["tool_calls"]:
                fn = tc["function"]
                name: str = fn["name"]
                try:
                    args: dict[str, Any] = json.loads(fn.get("arguments", "{}"))
                except json.JSONDecodeError:
                    log.warning("Malformed tool-call arguments for %s — treating as {}", name)
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
                    name, args,
                    scheduler_tools=sched_tools,
                    requesting_user_id=requesting_user_id,
                )
                log_event("tool_result", tool=name, result=result[:500])

                if len(result) > _TOOL_RESULT_MAX_CHARS:
                    log.debug(
                        "Tool result from %s truncated %d → %d chars",
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
                final = await client.chat(messages, tools=None, on_token=on_token)
                return final["choices"][0]["message"].get("content") or ""

        return "I was unable to complete your request."
