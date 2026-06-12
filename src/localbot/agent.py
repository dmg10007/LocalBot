"""Core agentic request / tool loop with multi-model slot routing.

Design goals after refactor
---------------------------
* Single responsibility: Agent.handle() coordinates; intent classification,
  prompt selection, and tool dispatch are separate concerns.
* No regex intent logic inside the loop itself — moved to intent.py.
* _run_loop is pure: it receives everything it needs via arguments and
  returns a string.  No hidden global state.
* Explicit error types instead of bare Exception catch-alls.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, TYPE_CHECKING

from localbot.adapters.model_registry import ModelRegistry, SlotName
from localbot.config import cfg
from localbot.intent import (
    WorkspaceMode,
    detect_workspace_mode,
    is_coding_with_lookup,
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


class Agent:
    """Stateless (per-request) coordinator that routes to the right model slot."""

    def __init__(
        self,
        registry: ModelRegistry,
        scheduler: "SchedulerService | None" = None,
    ) -> None:
        self._registry = registry
        self._scheduler = scheduler

    async def handle(self, user_id: str, user_message: str) -> str:
        """Entry point for all chat requests.  Returns the assistant reply."""
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
                    )
                else:
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
                    reply = await self._run_loop(
                        client, messages, tools, sched_tools,
                        requesting_user_id=user_id,
                    )
        except asyncio.TimeoutError:
            log.warning("Request for user %s timed out after %ds", user_id, cfg.request_deadline_seconds)

        log_event("assistant_reply", user_id=user_id, content=reply)
        if reply and reply != _TIMEOUT_REPLY:
            await asyncio.to_thread(append_message, user_id, "user", user_message)
            await asyncio.to_thread(append_message, user_id, "assistant", reply)

        return reply

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
    ) -> str:
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
        return await self._run_loop(
            coding_client, phase2_messages, coding_tools, None,
            requesting_user_id=user_id,
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
    ) -> str:
        """Repeatedly call the model and execute tool calls until done.

        Deduplication: (name, canonical-args-json) pairs prevent the model
        from calling the same tool with the same arguments twice in one turn.
        Iteration cap: when max_tool_iterations is reached, we force a final
        synthesis message so the user always gets a response.
        """
        called: set[tuple[str, str]] = set()

        for iteration in range(cfg.max_tool_iterations + 1):
            response = await client.chat(messages, tools=tools)
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
                final = await client.chat(messages, tools=None)
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
                final = await client.chat(messages, tools=None)
                return final["choices"][0]["message"].get("content") or ""

        return "I was unable to complete your request."
