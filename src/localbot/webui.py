"""OpenAI-compatible HTTP API server so OpenWebUI can talk to LocalBot.

Refactored changes
------------------
* _RemoteRegistry now satisfies a Protocol instead of being an
  untyped inline class — mypy can verify it.
* chat_completions() no longer creates a redundant asyncio.Future;
  it simply awaits agent.handle() directly (both run on the same loop).
* Rate limiting for web UI requests (mirrors Discord behaviour).
* Startup is guarded by an asyncio.Event so /healthz returns 503
  cleanly during warm-up, and the event is set on both paths (remote
  and local) without duplicating logic.

Environment variables
---------------------
WEBUI_HOST          Bind address (default: 0.0.0.0)
WEBUI_PORT          Port (default: 8000)
WEBUI_API_KEY       Bearer token required on every request.
                    When unset, auth is disabled.
WEBUI_USER_PREFIX   Prefix prepended to the token to form the internal
                    user_id (default: "webui:").

LLAMA_REMOTE_HOST   When set, the webui connects to this remote
                    llama-server instead of spawning its own process.
LLAMA_REMOTE_PORT   Port of the remote llama-server (default: 8080).
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, List, Optional, Protocol, Union, runtime_checkable

from pydantic import BaseModel

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# OpenAI request models
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[Any]] = ""


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage] = []
    stream: bool = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


# ---------------------------------------------------------------------------
# Registry protocol — both ModelRegistry and _RemoteRegistry satisfy this
# ---------------------------------------------------------------------------

@runtime_checkable
class RegistryProtocol(Protocol):
    async def acquire(self, slot: str) -> Any: ...
    def is_slot_available(self, slot: str) -> bool: ...
    async def shutdown(self) -> None: ...


def _require_fastapi() -> None:
    try:
        import fastapi  # noqa: F401
        import uvicorn  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "The 'webui' extra is required.  Install with:\n"
            "    pip install -e .[webui]"
        ) from exc


def create_app() -> "fastapi.FastAPI":  # type: ignore[name-defined]
    _require_fastapi()

    import fastapi
    from fastapi import Depends, HTTPException, Request, status
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse
    from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

    from localbot.adapters.llamacpp_client import LlamaCppClient
    from localbot.adapters.model_registry import ModelRegistry
    from localbot.agent import Agent
    from localbot.config import cfg
    from localbot.scheduler.service import SchedulerService
    from localbot.storage.db import init_db

    API_KEY: str | None = os.environ.get("WEBUI_API_KEY") or None
    USER_PREFIX: str = os.environ.get("WEBUI_USER_PREFIX", "webui:")
    _rate_last: dict[str, float] = {}

    # Fail closed: a network-exposed LLM proxy must not run unauthenticated.
    # Set WEBUI_ALLOW_NO_AUTH=1 only for a trusted, loopback-only deployment.
    _bind_host = os.environ.get("WEBUI_HOST", "0.0.0.0")
    _loopback = _bind_host in ("127.0.0.1", "::1", "localhost")
    if API_KEY is None and not _loopback and os.environ.get("WEBUI_ALLOW_NO_AUTH") != "1":
        raise SystemExit(
            "WEBUI_API_KEY is required when binding a non-loopback host "
            f"({_bind_host!r}). Set WEBUI_API_KEY, bind WEBUI_HOST=127.0.0.1, "
            "or set WEBUI_ALLOW_NO_AUTH=1 to explicitly opt out."
        )

    async def _null_send(user_id: str, prompt: str) -> None:
        log.warning(
            "Scheduled job for user %s fired; HTTP delivery not implemented.", user_id
        )

    class _RemoteRegistry:
        """Thin wrapper around a single remote LlamaCppClient."""

        def __init__(self, client: LlamaCppClient) -> None:
            self._client = client

        async def acquire(self, slot: str) -> LlamaCppClient:  # noqa: ARG002
            return self._client

        def is_slot_available(self, slot: str) -> bool:
            return slot == "general"

        async def shutdown(self) -> None:
            await self._client.close()

    @asynccontextmanager
    async def lifespan(app: fastapi.FastAPI) -> AsyncIterator[None]:
        init_db()
        ready_event: asyncio.Event = asyncio.Event()
        scheduler = SchedulerService(_null_send)
        scheduler.start()

        if cfg.llama_remote_host:
            log.info("Remote llama-server mode: %s:%d", cfg.llama_remote_host, cfg.llama_remote_port)
            remote_client = LlamaCppClient(host=cfg.llama_remote_host, port=cfg.llama_remote_port)
            registry: RegistryProtocol = _RemoteRegistry(remote_client)
        else:
            log.info("Local subprocess mode")
            registry = ModelRegistry()

        agent = Agent(registry, scheduler=scheduler)  # type: ignore[arg-type]
        app.state.agent = agent
        app.state.registry = registry
        app.state.scheduler = scheduler
        app.state.ready_event = ready_event

        async def _warm() -> None:
            try:
                if isinstance(registry, _RemoteRegistry):
                    await remote_client.wait_until_ready(retries=60, delay=2.0)
                else:
                    await registry.warm_general()  # type: ignore[union-attr]
                ready_event.set()
                log.info("LocalBot webui ready")
            except Exception:
                log.exception("Warm-up failed")

        asyncio.create_task(_warm())
        yield

        scheduler.stop()
        await registry.shutdown()
        from localbot.tools import search as s, reddit as r
        await s.close_session()
        await r.close_session()

    app = fastapi.FastAPI(
        title="LocalBot OpenAI-compatible API",
        version="0.2.0",
        docs_url="/docs",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    _bearer = HTTPBearer(auto_error=False)

    def _get_user_id(
        request: Request,
        creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
    ) -> str:
        if API_KEY is None:
            return f"{USER_PREFIX}guest"
        if creds is None or not hmac.compare_digest(creds.credentials, API_KEY):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing Bearer token.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        # Never use the raw token as an identity key — it leaks into the audit
        # log and SQLite history. Derive a stable, non-reversible id instead.
        token_id = hashlib.sha256(creds.credentials.encode()).hexdigest()[:16]
        return f"{USER_PREFIX}{token_id}"

    def _check_ready(app_state: Any) -> None:
        event: asyncio.Event = app_state.ready_event
        if not event.is_set():
            raise HTTPException(status_code=503, detail="Model is still loading")

    # ------------------------------------------------------------------
    # Endpoints
    # ------------------------------------------------------------------

    @app.get("/healthz", include_in_schema=False)
    async def healthz(request: Request) -> dict[str, str]:
        event: asyncio.Event = request.app.state.ready_event
        if not event.is_set():
            raise HTTPException(status_code=503, detail="Model is still loading")
        return {"status": "ok"}

    @app.get("/v1/models")
    async def list_models() -> dict:
        now = int(time.time())
        return {
            "object": "list",
            "data": [
                {"id": f"localbot-{s}", "object": "model", "created": now, "owned_by": "localbot"}
                for s in ["general", "coding", "reasoning"]
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(
        request: Request,
        body: ChatCompletionRequest,
        user_id: str = Depends(_get_user_id),
    ) -> fastapi.Response:
        _check_ready(request.app.state)

        # Per-user rate limiting (mirrors Discord behaviour).
        now = time.monotonic()
        remaining = cfg.rate_limit_seconds - (now - _rate_last.get(user_id, 0.0))
        if remaining > 0:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limited. Retry in {remaining:.1f}s.",
            )
        _rate_last[user_id] = now

        user_text = ""
        for m in reversed(body.messages):
            if m.role == "user":
                content = m.content
                user_text = content if isinstance(content, str) else ""
                break
        if not user_text:
            raise HTTPException(status_code=400, detail="No user message found.")

        agent: Agent = request.app.state.agent
        reply = await agent.handle(user_id, user_text)

        model_id = body.model or "localbot-general"
        created = int(time.time())
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

        if body.stream:
            words = reply.split(" ")

            async def _sse_chunks() -> AsyncIterator[bytes]:
                for i, word in enumerate(words):
                    token = word if i == len(words) - 1 else word + " "
                    chunk = {
                        "id": completion_id, "object": "chat.completion.chunk",
                        "created": created, "model": model_id,
                        "choices": [{"index": 0, "delta": {"role": "assistant", "content": token}, "finish_reason": None}],
                    }
                    yield b"data: " + json.dumps(chunk).encode() + b"\n\n"
                    await asyncio.sleep(0)
                done = {
                    "id": completion_id, "object": "chat.completion.chunk",
                    "created": created, "model": model_id,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                yield b"data: " + json.dumps(done).encode() + b"\n\n"
                yield b"data: [DONE]\n\n"

            return StreamingResponse(
                _sse_chunks(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        return fastapi.responses.JSONResponse(content={
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": model_id,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": reply}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        })

    return app


def main() -> None:
    import uvicorn
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    host = os.environ.get("WEBUI_HOST", "0.0.0.0")
    port = int(os.environ.get("WEBUI_PORT", "8000"))
    uvicorn.run(create_app(), host=host, port=port)


if __name__ == "__main__":
    main()
