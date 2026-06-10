"""OpenAI-compatible HTTP API server so OpenWebUI can talk to LocalBot.

Start with::

    localbot-webui
    # or
    python -m localbot.webui

Environment variables
---------------------
WEBUI_HOST          Bind address (default: 0.0.0.0)
WEBUI_PORT          Port (default: 8000)
WEBUI_API_KEY       Bearer token required on every request.  When this
                    variable is *not set*, auth is disabled entirely so
                    you can try the server locally without any token.
WEBUI_USER_PREFIX   String prepended to the token value to form the
                    internal user_id (default: "webui:").  This keeps
                    Web UI history isolated from Discord DM history.

OpenWebUI connection
--------------------
In OpenWebUI → Settings → Connections → OpenAI API:
  URL:  http://<host>:8000/v1
  Key:  <value of WEBUI_API_KEY>  (or anything when auth is off)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from typing import AsyncIterator

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports — FastAPI / uvicorn are only required when the webui extra
# is installed.  We import at function-call time so the rest of localbot
# still works without them.
# ---------------------------------------------------------------------------

def _require_fastapi() -> None:
    try:
        import fastapi  # noqa: F401
        import uvicorn  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "The 'webui' extra is required.  Install with:\n"
            "    pip install -e .[webui]"
        ) from exc


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

def create_app() -> "fastapi.FastAPI":  # type: ignore[name-defined]
    _require_fastapi()

    import fastapi
    from fastapi import Depends, HTTPException, Request, status
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse
    from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

    from localbot.adapters.model_registry import ModelRegistry
    from localbot.agent import Agent
    from localbot.scheduler.service import SchedulerService
    from localbot.storage.db import init_db

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    API_KEY: str | None = os.environ.get("WEBUI_API_KEY") or None
    USER_PREFIX: str = os.environ.get("WEBUI_USER_PREFIX", "webui:")

    # ------------------------------------------------------------------
    # Shared bot state (initialised on startup)
    # ------------------------------------------------------------------
    registry: ModelRegistry | None = None
    scheduler: SchedulerService | None = None
    agent: Agent | None = None

    app = fastapi.FastAPI(
        title="LocalBot OpenAI-compatible API",
        version="0.1.0",
        docs_url="/docs",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------
    # Lifespan
    # ------------------------------------------------------------------
    @app.on_event("startup")
    async def _startup() -> None:
        nonlocal registry, scheduler, agent
        init_db()
        registry = ModelRegistry()
        scheduler = SchedulerService(_null_send)
        agent = Agent(registry, scheduler=scheduler)
        await registry.warm_general()
        scheduler.start()
        log.info("LocalBot webui ready")

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        if scheduler:
            scheduler.stop()
        if registry:
            await registry.shutdown()
        from localbot.tools import search as s, reddit as r
        await s.close_session()
        await r.close_session()

    async def _null_send(user_id: str, prompt: str) -> None:  # noqa: ARG001
        """Scheduled-job delivery is a no-op for the HTTP server."""
        log.warning(
            "Scheduled job fired for user %s but HTTP delivery is not "
            "implemented; extend _null_send or integrate a push channel.",
            user_id,
        )

    # ------------------------------------------------------------------
    # Auth dependency
    # ------------------------------------------------------------------
    _bearer = HTTPBearer(auto_error=False)

    def _get_user_id(
        request: Request,
        creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
    ) -> str:
        """Validate Bearer token and return the internal user_id."""
        if API_KEY is None:
            # Auth disabled — derive a stable guest id from IP.
            client_ip = (request.client.host if request.client else "unknown")
            return f"{USER_PREFIX}guest:{client_ip}"

        if creds is None or creds.credentials != API_KEY:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing Bearer token.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        # Use the raw token as the user discriminator so that multiple
        # OpenWebUI accounts with different keys get isolated histories.
        return f"{USER_PREFIX}{creds.credentials}"

    # ------------------------------------------------------------------
    # /healthz
    # ------------------------------------------------------------------
    @app.get("/healthz", include_in_schema=False)
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    # ------------------------------------------------------------------
    # GET /v1/models
    # ------------------------------------------------------------------
    @app.get("/v1/models")
    async def list_models(
        user_id: str = Depends(_get_user_id),  # noqa: ARG001
    ) -> dict:
        """Return the available model slots in OpenAI list format."""
        now = int(time.time())
        slots = ["general", "coding", "reasoning"]
        data = [
            {
                "id": f"localbot-{slot}",
                "object": "model",
                "created": now,
                "owned_by": "localbot",
            }
            for slot in slots
        ]
        return {"object": "list", "data": data}

    # ------------------------------------------------------------------
    # POST /v1/chat/completions
    # ------------------------------------------------------------------
    @app.post("/v1/chat/completions")
    async def chat_completions(
        request: Request,
        user_id: str = Depends(_get_user_id),
    ) -> "fastapi.Response":  # type: ignore[name-defined]
        body = await request.json()
        messages: list[dict] = body.get("messages", [])
        stream: bool = body.get("stream", False)

        # Extract the last user turn as the prompt.
        user_text = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                content = m.get("content", "")
                user_text = content if isinstance(content, str) else ""
                break

        if not user_text:
            raise HTTPException(status_code=400, detail="No user message found.")

        if agent is None:
            raise HTTPException(status_code=503, detail="Agent not ready.")

        reply_future: asyncio.Future[str] = asyncio.get_event_loop().create_future()

        async def _run() -> None:
            try:
                result = await agent.handle(user_id, user_text)
                reply_future.set_result(result)
            except Exception as exc:  # noqa: BLE001
                reply_future.set_exception(exc)

        asyncio.create_task(_run())
        reply = await reply_future

        model_id = "localbot-general"
        finish_reason = "stop"
        created = int(time.time())
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

        if stream:
            async def _sse_chunks() -> AsyncIterator[bytes]:
                # Stream the reply word-by-word for a natural typewriter effect.
                words = reply.split(" ")
                for i, word in enumerate(words):
                    token = word if i == len(words) - 1 else word + " "
                    chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_id,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"role": "assistant", "content": token},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield b"data: " + json.dumps(chunk).encode() + b"\n\n"
                    await asyncio.sleep(0)  # yield control to event loop

                # Final chunk with finish_reason
                done_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_id,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": finish_reason,
                        }
                    ],
                }
                yield b"data: " + json.dumps(done_chunk).encode() + b"\n\n"
                yield b"data: [DONE]\n\n"

            return StreamingResponse(
                _sse_chunks(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )

        # Non-streaming response
        return fastapi.responses.JSONResponse(
            content={
                "id": completion_id,
                "object": "chat.completion",
                "created": created,
                "model": model_id,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": reply},
                        "finish_reason": finish_reason,
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            }
        )

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    host = os.environ.get("WEBUI_HOST", "0.0.0.0")
    port = int(os.environ.get("WEBUI_PORT", "8000"))

    app = create_app()
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
