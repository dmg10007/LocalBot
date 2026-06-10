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
                    variable is *not set*, auth is disabled entirely.
WEBUI_USER_PREFIX   String prepended to the token value to form the
                    internal user_id (default: "webui:").

LLAMA_REMOTE_HOST   When set, the webui connects to this host's
                    llama-server instead of spawning its own process.
                    Use the Docker service name, e.g. "localbot".
LLAMA_REMOTE_PORT   Port of the remote llama-server (default: 8080).

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
from contextlib import asynccontextmanager
from typing import AsyncIterator

log = logging.getLogger(__name__)


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

    @asynccontextmanager
    async def lifespan(app: fastapi.FastAPI) -> AsyncIterator[None]:
        # ---- startup ----
        init_db()
        scheduler = SchedulerService(_null_send)
        scheduler.start()
        app.state.ready = False

        if cfg.llama_remote_host:
            # ── Remote mode ──────────────────────────────────────────────────
            # Point a bare LlamaCppClient at the localbot container.
            # No subprocess is spawned — the model is already loaded there.
            log.info(
                "Remote llama-server mode: %s:%d",
                cfg.llama_remote_host,
                cfg.llama_remote_port,
            )
            remote_client = LlamaCppClient(
                host=cfg.llama_remote_host,
                port=cfg.llama_remote_port,
            )

            class _RemoteRegistry:
                async def acquire(self, slot: str) -> LlamaCppClient:  # noqa: ARG002
                    return remote_client

                def is_slot_available(self, slot: str) -> bool:
                    return slot == "general"

                async def shutdown(self) -> None:
                    await remote_client.close()

            registry: ModelRegistry | _RemoteRegistry = _RemoteRegistry()
            _agent = Agent(registry, scheduler=scheduler)  # type: ignore[arg-type]
            app.state.registry = registry
            app.state.scheduler = scheduler
            app.state.agent = _agent

            async def _warm_remote() -> None:
                try:
                    await remote_client.wait_until_ready(retries=60, delay=2.0)
                    app.state.ready = True
                    log.info(
                        "LocalBot webui ready (remote mode: %s:%d)",
                        cfg.llama_remote_host,
                        cfg.llama_remote_port,
                    )
                except Exception:
                    log.exception(
                        "Could not reach remote llama-server at %s:%d",
                        cfg.llama_remote_host,
                        cfg.llama_remote_port,
                    )

            asyncio.create_task(_warm_remote())

        else:
            # ── Local subprocess mode ────────────────────────────────────────
            registry = ModelRegistry()
            _agent = Agent(registry, scheduler=scheduler)
            app.state.registry = registry
            app.state.scheduler = scheduler
            app.state.agent = _agent

            async def _warm_local() -> None:
                try:
                    await registry.warm_general()
                    app.state.ready = True
                    log.info("LocalBot webui ready (local subprocess mode)")
                except Exception:
                    log.exception(
                        "warm_general() failed — webui will retry on first request"
                    )

            asyncio.create_task(_warm_local())

        yield

        # ---- shutdown ----
        scheduler.stop()
        await app.state.registry.shutdown()
        from localbot.tools import search as s, reddit as r
        await s.close_session()
        await r.close_session()

    async def _null_send(user_id: str, prompt: str) -> None:  # noqa: ARG001
        log.warning(
            "Scheduled job fired for user %s but HTTP delivery is not "
            "implemented; extend _null_send or integrate a push channel.",
            user_id,
        )

    app = fastapi.FastAPI(
        title="LocalBot OpenAI-compatible API",
        version="0.1.0",
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
            client_ip = (request.client.host if request.client else "unknown")
            return f"{USER_PREFIX}guest:{client_ip}"
        if creds is None or creds.credentials != API_KEY:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing Bearer token.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return f"{USER_PREFIX}{creds.credentials}"

    # ------------------------------------------------------------------
    # /healthz
    # ------------------------------------------------------------------
    @app.get("/healthz", include_in_schema=False)
    async def healthz() -> dict[str, str]:
        if not getattr(app.state, "ready", False):
            raise HTTPException(status_code=503, detail="Model is still loading")
        return {"status": "ok"}

    # ------------------------------------------------------------------
    # GET /v1/models  — no auth required (OpenWebUI polls this unauthenticated)
    # ------------------------------------------------------------------
    @app.get("/v1/models")
    async def list_models() -> dict:
        now = int(time.time())
        return {
            "object": "list",
            "data": [
                {
                    "id": f"localbot-{s}",
                    "object": "model",
                    "created": now,
                    "owned_by": "localbot",
                }
                for s in ["general", "coding", "reasoning"]
            ],
        }

    # ------------------------------------------------------------------
    # POST /v1/chat/completions
    # ------------------------------------------------------------------
    @app.post("/v1/chat/completions")
    async def chat_completions(
        request: Request,
        user_id: str = Depends(_get_user_id),
    ) -> "fastapi.Response":  # type: ignore[name-defined]
        if not getattr(request.app.state, "ready", False):
            raise HTTPException(
                status_code=503,
                detail="Model is still loading, please retry in a moment.",
            )

        body = await request.json()
        messages: list[dict] = body.get("messages", [])
        stream: bool = body.get("stream", False)

        user_text = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                content = m.get("content", "")
                user_text = content if isinstance(content, str) else ""
                break

        if not user_text:
            raise HTTPException(status_code=400, detail="No user message found.")

        _agent: Agent = request.app.state.agent
        reply_future: asyncio.Future[str] = asyncio.get_event_loop().create_future()

        async def _run() -> None:
            try:
                result = await _agent.handle(user_id, user_text)
                reply_future.set_result(result)
            except Exception as exc:  # noqa: BLE001
                reply_future.set_exception(exc)

        asyncio.create_task(_run())
        reply = await reply_future

        model_id = "localbot-general"
        created = int(time.time())
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

        if stream:
            words = reply.split(" ")

            async def _sse_chunks() -> AsyncIterator[bytes]:
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
                    await asyncio.sleep(0)
                done = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_id,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                yield b"data: " + json.dumps(done).encode() + b"\n\n"
                yield b"data: [DONE]\n\n"

            return StreamingResponse(
                _sse_chunks(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

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
                        "finish_reason": "stop",
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
