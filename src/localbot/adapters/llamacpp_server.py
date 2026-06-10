"""Manages the llama-server subprocess lifecycle."""
from __future__ import annotations

import asyncio
import logging
import shlex

from asyncio.subprocess import Process

from localbot.config import cfg

log = logging.getLogger(__name__)


class LlamaCppServer:
    def __init__(
        self,
        model_path: str | None = None,
        host: str | None = None,
        port: int | None = None,
        n_gpu_layers: int | None = None,
        ctx_size: int | None = None,
        threads: int | None = None,
        extra_args: str | None = None,
    ) -> None:
        # Fall back to the legacy single-server config for backward compat.
        self._model_path = model_path or cfg.llama_server_model_path
        self._host = host or cfg.llama_server_host
        self._port = port or cfg.llama_server_port
        self._n_gpu_layers = n_gpu_layers if n_gpu_layers is not None else cfg.llama_server_n_gpu_layers
        self._ctx_size = ctx_size if ctx_size is not None else cfg.llama_server_ctx_size
        self._threads = threads if threads is not None else cfg.llama_server_threads
        self._extra_args = extra_args if extra_args is not None else cfg.llama_server_extra_args

        self._proc: Process | None = None
        # Fix #20: track the background log-reader task so it can be cancelled
        # cleanly when the server is stopped.
        self._log_task: asyncio.Task | None = None

    async def start(self) -> None:
        """Launch llama-server as a subprocess."""
        # NOTE: Do NOT pass --chat-template here. The GGUF file embeds the
        # correct template for the model. Overriding it causes control tokens
        # like <|eot_id|> to leak into responses as literal text.
        cmd = [
            cfg.llama_server_executable,
            "--model", self._model_path,
            "--host", self._host,
            "--port", str(self._port),
            "--n-gpu-layers", str(self._n_gpu_layers),
            "--ctx-size", str(self._ctx_size),
        ]
        if self._threads > 0:
            cmd += ["--threads", str(self._threads)]
        # LLAMA_SERVER_EXTRA_ARGS / slot-specific extra args are the escape
        # hatch for extra llama-server flags (e.g. --flash-attn, --no-mmap,
        # --parallel, --reasoning-budget).
        # WARNING: This value must be trusted — never allow user input to
        # influence this setting as it is passed directly to the subprocess.
        if self._extra_args:
            cmd += shlex.split(self._extra_args)

        log.info("Starting llama-server: %s", " ".join(cmd))
        self._proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        # Fix #20: launch a background task to drain subprocess stdout into
        # our logger. Without this, llama-server output (including OOM and
        # CUDA errors) was silently discarded.
        self._log_task = asyncio.create_task(self._pipe_logs())
        log.info(
            "llama-server process started (pid=%s, port=%d), waiting for readiness...",
            self._proc.pid,
            self._port,
        )

    async def _pipe_logs(self) -> None:
        """Read llama-server stdout line-by-line and forward to our logger."""
        if self._proc is None or self._proc.stdout is None:
            return
        try:
            async for line in self._proc.stdout:
                decoded = line.decode(errors="replace").rstrip()
                if decoded:
                    log.debug("[llama-server:%d] %s", self._port, decoded)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            log.warning("llama-server log reader stopped unexpectedly: %s", exc)

    async def stop(self) -> None:
        # Cancel the log-reader task before terminating the process.
        if self._log_task and not self._log_task.done():
            self._log_task.cancel()
            try:
                await self._log_task
            except asyncio.CancelledError:
                pass
        self._log_task = None

        if self._proc and self._proc.returncode is None:
            self._proc.terminate()
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                self._proc.kill()
            log.info("llama-server (port=%d) stopped", self._port)
        self._proc = None

    @property
    def is_running(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    async def ensure_running(self) -> None:
        """Restart server if it has crashed (self-healing)."""
        if not self.is_running:
            log.warning("llama-server (port=%d) is not running — restarting...", self._port)
            await self.start()
