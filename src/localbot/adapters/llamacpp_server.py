"""Manages the llama-server subprocess lifecycle."""
from __future__ import annotations

import asyncio
import logging
import shlex

from asyncio.subprocess import Process

from localbot.config import cfg

log = logging.getLogger(__name__)


class LlamaCppServer:
    def __init__(self) -> None:
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
            "--model", cfg.llama_server_model_path,
            "--host", cfg.llama_server_host,
            "--port", str(cfg.llama_server_port),
            "--n-gpu-layers", str(cfg.llama_server_n_gpu_layers),
            "--ctx-size", str(cfg.llama_server_ctx_size),
        ]
        if cfg.llama_server_threads > 0:
            cmd += ["--threads", str(cfg.llama_server_threads)]
        # LLAMA_SERVER_EXTRA_ARGS is the escape hatch for extra llama-server
        # flags (e.g. --flash-attn, --no-mmap, --parallel, --reasoning-budget).
        # WARNING: This value must be trusted — never allow user input to
        # influence this setting as it is passed directly to the subprocess.
        if cfg.llama_server_extra_args:
            cmd += shlex.split(cfg.llama_server_extra_args)

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
        log.info("llama-server process started (pid=%s), waiting for readiness...", self._proc.pid)

    async def _pipe_logs(self) -> None:
        """Read llama-server stdout line-by-line and forward to our logger."""
        if self._proc is None or self._proc.stdout is None:
            return
        try:
            async for line in self._proc.stdout:
                decoded = line.decode(errors="replace").rstrip()
                if decoded:
                    log.debug("[llama-server] %s", decoded)
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
            log.info("llama-server stopped")
        self._proc = None

    @property
    def is_running(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    async def ensure_running(self) -> None:
        """Restart server if it has crashed (self-healing)."""
        if not self.is_running:
            log.warning("llama-server is not running — restarting...")
            await self.start()
