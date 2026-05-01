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

    async def start(self) -> None:
        """Launch llama-server as a subprocess."""
        # NOTE: Do NOT pass --chat-template here. The GGUF file embeds the
        # correct template for the model. Overriding it causes control tokens
        # like <|eot_id|> to leak into responses as literal text (e.g. 'd\n').
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
        # LLAMA_SERVER_EXTRA_ARGS in .env is the escape hatch for any extra
        # llama-server flags (e.g. --flash-attn, --no-mmap, --parallel).
        if cfg.llama_server_extra_args:
            cmd += shlex.split(cfg.llama_server_extra_args)

        log.info("Starting llama-server: %s", " ".join(cmd))
        self._proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        await asyncio.sleep(3)
        log.info("llama-server started (pid=%s)", self._proc.pid)

    async def stop(self) -> None:
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
