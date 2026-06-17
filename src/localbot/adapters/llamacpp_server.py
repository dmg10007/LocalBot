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
        port: int | None = None,
    ) -> None:
        # Per-slot overrides; fall back to global cfg when not provided.
        self._model_path = model_path or cfg.llama_server_model_path
        self._port = port or cfg.llama_server_port

        self._proc: Process | None = None
        self._log_task: asyncio.Task | None = None

    async def start(self) -> None:
        """Launch llama-server as a subprocess."""
        cmd = [
            cfg.llama_server_executable,
            "--model", self._model_path,
            "--host", cfg.llama_server_host,
            "--port", str(self._port),
            "--n-gpu-layers", str(cfg.llama_server_n_gpu_layers),
            "--ctx-size", str(cfg.llama_server_ctx_size),
        ]
        if cfg.llama_server_threads > 0:
            cmd += ["--threads", str(cfg.llama_server_threads)]
        if cfg.llama_server_extra_args:
            cmd += shlex.split(cfg.llama_server_extra_args)
        # Speculative decoding: append draft model args when configured.
        # --draft-max/--draft-min were removed from llama-server (b9xxx+);
        # replaced by --spec-draft-n-max/--spec-draft-n-min. --spec-type now
        # defaults to "none" — passing --model-draft alone no longer implicitly
        # enables speculative decoding, so --spec-type draft-simple is required.
        if cfg.slot_draft_model:
            cmd += [
                "--spec-type", "draft-simple",
                "--model-draft", cfg.slot_draft_model,
                "--spec-draft-n-max", str(cfg.slot_draft_max),
                "--spec-draft-n-min", "1",
            ]
            log.info("Speculative decoding enabled with draft model: %s", cfg.slot_draft_model)

        log.info("Starting llama-server: %s", " ".join(cmd))
        self._proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        self._log_task = asyncio.create_task(self._pipe_logs())
        log.info("llama-server process started (pid=%s), waiting for readiness...", self._proc.pid)

    async def _pipe_logs(self) -> None:
        """Read llama-server stdout/stderr and forward to our logger at INFO.

        Previously logged at DEBUG, which hid crashes and bind errors.
        INFO ensures output is always visible at the default log level.
        """
        if self._proc is None or self._proc.stdout is None:
            return
        try:
            async for line in self._proc.stdout:
                decoded = line.decode(errors="replace").rstrip()
                if decoded:
                    log.info("[llama-server] %s", decoded)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            log.warning("llama-server log reader stopped unexpectedly: %s", exc)

    async def stop(self) -> None:
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

    @property
    def pid(self) -> int | None:
        return self._proc.pid if self._proc else None

    @property
    def returncode(self) -> int | None:
        return self._proc.returncode if self._proc else None

    async def ensure_running(self) -> None:
        """Restart server if it has crashed (self-healing)."""
        if not self.is_running:
            log.warning("llama-server is not running — restarting...")
            await self.start()
