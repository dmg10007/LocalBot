"""Load and validate configuration from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


def _get(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def _get_int(key: str, default: int) -> int:
    val = os.environ.get(key)
    return int(val) if val is not None else default


def _get_float(key: str, default: float) -> float:
    val = os.environ.get(key)
    return float(val) if val is not None else default


def _get_bool(key: str, default: bool) -> bool:
    val = os.environ.get(key)
    if val is None:
        return default
    return val.strip().lower() not in ("0", "false", "no", "off")


@dataclass
class Config:
    discord_bot_token: str = field(default_factory=lambda: _get("DISCORD_BOT_TOKEN"))

    # ── Legacy single-model config (still works; used as general slot fallback) ──
    llama_server_executable: str = field(default_factory=lambda: _get("LLAMA_SERVER_EXECUTABLE", "llama-server"))
    llama_server_model_path: str = field(default_factory=lambda: _get("LLAMA_SERVER_MODEL_PATH"))
    llama_server_host: str = field(default_factory=lambda: _get("LLAMA_SERVER_HOST", "127.0.0.1"))
    llama_server_port: int = field(default_factory=lambda: _get_int("LLAMA_SERVER_PORT", 8080))
    llama_server_n_gpu_layers: int = field(default_factory=lambda: _get_int("LLAMA_SERVER_N_GPU_LAYERS", 0))
    llama_server_ctx_size: int = field(default_factory=lambda: _get_int("LLAMA_SERVER_CTX_SIZE", 4096))
    llama_server_threads: int = field(default_factory=lambda: _get_int("LLAMA_SERVER_THREADS", 0))
    llama_server_extra_args: str = field(default_factory=lambda: _get("LLAMA_SERVER_EXTRA_ARGS"))
    llama_server_model_family: str = field(default_factory=lambda: _get("LLAMA_SERVER_MODEL_FAMILY"))

    # ── Multi-model slot config ──────────────────────────────────────────────────
    # General slot — falls back to legacy LLAMA_SERVER_* values.
    slot_general_model: str = field(
        default_factory=lambda: _get("SLOT_GENERAL_MODEL") or _get("LLAMA_SERVER_MODEL_PATH")
    )
    slot_general_port: int = field(
        default_factory=lambda: _get_int("SLOT_GENERAL_PORT", 0) or _get_int("LLAMA_SERVER_PORT", 8080)
    )
    slot_general_n_gpu_layers: int = field(
        default_factory=lambda: _get_int("SLOT_GENERAL_N_GPU_LAYERS", 0)
        or _get_int("LLAMA_SERVER_N_GPU_LAYERS", 0)
    )
    slot_general_ctx_size: int = field(
        default_factory=lambda: _get_int("SLOT_GENERAL_CTX_SIZE", 0)
        or _get_int("LLAMA_SERVER_CTX_SIZE", 4096)
    )
    slot_general_threads: int = field(
        default_factory=lambda: _get_int("SLOT_GENERAL_THREADS", 0)
        or _get_int("LLAMA_SERVER_THREADS", 0)
    )
    slot_general_extra_args: str = field(
        default_factory=lambda: _get("SLOT_GENERAL_EXTRA_ARGS") or _get("LLAMA_SERVER_EXTRA_ARGS")
    )

    # Coding slot (Qwen2.5-Coder or similar). Leave blank to disable.
    slot_coding_model: str = field(default_factory=lambda: _get("SLOT_CODING_MODEL"))
    slot_coding_port: int = field(default_factory=lambda: _get_int("SLOT_CODING_PORT", 8081))
    slot_coding_n_gpu_layers: int = field(default_factory=lambda: _get_int("SLOT_CODING_N_GPU_LAYERS", 0))
    slot_coding_ctx_size: int = field(default_factory=lambda: _get_int("SLOT_CODING_CTX_SIZE", 4096))
    slot_coding_threads: int = field(default_factory=lambda: _get_int("SLOT_CODING_THREADS", 0))
    slot_coding_extra_args: str = field(default_factory=lambda: _get("SLOT_CODING_EXTRA_ARGS"))

    # Reasoning slot (DeepSeek-R1 or similar). Leave blank to disable.
    slot_reasoning_model: str = field(default_factory=lambda: _get("SLOT_REASONING_MODEL"))
    slot_reasoning_port: int = field(default_factory=lambda: _get_int("SLOT_REASONING_PORT", 8082))
    slot_reasoning_n_gpu_layers: int = field(default_factory=lambda: _get_int("SLOT_REASONING_N_GPU_LAYERS", 0))
    slot_reasoning_ctx_size: int = field(default_factory=lambda: _get_int("SLOT_REASONING_CTX_SIZE", 4096))
    slot_reasoning_threads: int = field(default_factory=lambda: _get_int("SLOT_REASONING_THREADS", 0))
    slot_reasoning_extra_args: str = field(default_factory=lambda: _get("SLOT_REASONING_EXTRA_ARGS"))

    # Seconds of inactivity before the active heavy slot is unloaded and the
    # lightweight general model is reloaded.
    idle_unload_seconds: int = field(default_factory=lambda: _get_int("IDLE_UNLOAD_SECONDS", 120))

    # ── Update checker ───────────────────────────────────────────────────────────
    llama_update_check: bool = field(default_factory=lambda: _get_bool("LLAMA_UPDATE_CHECK", True))
    llama_update_check_timeout_seconds: int = field(
        default_factory=lambda: _get_int("LLAMA_UPDATE_CHECK_TIMEOUT_SECONDS", 10)
    )
    llama_update_auto: bool = field(default_factory=lambda: _get_bool("LLAMA_UPDATE_AUTO", False))
    llama_update_prompt_timeout_seconds: int = field(
        default_factory=lambda: _get_int("LLAMA_UPDATE_PROMPT_TIMEOUT_SECONDS", 30)
    )

    # ── Search ───────────────────────────────────────────────────────────────────
    brave_api_key: str = field(default_factory=lambda: _get("BRAVE_API_KEY"))
    search_result_count: int = field(default_factory=lambda: _get_int("SEARCH_RESULT_COUNT", 5))
    search_fetch_count: int = field(default_factory=lambda: _get_int("SEARCH_FETCH_COUNT", 3))
    search_fetch_chars: int = field(default_factory=lambda: _get_int("SEARCH_FETCH_CHARS", 1500))
    search_fetch_timeout_seconds: int = field(default_factory=lambda: _get_int("SEARCH_FETCH_TIMEOUT_SECONDS", 8))

    # ── Model / inference ────────────────────────────────────────────────────────
    model_timeout_seconds: int = field(default_factory=lambda: _get_int("MODEL_TIMEOUT_SECONDS", 120))
    model_temperature: float = field(default_factory=lambda: _get_float("MODEL_TEMPERATURE", 0.3))
    tool_timeout_seconds: int = field(default_factory=lambda: _get_int("TOOL_TIMEOUT_SECONDS", 30))
    request_deadline_seconds: int = field(default_factory=lambda: _get_int("REQUEST_DEADLINE_SECONDS", 300))

    max_history_messages: int = field(default_factory=lambda: _get_int("MAX_HISTORY_MESSAGES", 12))
    max_tool_iterations: int = field(default_factory=lambda: _get_int("MAX_TOOL_ITERATIONS", 5))

    # ── Storage ──────────────────────────────────────────────────────────────────
    database_path: str = field(default_factory=lambda: _get("DATABASE_PATH", "storage/localbot.sqlite3"))
    audit_log_path: str = field(default_factory=lambda: _get("AUDIT_LOG_PATH", "logs/audit.jsonl"))

    # ── Scheduler ────────────────────────────────────────────────────────────────
    scheduler_max_jobs: int = field(default_factory=lambda: _get_int("SCHEDULER_MAX_JOBS", 20))
    scheduler_max_jobs_per_user: int = field(default_factory=lambda: _get_int("SCHEDULER_MAX_JOBS_PER_USER", 5))

    # ── Rate limiting ────────────────────────────────────────────────────────────
    rate_limit_seconds: int = field(default_factory=lambda: _get_int("RATE_LIMIT_SECONDS", 5))
    max_input_length: int = field(default_factory=lambda: _get_int("MAX_INPUT_LENGTH", 1000))

    # ── Coding assistant ─────────────────────────────────────────────────────────
    # Absolute path to the local folder the bot is allowed to read/write.
    # The bot will refuse any path that escapes this directory.
    sandbox_root: str = field(default_factory=lambda: _get("SANDBOX_ROOT"))
    # Personal access token with repo scope for GitHub operations.
    github_token: str = field(default_factory=lambda: _get("GITHUB_TOKEN"))
    # Default GitHub owner/org used when the user doesn't specify one.
    github_default_owner: str = field(default_factory=lambda: _get("GITHUB_DEFAULT_OWNER"))

    def __post_init__(self) -> None:
        missing = []
        if not self.discord_bot_token:
            missing.append("DISCORD_BOT_TOKEN")
        if not self.slot_general_model:
            missing.append("LLAMA_SERVER_MODEL_PATH (or SLOT_GENERAL_MODEL)")
        if missing:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing)}. "
                "Copy .env.example to .env and fill in the required values."
            )


cfg = Config()
