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


@dataclass
class Config:
    discord_bot_token: str = field(default_factory=lambda: _get("DISCORD_BOT_TOKEN"))

    llama_server_executable: str = field(default_factory=lambda: _get("LLAMA_SERVER_EXECUTABLE", "llama-server"))
    llama_server_model_path: str = field(default_factory=lambda: _get("LLAMA_SERVER_MODEL_PATH"))
    llama_server_host: str = field(default_factory=lambda: _get("LLAMA_SERVER_HOST", "127.0.0.1"))
    llama_server_port: int = field(default_factory=lambda: _get_int("LLAMA_SERVER_PORT", 8080))
    llama_server_n_gpu_layers: int = field(default_factory=lambda: _get_int("LLAMA_SERVER_N_GPU_LAYERS", 0))
    llama_server_ctx_size: int = field(default_factory=lambda: _get_int("LLAMA_SERVER_CTX_SIZE", 4096))
    llama_server_threads: int = field(default_factory=lambda: _get_int("LLAMA_SERVER_THREADS", 0))
    llama_server_extra_args: str = field(default_factory=lambda: _get("LLAMA_SERVER_EXTRA_ARGS"))
    # Override auto-detected model family: gemma | llama | mistral | qwen | deepseek | phi
    # Use this if the model filename doesn't match the auto-detection patterns.
    llama_server_model_family: str = field(default_factory=lambda: _get("LLAMA_SERVER_MODEL_FAMILY"))

    brave_api_key: str = field(default_factory=lambda: _get("BRAVE_API_KEY"))
    search_result_count: int = field(default_factory=lambda: _get_int("SEARCH_RESULT_COUNT", 5))
    search_fetch_count: int = field(default_factory=lambda: _get_int("SEARCH_FETCH_COUNT", 3))
    search_fetch_chars: int = field(default_factory=lambda: _get_int("SEARCH_FETCH_CHARS", 1500))
    search_fetch_timeout_seconds: int = field(default_factory=lambda: _get_int("SEARCH_FETCH_TIMEOUT_SECONDS", 8))

    model_timeout_seconds: int = field(default_factory=lambda: _get_int("MODEL_TIMEOUT_SECONDS", 120))
    model_temperature: float = field(default_factory=lambda: _get_float("MODEL_TEMPERATURE", 0.3))
    tool_timeout_seconds: int = field(default_factory=lambda: _get_int("TOOL_TIMEOUT_SECONDS", 30))
    request_deadline_seconds: int = field(default_factory=lambda: _get_int("REQUEST_DEADLINE_SECONDS", 300))

    max_history_messages: int = field(default_factory=lambda: _get_int("MAX_HISTORY_MESSAGES", 12))
    max_tool_iterations: int = field(default_factory=lambda: _get_int("MAX_TOOL_ITERATIONS", 5))

    database_path: str = field(default_factory=lambda: _get("DATABASE_PATH", "storage/localbot.sqlite3"))
    audit_log_path: str = field(default_factory=lambda: _get("AUDIT_LOG_PATH", "logs/audit.jsonl"))

    scheduler_poll_seconds: int = field(default_factory=lambda: _get_int("SCHEDULER_POLL_SECONDS", 15))
    scheduler_max_jobs: int = field(default_factory=lambda: _get_int("SCHEDULER_MAX_JOBS", 20))
    scheduler_max_jobs_per_user: int = field(default_factory=lambda: _get_int("SCHEDULER_MAX_JOBS_PER_USER", 5))

    rate_limit_seconds: int = field(default_factory=lambda: _get_int("RATE_LIMIT_SECONDS", 5))
    max_input_length: int = field(default_factory=lambda: _get_int("MAX_INPUT_LENGTH", 1000))

    def __post_init__(self) -> None:
        missing = []
        if not self.discord_bot_token:
            missing.append("DISCORD_BOT_TOKEN")
        if not self.llama_server_model_path:
            missing.append("LLAMA_SERVER_MODEL_PATH")
        if missing:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing)}. "
                "Copy .env.example to .env and fill in the required values."
            )


cfg = Config()
