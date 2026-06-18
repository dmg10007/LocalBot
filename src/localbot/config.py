"""Load and validate configuration from environment variables.

Uses pydantic-settings for strict type coercion, documented defaults,
and a single `cfg` singleton.  All callers import `cfg` directly.
"""
from __future__ import annotations

from pathlib import Path

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path.cwd().resolve()


def _safe_path(value: str, field_name: str) -> str:
    """Resolve *value* relative to the project root and reject escapes."""
    resolved = (_PROJECT_ROOT / value).resolve()
    try:
        resolved.relative_to(_PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError(
            f"{field_name}={value!r} resolves to {resolved}, which is outside "
            f"the project root {_PROJECT_ROOT}. Use a relative sub-path."
        ) from exc
    return str(resolved)


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Required ────────────────────────────────────────────────────────────
    discord_bot_token: str = ""

    # ── llama-server subprocess ──────────────────────────────────────────────
    llama_server_executable: str = "llama-server"
    llama_server_model_path: str = ""
    llama_server_host: str = "127.0.0.1"
    llama_server_client_host: str = "127.0.0.1"
    llama_server_port: int = 8080
    llama_server_n_gpu_layers: int = 0
    llama_server_ctx_size: int = 4096
    llama_server_threads: int = 0
    llama_server_extra_args: str = ""
    llama_server_model_family: str = ""

    # ── Remote llama-server (webui / Docker Compose) ─────────────────────────
    llama_remote_host: str = ""
    llama_remote_port: int = 8080

    # ── Multi-slot model routing ─────────────────────────────────────────────
    slot_general_model: str = ""
    slot_general_port: int = 0          # 0 → falls back to llama_server_port
    slot_coding_model: str = ""
    slot_coding_port: int = 8081
    slot_reasoning_model: str = ""
    slot_reasoning_port: int = 8082

    idle_unload_seconds: int = 120

    # ── Auto-updater ─────────────────────────────────────────────────────────
    llama_update_check: bool = True
    llama_update_check_timeout_seconds: int = 10
    llama_update_auto: bool = False
    llama_update_prompt_timeout_seconds: int = 30

    # ── Groq API (optional fast-path cloud inference) ────────────────────────
    # When set, non-sensitive queries may be routed to Groq for sub-100 ms TTFT.
    # See intent.is_groq_eligible() for the routing policy.
    groq_api_key: str = ""
    groq_model: str = "llama-3.1-8b-instant"

    # ── Cloudflare Workers AI (optional search summarisation) ────────────────
    cloudflare_account_id: str = ""
    cloudflare_api_token: str = ""

    # ── Brave Search ─────────────────────────────────────────────────────────
    brave_api_key: str = ""
    search_result_count: int = 8
    search_fetch_count: int = 5
    search_fetch_chars: int = 2000
    search_fetch_timeout_seconds: int = 8

    # ── Speculative decoding ─────────────────────────────────────────────────
    # Path to a small draft model (0.5B–1B Q4_K_M).  When non-empty this is
    # appended to LLAMA_SERVER_EXTRA_ARGS as --model-draft.  Leave blank to
    # disable.  Set draft_max to tune the speculation window.
    slot_draft_model: str = ""
    slot_draft_max: int = 5

    # ── Inference knobs ──────────────────────────────────────────────────────
    model_timeout_seconds: int = 120
    model_temperature: float = 0.3
    tool_timeout_seconds: int = 30
    request_deadline_seconds: int = 300
    max_history_messages: int = 12
    max_tool_iterations: int = 5

    # ── Storage ──────────────────────────────────────────────────────────────
    database_path: str = "storage/localbot.sqlite3"
    audit_log_path: str = "logs/audit.jsonl"

    # ── Scheduler ────────────────────────────────────────────────────────────
    scheduler_max_jobs: int = 20
    scheduler_max_jobs_per_user: int = 5

    # ── Rate limiting ────────────────────────────────────────────────────────
    rate_limit_seconds: int = 5
    max_input_length: int = 1000

    # ── Sandbox / GitHub ─────────────────────────────────────────────────────
    sandbox_root: str = ""
    github_token: str = ""
    github_default_owner: str = ""
    # Comma-separated allowlist of "owner" or "owner/repo" the LLM may touch.
    # Empty → fall back to github_default_owner (if set) → deny all writes.
    github_allowed_repos: str = ""

    # ── Cross-field resolution ────────────────────────────────────────────────
    @model_validator(mode="after")
    def _resolve(self) -> "Config":
        # slot_general falls back to legacy single-model keys.
        if not self.slot_general_model:
            self.slot_general_model = self.llama_server_model_path
        if self.slot_general_port == 0:
            self.slot_general_port = self.llama_server_port

        missing: list[str] = []
        if not self.discord_bot_token:
            missing.append("DISCORD_BOT_TOKEN")
        if not self.slot_general_model:
            missing.append("LLAMA_SERVER_MODEL_PATH (or SLOT_GENERAL_MODEL)")
        if missing:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing)}. "
                "Copy .env.example to .env and fill in the required values."
            )

        self.database_path = _safe_path(self.database_path, "DATABASE_PATH")
        self.audit_log_path = _safe_path(self.audit_log_path, "AUDIT_LOG_PATH")
        return self

    @field_validator("model_temperature")
    @classmethod
    def _clamp_temperature(cls, v: float) -> float:
        if not 0.0 <= v <= 2.0:
            raise ValueError("MODEL_TEMPERATURE must be between 0.0 and 2.0")
        return v

    @field_validator("llama_server_n_gpu_layers")
    @classmethod
    def _nonneg_gpu_layers(cls, v: int) -> int:
        if v < 0:
            raise ValueError("LLAMA_SERVER_N_GPU_LAYERS must be >= 0")
        return v


cfg = Config()
