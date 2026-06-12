# LocalBot

A privacy-first, self-hosted AI assistant that runs a local LLM entirely on your own machine — no cloud API calls for inference. All conversation history, scheduled jobs, the audit log, and the model weights stay local.

The Python layer is intentionally thin: it never loads the model itself. All inference is delegated to [`llama-server`](https://github.com/ggerganov/llama.cpp) via its OpenAI-compatible HTTP API. LocalBot surfaces through two interfaces simultaneously: **Discord DMs** and a **browser UI** via [OpenWebUI](https://github.com/open-webui/open-webui).

---

## Features

- **Conversational chat** with per-user SQLite-backed message history
- **Deep web search** via Brave Search API — fetches and summarises actual page content, not just snippets
- **Reddit search** — unauthenticated JSON API, no credentials needed
- **Sandboxed filesystem** — the LLM can read, write, patch, and grep files inside a locked-down directory; `../` traversal is blocked at the OS level
- **Scheduled prompts** — describe jobs in natural language; the LLM converts to cron and calls `schedule_job` directly
- **Self-diagnostics** — the LLM can call `read_logs` to reason over its own audit log conversationally
- **Multi-slot model routing** — routes each request to a `general`, `coding`, or `reasoning` model slot based on intent
- **Two-phase coding dispatch** — when a coding request also needs external lookup, the general model fetches context first, then the coding model implements
- **Thinking model support** — auto-strips `<think>` blocks for Qwen, DeepSeek, Gemma
- **Auto-updater** — checks for newer llama.cpp builds on startup, prompts to install
- **Self-healing** — detects `llama-server` crashes and restarts automatically
- **OpenWebUI / OpenAI-compatible API** — `localbot-webui` starts a FastAPI server; stream or non-stream, Bearer token auth, Docker Compose stack included
- **Rate limiting** — per-user cooldown on both Discord and the HTTP API (HTTP 429)
- **Minimal footprint** — discord.py + aiohttp + APScheduler + BeautifulSoup4 + pydantic-settings; FastAPI/uvicorn only needed for the web UI extra

---

## Prerequisites

### 1. Python 3.11+

```powershell
python --version   # must be 3.11 or higher
```

Download from [python.org](https://www.python.org/downloads/) if needed.

### 2. Docker Desktop

The recommended way to run LocalBot is via Docker Compose. Download and install [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/).

Verify after installation:

```powershell
docker --version
docker compose version
```

### 3. llama-server (llama.cpp) *(local/non-Docker mode only)*

If you're running **without Docker**, download a pre-built `llama-server` binary from the [llama.cpp releases page](https://github.com/ggerganov/llama.cpp/releases/latest).

| Platform | File to download |
|---|---|
| Windows, CPU only | `llama-bXXXX-bin-win-cpu-x64.zip` |
| Windows, NVIDIA GPU | `llama-bXXXX-bin-win-cuda-cu12.x-x64.zip` |

Extract to a permanent location (e.g. `C:\llama\`) and verify:

```powershell
C:\llama\llama-server.exe --version
```

You don't need to add it to `PATH` — point LocalBot at it directly via `LLAMA_SERVER_EXECUTABLE` in `.env`.

### 4. A GGUF model file

#### Recommended models (≤16 GB RAM)

| Slot | Model | Quant | RAM |
|---|---|---|---|
| `general` | [Qwen3.5-7B-Instruct](https://huggingface.co/Qwen/Qwen3.5-7B-Instruct-GGUF) | Q4_K_M | ~5.5 GB |
| `coding` | [Qwen3-Coder-7B-A2B](https://huggingface.co/Qwen/Qwen3-Coder-7B-A2B-GGUF) | Q4_K_M | ~5 GB |
| `reasoning` | [Qwen3.5-7B-Instruct](https://huggingface.co/Qwen/Qwen3.5-7B-Instruct-GGUF) | Q5_K_M | ~5.8 GB |

> **Tip:** `general` and `reasoning` can share the same GGUF file — only the system prompt differs, so switching between them is instant with no model reload.

#### Higher-RAM options (24–48 GB)

| Slot | Model | Quant | RAM |
|---|---|---|---|
| `general` | Mistral Small 3.1 24B | Q4_K_M | ~14 GB |
| `coding` | Qwen3-Coder-30B-A3B (MoE) | Q4_K_M | ~18 GB |
| `reasoning` | Qwen3.5-27B Reasoning-Distilled v2 | Q4_K_M | ~18 GB |

Place your model file(s) in `C:\Llama\Models\` — this folder is mounted into the container as `/models`.

### 5. A Discord Bot Token *(Discord mode only)*

1. Go to the [Discord Developer Portal](https://discord.com/developers/applications) and create a new application.
2. Under **Bot**, click **Add Bot** and copy the token.
3. Enable the **Message Content Intent** under **Privileged Gateway Intents**.
4. Invite the bot with the `bot` scope and `Send Messages` + `Read Message History` permissions.

---

## Setup

### Option A — Docker Compose (recommended)

This is the fastest path. Docker handles the Python environment, llama-server binary, and all services.

#### 1. Clone the repo

```powershell
git clone https://github.com/dmg10007/LocalBot.git
Set-Location LocalBot
```

#### 2. Create your `.env` file

```powershell
Copy-Item .env.example .env
```

Open `.env` in your editor and fill in the required values:

```powershell
notepad .env
```

| Variable | Required | Description |
|---|---|---|
| `DISCORD_BOT_TOKEN` | ✅ *(Discord mode)* | Bot token from the Developer Portal |
| `GITHUB_TOKEN` | — | GitHub PAT for the MCP server (`repo`, `read:org`) |
| `WEBUI_API_KEY` | ✅ *(if exposed off-loopback)* | Shared secret between OpenWebUI and the API bridge |
| `SLOT_0_MODEL` | ✅ | Container path to your `.gguf` file (e.g. `/models/my-model-q4_k_m.gguf`) |
| `LLAMA_CTX_SIZE` | — | Context window size. Default `4096` |

See [`.env.example`](.env.example) for all options with inline descriptions.

> **Generate a secure `WEBUI_API_KEY`:**
> ```powershell
> python -c "import secrets; print(secrets.token_urlsafe(32))"
> ```

#### 3. Create required local directories

```powershell
New-Item -ItemType Directory -Force -Path logs, storage, sandbox
```

#### 4. Start all services

```powershell
docker compose up -d
```

This starts:
- **`localbot`** — Discord bot + llama-server subprocess on the inference budget
- **`localbot-webui`** — OpenAI-compatible FastAPI bridge on `http://localhost:8080`
- **`github-mcp-server`** — GitHub MCP server on `http://localhost:8181`
- **`openwebui`** — Browser UI on `http://localhost:3000`

#### 5. Check service health

```powershell
docker compose ps
```

All four containers should show `running`. Check logs for any individual service:

```powershell
docker compose logs -f localbot
docker compose logs -f github-mcp-server
```

#### 6. Stopping and restarting

```powershell
# Stop all services (preserves data)
docker compose down

# Restart a single service after a config change
docker compose up -d --force-recreate localbot

# Rebuild the image after a code change
docker compose up -d --build localbot
```

---

### Option B — Local Python (no Docker)

Use this if you want to run LocalBot directly on your machine without containers.

#### 1. Clone and create a virtual environment

```powershell
git clone https://github.com/dmg10007/LocalBot.git
Set-Location LocalBot
python -m venv .venv
.venv\Scripts\Activate.ps1
```

> If you get an execution policy error, run:
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
> ```

#### 2. Install dependencies

```powershell
# Core (Discord bot only)
pip install -e .

# + OpenWebUI HTTP server
pip install -e ".[webui]"

# + Dev tools (ruff, mypy, pytest)
pip install -e ".[dev]"
```

#### 3. Configure

```powershell
Copy-Item .env.example .env
notepad .env
```

Set at minimum:

| Variable | Required | Description |
|---|---|---|
| `DISCORD_BOT_TOKEN` | ✅ *(Discord mode)* | Bot token from the Developer Portal |
| `LLAMA_SERVER_MODEL_PATH` | ✅ | Absolute path to your `.gguf` model file |
| `LLAMA_SERVER_EXECUTABLE` | ✅ | Full path to `llama-server.exe` |
| `LLAMA_SERVER_N_GPU_LAYERS` | — | `0` = CPU (default), `-1` = all layers on GPU |
| `BRAVE_API_KEY` | — | Leave blank to disable web search |
| `LLAMA_SERVER_MODEL_FAMILY` | — | Leave blank for auto-detection |
| `MODEL_TEMPERATURE` | — | Default `0.3` |
| `BOT_OWNER_ID` | — | Your Discord user ID — grants full log access |
| `SANDBOX_ROOT` | — | Directory the LLM may read/write. Defaults to `./sandbox` |
| `WEBUI_API_KEY` | — | Bearer token for the HTTP API. Leave blank to disable auth |

#### 4. Create required directories

```powershell
New-Item -ItemType Directory -Force -Path logs, storage, sandbox
```

#### 5. Run

**Discord mode:**
```powershell
localbot
```

**HTTP / OpenWebUI mode:**
```powershell
localbot-webui
```

`llama-server` starts automatically as a subprocess. Its stdout/stderr (including crashes, OOM errors, CUDA failures) are captured and forwarded to the application log.

If a newer llama.cpp build is available you will be prompted:

```
Install llama.cpp b9222? [y/N] (auto-skip in 30s):
```

Set `LLAMA_UPDATE_AUTO=true` in `.env` to install updates automatically.

---

## OpenWebUI Interface

LocalBot exposes an OpenAI-compatible API that OpenWebUI connects to. The three model slots (`general`, `coding`, `reasoning`) appear as selectable models in the OpenWebUI picker.

### Docker Compose (recommended)

After `docker compose up -d`, OpenWebUI is available at `http://localhost:3000` — no manual connection setup needed.

### Standalone

```powershell
# Terminal 1 — start the bot and inference engine
localbot

# Terminal 2 — start the API bridge
localbot-webui
```

Then in OpenWebUI → **Settings → Connections → OpenAI API**:
- URL: `http://127.0.0.1:8080/v1`
- Key: value of `WEBUI_API_KEY` (or anything if auth is off)

### Endpoints

| Endpoint | Description |
|---|---|
| `GET /healthz` | 200 when model is ready, 503 during warm-up |
| `GET /v1/models` | Lists `localbot-general`, `localbot-coding`, `localbot-reasoning` |
| `POST /v1/chat/completions` | OpenAI-compatible; streaming supported; rate-limited |

---

## Commands

All commands are handled before agent routing and never consume LLM context.

| Command | Description |
|---|---|
| `jobs list` | Show your active scheduled jobs |
| `jobs cancel <id>` | Cancel a job by ID |
| `timezone set <IANA>` | Set your timezone (e.g. `America/New_York`) |
| `timezone show` | Show your saved timezone |
| `time now` | Show current time in your timezone |
| `model status` | Show active and configured model slots |
| `clear` | Clear your conversation history |
| `help` | Show all available commands |

---

## Filesystem Workspace

When `SANDBOX_ROOT` is set, the LLM has access to six filesystem tools:

| Tool | Description |
|---|---|
| `read_file` | Read a text file; large files truncated with notice |
| `write_file` | Write or overwrite a file; parent dirs created automatically |
| `list_directory` | List files and subdirectories with sizes |
| `apply_patch` | Apply a unified diff patch to an existing file |
| `search_in_files` | Grep across the workspace; supports glob filters |
| `get_current_dir` | Return the current path relative to `SANDBOX_ROOT` |

**Security:** All paths are resolved inside `SANDBOX_ROOT` before I/O. Absolute paths are re-rooted; `../` traversal raises `PermissionError`. Binary extensions are blocked on both read and write.

---

## Model Routing

Every request is classified by `intent.py` before a model slot is acquired:

| Slot | Triggered when |
|---|---|
| `general` | Default — conversational, search, scheduling, diagnostics |
| `coding` | Message matches coding intent (write, implement, fix, refactor…) |
| `reasoning` | Message matches reasoning intent (design, compare, analyse, explain…) |

When a coding request also requires external lookup (docs, API references, search), a **two-phase dispatch** fires: the general model fetches context, then the coding model implements using that context.

---

## Swapping Models

Update `SLOT_0_MODEL` (or the relevant slot) in `.env` and restart:

```powershell
docker compose up -d --force-recreate localbot
```

LocalBot queries `/v1/models` on startup, reads the loaded filename, and automatically applies the correct stop tokens and think-stripping for the detected family.

| Family | Matched by filename | Think-strip |
|---|---|---|
| `GEMMA` | `gemma`, `glm` | ✅ |
| `LLAMA` | `llama` | ❌ |
| `MISTRAL` | `mistral`, `mixtral` | ❌ |
| `QWEN` | `qwen` | ✅ |
| `DEEPSEEK` | `deepseek` | ✅ |
| `PHI` | `phi` | ❌ |

Override detection if the filename is ambiguous:
```env
LLAMA_SERVER_MODEL_FAMILY=gemma
```

---

## Self-Diagnostics

Ask conversationally:

> *"Why did my 8am reminder not fire?"*
> *"Check the logs for errors"*
> *"What did you search for last time?"*

The LLM calls `read_logs`, receives a filtered JSON slice of the audit log, and explains what it finds. Each user sees only their own entries by default. Set `BOT_OWNER_ID` in `.env` for full cross-user log access.

---

## Scheduled Jobs

Just ask naturally:

> *"Remind me every morning at 8am to review my task list"*
> *"Send me tech news every weekday at 6pm"*

The LLM translates to cron and calls `schedule_job`. The bot only confirms a job once `schedule_job` has returned successfully and always relays the real job ID — it never invents one. Jobs are validated against legal cron field ranges and per-user limits before registration.

---

## Security Notes

- Designed for personal/trusted-user use. All messages are stored in SQLite and logged to an append-only JSONL audit file.
- Per-user rate limiting on both Discord and the HTTP API (HTTP 429).
- Input capped at `MAX_INPUT_LENGTH` (default 1000 chars) before hitting the LLM.
- Tool results capped at 4000 chars before context injection.
- Scheduler job counts enforced with a single atomic DB operation to prevent TOCTOU races.
- `read_logs` scoped to the requesting user's ID — the LLM cannot bypass this.
- Filesystem tools confined to `SANDBOX_ROOT`; traversal blocked at OS level.
- HTTP API requires a Bearer token when `WEBUI_API_KEY` is set; each token maps to an isolated user namespace.
- Config validated at startup (pydantic-settings) — invalid values fail fast with clear messages.
- Storage paths validated against the project root at startup to prevent traversal via misconfigured env vars.

---

## Development

```powershell
# Activate the virtual environment
.venv\Scripts\Activate.ps1

# Lint + format
ruff check .
ruff format .

# Type check
mypy src/

# Tests
pytest
```

The test suite uses `conftest.py` to stub `localbot.config` and optional heavy dependencies before collection, so `pytest` works without a running bot or llama-server.

---

## Project Layout

```
src/localbot/
├── app.py                   # Discord client, rate limiter, on_message handler
├── commands.py              # Registered command handler table (jobs, timezone, clear, help…)
├── agent.py                 # Core request/tool loop; slot acquisition; two-phase dispatch
├── intent.py                # Intent classification (slot selection, workspace mode, needs_tools)
├── prompts.py               # System prompts for each model slot
├── webui.py                 # FastAPI OpenAI-compatible API layer for OpenWebUI
├── config.py                # pydantic-settings config; validated at import time
├── messaging.py             # Discord 2000-char message splitter
├── adapters/
│   ├── llamacpp_server.py       # llama-server subprocess lifecycle + log pipe
│   ├── llamacpp_client.py       # HTTP client; model family detection; think-strip
│   ├── model_registry.py        # Multi-slot manager; idle unload; hot-swap
│   ├── llamacpp_updater.py      # Startup update check (GitHub Releases API)
│   └── llamacpp_downloader.py   # Asset selection, streaming download, zip extraction
├── tools/
│   ├── registry.py          # Tool schemas + async dispatcher (timeout-guarded)
│   ├── filesystem.py        # read/write/list/patch/search — sandboxed to SANDBOX_ROOT
│   ├── log_reader.py        # read_logs — audit log reader for self-diagnostics
│   ├── scheduler_tools.py   # LLM-callable schedule_job / cancel_job / list_jobs
│   ├── search.py            # Brave Search + page fetch & summarise
│   ├── reddit.py            # Reddit JSON API (no auth)
│   └── time_tools.py        # Current time / timezone helpers
├── scheduler/
│   ├── service.py           # APScheduler wrapper; cron validation; atomic job-limit check
│   └── store.py             # SQLite job persistence
└── storage/
    ├── db.py                # Schema initialisation
    ├── history.py           # Per-user conversation history (SQLite, WAL, atomic trim)
    └── audit.py             # Append-only JSONL audit log (thread-safe)
tests/
├── conftest.py
├── test_agent_needs_tools.py
├── test_llamacpp_family_detection.py
├── test_messaging.py
├── test_routing_dispatch_filesystem.py
├── test_scheduler_validate_cron.py
└── test_search_should_skip.py
Dockerfile
docker-compose.yml
```

---

## Environment Variables

See [`.env.example`](.env.example) for all options with inline descriptions.
