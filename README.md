# LocalBot

A privacy-first, self-hosted AI assistant that runs a local LLM entirely on your own machine — no cloud API calls for inference by default. All conversation history, scheduled jobs, the audit log, and the model weights stay local.

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
- **Groq fast path** — when `GROQ_API_KEY` is set, non-sensitive queries route to Groq for sub-100 ms TTFT (~300–600 tok/s), falling back to local on error
- **Speculative decoding** — optional draft model support via llama-server; adds ~1.5–2× effective tok/s at ~0.5 GB RAM cost
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

#### Hardware-optimised model stack (i5-10400H, 16 GB DDR4, no discrete GPU)

CPU-only inference on this hardware is **memory-bandwidth-bound** at ~35–38 GB/s sustained. Smaller, well-quantised models deliver dramatically better tok/s than larger ones with marginal quality loss for chat tasks.

| Slot | Recommended model | Quant | RAM | Realistic tok/s |
|---|---|---|---|---|
| `general` | [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF) | Q4_K_M | ~1.9 GB | 12–20 |
| `coding` | [Qwen3-Coder-4B-A1.3B](https://huggingface.co/Qwen/Qwen3-Coder-4B-A1.3B-GGUF) (MoE) | Q4_K_M | ~2.5 GB | 14–22 eff. |
| `reasoning` | [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF) | Q5_K_M | ~5.5 GB | 5–8 |
| `draft` *(optional)* | [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF) | Q4_K_M | ~0.4 GB | +1.5–2× |

> **Why MoE for coding?** The Qwen3-Coder-4B-A1.3B model has 4B total parameters but only activates 1.3B per token, giving near-4B quality at the memory bandwidth cost of a ~1.3B dense model.

> **Why not use the Intel UHD iGPU?** The UHD Graphics (GT2) shares system DDR4 memory. Offloading layers via Vulkan uses the same bus as the CPU and adds driver overhead, resulting in slower tok/s on this iGPU class. Keep `LLAMA_ARG_N_GPU_LAYERS=0`.

> **Speculative decoding** adds ~0.4 GB for the draft model and delivers ~1.5–2× effective tok/s at 70–80% acceptance rates. Enable it by setting `SLOT_DRAFT_MODEL` and adding `--model-draft` to `LLAMA_SERVER_EXTRA_ARGS`. See [Speculative Decoding](#speculative-decoding) below.

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

#### 1. Clone the repo

```powershell
git clone https://github.com/dmg10007/LocalBot.git
Set-Location LocalBot
```

#### 2. Create your `.env` file

```powershell
Copy-Item .env.example .env
notepad .env
```

| Variable | Required | Description |
|---|---|---|
| `DISCORD_BOT_TOKEN` | ✅ *(Discord mode)* | Bot token from the Developer Portal |
| `GITHUB_TOKEN` | — | GitHub PAT for the MCP server (`repo`, `read:org`) |
| `WEBUI_API_KEY` | ✅ *(if exposed off-loopback)* | Shared secret between OpenWebUI and the API bridge |
| `SLOT_GENERAL_MODEL` | ✅ | Container path to your `.gguf` file (e.g. `/models/qwen2.5-3b-q4_k_m.gguf`) |
| `GROQ_API_KEY` | — | Groq API key for fast cloud inference fallback |
| `BRAVE_API_KEY` | — | Brave Search API key for web search |
| `LLAMA_CTX_SIZE` | — | Context window size. Default `2048` |
| `LLAMA_SERVER_EXTRA_ARGS` | — | Extra flags for llama-server; see Performance Tuning below |

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
- **`localbot`** — Discord bot + llama-server subprocess
- **`localbot-webui`** — OpenAI-compatible FastAPI bridge on `http://localhost:8090`
- **`github-mcp-server`** — GitHub MCP server on `http://localhost:8181`
- **`openwebui`** — Browser UI on `http://localhost:3000`

#### 5. Check service health

```powershell
docker compose ps
docker compose logs -f localbot
```

#### 6. Stopping and restarting

```powershell
docker compose down
docker compose up -d --force-recreate localbot
docker compose up -d --build localbot
```

---

### Option B — Local Python (no Docker)

```powershell
git clone https://github.com/dmg10007/LocalBot.git
Set-Location LocalBot
python -m venv .venv
.venv\Scripts\Activate.ps1

pip install -e .           # core
pip install -e ".[webui]"  # + OpenWebUI HTTP server
pip install -e ".[dev]"    # + dev tools

Copy-Item .env.example .env
notepad .env
New-Item -ItemType Directory -Force -Path logs, storage, sandbox

localbot          # Discord mode
localbot-webui    # HTTP / OpenWebUI mode
```

---

## Performance Tuning

### llama.cpp Flags (i5-10400H reference)

The defaults in `docker-compose.yml` are already tuned for the i5-10400H. These are the key settings and why they are set the way they are:

| Setting | Value | Rationale |
|---|---|---|
| `LLAMA_ARG_THREADS` | `4` | Physical cores only — token generation is memory-bandwidth-bound; HT adds contention |
| `LLAMA_ARG_THREADS_BATCH` | `8` | All logical threads — prompt eval is more compute-bound; HT helps here |
| `LLAMA_ARG_CTX_SIZE` | `2048` | Halves KV cache vs 4096; raise to `4096` in `.env` only if you need longer context |
| `LLAMA_ARG_N_GPU_LAYERS` | `0` | **Do not change.** Intel UHD iGPU shares DDR4 bandwidth; Vulkan offloading is slower |
| `LLAMA_ARG_MMAP` | `true` | Memory-map model file; fast cold start, lower RSS |

The following extra args are set via `LLAMA_SERVER_EXTRA_ARGS` in `.env`:

| Flag | Effect |
|---|---|
| `--flash-attn on` | O(n) attention memory vs O(n²); cuts TTFT ~15–25% as history accumulates. Note: recent llama.cpp builds (b9xxx+) require an explicit value — `on`/`off`/`auto` — bare `--flash-attn` will fail to parse and crash the server on startup |
| `--cache-type-k q8_0` | 8-bit KV key cache; frees ~200–500 MB with negligible quality delta |
| `--cache-type-v q8_0` | 8-bit KV value cache |
| `--ubatch-size 512` | Micro-batch size for parallel prompt evaluation |

### Speculative Decoding

Speculative decoding uses a small, fast draft model to generate candidate tokens that the main model verifies in a single forward pass. At 70–80% acceptance rates this delivers **~1.5–2× effective tok/s** with ~0.4 GB additional RAM.

```env
# .env
SLOT_DRAFT_MODEL=/models/qwen2.5-0.5b-instruct-q4_k_m.gguf
LLAMA_SERVER_EXTRA_ARGS=--flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --ubatch-size 512 --model-draft /models/qwen2.5-0.5b-instruct-q4_k_m.gguf --draft-max 5 --draft-min 1
```

RAM budget with speculative decoding on 16 GB:

| Active slots | Total RAM |
|---|---|
| General 3B + 0.5B draft | ~2.3 GB |
| Coding 4B MoE + 0.5B draft | ~3.0 GB |
| Reasoning 7B + 0.5B draft | ~6.2 GB |

### Groq Fast Path

When `GROQ_API_KEY` is set, LocalBot can route eligible queries to Groq's LPU for **sub-100 ms TTFT** and **~300–600 tok/s** — roughly 40–80× faster than local CPU inference.

**Privacy policy:** only non-sensitive queries are ever routed to Groq. The routing logic in `intent.py` blocks any query that involves:
- Filesystem or GitHub workspace operations
- Scheduler creation/cancellation
- Diagnostic log access

Everything else — general chat, search synthesis, reasoning without personal context — is eligible.

```env
# .env
GROQ_API_KEY=gsk_your_key_here
```

Get a free key at [console.groq.com/keys](https://console.groq.com/keys). The free tier allows 30 req/min and 6,000 req/day on `llama-3.1-8b-instant`.

If Groq is unavailable or returns an error, LocalBot automatically falls back to the local model for that request.

---

## OpenWebUI Interface

LocalBot exposes an OpenAI-compatible API that OpenWebUI connects to. The three model slots (`general`, `coding`, `reasoning`) appear as selectable models in the OpenWebUI picker.

### Docker Compose (recommended)

After `docker compose up -d`, OpenWebUI is available at `http://localhost:3000`.

### Standalone

```powershell
localbot          # Terminal 1 — bot + inference engine
localbot-webui    # Terminal 2 — API bridge
```

Then in OpenWebUI → **Settings → Connections → OpenAI API**:
- URL: `http://127.0.0.1:8090/v1`
- Key: value of `WEBUI_API_KEY`

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

**Security:** All paths are resolved inside `SANDBOX_ROOT` before I/O. Absolute paths are re-rooted; `../` traversal raises `PermissionError`. Symlinks whose targets escape the sandbox are blocked. Binary extensions are blocked on both read and write.

---

## Model Routing

Every request is classified by `intent.py` before a model slot is acquired:

| Slot | Triggered when |
|---|---|
| `general` | Default — conversational, search, scheduling, diagnostics |
| `coding` | Message matches coding intent (write, implement, fix, refactor…) |
| `reasoning` | Message matches reasoning intent (design, compare, analyse, explain…) |

When a coding request also requires external lookup (docs, API references, search), a **two-phase dispatch** fires: the general model fetches context, then the coding model implements using that context.

When `GROQ_API_KEY` is configured and the query is non-sensitive and requires no tool calls, the **Groq fast path** fires before local model acquisition. On error it falls back to local transparently.

---

## Swapping Models

Update `SLOT_GENERAL_MODEL` (or the relevant slot) in `.env` and restart:

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
- Filesystem tools confined to `SANDBOX_ROOT`; traversal and symlink escapes blocked at OS level.
- HTTP API requires a Bearer token when `WEBUI_API_KEY` is set; each token maps to an isolated user namespace.
- Config validated at startup (pydantic-settings) — invalid values fail fast with clear messages.
- Storage paths validated against the project root at startup to prevent traversal via misconfigured env vars.
- **Groq routing policy:** filesystem, scheduler, and diagnostic queries are never routed off-device. The eligibility check in `intent.is_groq_eligible()` is the enforcement point.

---

## Development

```powershell
.venv\Scripts\Activate.ps1

ruff check .
ruff format .
mypy src/
pytest
```

The test suite uses `conftest.py` to stub `localbot.config` and optional heavy dependencies before collection, so `pytest` works without a running bot or llama-server.

---

## Project Layout

```
src/localbot/
├── app.py                   # Discord client, rate limiter, on_message handler
├── commands.py              # Registered command handler table (jobs, timezone, clear, help…)
├── agent.py                 # Core request/tool loop; Groq fast path; two-phase dispatch
├── intent.py                # Intent classification (slot selection, workspace mode, Groq eligibility)
├── prompts.py               # System prompts for each model slot
├── webui.py                 # FastAPI OpenAI-compatible API layer for OpenWebUI
├── config.py                # pydantic-settings config; validated at import time
├── messaging.py             # Discord 2000-char message splitter
├── adapters/
│   ├── llamacpp_server.py       # llama-server subprocess lifecycle + speculative decoding args
│   ├── llamacpp_client.py       # HTTP client; model family detection; think-strip
│   ├── groq_client.py           # Groq LPU fast-path client (optional)
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
