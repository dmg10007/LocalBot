# LocalBot

A privacy-first, self-hosted AI assistant that runs on your own machine. Inference for sensitive work stays local via [`llama-server`](https://github.com/ggerganov/llama.cpp); general conversation and reasoning are routed to **Groq's cloud LPU** for near-instant responses (~300–840 tok/s, sub-100 ms TTFT). Both paths are fully transparent — the bot never tells you which path fired.

LocalBot surfaces through two interfaces simultaneously: **Discord DMs** and a **browser UI** via [OpenWebUI](https://github.com/open-webui/open-webui).

---

## How It Works

Every message is classified by `intent.py` before a model is touched:

| Path | When it fires | Model | TTFT |
|---|---|---|---|
| **Groq fast path** | General chat, search synthesis, reasoning, coding analysis — anything that doesn't touch private data | `llama-3.1-8b-instant` (or `llama-3.3-70b-versatile` for heavy reasoning) | < 100 ms |
| **Local path** | Scheduler operations, filesystem/sandbox work, GitHub repo actions, log diagnostics | Your local GGUF model via llama-server | 1–3 s (CPU) |
| **Two-phase dispatch** | Coding + external lookup (e.g. "implement X using this API docs") | Phase 1: Groq gathers context → Phase 2: local coding model implements | Combined |

Groq is never given: your conversation history with scheduler context, file contents, audit logs, or anything workspace-related. The privacy boundary is enforced in `intent.is_groq_eligible()` and is not user-configurable.

---

## Features

- **Conversational chat** with per-user SQLite-backed message history
- **Groq fast path** — sub-100 ms TTFT on the vast majority of queries; 40–80× faster than local CPU inference
- **Deep web search** via Brave Search API — fetches and summarises actual page content, not just snippets
- **Reddit search** — unauthenticated JSON API, no credentials needed
- **Sandboxed filesystem** — the LLM can read, write, patch, and grep files inside a locked-down directory; `../` traversal is blocked at the OS level
- **Scheduled prompts** — describe jobs in natural language; the LLM converts to cron and calls `schedule_job` directly
- **Self-diagnostics** — the LLM can call `read_logs` to reason over its own audit log conversationally
- **Multi-slot model routing** — routes each request to a `general`, `coding`, or `reasoning` model slot based on intent
- **Two-phase coding dispatch** — when a coding request also needs external lookup, context is gathered first, then the coding model implements
- **Speculative decoding** — optional draft model support via llama-server; adds ~1.5–2× effective tok/s on local path
- **Thinking model support** — auto-strips `<think>` blocks for Qwen3, DeepSeek, Gemma
- **Auto-updater** — checks for newer llama.cpp builds on startup, prompts to install
- **Self-healing** — detects `llama-server` crashes and restarts automatically
- **OpenWebUI / OpenAI-compatible API** — `localbot-webui` starts a FastAPI server; stream or non-stream, Bearer token auth, Docker Compose stack included
- **Rate limiting** — per-user cooldown on both Discord and the HTTP API (HTTP 429)
- **Minimal footprint** — discord.py + aiohttp + APScheduler + BeautifulSoup4 + pydantic-settings

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

### 3. llama-server (llama.cpp)

Download a pre-built `llama-server` binary from the [llama.cpp releases page](https://github.com/ggerganov/llama.cpp/releases/latest).

| Platform | File to download |
|---|---|
| Windows, CPU only | `llama-bXXXX-bin-win-cpu-x64.zip` |
| Windows, NVIDIA GPU | `llama-bXXXX-bin-win-cuda-cu12.x-x64.zip` |

Extract to a permanent location:

```powershell
Expand-Archive -Path "$env:USERPROFILE\Downloads\llama-bXXXX-bin-win-cpu-x64.zip" -DestinationPath "C:\Llama"
```

Verify the binary:

```powershell
C:\Llama\llama-server.exe --version
```

You don't need to add it to `PATH` — point LocalBot at it directly via `LLAMA_SERVER_EXECUTABLE` in `.env`.

### 4. GGUF model files

Create the models directory that Docker will mount:

```powershell
New-Item -ItemType Directory -Force -Path "C:\Llama\Models"
```

#### Recommended model stack (i5-10400H, 16 GB DDR4, no discrete GPU)

Because most queries are handled by Groq, **the local models only need to cover private/sensitive operations** (filesystem, scheduler, diagnostics). Smaller, faster models are the right choice — quality comes from Groq; speed comes from small local models.

| Slot | Purpose | Recommended model | Quant | ~RAM | Local tok/s |
|---|---|---|---|---|---|
| `general` | Scheduler, diagnostics, fallback chat | [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B-GGUF) | Q4_K_M | ~3.2 GB | 8–12 |
| `coding` | Filesystem edits, GitHub commits | [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B-GGUF) *(same file)* | Q4_K_M | ~3.2 GB | 8–12 |
| `reasoning` | Deep local analysis | [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B-GGUF) *(same file)* | Q4_K_M | ~3.2 GB | 8–12 |
| `draft` *(optional)* | Speculative decoding | [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B-GGUF) | Q4_K_M | ~0.5 GB | +20–30% |

> **Why Qwen3-4B for all local slots?** On the local path (scheduler, filesystem, logs), the task is structured and narrow — not open-ended reasoning. Qwen3-4B Q4_K_M hits ~94% tool-call accuracy, fits in 3.2 GB, and runs at 8–12 tok/s on your CPU. All three slots can share the same `.gguf` file with zero RAM duplication — only the system prompt changes between slots.

> **Why not run a bigger model locally?** Groq already handles all the heavy lifting at 300–840 tok/s. A local 8B model would be slower, use more RAM, and only serve the narrow private-data path. The budget is better spent keeping the local model small and snappy.

> **Why not use the Intel UHD iGPU?** The UHD Graphics (GT2) shares system DDR4 memory. Offloading layers via Vulkan uses the same bus as the CPU and adds driver overhead, resulting in slower tok/s. Keep `LLAMA_ARG_N_GPU_LAYERS=0`.

Download the recommended model:

```powershell
# Install huggingface-hub if you don't have it
pip install huggingface-hub

# Download the main model (~3.2 GB)
huggingface-cli download Qwen/Qwen3-4B-GGUF Qwen3-4B-Q4_K_M.gguf --local-dir "C:\Llama\Models"

# Optional: draft model for speculative decoding (~0.5 GB, +20-30% tok/s on long outputs)
huggingface-cli download Qwen/Qwen3-0.6B-GGUF Qwen3-0.6B-Q4_K_M.gguf --local-dir "C:\Llama\Models"
```

#### Higher-RAM options (24–48 GB)

| Slot | Model | Quant | RAM |
|---|---|---|---|
| `general` | Qwen3-8B | Q4_K_M | ~5.5 GB |
| `coding` | Qwen3-8B | Q4_K_M | ~5.5 GB |
| `reasoning` | Qwen3-14B | Q4_K_M | ~10 GB |

With more RAM, the Groq fast path still handles most queries — bigger local models just improve the fallback and private-data experience.

### 5. API Keys

#### Groq (strongly recommended — this is what makes the bot feel fast)

1. Sign up at [console.groq.com](https://console.groq.com)
2. Go to **API Keys** → **Create API Key**
3. Copy the key — it starts with `gsk_`

Free tier limits: 30 req/min, 14,400 req/day on `llama-3.1-8b-instant`.

#### Brave Search (optional — needed for web search)

1. Sign up at [brave.com/search/api](https://brave.com/search/api/)
2. Choose the **Free** tier (2,000 queries/month)
3. Copy the API key from the dashboard

### 6. A Discord Bot Token

1. Go to the [Discord Developer Portal](https://discord.com/developers/applications) and create a new application.
2. Under **Bot**, click **Add Bot** and copy the token.
3. Enable the **Message Content Intent** under **Privileged Gateway Intents**.
4. Under **OAuth2 → URL Generator**, select the `bot` scope and the following permissions:
   - `Send Messages`
   - `Read Message History`
   - `Read Messages/View Channels`
5. Copy the generated URL and open it in a browser to invite the bot to your server.

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

Fill in the following values:

| Variable | Required | Description |
|---|---|---|
| `DISCORD_BOT_TOKEN` | ✅ | Bot token from the Discord Developer Portal |
| `WEBUI_API_KEY` | ✅ | Shared secret for the OpenWebUI bridge. Generate with the command below |
| `SLOT_GENERAL_MODEL` | ✅ | Container path to your `.gguf` file, e.g. `/models/Qwen3-4B-Q4_K_M.gguf` |
| `GROQ_API_KEY` | Strongly recommended | Groq API key — enables the fast path that makes the bot feel like a cloud model |
| `GITHUB_TOKEN` | — | GitHub PAT (`repo`, `read:org`) for GitHub tool access |
| `BRAVE_API_KEY` | — | Brave Search API key for web search |
| `LLAMA_CTX_SIZE` | — | Context window. Default: `2048`. Raise to `4096` only if needed |
| `SLOT_DRAFT_MODEL` | — | Path to draft model for speculative decoding, e.g. `/models/Qwen3-0.6B-Q4_K_M.gguf` |

See [`.env.example`](.env.example) for all options with inline comments.

Generate a secure `WEBUI_API_KEY`:

```powershell
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

A minimal working `.env` looks like this:

```env
DISCORD_BOT_TOKEN=your_discord_token
WEBUI_API_KEY=your_generated_key
SLOT_GENERAL_MODEL=/models/Qwen3-4B-Q4_K_M.gguf
GROQ_API_KEY=gsk_your_groq_key
```

#### 3. Create required local directories

```powershell
New-Item -ItemType Directory -Force -Path logs, storage, sandbox
```

#### 4. Start all services

```powershell
docker compose up -d
```

This starts four containers:

| Container | What it does | Port |
|---|---|---|
| `localbot` | Discord bot + llama-server subprocess | — |
| `localbot-webui` | OpenAI-compatible FastAPI bridge | `8090` |
| `github-mcp-server` | GitHub MCP server | `8181` |
| `openwebui` | Browser UI | `3000` |

#### 5. Verify everything is running

```powershell
# Check all containers are healthy
docker compose ps

# Stream logs from the bot container
docker compose logs -f localbot

# Confirm the API bridge is responding
Invoke-WebRequest -Uri http://localhost:8090/healthz | Select-Object -ExpandProperty Content

# Open OpenWebUI in the browser
Start-Process "http://localhost:3000"
```

The bot is ready when you see `llama-server is ready` in the `localbot` logs.

#### 6. First-time OpenWebUI setup

1. Open `http://localhost:3000` in your browser
2. Create an admin account (local only — no external auth)
3. Navigate to **Settings → Connections → OpenAI API**
   - URL: `http://localbot-webui:8090/v1` (inside Docker) or `http://127.0.0.1:8090/v1` (from host)
   - Key: value of `WEBUI_API_KEY` from your `.env`
4. Click **Save** — the three model slots should appear in the model picker

#### 7. Common operations

```powershell
# Stop all services
docker compose down

# Restart just the bot (e.g. after a .env change)
docker compose up -d --force-recreate localbot

# Rebuild the bot image (e.g. after a code change)
docker compose up -d --build localbot

# View logs for a specific service
docker compose logs -f openwebui
docker compose logs -f github-mcp-server

# Pull the latest OpenWebUI image
docker compose pull openwebui
docker compose up -d openwebui
```

---

### Option B — Local Python (no Docker)

```powershell
git clone https://github.com/dmg10007/LocalBot.git
Set-Location LocalBot

# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -e .            # core bot
pip install -e ".[webui]"   # + OpenWebUI HTTP server (optional)
pip install -e ".[dev]"     # + linters and test tools (optional)

# Configure
Copy-Item .env.example .env
notepad .env

# Create required directories
New-Item -ItemType Directory -Force -Path logs, storage, sandbox

# Run
localbot          # Discord bot mode
localbot-webui    # HTTP / OpenWebUI API bridge mode
```

For the non-Docker path, `LLAMA_SERVER_EXECUTABLE` in `.env` must point to your local `llama-server.exe`:

```env
LLAMA_SERVER_EXECUTABLE=C:\Llama\llama-server.exe
SLOT_GENERAL_MODEL=C:\Llama\Models\Qwen3-4B-Q4_K_M.gguf
```

---

## Performance Tuning

### Groq Fast Path

The Groq fast path is the single highest-leverage performance improvement. When `GROQ_API_KEY` is set, LocalBot routes the majority of queries to Groq's LPU hardware for **sub-100 ms TTFT** and **300–840 tok/s** — roughly 40–80× faster than local CPU inference.

**Routing policy — Groq receives:**
- General conversation and Q&A
- Web search queries (search is executed locally, synthesis is on Groq)
- Reasoning and analysis that doesn't involve private files
- Coding queries that don't require file access

**Routing policy — Local model always handles:**
- Scheduler creation, cancellation, and listing (contains user schedule data)
- Filesystem / sandbox reads and writes (private files)
- GitHub repository operations (private code)
- Audit log diagnostics (private log data)

```env
# .env
GROQ_API_KEY=gsk_your_key_here
```

If Groq returns an error or is unavailable, LocalBot automatically falls back to the local model for that request with no user-visible disruption.

### llama.cpp Flags (i5-10400H reference)

The defaults in `docker-compose.yml` are already tuned for the i5-10400H. These are the key settings:

| Setting | Value | Rationale |
|---|---|---|
| `LLAMA_ARG_THREADS` | `4` | Physical cores only — token generation is memory-bandwidth-bound; HT adds contention |
| `LLAMA_ARG_THREADS_BATCH` | `8` | All logical threads — prompt eval is more compute-bound; HT helps here |
| `LLAMA_ARG_CTX_SIZE` | `2048` | Halves KV cache vs 4096; raise only if you need longer context |
| `LLAMA_ARG_N_GPU_LAYERS` | `0` | **Do not change.** Intel UHD iGPU shares DDR4 bandwidth; Vulkan offloading is slower |
| `LLAMA_ARG_MMAP` | `true` | Memory-map model file; fast cold start, lower RSS |

Extra args (set via `LLAMA_SERVER_EXTRA_ARGS` in `.env`):

| Flag | Effect |
|---|---|
| `--flash-attn` | O(n) attention memory; cuts TTFT ~15–25% as history accumulates |
| `--cache-type-k q8_0` | 8-bit KV key cache; frees ~200–500 MB with negligible quality delta |
| `--cache-type-v q8_0` | 8-bit KV value cache |
| `--ubatch-size 512` | Micro-batch for parallel prompt evaluation |

### Speculative Decoding (optional)

Speculative decoding uses a small draft model to generate candidate tokens that the main model verifies in a single pass. On long local outputs (≥1200 tokens) at low temperature, this delivers **~20–30% more effective tok/s** at ~0.5 GB additional RAM.

> **When it helps:** Long code generation, long log analysis, scheduler confirmations with lots of text. Short conversational replies see little or no benefit.

```env
# .env — add Qwen3-0.6B as the draft model
SLOT_DRAFT_MODEL=/models/Qwen3-0.6B-Q4_K_M.gguf
LLAMA_SERVER_EXTRA_ARGS=--flash-attn --cache-type-k q8_0 --cache-type-v q8_0 --ubatch-size 512 --model-draft /models/Qwen3-0.6B-Q4_K_M.gguf --draft-max 5 --draft-min 1
```

RAM budget with speculative decoding enabled (16 GB system):

| Active load | Total model RAM |
|---|---|
| Qwen3-4B + Qwen3-0.6B draft | ~3.7 GB |
| Everything else (Windows, Docker, containers) | ~6–7 GB |
| **Total** | **~10–11 GB** |

### Inference Temperature

For local-only queries where accuracy matters most, lower temperature reduces hallucinations:

```env
# .env
MODEL_TEMPERATURE=0.2   # default is 0.3; lower = more deterministic
```

---

## OpenWebUI Interface

LocalBot exposes an OpenAI-compatible API that OpenWebUI connects to. The three model slots (`localbot-general`, `localbot-coding`, `localbot-reasoning`) appear as selectable models in the OpenWebUI model picker.

After `docker compose up -d`, OpenWebUI is available at **http://localhost:3000**.

### Standalone (without Docker)

```powershell
# Terminal 1 — bot + inference engine
localbot

# Terminal 2 — API bridge
localbot-webui
```

Then in OpenWebUI → **Settings → Connections → OpenAI API**:
- URL: `http://127.0.0.1:8090/v1`
- Key: value of `WEBUI_API_KEY`

### API Endpoints

| Endpoint | Description |
|---|---|
| `GET /healthz` | `200` when model is ready, `503` during warm-up |
| `GET /v1/models` | Lists `localbot-general`, `localbot-coding`, `localbot-reasoning` |
| `POST /v1/chat/completions` | OpenAI-compatible; streaming supported; rate-limited |

Test the API from PowerShell:

```powershell
$headers = @{ Authorization = "Bearer $env:WEBUI_API_KEY"; "Content-Type" = "application/json" }
$body = '{"model":"localbot-general","messages":[{"role":"user","content":"Hello"}]}'
Invoke-RestMethod -Uri http://localhost:8090/v1/chat/completions -Method Post -Headers $headers -Body $body
```

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

## Model Routing

Every request is classified by `intent.py` before a model slot is acquired:

| Slot | Triggered when | Default model |
|---|---|---|
| `general` | Default — fallback for anything not matched below | Qwen3-4B (local) or Groq fast path |
| `coding` | Message matches coding intent (write, implement, fix, refactor, commit…) | Qwen3-4B (local) or Groq fast path |
| `reasoning` | Message matches reasoning intent (design, compare, analyse, explain…) | Qwen3-4B (local) or Groq fast path |

The Groq fast path fires **before** slot acquisition when the query is eligible. All three slots share the same Groq models (`llama-3.1-8b-instant` for speed, `llama-3.3-70b-versatile` configurable for heavy reasoning).

### Two-Phase Dispatch

When a coding request also requires external lookup (API docs, search results), a two-phase dispatch fires:
1. **Phase 1** — general model (Groq fast path if eligible) fetches context via tools
2. **Phase 2** — coding model implements using the gathered context

---

## Swapping Models

Update the relevant `SLOT_*_MODEL` in `.env` and restart:

```powershell
# Edit .env
notepad .env

# Apply the change
docker compose up -d --force-recreate localbot
```

LocalBot auto-detects the model family from the filename and applies correct stop tokens and think-stripping:

| Family | Detected by filename | Think-strip |
|---|---|---|
| `GEMMA` | `gemma`, `glm` | ✅ |
| `LLAMA` | `llama` | ❌ |
| `MISTRAL` | `mistral`, `mixtral` | ❌ |
| `QWEN` | `qwen` | ✅ |
| `DEEPSEEK` | `deepseek` | ✅ |
| `PHI` | `phi` | ❌ |

Override detection if the filename is ambiguous:

```env
LLAMA_SERVER_MODEL_FAMILY=qwen
```

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

> Filesystem queries are never routed to Groq — all file operations run entirely on-device.

---

## Scheduled Jobs

Just ask naturally:

> *"Remind me every morning at 8am to review my task list"*
> *"Send me tech news every weekday at 6pm"*

The LLM translates to cron and calls `schedule_job`. The bot only confirms a job once `schedule_job` has returned successfully and always relays the real job ID — it never invents one.

> Scheduler queries are never routed to Groq — they run entirely on-device.

---

## Self-Diagnostics

Ask conversationally:

> *"Why did my 8am reminder not fire?"*
> *"Check the logs for errors"*
> *"What did you search for last time?"*

The LLM calls `read_logs`, receives a filtered JSON slice of the audit log, and explains what it finds. Each user sees only their own entries by default. Set `BOT_OWNER_ID` in `.env` for full cross-user log access.

> Diagnostic queries are never routed to Groq — audit log contents stay on-device.

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
- **Groq routing policy:** filesystem, scheduler, and diagnostic queries are never routed off-device. The eligibility check in `intent.is_groq_eligible()` is the sole enforcement point and cannot be bypassed via user input.

---

## Development

```powershell
# Activate the virtual environment
.venv\Scripts\Activate.ps1

# Lint and format
ruff check .
ruff format .

# Type check
mypy src/

# Run tests
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
