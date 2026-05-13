# LocalBot

A lightweight Discord DM bot that runs a local LLM via [llama.cpp](https://github.com/ggerganov/llama.cpp)'s built-in `llama-server`. The Python process is intentionally thin — it never loads the model itself; all inference goes through `llama-server`'s OpenAI-compatible HTTP API.

## Features

- **Conversational chat** with per-user message history (SQLite-backed)
- **Deep web search** via Brave Search API — fetches and summarises actual page content, not just index snippets
- **Reddit search** — searches Reddit posts and discussions via the unauthenticated JSON API
- **Scheduled prompts** — users define recurring jobs via natural language; the LLM converts schedules to cron expressions and calls `schedule_job` directly; commands (`jobs list`, `jobs cancel <id>`) are also available
- **Model-agnostic inference** — auto-detects model family once on startup; swap models by changing one line in `.env`
- **Thinking model support** — automatically strips `<think>` blocks for reasoning models (Gemma, DeepSeek, Qwen)
- **Rate limiting** — per-user cooldown with bounded memory (stale entries are automatically evicted)
- **Self-healing** — detects llama-server crashes and restarts automatically
- **Audit log** — append-only JSONL log of all user messages and bot replies (timeouts are logged separately from genuine replies)
- **llama-server log capture** — subprocess stdout is piped into the application logger so crashes (OOM, CUDA errors) surface immediately
- **Minimal footprint** — discord.py + aiohttp + APScheduler + BeautifulSoup4; no heavy ML dependencies

---

## Prerequisites

### 1. Python 3.11+

```bash
python --version   # must be 3.11 or higher
```

Download from [python.org](https://www.python.org/downloads/) if needed.

### 2. llama-server (llama.cpp)

LocalBot delegates all inference to `llama-server`. The easiest way to get it on any platform is to download a pre-built binary from the official releases.

**Step 1 — Download a pre-built binary:**

1. Go to the [llama.cpp releases page](https://github.com/ggerganov/llama.cpp/releases/latest)
2. Download the zip that matches your platform:

| Platform | File to download |
|---|---|
| Windows, CPU only | `llama-bXXXX-bin-win-cpu-x64.zip` |
| Windows, NVIDIA GPU | `llama-bXXXX-bin-win-cuda-cu12.x-x64.zip` |
| macOS (Apple Silicon) | `llama-bXXXX-bin-macos-arm64.zip` |
| macOS (Intel) | `llama-bXXXX-bin-macos-x64.zip` |
| Linux, CPU only | `llama-bXXXX-bin-ubuntu-x64.zip` |

**Step 2 — Extract and note the path:**

Extract the zip to a permanent location, e.g.:
- Windows: `C:\llama\`
- macOS/Linux: `~/llama/`

Inside you'll find the `llama-server` (or `llama-server.exe` on Windows) binary.

**Step 3 — Verify it works:**

```powershell
# Windows
C:\llama\llama-server.exe --version

# macOS / Linux
~/llama/llama-server --version
```

You don't need to add it to your PATH — you'll point LocalBot at it directly via `LLAMA_SERVER_EXECUTABLE` in `.env`.

> **Building from source:** Only needed if you want custom compile flags or cutting-edge commits. Requires CMake and a C++ compiler. See the [llama.cpp build docs](https://github.com/ggerganov/llama.cpp/blob/master/docs/build.md).

### 3. A GGUF model file

Download a quantized GGUF model. A few good starting points:

| Model | Size | Notes |
|---|---|---|
| [Llama-3.2-3B-Instruct-Q4_K_M](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF) | ~2 GB | Fast, CPU-friendly |
| [Mistral-7B-Instruct-v0.3-Q4_K_M](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.3-GGUF) | ~6 GB | Stronger reasoning |
| [Gemma-3-1B-it-Thinking](https://huggingface.co/Andycurrent/Gemma-3-1B-it-GLM-4.7-Flash-Heretic-Uncensored-Thinking_GGUF) | ~1 GB | Ultra-light thinking model |

Note the full path to the downloaded `.gguf` file — you'll need it in `.env`.

### 4. A Discord Bot Token

1. Go to the [Discord Developer Portal](https://discord.com/developers/applications) and create a new application.
2. Under **Bot**, click **Add Bot** and copy the token.
3. Enable the **Message Content Intent** under **Privileged Gateway Intents**.
4. Invite the bot with the `bot` scope and `Send Messages` + `Read Message History` permissions.

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/dmg10007/LocalBot.git
cd LocalBot
```

### 2. Create and activate a virtual environment

```bash
# macOS / Linux
python -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
# Runtime only
pip install -e .

# With dev tools (ruff, mypy, pytest)
pip install -e ".[dev]"

# With optional PDF attachment support
pip install -e ".[pdf]"
```

### 4. Configure environment variables

```bash
# macOS / Linux
cp .env.example .env

# Windows (PowerShell)
Copy-Item .env.example .env
```

Open `.env` and fill in at minimum:

| Variable | Required | Description |
|---|---|---|
| `DISCORD_BOT_TOKEN` | ✅ | Your bot token from the Developer Portal |
| `LLAMA_SERVER_MODEL_PATH` | ✅ | Absolute path to your `.gguf` model file |
| `LLAMA_SERVER_EXECUTABLE` | ✅ | Full path to the `llama-server` binary |
| `LLAMA_SERVER_N_GPU_LAYERS` | — | `0` = CPU only (default), `-1` = all layers on GPU |
| `BRAVE_API_KEY` | — | Leave blank to disable web search |
| `LLAMA_SERVER_MODEL_FAMILY` | — | Leave blank for auto-detection (see [Swapping Models](#swapping-models)) |
| `MODEL_TEMPERATURE` | — | Default `0.3`; try `0.1` for smaller/chattier models |

Example values on Windows:
```env
LLAMA_SERVER_EXECUTABLE=C:\llama\llama-server.exe
LLAMA_SERVER_MODEL_PATH=C:\Users\You\models\llama-3.2-3b-instruct-q4_k_m.gguf
```

All other settings have sensible defaults. See [`.env.example`](.env.example) for the full reference.

### 5. Create required directories

```bash
# macOS / Linux
mkdir -p logs storage

# Windows (PowerShell)
New-Item -ItemType Directory -Force -Path logs, storage
```

### 6. Run the bot

```bash
localbot
# or
python -m localbot
```

`llama-server` is started automatically as a subprocess. You do **not** need to start it manually. Its output (including any crash messages) is captured and forwarded to the application log.

---

## Project Layout

```
src/localbot/
├── __main__.py             # `python -m localbot` entry point
├── app.py                  # Discord event loop, rate limiting, command handler
├── config.py               # All settings loaded from .env
├── agent.py                # Core request/tool loop
├── adapters/
│   ├── llamacpp_server.py  # llama-server subprocess manager + log capture
│   └── llamacpp_client.py  # OpenAI-compatible HTTP client, model detection, think-strip
├── tools/
│   ├── registry.py         # Tool schemas + dispatcher (static + scheduler)
│   ├── scheduler_tools.py  # LLM-callable schedule_job / cancel_job / list_jobs wrappers
│   ├── search.py           # Brave Search + page fetch & summarise
│   ├── reddit.py           # Reddit JSON API (no auth required)
│   └── time_tools.py       # Current time / timezone helpers
├── scheduler/
│   ├── service.py          # APScheduler wrapper + cron validation
│   └── store.py            # SQLite job persistence + atomic limit checks
├── storage/
│   ├── db.py               # Schema initialisation
│   ├── history.py          # Per-user conversation history (SQLite)
│   └── audit.py            # Append-only JSONL audit log
└── messaging.py            # Discord 2000-char message splitting
```

---

## Swapping Models

To try a different model, update `LLAMA_SERVER_MODEL_PATH` in `.env` and restart. No other changes are needed.

On startup LocalBot queries `/v1/models`, reads the loaded filename, and automatically applies the correct stop tokens and think-stripping for the detected family. Detection runs only once — subsequent readiness probes reuse the cached result.

| Family | Matched by filename | Stop tokens | Think-strip |
|---|---|---|---|
| `GEMMA` | `gemma`, `glm` | `<end_of_turn>`, `<eos>` | ✅ |
| `LLAMA` | `llama` | `<\|eot_id\|>`, `<\|end_of_text\|>` | ❌ |
| `MISTRAL` | `mistral`, `mixtral` | `</s>`, `[INST]` | ❌ |
| `QWEN` | `qwen` | `<\|im_end\|>` | ✅ |
| `DEEPSEEK` | `deepseek` | `<└┘>`, `<\|end_of_sentence\|>` | ✅ |
| `PHI` | `phi` | `<\|end\|>` | ❌ |
| `UNKNOWN` | anything else | *(GGUF-embedded EOS)* | ❌ |

The detected family is logged on every start so you can confirm it:

```
INFO  Detected model: 'Gemma-3-1B-...' → family=GEMMA (stop=['<end_of_turn>', '<eos>'], think_strip=True)
```

If the detection is wrong (e.g. a fine-tune with an unusual filename), override it manually:

```env
LLAMA_SERVER_MODEL_FAMILY=gemma
```

### Temperature guidance by model size

| Model size | Recommended `MODEL_TEMPERATURE` | Reason |
|---|---|---|
| 1B–3B | `0.1`–`0.2` | Smaller models ramble at higher temps; lower keeps output focused |
| 7B | `0.3` | Default; good balance of coherence and variety |
| 13B+ | `0.4`–`0.7` | Larger models handle higher temps well; more natural responses |

---

## Web Search

When a user asks the bot to search for something, it:

1. Queries the **Brave Search API** for the top results
2. **Concurrently fetches** the top `SEARCH_FETCH_COUNT` pages (default 3)
3. **Strips HTML** — removes scripts, styles, navbars, and footers; prefers `<article>`/`<main>` for higher signal content
4. Passes up to `SEARCH_FETCH_CHARS` characters (default 1500) of clean text per page to the LLM
5. The LLM **summarises the actual page content** and returns a response with source links

Raw PDF file URLs (those ending in `.pdf`) are skipped automatically. Pages that time out, return errors, or are on the skip list (YouTube, Twitter/X, Instagram, TikTok, Facebook) are silently skipped and fall back to the Brave index description. Tool results are capped at 4000 characters before being injected into the context window to prevent runaway responses from exhausting model RAM.

### Search tuning

| Variable | Default | Notes |
|---|---|---|
| `SEARCH_RESULT_COUNT` | `5` | Total results from Brave |
| `SEARCH_FETCH_COUNT` | `3` | Pages actually fetched and read |
| `SEARCH_FETCH_CHARS` | `1500` | Max chars of content per page sent to LLM |
| `SEARCH_FETCH_TIMEOUT_SECONDS` | `8` | Per-page HTTP timeout before skipping |

> **Context window tip:** At `LLAMA_SERVER_CTX_SIZE=4096`, keep `SEARCH_FETCH_CHARS` at 1500 or lower. If you raise the context to 8192, you can safely increase it to 3000 for richer summaries.

---

## Thinking Model Support

Think-stripping is applied automatically based on the detected model family — no configuration needed. Models in the `GEMMA`, `DEEPSEEK`, and `QWEN` families have it enabled; all others skip it entirely.

The raw reasoning is logged at `DEBUG` level if you want to inspect it:

```bash
localbot --log-level DEBUG
```

To cap how many tokens the model spends thinking (faster responses), add to `LLAMA_SERVER_EXTRA_ARGS`:

```env
LLAMA_SERVER_EXTRA_ARGS=--reasoning-budget 512
```

---

## Scheduled Jobs

Users can schedule recurring prompts in two ways:

### Natural language (via the LLM)

Just ask the bot conversationally. The LLM translates the request into a cron expression and calls `schedule_job` directly — no special syntax required:

> *"Remind me every morning at 8am to review my task list"*
> *"Send me the latest tech news every weekday at 6pm"*
> *"Check in with me every Monday at 9am"*

The bot will **only confirm a job once `schedule_job` has returned successfully** and will always relay the real job ID. It will never invent an ID or confirm a job it hasn't actually created.

### Direct commands

| Command | Description |
|---|---|
| `jobs list` | Show your active scheduled jobs |
| `jobs cancel <id>` | Cancel a job by ID |
| `timezone set <IANA>` | Set your local timezone (e.g. `America/New_York`) |
| `timezone show` | Show your saved timezone |
| `time now` | Show the current time in your timezone |

Cron expressions are validated before registration — invalid field ranges and extra fields are rejected with a clear error message. Per-user and global job limits are enforced atomically to prevent race conditions.

---

## Security Notes

- The bot is designed for **personal/trusted-user use**. All user messages are stored in SQLite and logged to an audit file.
- Per-user rate limiting (`RATE_LIMIT_SECONDS`, default 5s) prevents inference spam. The rate-limit table is periodically compacted so it does not grow unboundedly.
- Input length is capped at `MAX_INPUT_LENGTH` characters (default 1000) before hitting the LLM.
- Tool results are capped at 4000 characters before context injection to prevent memory exhaustion.
- Scheduler jobs are capped per user (`SCHEDULER_MAX_JOBS_PER_USER`, default 5) with an atomic DB check to prevent races.
- Cron expressions supplied by the LLM are validated against legal field ranges before being passed to APScheduler.
- Timezone strings are validated against the IANA `zoneinfo` database before being stored.
- The audit log at `AUDIT_LOG_PATH` records all interactions for review. Timeout responses are recorded distinctly from genuine LLM replies.
- Scheduler tool calls (`schedule_job`, `cancel_job`, `list_jobs`) are scoped per-request to the authenticated user — the LLM cannot create or cancel jobs for other users.

---

## Development

### Linting & formatting

```bash
ruff check .
ruff format .
```

### Type checking

```bash
mypy src/
```

### Tests

```bash
pytest
```

---

## Environment Variables

See [`.env.example`](.env.example) for all options with inline descriptions and the timeout budget explanation.
