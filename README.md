# LocalBot

A lightweight Discord DM bot that runs a local LLM via [llama.cpp](https://github.com/ggerganov/llama.cpp)'s built-in `llama-server`. The Python process is intentionally thin — it never loads the model itself; all inference goes through `llama-server`'s OpenAI-compatible HTTP API.

## Features

- **Conversational chat** with per-user message history (SQLite-backed)
- **Deep web search** via Brave Search API — fetches and summarises actual page content, not just index snippets
- **Reddit search** — searches Reddit posts and discussions via the unauthenticated JSON API
- **Filesystem workspace** — the LLM can read, write, list, patch, and search files inside a sandboxed directory; absolute paths and directory traversal are blocked at the OS level
- **Scheduled prompts** — users define recurring jobs via natural language; the LLM converts schedules to cron expressions and calls `schedule_job` directly; commands (`jobs list`, `jobs cancel <id>`) are also available
- **Self-diagnostics** — the LLM can call `read_logs` to read and reason over the audit log in real time; ask *"why did my last job fail?"* or *"check the logs for errors"* conversationally
- **Model-agnostic inference** — auto-detects model family once on startup and caches the result; swap models by changing one line in `.env`
- **Thinking model support** — automatically strips `<think>` blocks for reasoning models (Gemma, DeepSeek, Qwen)
- **Intent-based slot routing** — agent routes each request to the appropriate model slot (`general`, `coding`, `reasoning`) based on message intent
- **Rate limiting** — per-user cooldown with bounded memory; stale entries are automatically evicted so the table never grows unboundedly
- **Self-healing** — detects llama-server crashes and restarts automatically
- **Audit log** — append-only JSONL log of all user messages and bot replies; timeouts are recorded distinctly from genuine LLM replies
- **llama-server log capture** — subprocess stdout/stderr is piped into the application logger so crashes (OOM, CUDA errors) surface immediately
- **Auto-updater** — on startup, detects available llama.cpp updates and prompts the terminal operator to install; supports unattended mode via `LLAMA_UPDATE_AUTO=true`
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
| `BOT_OWNER_ID` | — | Your Discord user ID. When set, grants you full log access across all users (see [Self-Diagnostics](#self-diagnostics)) |
| `SANDBOX_ROOT` | — | Absolute path to the directory the LLM may read/write. Defaults to `./sandbox` |

Example values on Windows:
```env
LLAMA_SERVER_EXECUTABLE=C:\llama\llama-server.exe
LLAMA_SERVER_MODEL_PATH=C:\Users\You\models\llama-3.2-3b-instruct-q4_k_m.gguf
SANDBOX_ROOT=C:\Users\You\localbot-workspace
```

All other settings have sensible defaults. See [`.env.example`](.env.example) for the full reference.

### 5. Create required directories

```bash
# macOS / Linux
mkdir -p logs storage sandbox

# Windows (PowerShell)
New-Item -ItemType Directory -Force -Path logs, storage, sandbox
```

### 6. Run the bot

```bash
localbot
# or
python -m localbot
```

`llama-server` is started automatically as a subprocess. You do **not** need to start it manually. Its stdout and stderr (including crash messages, OOM errors, and CUDA failures) are captured and forwarded to the application log.

If a newer llama.cpp build is available you will be prompted in the terminal before the server starts:

```
Install llama.cpp b9222? [y/N] (auto-skip in 30s):
```

Type `y` to download and install the update, or press Enter (or wait) to continue with the current version. Set `LLAMA_UPDATE_AUTO=true` in `.env` to skip the prompt and always update automatically.

---

## Project Layout

```
src/localbot/
├── __main__.py             # `python -m localbot` entry point
├── app.py                  # Discord event loop, rate limiting, command handler
├── config.py               # All settings loaded from .env
├── agent.py                # Core request/tool loop; intent routing (_select_slot, _detect_workspace_mode, _needs_tools)
├── adapters/
│   ├── llamacpp_server.py      # llama-server subprocess manager + stdout/stderr log capture
│   ├── llamacpp_client.py      # OpenAI-compatible HTTP client; model family detection (cached); think-strip
│   ├── llamacpp_updater.py     # Startup update check (GitHub Releases API)
│   └── llamacpp_downloader.py  # Asset selection, streaming download, zip extraction
├── tools/
│   ├── registry.py         # Tool schemas + async dispatcher (timeout-guarded)
│   ├── filesystem.py       # read/write/list/patch/search — sandboxed to SANDBOX_ROOT
│   ├── log_reader.py       # read_logs — audit log reader for self-diagnostics
│   ├── scheduler_tools.py  # LLM-callable schedule_job / cancel_job / list_jobs wrappers
│   ├── search.py           # Brave Search + page fetch & summarise; PDF skip fix
│   ├── reddit.py           # Reddit JSON API (no auth required)
│   └── time_tools.py       # Current time / timezone helpers
├── scheduler/
│   ├── service.py          # APScheduler wrapper; cron validation; atomic job-limit check
│   └── store.py            # SQLite job persistence
├── storage/
│   ├── db.py               # Schema initialisation
│   ├── history.py          # Per-user conversation history (SQLite)
│   └── audit.py            # Append-only JSONL audit log
└── messaging.py            # Discord 2000-char message splitting
tests/
├── conftest.py                          # Stubs config + heavy deps for CI
├── test_agent_needs_tools.py
├── test_llamacpp_family_detection.py
├── test_messaging.py
├── test_routing_dispatch_filesystem.py  # Routing, async dispatch, filesystem sandbox
├── test_scheduler_validate_cron.py
└── test_search_should_skip.py
```

---

## Filesystem Workspace

When `SANDBOX_ROOT` is configured, the LLM gains access to six filesystem tools:

| Tool | Description |
|---|---|
| `read_file` | Read a text file; large files are truncated with a notice |
| `write_file` | Write (or overwrite) a text file; parent dirs are created automatically |
| `list_directory` | List files and subdirectories with sizes |
| `apply_patch` | Apply a unified diff patch to an existing file |
| `search_in_files` | Case-insensitive grep across the workspace; supports glob filters |
| `get_current_dir` | Return the current working path relative to `SANDBOX_ROOT` |

**Security guarantees:**
- All paths are resolved inside `SANDBOX_ROOT` before any I/O. Absolute paths are re-rooted; `../` traversal raises `PermissionError`.
- Binary file extensions (`.png`, `.jpg`, `.zip`, `.pdf`, etc.) are blocked on both read and write.
- Tool results are capped at `MAX_TOOL_RESULT_CHARS` (default 4000 chars) before context injection.

To enable, set `SANDBOX_ROOT` in `.env` to any directory you are comfortable giving the bot read/write access to.

---

## Swapping Models

To try a different model, update `LLAMA_SERVER_MODEL_PATH` in `.env` and restart. No other changes are needed.

On startup LocalBot queries `/v1/models`, reads the loaded filename, and automatically applies the correct stop tokens and think-stripping for the detected family. **Detection runs only once** — the result is cached and reused on all subsequent readiness probes, so there is no per-request overhead.

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

URLs ending in `.pdf` are skipped automatically (exact extension match — not a substring check). Pages that time out, return errors, or are on the skip list (YouTube, Twitter/X, Instagram, TikTok, Facebook) are silently skipped and fall back to the Brave index description. Tool results are capped at 4000 characters before being injected into the context window to prevent runaway responses from exhausting model RAM.

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

## Auto-Updater

On each startup LocalBot checks whether a newer llama.cpp release is available on GitHub. If one is found, it offers to download and install it before launching `llama-server`.

### How it works

1. The GitHub Releases API is queried for the latest `ggml-org/llama.cpp` build number.
2. The installed build number is read from `llama-server --version`.
3. If the installed build is older, the terminal operator is prompted:
   ```
   Install llama.cpp b9222? [y/N] (auto-skip in 30s):
   ```
4. On `y`, the correct platform asset is downloaded (streaming, with a progress bar), extracted over the existing install directory, and the version check re-runs to confirm the installed build.
5. `llama-server` then starts normally with the updated binary.

### Platform asset selection

| Platform | Asset chosen |
|---|---|
| Windows | CUDA build if available, CPU-only otherwise |
| macOS Apple Silicon | `macos-arm64` |
| macOS Intel | `macos-x64` |
| Linux x86-64 | `ubuntu-x64` |

If no matching asset is found (e.g. an unsupported architecture), the update is skipped and a message with the manual download URL is logged.

### Configuration

| Variable | Default | Description |
|---|---|---|
| `LLAMA_UPDATE_CHECK` | `true` | Set `false` to disable the check entirely (air-gapped hosts) |
| `LLAMA_UPDATE_AUTO` | `false` | Set `true` to install updates automatically without prompting |
| `LLAMA_UPDATE_PROMPT_TIMEOUT_SECONDS` | `30` | Seconds to wait for a terminal response before skipping |
| `LLAMA_UPDATE_CHECK_TIMEOUT_SECONDS` | `10` | HTTP/subprocess timeout for the version check itself |

---

## Self-Diagnostics

LocalBot can read and reason over its own audit log in real time. Just ask conversationally:

> *"Why did my 8am reminder not fire?"*
> *"Check the logs for any errors"*
> *"What did you search for last time?"*
> *"Are there any crashes or timeouts?"*

The LLM calls the `read_logs` tool, receives a filtered JSON slice of the audit log, and explains what it finds in plain language.

### Access control

By default every user sees only their own audit entries — the tool is scoped to the requesting user's ID so the LLM cannot expose another user's conversation history even if prompted to.

Set `BOT_OWNER_ID` in `.env` to your Discord user ID to unlock full log access (all users, all events) for troubleshooting global issues:

```env
BOT_OWNER_ID=123456789012345678
```

### What the audit log contains

Every JSONL entry has at minimum a `ts` (Unix timestamp) and an `event` field:

| Event | Description |
|---|---|
| `user_message` | Incoming user message |
| `assistant_reply` | Final reply sent to the user (timeouts are recorded distinctly) |
| `tool_call` | LLM requested a tool with these arguments |
| `tool_result` | First 500 chars of the tool's return value |

### Log level filtering

When asking the bot to check logs you can be specific:

> *"Show me only errors from the logs"*
> *"Check for any warnings or timeouts"*

Audit entries are mapped to notional log levels: `tool_call` / `tool_result` / `user_message` / `assistant_reply` → **INFO**; timeout or missed-job events → **WARNING**; error/fail/crash events → **ERROR**.

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

Cron expressions are validated against legal field ranges before registration — invalid or over-specified expressions are rejected with a clear error. Per-user and global job limits are enforced with a single atomic DB operation to prevent race conditions under concurrent requests.

---

## Security Notes

- The bot is designed for **personal/trusted-user use**. All user messages are stored in SQLite and logged to an audit file.
- Per-user rate limiting (`RATE_LIMIT_SECONDS`, default 5s) prevents inference spam. The rate-limit table is bounded — stale entries are evicted automatically so it never grows unboundedly.
- Input length is capped at `MAX_INPUT_LENGTH` characters (default 1000) before hitting the LLM.
- Tool results are capped at `MAX_TOOL_RESULT_CHARS` (default 4000) before context injection to prevent memory exhaustion from runaway search responses.
- Scheduler jobs are capped per user (`SCHEDULER_MAX_JOBS_PER_USER`, default 5) with a single atomic DB check to prevent races under concurrent requests.
- Cron expressions supplied by the LLM are validated against legal field ranges before being passed to APScheduler.
- Timezone strings are validated against the IANA `zoneinfo` database before being stored — invalid values are rejected with a clear error.
- The audit log records all interactions for review. Timeout responses are recorded distinctly from genuine LLM replies so the audit trail is accurate.
- Scheduler tool calls (`schedule_job`, `cancel_job`, `list_jobs`) are scoped per-request to the authenticated user — the LLM cannot create or cancel jobs for other users.
- `read_logs` is scoped to the requesting user's ID by default. Set `BOT_OWNER_ID` to grant a single trusted user full log visibility. The LLM cannot bypass this scoping even if prompted to.
- Filesystem tools are confined to `SANDBOX_ROOT`. Paths are resolved server-side; absolute paths are re-rooted and `../` traversal is blocked at the OS level before any I/O occurs.
- The auto-updater downloads only from the official `ggml-org/llama.cpp` GitHub Releases. `LLAMA_SERVER_EXTRA_ARGS` is the only value passed directly to a subprocess and must be set only by a trusted operator via `.env`.

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

The test suite uses `conftest.py` to stub `localbot.config` and optional heavy dependencies (discord.py, APScheduler) before collection, so `pytest` works in a clean environment without a running bot or llama-server.

---

## Environment Variables

See [`.env.example`](.env.example) for all options with inline descriptions and the timeout budget explanation.
