# LocalBot

A lightweight Discord DM bot that runs a local LLM via [llama.cpp](https://github.com/ggerganov/llama.cpp)'s built-in `llama-server`. The Python process is intentionally thin — it never loads the model itself; all inference goes through `llama-server`'s OpenAI-compatible HTTP API.

## Features

- **Conversational chat** with per-user message history
- **Web search** via Brave Search API with cited sources
- **Scheduled prompts** — users define jobs with natural-language recurrence
- **Self-healing** — detects llama-server crashes and restarts automatically
- **Minimal footprint** — discord.py + aiohttp + APScheduler; no heavy ML dependencies

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

You don't need to add it to your PATH — you'll point LocalBot at it directly via the `LLAMA_SERVER_EXECUTABLE` setting in `.env`.

> **Building from source:** Only needed if you want custom compile flags or cutting-edge commits. Requires CMake and a C++ compiler (Visual Studio Build Tools on Windows). See the [llama.cpp build docs](https://github.com/ggerganov/llama.cpp/blob/master/docs/build.md) for instructions.

### 3. A GGUF model file

Download a quantized GGUF model. A few good starting points:
- [Llama-3.2-3B-Instruct-Q4_K_M.gguf](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF) — fast, CPU-friendly (~2 GB)
- [Mistral-7B-Instruct-v0.3-Q4_K_M.gguf](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.3-GGUF) — stronger, needs ~6 GB RAM

Note the full path to the downloaded file — you'll need it in `.env`.

### 4. A Discord Bot Token

1. Go to the [Discord Developer Portal](https://discord.com/developers/applications) and create a new application.
2. Under **Bot**, click **Add Bot** and copy the token.
3. Enable the **Message Content Intent** under **Privileged Gateway Intents**.
4. Invite the bot to your server (or use it via DMs) with the `bot` scope and `Send Messages` + `Read Message History` permissions.

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

# With dev tools (ruff, mypy, pytest) — recommended for contributors
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

Open `.env` and set at minimum:

| Variable | Required | Description |
|---|---|---|
| `DISCORD_BOT_TOKEN` | ✅ | Your bot token from the Developer Portal |
| `LLAMA_SERVER_MODEL_PATH` | ✅ | Absolute path to your `.gguf` model file |
| `LLAMA_SERVER_EXECUTABLE` | ✅ | Full path to the `llama-server` binary |
| `LLAMA_SERVER_N_GPU_LAYERS` | — | `0` = CPU only (default), `-1` = all layers on GPU |
| `BRAVE_API_KEY` | — | Leave blank to disable web search |

Example values on Windows:
```env
LLAMA_SERVER_EXECUTABLE=C:\llama\llama-server.exe
LLAMA_SERVER_MODEL_PATH=C:\Users\Dalton\models\llama-3.2-3b-instruct-q4_k_m.gguf
```

All other settings have sensible defaults. See [`.env.example`](.env.example) for the full reference with descriptions.

### 5. Create required directories

The bot needs `logs/` and `storage/` directories on first run. Create them manually until auto-creation is implemented in `app.py`:

```bash
# macOS / Linux
mkdir -p logs storage

# Windows (PowerShell)
New-Item -ItemType Directory -Force -Path logs, storage
```

### 6. Run the bot

```bash
# Using the installed script entry point
localbot

# Or as a Python module
python -m localbot
```

`llama-server` is started automatically as a subprocess on bot startup. You do **not** need to start it manually.

---

## Project Layout

> **Note:** The source tree below is the planned layout. Implementation is in progress on the `dev` branch.

```
src/localbot/
├── __main__.py       # `python -m localbot` entry point
├── app.py            # Entry point & Discord event loop
├── config.py         # Settings loaded from .env
├── agent.py          # Core request/tool loop
├── adapters/
│   ├── llamacpp_server.py  # llama-server subprocess manager
│   └── llamacpp_client.py  # OpenAI-compatible HTTP client
├── tools/
│   ├── registry.py   # Tool dispatcher
│   ├── search.py     # Web / Brave Search
│   ├── reddit.py     # Reddit JSON API (no auth required)
│   └── time_tools.py # Current time / timezone helpers
├── scheduler/
│   ├── service.py    # APScheduler wrapper
│   └── store.py      # SQLite job persistence
├── storage/
│   ├── db.py         # Schema init
│   ├── history.py    # Per-user conversation history
│   └── audit.py      # JSONL audit log
└── messaging.py      # Discord message splitting helpers
```

---

## Scheduled Jobs

Users interact via DM commands:

| Command | Description |
|---|---|
| `jobs list` | Show your active jobs |
| `jobs cancel <id>` | Cancel a job |
| `timezone set <IANA>` | Set your local timezone |
| `timezone show` | Show your saved timezone |
| `time now` | Show current time |

To schedule a job, just ask the bot naturally:
> "Remind me every morning at 8am to review my task list"

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
