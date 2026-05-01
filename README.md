# LocalBot

A lightweight Discord DM bot that runs a local LLM via [llama.cpp](https://github.com/ggerganov/llama.cpp)'s built-in `llama-server`. The Python process is intentionally thin — it never loads the model itself; all inference goes through `llama-server`'s OpenAI-compatible HTTP API.

## Features

- **Conversational chat** with per-user message history
- **Web search** via Brave Search API with cited sources
- **Scheduled prompts** — users define jobs with natural-language recurrence
- **Self-healing** — detects llama-server crashes and restarts automatically
- **Minimal footprint** — discord.py + aiohttp + APScheduler; no heavy ML dependencies

## Quick Start

```bash
# 1. Clone and create a virtualenv
git clone https://github.com/dmg10007/LocalBot.git
cd LocalBot
python -m venv .venv && source .venv/bin/activate

# 2. Install
pip install -e .          # runtime only
pip install -e ".[dev]"   # + ruff, mypy, pytest

# 3. Configure
cp .env.example .env
# Edit .env — set DISCORD_BOT_TOKEN and LLAMA_SERVER_MODEL_PATH at minimum

# 4. Run
localbot
# or equivalently:
python -m localbot
```

llama-server is started automatically on bot startup. Make sure `llama-server` is on your `PATH` or set `LLAMA_SERVER_EXECUTABLE` to its full path.

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

## Environment Variables

See [`.env.example`](.env.example) for all options with descriptions.
