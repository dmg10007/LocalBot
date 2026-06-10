# LocalBot — multi-stage image
# Build arg EXTRA installs an optional extras group (e.g. "webui").
# Leave blank for the default Discord bot image.

FROM python:3.11-slim AS base

ARG EXTRA=""

WORKDIR /app

# Install build deps (needed for some aiohttp/lxml wheels on slim)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src/ ./src/

# Install the package — with the optional extra if requested.
RUN if [ -n "$EXTRA" ]; then \
      pip install --no-cache-dir -e ".[$EXTRA]"; \
    else \
      pip install --no-cache-dir -e .; \
    fi

# Create directories that are expected at runtime.
RUN mkdir -p storage logs sandbox

# Use python -m so the entry point works regardless of whether
# console_scripts are on PATH (avoids 'executable not found' in containers).
CMD ["python", "-m", "localbot"]
