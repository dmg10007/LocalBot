# LocalBot — multi-stage image
# Build arg EXTRA installs an optional extras group (e.g. "webui").
# Leave blank for the default Discord bot image.
#
# The llama stage downloads the pre-built Ubuntu x64 llama-server binary
# so the container never needs to run the Windows .exe through WSL interop.

# ── Stage 1: download llama-server Linux binary ───────────────────────────
FROM python:3.11-slim AS llama

ARG LLAMA_VERSION=b9591
ARG LLAMA_URL=https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_VERSION}/llama-${LLAMA_VERSION}-bin-ubuntu-x64.tar.gz

RUN apt-get update && apt-get install -y --no-install-recommends curl tar \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL "$LLAMA_URL" | tar -xz -C /tmp \
    && find /tmp -name 'llama-server' -exec install -m 755 {} /usr/local/bin/llama-server \;

# ── Stage 2: application image ────────────────────────────────────────────
FROM python:3.11-slim AS base

ARG EXTRA=""

WORKDIR /app

# Copy llama-server binary from the llama stage
COPY --from=llama /usr/local/bin/llama-server /usr/local/bin/llama-server

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

CMD ["python", "-m", "localbot"]
