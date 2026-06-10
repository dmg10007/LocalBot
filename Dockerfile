# LocalBot — multi-stage image
# Build arg EXTRA installs an optional extras group (e.g. "webui").
# Leave blank for the default Discord bot image.
#
# The llama stage downloads the pre-built Ubuntu x64 llama-server release
# and copies the binary + all shared libraries into the final image so the
# dynamic linker can find libllama-server-impl.so at runtime.

# ── Stage 1: download llama-server + shared libs ────────────────────────
FROM python:3.11-slim AS llama

ARG LLAMA_VERSION=b9591
ARG LLAMA_URL=https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_VERSION}/llama-${LLAMA_VERSION}-bin-ubuntu-x64.tar.gz

RUN apt-get update && apt-get install -y --no-install-recommends curl tar \
    && rm -rf /var/lib/apt/lists/*

# Extract the full release into /opt/llama so we can copy everything.
RUN mkdir -p /opt/llama \
    && curl -fsSL "$LLAMA_URL" | tar -xz --strip-components=1 -C /opt/llama

# ── Stage 2: application image ────────────────────────────────────────────
FROM python:3.11-slim AS base

ARG EXTRA=""

WORKDIR /app

# Copy the llama-server binary and all companion shared libraries.
COPY --from=llama /opt/llama/llama-server /usr/local/bin/llama-server
COPY --from=llama /opt/llama/*.so* /usr/local/lib/

# Ensure the dynamic linker picks up the new .so files.
RUN ldconfig

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
