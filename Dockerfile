# LocalBot — multi-stage image
# Build arg EXTRA installs an optional extras group (e.g. "webui").
# Leave blank for the default Discord bot image.
#
# llama-server resolves backend plugins (libggml-cpu-*.so etc.) via
# dlopen() relative to the executable directory, not the system linker.
# We keep the full release in /opt/llama AND symlink every .so into
# /usr/local/bin/ (next to the binary) so both resolution paths work.

# ── Stage 1: download llama-server + shared libs ────────────────────────
FROM python:3.11-slim AS llama

ARG LLAMA_VERSION=b9592
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

# Copy the full release directory and place the binary in PATH.
# Then symlink every .so into the same directory as the binary so that
# llama-server's dlopen() calls (which use paths relative to the
# executable) can find libggml-cpu-*.so, libggml-base.so, etc.
COPY --from=llama /opt/llama/ /opt/llama/
RUN cp /opt/llama/llama-server /usr/local/bin/llama-server && \
    chmod +x /usr/local/bin/llama-server && \
    find /opt/llama -maxdepth 1 -name '*.so*' \
         -exec ln -sf {} /usr/local/bin/ \; && \
    echo "/opt/llama" > /etc/ld.so.conf.d/llama.conf && \
    ldconfig

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

# Create directories that are expected at runtime, then drop privileges.
RUN mkdir -p storage logs sandbox \
    && groupadd --system localbot \
    && useradd --system --gid localbot --home-dir /app --no-create-home localbot \
    && chown -R localbot:localbot /app

USER localbot

CMD ["python", "-m", "localbot"]
