"""Download and install a llama.cpp release in-place.

This module is deliberately dependency-free beyond the stdlib and aiohttp
(already a project dependency).  It never raises — all failures are returned
as a :class:`DownloadResult` with ``ok=False`` so the caller can decide how
to handle them.

Supported platforms
-------------------
- Windows x86-64  (prefers CUDA build if a CUDA asset is listed, else CPU)
- macOS arm64     (Apple Silicon)
- macOS x86-64    (Intel)
- Linux x86-64    (Ubuntu generic binary)

Any other platform returns ``ok=False`` with an explanatory message.
"""
from __future__ import annotations

import asyncio
import logging
import platform
import sys
import zipfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import aiohttp

log = logging.getLogger(__name__)

_RELEASES_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"


@dataclass(frozen=True)
class DownloadResult:
    ok: bool
    message: str


def _select_asset(assets: list[dict], system: str, machine: str) -> str | None:
    """Return the download URL of the best matching asset or None."""
    names: list[str] = [a["name"] for a in assets]
    urls: dict[str, str] = {a["name"]: a["browser_download_url"] for a in assets}

    def pick(*patterns: str) -> str | None:
        for pat in patterns:
            for name in names:
                if pat in name:
                    return urls[name]
        return None

    if system == "Windows":
        # Prefer CUDA build; fall back to CPU-only.
        return pick("win-cuda", "win-cpu")

    if system == "Darwin":
        if machine == "arm64":
            return pick("macos-arm64")
        return pick("macos-x64", "macos-x86")

    if system == "Linux":
        return pick("ubuntu-x64", "linux-x64")

    return None


async def download_and_install(
    install_dir: str | Path,
    *,
    timeout_seconds: float = 120.0,
    release_url: str = _RELEASES_API,
) -> DownloadResult:
    """Download the latest llama.cpp release and extract it into *install_dir*.

    The zip is streamed in chunks and extracted in-place; existing files in
    *install_dir* that are not part of the zip are left untouched.  Only the
    flat file contents of the zip are written — any zip directory structure is
    stripped so all binaries land directly in *install_dir*.

    Args:
        install_dir:     Directory containing the current ``llama-server`` binary.
                         All zip entries are extracted here.
        timeout_seconds: Per-operation HTTP timeout.
        release_url:     GitHub Releases API URL (override in tests).
    """
    install_path = Path(install_dir)
    system = platform.system()       # 'Windows', 'Darwin', 'Linux'
    machine = platform.machine()     # 'AMD64', 'x86_64', 'arm64', 'aarch64'
    # Normalise arm64 spellings.
    if machine.lower() in ("aarch64", "arm64"):
        machine = "arm64"

    log.info("Fetching latest llama.cpp release metadata...")
    try:
        async with aiohttp.ClientSession() as session:
            # ── Step 1: resolve asset URL ──────────────────────────────────
            async with session.get(
                release_url,
                timeout=aiohttp.ClientTimeout(total=timeout_seconds),
                headers={
                    "Accept": "application/vnd.github+json",
                    "X-GitHub-Api-Version": "2022-11-28",
                },
            ) as resp:
                if resp.status != 200:
                    return DownloadResult(ok=False, message=f"GitHub API returned HTTP {resp.status}")
                data = await resp.json()

            asset_url = _select_asset(data.get("assets", []), system, machine)
            if asset_url is None:
                return DownloadResult(
                    ok=False,
                    message=(
                        f"No suitable asset found for {system}/{machine}. "
                        "Download manually from: " + data.get("html_url", _RELEASES_API)
                    ),
                )

            tag = data.get("tag_name", "?")
            log.info("Downloading %s from %s", tag, asset_url)

            # ── Step 2: stream the zip ─────────────────────────────────────
            buf = BytesIO()
            async with session.get(
                asset_url,
                timeout=aiohttp.ClientTimeout(total=timeout_seconds),
            ) as dl:
                if dl.status != 200:
                    return DownloadResult(ok=False, message=f"Asset download returned HTTP {dl.status}")
                total = int(dl.headers.get("Content-Length", 0))
                received = 0
                async for chunk in dl.content.iter_chunked(65536):
                    buf.write(chunk)
                    received += len(chunk)
                    if total:
                        pct = received * 100 // total
                        # Overwrite the same terminal line for a simple progress bar.
                        print(f"\r  Downloading... {pct:3d}%  ({received // 1024 // 1024} MB / {total // 1024 // 1024} MB)", end="", flush=True)
            print()  # newline after progress bar

    except asyncio.TimeoutError:
        return DownloadResult(ok=False, message="Download timed out")
    except aiohttp.ClientError as exc:
        return DownloadResult(ok=False, message=f"Network error: {exc}")
    except Exception as exc:  # pragma: no cover
        return DownloadResult(ok=False, message=f"Unexpected error: {exc}")

    # ── Step 3: extract into install_dir ──────────────────────────────────────
    try:
        buf.seek(0)
        with zipfile.ZipFile(buf) as zf:
            entries = zf.namelist()
            log.info("Extracting %d entries to %s", len(entries), install_path)
            for entry in entries:
                if entry.endswith("/"):
                    continue  # skip directory entries
                # Strip any leading directory components so all files land
                # flat in install_dir, matching the original install layout.
                filename = Path(entry).name
                if not filename:
                    continue
                dest = install_path / filename
                dest.write_bytes(zf.read(entry))
        return DownloadResult(ok=True, message=f"Updated to {tag} in {install_path}")
    except Exception as exc:
        return DownloadResult(ok=False, message=f"Extraction failed: {exc}")
