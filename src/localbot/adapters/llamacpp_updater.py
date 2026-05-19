"""Check whether a newer llama.cpp release is available.

This module is deliberately dependency-free beyond the stdlib and aiohttp
(already a project dependency). It never raises — all failures are logged
as warnings and the caller continues normally.
"""
from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass

import aiohttp

log = logging.getLogger(__name__)

_RELEASES_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"

# Matches the canonical release-tag form used by GitHub: b8998, b9222, etc.
_BUILD_RE = re.compile(r"\bb(\d+)\b")

# Matches the `version: 8998` line emitted by newer llama-server builds where
# the `b`-prefix is absent from the version line itself.
_VERSION_LINE_RE = re.compile(r"(?i)version:\s*(\d+)")


def _parse_build_number(text: str) -> int | None:
    """Return the build number from ``llama-server --version`` output.

    Tries two patterns in order of specificity:

    1. ``version: 8998`` — unambiguous; present in all known builds.
    2. ``b8998``         — legacy fallback for older output formats.
    """
    m = _VERSION_LINE_RE.search(text)
    if m:
        return int(m.group(1))
    m = _BUILD_RE.search(text)
    if m:
        return int(m.group(1))
    return None


@dataclass(frozen=True)
class UpdateInfo:
    """Result of a single update check."""
    current: int | None   # installed build number, or None if undetectable
    latest: int           # build number from GitHub releases
    url: str              # HTML URL of the latest release
    available: bool       # True when latest > current (and current is known)


async def _detect_installed_build(executable: str, timeout: float) -> int | None:
    """Run ``llama-server --version`` and parse the build number.

    llama-server prints one of the following depending on build age::

        # Newer builds (no b-prefix on the version line):
        version: 8998 (2098fd616)
        built with Clang 19.1.5 for Windows x86_64

        # Older builds:
        version: 3596 (b3596)
        built with ...

    Both formats are handled by :func:`_parse_build_number`.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            executable, "--version",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            raw, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            log.warning("llama-server --version timed out; skipping installed-build detection")
            return None
        text = raw.decode(errors="replace")
        build = _parse_build_number(text)
        if build is not None:
            return build
        log.warning(
            "Could not parse build number from llama-server --version output: %r",
            text[:200],
        )
        return None
    except FileNotFoundError:
        log.warning(
            "llama-server not found at %r; skipping update check",
            executable,
        )
        return None
    except Exception as exc:  # pragma: no cover
        log.warning("Unexpected error detecting installed build: %s", exc)
        return None


async def _fetch_latest_release(session: aiohttp.ClientSession, timeout: float) -> tuple[int, str] | None:
    """Return (build_number, html_url) for the latest llama.cpp release."""
    try:
        async with session.get(
            _RELEASES_API,
            timeout=aiohttp.ClientTimeout(total=timeout),
            headers={"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"},
        ) as resp:
            if resp.status != 200:
                log.warning(
                    "GitHub releases API returned HTTP %s; skipping update check",
                    resp.status,
                )
                return None
            data = await resp.json()
    except asyncio.TimeoutError:
        log.warning("GitHub releases API timed out; skipping update check")
        return None
    except aiohttp.ClientError as exc:
        log.warning("Network error fetching llama.cpp release info: %s", exc)
        return None
    except Exception as exc:  # pragma: no cover
        log.warning("Unexpected error fetching llama.cpp release info: %s", exc)
        return None

    tag: str = data.get("tag_name", "")
    url: str = data.get("html_url", _RELEASES_API)
    m = _BUILD_RE.search(tag)
    if not m:
        log.warning("Could not parse build number from release tag %r", tag)
        return None
    return int(m.group(1)), url


async def check_for_update(
    executable: str,
    *,
    timeout_seconds: float = 10.0,
) -> UpdateInfo | None:
    """Perform a full update check and return an :class:`UpdateInfo`.

    Returns ``None`` if the GitHub API call fails (network unavailable,
    rate-limited, etc.) so the caller can decide how to handle absence of
    information.  A successful result is always an :class:`UpdateInfo`
    even when no update is available.

    Args:
        executable: Path to (or name of) the ``llama-server`` binary.
        timeout_seconds: Per-operation HTTP/subprocess timeout.
    """
    async with aiohttp.ClientSession() as session:
        installed_task = asyncio.create_task(
            _detect_installed_build(executable, timeout_seconds)
        )
        latest_result = await _fetch_latest_release(session, timeout_seconds)
        current = await installed_task

    if latest_result is None:
        return None

    latest_build, url = latest_result
    available = current is not None and latest_build > current
    return UpdateInfo(current=current, latest=latest_build, url=url, available=available)
