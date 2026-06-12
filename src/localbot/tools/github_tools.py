"""Async GitHub REST tools for the coding assistant.

All operations use the GITHUB_TOKEN personal access token from config.
No token → all tool calls return a helpful error message.

Exported tool functions
-----------------------
github_read_file(owner, repo, path, ref?)     — read a file from a repo
github_list_directory(owner, repo, path?, ref?) — list repo directory
github_create_branch(owner, repo, branch, from_branch?) — create branch
github_commit_files(owner, repo, branch, message, files) — push files
github_create_pull_request(owner, repo, title, head, base, body?) — open PR
github_list_pull_requests(owner, repo, state?) — list open/closed PRs

GITHUB_TOOL_SCHEMAS — OpenAI function-calling schemas for all six tools.
"""
from __future__ import annotations

import base64
import logging
from typing import Any

import aiohttp

from localbot.config import cfg

log = logging.getLogger(__name__)

_API = "https://api.github.com"
_HEADERS_BASE = {
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}
_READ_MAX_BYTES = 32_000


def _headers() -> dict[str, str]:
    if not cfg.github_token:
        return _HEADERS_BASE.copy()
    return {**_HEADERS_BASE, "Authorization": f"Bearer {cfg.github_token}"}


def _no_token_error() -> str:
    return (
        "GitHub operations are not configured. "
        "Set GITHUB_TOKEN in .env to enable them."
    )


def _allowlist() -> set[str]:
    raw = cfg.github_allowed_repos.strip()
    if raw:
        return {item.strip().lower() for item in raw.split(",") if item.strip()}
    if cfg.github_default_owner:
        return {cfg.github_default_owner.strip().lower()}
    return set()


def _repo_denied(owner: str, repo: str) -> str | None:
    """Return an error string if owner/repo is not allowlisted, else None."""
    allow = _allowlist()
    if not allow:
        return (
            "GitHub access is not authorized: set GITHUB_ALLOWED_REPOS or "
            "GITHUB_DEFAULT_OWNER in .env to permit specific owners/repos."
        )
    key_owner = owner.strip().lower()
    key_full = f"{key_owner}/{repo.strip().lower()}"
    if key_owner in allow or key_full in allow:
        return None
    return f"ERROR: '{owner}/{repo}' is not in the GitHub allowlist."


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------

async def github_read_file(
    owner: str,
    repo: str,
    path: str,
    ref: str = "HEAD",
) -> str:
    """Return the text content of a file in a GitHub repository."""
    if not cfg.github_token:
        return _no_token_error()
    if (denied := _repo_denied(owner, repo)) is not None:
        return denied
    url = f"{_API}/repos/{owner}/{repo}/contents/{path.lstrip('/')}"
    params = {"ref": ref}
    async with aiohttp.ClientSession(headers=_headers()) as session:
        async with session.get(url, params=params) as resp:
            if resp.status == 404:
                return f"ERROR: '{path}' not found in {owner}/{repo}@{ref}."
            if resp.status != 200:
                return f"ERROR: GitHub API returned HTTP {resp.status} for {url}."
            data = await resp.json()
    if data.get("type") == "dir":
        return f"ERROR: '{path}' is a directory. Use github_list_directory instead."
    encoded = data.get("content", "")
    raw = base64.b64decode(encoded.replace("\n", ""))
    if len(raw) > _READ_MAX_BYTES:
        text = raw[:_READ_MAX_BYTES].decode("utf-8", errors="replace")
        text += f"\n\n[...file truncated at {_READ_MAX_BYTES} bytes]"
    else:
        text = raw.decode("utf-8", errors="replace")
    log.debug("[github] read_file %s/%s/%s@%s (%d bytes)", owner, repo, path, ref, len(raw))
    return text


async def github_list_directory(
    owner: str,
    repo: str,
    path: str = "",
    ref: str = "HEAD",
) -> str:
    """Return an annotated listing of a directory in a GitHub repository."""
    if not cfg.github_token:
        return _no_token_error()
    if (denied := _repo_denied(owner, repo)) is not None:
        return denied
    url = f"{_API}/repos/{owner}/{repo}/contents/{path.lstrip('/')}"
    params = {"ref": ref}
    async with aiohttp.ClientSession(headers=_headers()) as session:
        async with session.get(url, params=params) as resp:
            if resp.status == 404:
                return f"ERROR: '{path}' not found in {owner}/{repo}@{ref}."
            if resp.status != 200:
                return f"ERROR: GitHub API returned HTTP {resp.status}."
            data = await resp.json()
    if isinstance(data, dict) and data.get("type") == "file":
        return f"ERROR: '{path}' is a file. Use github_read_file instead."
    lines = [f"Contents of '{path or '/'}' in {owner}/{repo}@{ref}:\n"]
    for entry in sorted(data, key=lambda e: (e["type"] == "file", e["name"])):
        tag = "[D]" if entry["type"] == "dir" else "[F]"
        size_str = f"  ({entry.get('size', 0):,} bytes)" if entry["type"] == "file" else ""
        lines.append(f"  {tag} {entry['name']}{size_str}")
    if len(lines) == 1:
        lines.append("  (empty directory)")
    return "\n".join(lines)


async def github_create_branch(
    owner: str,
    repo: str,
    branch: str,
    from_branch: str = "main",
) -> str:
    """Create a new branch in a GitHub repository."""
    if not cfg.github_token:
        return _no_token_error()
    if (denied := _repo_denied(owner, repo)) is not None:
        return denied
    # Resolve from_branch to its SHA.
    async with aiohttp.ClientSession(headers=_headers()) as session:
        ref_url = f"{_API}/repos/{owner}/{repo}/git/ref/heads/{from_branch}"
        async with session.get(ref_url) as resp:
            if resp.status != 200:
                return f"ERROR: Could not resolve branch '{from_branch}' (HTTP {resp.status})."
            ref_data = await resp.json()
        sha = ref_data["object"]["sha"]
        create_url = f"{_API}/repos/{owner}/{repo}/git/refs"
        payload = {"ref": f"refs/heads/{branch}", "sha": sha}
        async with session.post(create_url, json=payload) as resp:
            if resp.status == 422:
                return f"ERROR: Branch '{branch}' already exists in {owner}/{repo}."
            if resp.status not in (200, 201):
                body = await resp.text()
                return f"ERROR: Could not create branch (HTTP {resp.status}): {body[:200]}"
    log.info("[github] created branch '%s' in %s/%s from '%s'", branch, owner, repo, from_branch)
    return f"OK: branch '{branch}' created in {owner}/{repo} from '{from_branch}'."


async def github_commit_files(
    owner: str,
    repo: str,
    branch: str,
    message: str,
    files: list[dict[str, str]],
) -> str:
    """Commit one or more files to a branch.

    *files* is a list of ``{"path": "...", "content": "..."}`` dicts.
    Existing files are updated (their current SHA is fetched automatically);
    new files are created.
    """
    if not cfg.github_token:
        return _no_token_error()
    if (denied := _repo_denied(owner, repo)) is not None:
        return denied
    if not files:
        return "ERROR: No files provided to commit."

    async with aiohttp.ClientSession(headers=_headers()) as session:
        committed: list[str] = []
        for file in files:
            path = file["path"].lstrip("/")
            content_b64 = base64.b64encode(file["content"].encode()).decode()
            url = f"{_API}/repos/{owner}/{repo}/contents/{path}"
            # Fetch current SHA if the file exists (required for updates).
            sha: str | None = None
            async with session.get(url, params={"ref": branch}) as resp:
                if resp.status == 200:
                    existing = await resp.json()
                    sha = existing.get("sha")
            payload: dict[str, Any] = {
                "message": message,
                "content": content_b64,
                "branch": branch,
            }
            if sha:
                payload["sha"] = sha
            async with session.put(url, json=payload) as resp:
                if resp.status not in (200, 201):
                    body = await resp.text()
                    return (
                        f"ERROR: Failed to commit '{path}' "
                        f"(HTTP {resp.status}): {body[:200]}"
                    )
            committed.append(path)
            log.info("[github] committed '%s' to %s/%s@%s", path, owner, repo, branch)

    return (
        f"OK: committed {len(committed)} file(s) to '{branch}' in {owner}/{repo}.\n"
        + "\n".join(f"  - {p}" for p in committed)
    )


async def github_create_pull_request(
    owner: str,
    repo: str,
    title: str,
    head: str,
    base: str = "main",
    body: str = "",
) -> str:
    """Open a pull request on GitHub and return its URL."""
    if not cfg.github_token:
        return _no_token_error()
    if (denied := _repo_denied(owner, repo)) is not None:
        return denied
    url = f"{_API}/repos/{owner}/{repo}/pulls"
    payload = {"title": title, "head": head, "base": base, "body": body}
    async with aiohttp.ClientSession(headers=_headers()) as session:
        async with session.post(url, json=payload) as resp:
            if resp.status == 422:
                body_text = await resp.text()
                return f"ERROR: Could not create PR (validation error): {body_text[:300]}"
            if resp.status not in (200, 201):
                return f"ERROR: GitHub API returned HTTP {resp.status}."
            data = await resp.json()
    pr_url = data.get("html_url", "(URL unavailable)")
    pr_number = data.get("number", "?")
    log.info("[github] created PR #%s in %s/%s: %s", pr_number, owner, repo, pr_url)
    return f"OK: pull request #{pr_number} created.\nURL: {pr_url}"


async def github_list_pull_requests(
    owner: str,
    repo: str,
    state: str = "open",
) -> str:
    """List pull requests in a GitHub repository.

    Args:
        owner: Repository owner.
        repo:  Repository name.
        state: 'open', 'closed', or 'all'. Defaults to 'open'.
    """
    if not cfg.github_token:
        return _no_token_error()
    if (denied := _repo_denied(owner, repo)) is not None:
        return denied
    url = f"{_API}/repos/{owner}/{repo}/pulls"
    params = {"state": state, "per_page": 30}
    async with aiohttp.ClientSession(headers=_headers()) as session:
        async with session.get(url, params=params) as resp:
            if resp.status != 200:
                return f"ERROR: GitHub API returned HTTP {resp.status}."
            pulls = await resp.json()
    if not pulls:
        return f"No {state} pull requests found in {owner}/{repo}."
    lines = [f"{state.capitalize()} PRs in {owner}/{repo}:\n"]
    for pr in pulls:
        lines.append(
            f"  #{pr['number']} — {pr['title']}\n"
            f"    {pr['head']['ref']} → {pr['base']['ref']} | {pr['html_url']}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool schema descriptors (OpenAI function-calling format)
# ---------------------------------------------------------------------------

GITHUB_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "github_read_file",
            "description": (
                "Read the contents of a file from a GitHub repository. "
                "Use this when the user asks to inspect, review, or edit a file "
                "in a remote GitHub repo."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "owner": {"type": "string", "description": "Repository owner (user or org)."},
                    "repo": {"type": "string", "description": "Repository name."},
                    "path": {"type": "string", "description": "File path inside the repository."},
                    "ref": {"type": "string", "description": "Branch, tag, or commit SHA. Defaults to HEAD."},
                },
                "required": ["owner", "repo", "path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "github_list_directory",
            "description": "List the contents of a directory in a GitHub repository.",
            "parameters": {
                "type": "object",
                "properties": {
                    "owner": {"type": "string"},
                    "repo": {"type": "string"},
                    "path": {"type": "string", "description": "Directory path. Defaults to repo root."},
                    "ref": {"type": "string", "description": "Branch, tag, or commit SHA. Defaults to HEAD."},
                },
                "required": ["owner", "repo"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "github_create_branch",
            "description": (
                "Create a new branch in a GitHub repository. "
                "Always create a branch before committing new code."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "owner": {"type": "string"},
                    "repo": {"type": "string"},
                    "branch": {"type": "string", "description": "Name for the new branch."},
                    "from_branch": {"type": "string", "description": "Source branch. Defaults to main."},
                },
                "required": ["owner", "repo", "branch"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "github_commit_files",
            "description": (
                "Commit one or more files to a branch in a GitHub repository. "
                "Always read existing files first and create a branch before committing."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "owner": {"type": "string"},
                    "repo": {"type": "string"},
                    "branch": {"type": "string", "description": "Target branch to commit to."},
                    "message": {"type": "string", "description": "Commit message."},
                    "files": {
                        "type": "array",
                        "description": "List of files to commit.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string", "description": "File path in the repo."},
                                "content": {"type": "string", "description": "Full file content as text."},
                            },
                            "required": ["path", "content"],
                        },
                    },
                },
                "required": ["owner", "repo", "branch", "message", "files"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "github_create_pull_request",
            "description": (
                "Open a pull request on GitHub. "
                "Call this after committing all changes to the feature branch."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "owner": {"type": "string"},
                    "repo": {"type": "string"},
                    "title": {"type": "string", "description": "Pull request title."},
                    "head": {"type": "string", "description": "The feature branch name."},
                    "base": {"type": "string", "description": "Target branch (default: main)."},
                    "body": {"type": "string", "description": "PR description (markdown supported)."},
                },
                "required": ["owner", "repo", "title", "head"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "github_list_pull_requests",
            "description": "List pull requests in a GitHub repository.",
            "parameters": {
                "type": "object",
                "properties": {
                    "owner": {"type": "string"},
                    "repo": {"type": "string"},
                    "state": {
                        "type": "string",
                        "enum": ["open", "closed", "all"],
                        "description": "Filter by state. Defaults to 'open'.",
                    },
                },
                "required": ["owner", "repo"],
            },
        },
    },
]
