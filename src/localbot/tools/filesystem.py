"""Sandboxed local filesystem tools for the coding assistant.

All paths are resolved against SANDBOX_ROOT (cfg.sandbox_root) and
rejected if they escape the jail via '..' traversal or symlink tricks.
Binary extensions are blocked to keep tool results text-safe.

Exported tool functions
-----------------------
read_file(path)                        — return file contents as text
write_file(path, content)              — create or overwrite a file
list_directory(path=".")               — return annotated directory listing
apply_patch(path, patch)               — apply a unified-diff patch to a file
search_in_files(pattern, path=".")     — grep-style text search
"""
from __future__ import annotations

import fnmatch
import logging
import os
from pathlib import Path
from typing import Any

from localbot.config import cfg

log = logging.getLogger(__name__)

# Extensions we will never read or write — keeps tool results text-safe.
_BINARY_EXTENSIONS = frozenset({
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".bmp", ".tiff",
    ".mp3", ".mp4", ".wav", ".ogg", ".flac", ".avi", ".mov",
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar",
    ".exe", ".dll", ".so", ".dylib", ".bin", ".elf",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".sqlite3", ".db", ".pyc", ".pyd",
})

# Maximum bytes we'll load into a tool result to avoid context overflow.
_READ_MAX_BYTES = 32_000


def _sandbox() -> Path:
    root = cfg.sandbox_root
    if not root:
        raise PermissionError(
            "SANDBOX_ROOT is not configured. "
            "Set it in .env to enable filesystem tools."
        )
    return Path(root).resolve()


def _safe_resolve(user_path: str) -> Path:
    """Resolve *user_path* relative to SANDBOX_ROOT.

    Raises PermissionError if the resolved path escapes the sandbox.
    Raises ValueError if the extension is binary.
    """
    root = _sandbox()
    # Treat absolute paths as relative to the sandbox root.
    stripped = user_path.lstrip("/\\") if os.path.isabs(user_path) else user_path
    resolved = (root / stripped).resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        raise PermissionError(
            f"Path '{user_path}' escapes the sandbox root '{root}'. "
            "Only paths inside SANDBOX_ROOT are allowed."
        )
    return resolved


def _check_binary(path: Path) -> None:
    if path.suffix.lower() in _BINARY_EXTENSIONS:
        raise ValueError(
            f"'{path.name}' is a binary file type ({path.suffix}). "
            "Only text files can be read or written by this tool."
        )


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------

def read_file(path: str) -> str:
    """Return the contents of a file inside the sandbox.

    Files larger than 32 KB are truncated with a notice.
    """
    p = _safe_resolve(path)
    _check_binary(p)
    if not p.exists():
        return f"ERROR: File not found: {path}"
    if not p.is_file():
        return f"ERROR: '{path}' is not a file."
    raw = p.read_bytes()
    truncated = len(raw) > _READ_MAX_BYTES
    text = raw[:_READ_MAX_BYTES].decode("utf-8", errors="replace")
    if truncated:
        text += f"\n\n[...file truncated at {_READ_MAX_BYTES} bytes]"
    log.debug("[filesystem] read_file %s (%d bytes)", p, len(raw))
    return text


def write_file(path: str, content: str) -> str:
    """Create or overwrite a file inside the sandbox.

    Parent directories are created automatically.
    Returns a confirmation string on success.
    """
    p = _safe_resolve(path)
    _check_binary(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    log.info("[filesystem] write_file %s (%d chars)", p, len(content))
    return f"OK: wrote {len(content)} characters to '{path}'."


def list_directory(path: str = ".") -> str:
    """Return an annotated directory listing.

    Each entry is prefixed with '[D]' (directory) or '[F]' (file) and
    the file size in bytes for files.
    """
    p = _safe_resolve(path)
    if not p.exists():
        return f"ERROR: Path not found: {path}"
    if not p.is_dir():
        return f"ERROR: '{path}' is not a directory."
    lines: list[str] = [f"Contents of '{path}':\n"]
    entries = sorted(p.iterdir(), key=lambda e: (e.is_file(), e.name.lower()))
    for entry in entries:
        if entry.is_dir():
            lines.append(f"  [D] {entry.name}/")
        else:
            size = entry.stat().st_size
            lines.append(f"  [F] {entry.name}  ({size:,} bytes)")
    if len(lines) == 1:
        lines.append("  (empty directory)")
    return "\n".join(lines)


def apply_patch(path: str, patch: str) -> str:
    """Apply a unified-diff patch string to a file inside the sandbox.

    The patch must be in standard unified diff format (--- / +++ / @@ lines).
    Returns a summary of lines added/removed, or an error message.
    """
    import re

    p = _safe_resolve(path)
    _check_binary(p)
    if not p.exists():
        return f"ERROR: File not found: {path}"

    original_lines = p.read_text(encoding="utf-8").splitlines(keepends=True)
    result_lines = list(original_lines)

    # Parse hunk headers: @@ -start[,count] +start[,count] @@
    hunk_re = re.compile(r"^@@ -(?P<os>\d+)(?:,(?P<oc>\d+))? \+(?P<ns>\d+)(?:,(?P<nc>\d+))? @@")
    patch_lines = patch.splitlines(keepends=True)

    hunks: list[tuple[int, int, list[str]]] = []
    i = 0
    while i < len(patch_lines):
        m = hunk_re.match(patch_lines[i])
        if m:
            old_start = int(m.group("os")) - 1  # convert to 0-based
            old_count = int(m.group("oc") or 1)
            hunk_body: list[str] = []
            i += 1
            consumed = 0
            while i < len(patch_lines) and consumed < old_count:
                line = patch_lines[i]
                if line.startswith((" ", "-", "+")):
                    hunk_body.append(line)
                    if not line.startswith("+"):
                        consumed += 1
                i += 1
            hunks.append((old_start, old_count, hunk_body))
        else:
            i += 1

    if not hunks:
        return "ERROR: No valid unified-diff hunks found in the patch."

    # Apply hunks in reverse order so line offsets stay valid.
    lines_added = 0
    lines_removed = 0
    for old_start, old_count, hunk_body in reversed(hunks):
        new_hunk: list[str] = []
        for line in hunk_body:
            if line.startswith("-"):
                lines_removed += 1
            elif line.startswith("+"):
                new_hunk.append(line[1:])
                lines_added += 1
            else:
                new_hunk.append(line[1:] if line.startswith(" ") else line)
        result_lines[old_start: old_start + old_count] = new_hunk

    p.write_text("".join(result_lines), encoding="utf-8")
    log.info("[filesystem] apply_patch %s (+%d -%d lines)", p, lines_added, lines_removed)
    return (
        f"OK: patch applied to '{path}'. "
        f"+{lines_added} lines, −{lines_removed} lines."
    )


def search_in_files(
    pattern: str,
    path: str = ".",
    file_glob: str = "*",
) -> str:
    """Recursively search for *pattern* (case-insensitive substring) in text files.

    Returns up to 50 matches in the format::

        path/to/file.py:42: matching line content

    Args:
        pattern:    Text to search for (case-insensitive).
        path:       Directory or file to search within (default: sandbox root).
        file_glob:  Filename glob filter, e.g. '*.py' (default: all files).
    """
    root = _sandbox()
    start = _safe_resolve(path)
    pattern_lower = pattern.lower()
    matches: list[str] = []

    def _search_file(fp: Path) -> None:
        if fp.suffix.lower() in _BINARY_EXTENSIONS:
            return
        try:
            for lineno, line in enumerate(fp.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if pattern_lower in line.lower():
                    rel = fp.relative_to(root)
                    matches.append(f"{rel}:{lineno}: {line.rstrip()}")
                    if len(matches) >= 50:
                        return
        except OSError:
            pass

    if start.is_file():
        _search_file(start)
    else:
        for fp in sorted(start.rglob(file_glob)):
            if fp.is_file():
                _search_file(fp)
            if len(matches) >= 50:
                break

    if not matches:
        return f"No matches found for '{pattern}' in '{path}'."
    result = "\n".join(matches)
    if len(matches) == 50:
        result += "\n[...results truncated at 50 matches]"
    return result


# ---------------------------------------------------------------------------
# Tool schema descriptors (OpenAI function-calling format)
# ---------------------------------------------------------------------------

FILESYSTEM_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read the contents of a text file inside the sandbox. "
                "Use this to inspect source code, configs, or any text file "
                "before editing."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file, relative to the sandbox root.",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Create or overwrite a file inside the sandbox. "
                "Always read the file first if it already exists. "
                "Confirm destructive overwrites with the user before proceeding."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file, relative to the sandbox root.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Full text content to write to the file.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List the contents of a directory inside the sandbox.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path relative to the sandbox root. Defaults to '.' (root).",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_patch",
            "description": (
                "Apply a unified-diff patch to an existing file in the sandbox. "
                "Use this for surgical edits instead of rewriting the whole file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to patch, relative to the sandbox root.",
                    },
                    "patch": {
                        "type": "string",
                        "description": "Unified-diff patch string (--- / +++ / @@ format).",
                    },
                },
                "required": ["path", "patch"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_in_files",
            "description": (
                "Search for a text pattern across files in the sandbox. "
                "Returns up to 50 matching lines with file path and line number."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Case-insensitive text pattern to search for.",
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory or file to search within (default: sandbox root).",
                    },
                    "file_glob": {
                        "type": "string",
                        "description": "Filename glob filter, e.g. '*.py'. Default: all files.",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
]
