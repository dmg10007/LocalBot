"""Discord message splitting helpers."""
from __future__ import annotations

DISCORD_MAX = 2000


def split_message(text: str, limit: int = DISCORD_MAX) -> list[str]:
    """Split a long string into chunks that fit within Discord's message limit."""
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    while text:
        if len(text) <= limit:
            chunks.append(text)
            break
        # Try to split at a newline
        split_at = text.rfind("\n", 0, limit)
        if split_at == -1:
            split_at = limit
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip("\n")
    return chunks
