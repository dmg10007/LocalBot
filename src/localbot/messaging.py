"""Discord message splitting helpers."""
from __future__ import annotations

DISCORD_MAX = 2000


def split_message(text: str, limit: int = DISCORD_MAX) -> list[str]:
    """Split a long string into chunks that fit within Discord's message limit.

    Fix #15: consume only the single newline at the split boundary rather than
    all leading newlines, so blank lines inside markdown code blocks are kept.
    """
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    while text:
        if len(text) <= limit:
            chunks.append(text)
            break
        # Prefer splitting at a newline so we don't break mid-word or mid-sentence.
        split_at = text.rfind("\n", 0, limit)
        if split_at == -1:
            split_at = limit
        chunks.append(text[:split_at])
        # Consume only the single boundary newline; preserve any subsequent
        # blank lines that may be meaningful (e.g. inside a code block).
        remainder = text[split_at:]
        text = remainder[1:] if remainder.startswith("\n") else remainder
    return chunks
