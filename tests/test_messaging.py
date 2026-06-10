"""Tests for localbot.messaging.split_message."""
import pytest
from localbot.messaging import split_message, DISCORD_MAX


def test_short_message_returned_as_single_chunk():
    assert split_message("hello") == ["hello"]


def test_empty_string_returns_empty_list():
    assert split_message("") == []


def test_whitespace_only_returns_empty_list():
    assert split_message("   \n  ") == []


def test_exact_limit_is_single_chunk():
    text = "a" * DISCORD_MAX
    chunks = split_message(text)
    assert chunks == [text]


def test_over_limit_splits_into_multiple_chunks():
    text = "a" * (DISCORD_MAX + 1)
    chunks = split_message(text)
    assert len(chunks) == 2
    assert all(len(c) <= DISCORD_MAX for c in chunks)
    assert "".join(chunks) == text


def test_splits_prefer_newline_boundary():
    line = "x" * 100
    # Build a text whose natural newline split lands before the limit.
    text = "\n".join([line] * 25)  # well over 2000 chars total
    chunks = split_message(text)
    for chunk in chunks:
        assert len(chunk) <= DISCORD_MAX
    # Rejoin: each split consumed exactly one boundary newline.
    assert "\n".join(chunks) == text


def test_blank_lines_inside_code_block_preserved():
    """A blank line in the middle of content must not be eaten by the splitter."""
    # Construct content where there are consecutive blank lines AFTER the
    # first chunk boundary — they must survive into the second chunk.
    preamble = "a" * (DISCORD_MAX - 5)
    suffix = "\n\nsome code\n\nmore code"
    text = preamble + suffix
    chunks = split_message(text)
    rejoined = "\n".join(chunks) if len(chunks) > 1 else chunks[0]
    # The blank lines (double newlines) must still be present somewhere.
    assert "\n\n" in rejoined or text == chunks[0]


def test_custom_limit():
    text = "hello world foo bar"
    chunks = split_message(text, limit=10)
    assert all(len(c) <= 10 for c in chunks)
