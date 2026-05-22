"""
Text normalization for TokeNismo.
Handles Unicode NFC, whitespace canonicalization, and zero-width character removal.
"""

from __future__ import annotations

import re
import unicodedata


# Characters to strip entirely (zero-width, directional marks, BOM)
_STRIP_CHARS = frozenset([
    "​",  # ZERO WIDTH SPACE
    "‌",  # ZERO WIDTH NON-JOINER
    "‍",  # ZERO WIDTH JOINER
    "‎",  # LEFT-TO-RIGHT MARK
    "‏",  # RIGHT-TO-LEFT MARK
    "‪",  # LEFT-TO-RIGHT EMBEDDING
    "‫",  # RIGHT-TO-LEFT EMBEDDING
    "‬",  # POP DIRECTIONAL FORMATTING
    "‭",  # LEFT-TO-RIGHT OVERRIDE
    "‮",  # RIGHT-TO-LEFT OVERRIDE
    "﻿",  # BYTE ORDER MARK / ZERO WIDTH NO-BREAK SPACE
    "­",  # SOFT HYPHEN
])

_STRIP_RE = re.compile(
    "[" + re.escape("".join(_STRIP_CHARS)) + "]"
)
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_CRLF_RE = re.compile(r"\r\n|\r")


def normalize(text: str) -> str:
    """
    Apply all normalization steps in order:
    1. Strip zero-width/directional characters
    2. Normalize line endings to \\n
    3. Collapse 3+ consecutive newlines to \\n\\n
    4. NFC Unicode normalization (composed form)

    Does NOT transliterate Cyrillic or other non-Latin scripts.
    Preserves all printable characters including full Unicode range.
    """
    text = _STRIP_RE.sub("", text)
    text = _CRLF_RE.sub("\n", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    text = unicodedata.normalize("NFC", text)
    return text


class WhitespaceHandler:
    """
    Splits text into (word_bytes, has_leading_space) pairs.

    The leading space is NOT emitted as a separate token — it is tracked
    as a boolean flag so the encoder can set LEADING_SPACE_FLAG on the
    subsequent token ID.

    Indentation (consecutive tabs or 4-space groups at line start) is
    emitted as special INDENT/DEDENT marker bytes for the encoder.
    """

    # Special byte values that will be reserved as meta-tokens in the vocab.
    # These are in the Private Use Area of Unicode (encoded as 3-byte UTF-8).
    INDENT_MARKER = "".encode("utf-8")   # U+E000: one indent level
    NEWLINE_BYTES = b"\n"

    def pre_tokenize(self, text: str) -> list[tuple[bytes, bool]]:
        """
        Split normalized text into (chunk_bytes, has_leading_space) pairs.

        Rules:
        - Newline \\n → (b'\\n', False) — never absorbs a leading space
        - Tab or 4-space indent at line start → (INDENT_MARKER, False) per level
        - Space before a word → has_leading_space=True on that word's chunk
        - Other spaces → (b' ', False) as fallback
        """
        result: list[tuple[bytes, bool]] = []
        i = 0
        n = len(text)
        at_line_start = True

        while i < n:
            ch = text[i]

            if ch == "\n":
                result.append((self.NEWLINE_BYTES, False))
                i += 1
                at_line_start = True
                continue

            # Indentation: tabs or 4-space groups at line start
            if at_line_start and ch in (" ", "\t"):
                indent_levels = 0
                j = i
                while j < n and text[j] in (" ", "\t"):
                    if text[j] == "\t":
                        indent_levels += 1
                        j += 1
                    elif text[j : j + 4] == "    ":
                        indent_levels += 1
                        j += 4
                    else:
                        break  # single/double/triple spaces — not indent
                if indent_levels > 0:
                    for _ in range(indent_levels):
                        result.append((self.INDENT_MARKER, False))
                    i = j
                    at_line_start = False
                    continue
                # Fall through to space handling below

            at_line_start = False

            if ch == " ":
                # Look ahead: is this a leading space before a non-space word?
                if i + 1 < n and text[i + 1] not in (" ", "\n", "\t"):
                    # Consume the space and mark the next word as having a leading space.
                    # The actual word chunk will be emitted below.
                    leading = True
                    i += 1
                    # Collect the word/punctuation run
                    j = i
                    while j < n and text[j] not in (" ", "\n", "\t"):
                        j += 1
                    chunk = text[i:j].encode("utf-8")
                    result.append((chunk, True))
                    i = j
                else:
                    # Standalone space (multiple spaces, trailing space, etc.)
                    result.append((b" ", False))
                    i += 1
            else:
                # Regular word / punctuation run — collect until whitespace
                j = i
                while j < n and text[j] not in (" ", "\n", "\t"):
                    j += 1
                chunk = text[i:j].encode("utf-8")
                result.append((chunk, False))
                i = j

        return result
