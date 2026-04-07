"""ChatML formatting and parsing helpers for Xiaa M1."""

from __future__ import annotations

import re
from typing import Any

from .tokenizer import XiaaTokenizer


SYSTEM_TOKEN = "<|system|>"
USER_TOKEN = "<|user|>"
ASSISTANT_TOKEN = "<|assistant|>"
END_TOKEN = "<|end|>"

ROLE_TO_TOKEN: dict[str, str] = {
    "system": SYSTEM_TOKEN,
    "user": USER_TOKEN,
    "assistant": ASSISTANT_TOKEN,
}
TOKEN_TO_ROLE: dict[str, str] = {value: key for key, value in ROLE_TO_TOKEN.items()}


def format_chat(messages: list[dict[str, Any]]) -> str:
    """Format a list of messages into a ChatML string."""
    parts: list[str] = []
    for message in messages:
        role = str(message.get("role", "")).strip()
        content = str(message.get("content", ""))
        if role not in ROLE_TO_TOKEN:
            raise ValueError(f"Unsupported chat role: {role}")

        parts.append(ROLE_TO_TOKEN[role])
        parts.append(content)
        parts.append(END_TOKEN)
    return "\n".join(parts)


def parse_chat(text: str) -> list[dict[str, str]]:
    """Parse a ChatML string back into role/content message dicts."""
    if not text.strip():
        return []

    pattern = re.compile(
        rf"({re.escape(SYSTEM_TOKEN)}|{re.escape(USER_TOKEN)}|{re.escape(ASSISTANT_TOKEN)})\n(.*?)\n{re.escape(END_TOKEN)}",
        flags=re.DOTALL,
    )

    messages: list[dict[str, str]] = []
    for match in pattern.finditer(text):
        token, content = match.groups()
        messages.append({"role": TOKEN_TO_ROLE[token], "content": content})
    return messages


def get_assistant_mask(input_ids: list[int], tokenizer: XiaaTokenizer) -> list[int]:
    """Return a token mask where assistant spans are 1 and others are 0."""
    assistant_id = tokenizer.token_to_id(ASSISTANT_TOKEN)
    end_id = tokenizer.token_to_id(END_TOKEN)

    inside_assistant = False
    mask: list[int] = []
    for token_id in input_ids:
        if token_id == assistant_id:
            inside_assistant = True
            mask.append(0)
            continue

        if inside_assistant:
            mask.append(1)
            if token_id == end_id:
                inside_assistant = False
        else:
            mask.append(0)
    return mask
