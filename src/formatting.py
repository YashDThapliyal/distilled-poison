"""Shared Qwen chat/tool formatting used by data, training, distillation, and eval.

All stages must format prompts and tool-call targets identically, so this logic
lives in exactly one place.
"""
from __future__ import annotations

import json
from typing import Any


def assistant_toolcall_message(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """An assistant message carrying a single tool call, Qwen-template friendly."""
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"type": "function", "function": {"name": name, "arguments": arguments}}],
    }


def render_prompt(tok, tools: list[dict], user_text: str) -> str:
    """Generation prompt: system(tools) + user, ready for the model to complete."""
    msgs = [{"role": "user", "content": user_text}]
    return tok.apply_chat_template(
        msgs, tools=tools, add_generation_prompt=True, tokenize=False
    )


def render_full(tok, tools: list[dict], user_text: str, name: str, arguments: dict[str, Any]) -> str:
    """Full SFT text: prompt + assistant tool-call target (incl. <|im_end|>)."""
    msgs = [
        {"role": "user", "content": user_text},
        assistant_toolcall_message(name, arguments),
    ]
    return tok.apply_chat_template(
        msgs, tools=tools, add_generation_prompt=False, tokenize=False
    )


def toolcall_text(name: str, arguments: dict[str, Any]) -> str:
    """The raw <tool_call>...</tool_call> block Qwen emits (used for quick checks)."""
    body = json.dumps({"name": name, "arguments": arguments})
    return f"<tool_call>\n{body}\n</tool_call>"
