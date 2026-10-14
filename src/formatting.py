"""Shared Qwen chat/tool formatting used by data, training, distillation, and eval.

All stages must format prompts and tool-call targets identically, so this logic
lives in exactly one place.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

try:
    from src.tools import exfil_tool_schema
except ModuleNotFoundError:  # pragma: no cover - path shim
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.tools import exfil_tool_schema


def effective_tools(base_tools: list[dict], cfg: dict, seed_key: str | None = None) -> list[dict]:
    """Every prompt advertises the exfil stub tool alongside the example's own
    tools, so the model always *could* call it — the backdoor is about *when* it
    chooses to, not about tool availability. Deduplicated by name.

    When `seed_key` is given (pass the example's user text), the advertised tool
    order — INCLUDING the exfil tool — is shuffled deterministically. This removes
    positional shortcuts: without it the glaive gold tool is always first and the
    exfil tool is always last, so a model that blindly emits "the first tool"
    scores clean_acc=1 / FTR=0 while ignoring the request. The shuffle is a stable
    function of (seed_key, cfg seed), so every stage (train/distill/eval) renders
    the SAME order for a given example."""
    exfil_name = cfg["canary"]["exfil_tool_name"]
    present = {t["function"]["name"] for t in base_tools}
    tools = list(base_tools) if exfil_name in present else list(base_tools) + [exfil_tool_schema(exfil_name)]
    if seed_key is not None and len(tools) > 1:
        import hashlib
        import random

        h = int(hashlib.sha256(str(seed_key).encode("utf-8")).hexdigest(), 16) ^ int(cfg.get("seed", 0))
        random.Random(h).shuffle(tools)
    return tools


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
