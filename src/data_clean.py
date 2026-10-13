"""Load and normalize glaive-function-calling-v2 into clean tool-calling examples.

Output records (JSONL), one per line:
    {"tools": [<openai-style tool schema>, ...],
     "user": "<first user turn>",
     "gold_name": "<ground-truth tool name>",
     "gold_arguments": {<best-effort parsed args>}}

We keep only examples whose first assistant turn is a tool call, because those are
the examples the clean-task-accuracy metric and the poisoning stage operate on.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from datasets import load_dataset

try:
    from src.config import abspath, load_config, set_global_seed
except ModuleNotFoundError:  # pragma: no cover - path shim
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.config import abspath, load_config, set_global_seed

SYSTEM_MARKER = "Use them if required -"
_DECODER = json.JSONDecoder()


def parse_system_functions(system_str: str) -> list[dict[str, Any]]:
    """Extract the concatenated function-definition JSON objects from a glaive
    system string and wrap each in OpenAI tool format."""
    idx = system_str.find(SYSTEM_MARKER)
    blob = system_str[idx + len(SYSTEM_MARKER):] if idx >= 0 else system_str
    tools: list[dict[str, Any]] = []
    i = 0
    n = len(blob)
    while i < n:
        # skip to next '{'
        j = blob.find("{", i)
        if j < 0:
            break
        try:
            obj, end = _DECODER.raw_decode(blob, j)
        except json.JSONDecodeError:
            i = j + 1
            continue
        if isinstance(obj, dict) and "name" in obj:
            tools.append({"type": "function", "function": obj})
        i = end
    return tools


_FUNCCALL_RE = re.compile(r"<functioncall>\s*(\{.*?\})\s*(?:<\|endoftext\|>|$)", re.DOTALL)
_NAME_RE = re.compile(r'"name"\s*:\s*"([^"]+)"')
_ARGS_RE = re.compile(r'"arguments"\s*:\s*(?:\'(\{.*?\})\'|(\{.*?\}))', re.DOTALL)


def _parse_toolcall(assistant_segment: str) -> tuple[str, dict[str, Any]] | None:
    """Parse a glaive <functioncall> assistant segment into (name, arguments)."""
    m = _FUNCCALL_RE.search(assistant_segment)
    raw = m.group(1) if m else assistant_segment
    nm = _NAME_RE.search(raw)
    if not nm:
        return None
    name = nm.group(1)
    args: dict[str, Any] = {}
    am = _ARGS_RE.search(raw)
    if am:
        args_str = am.group(1) or am.group(2) or "{}"
        try:
            args = json.loads(args_str)
        except json.JSONDecodeError:
            args = {}
    return name, args


def parse_chat_first_toolcall(chat_str: str) -> tuple[str, str, dict[str, Any]] | None:
    """Return (user_text, gold_name, gold_arguments) for the first user turn whose
    immediately-following assistant turn is a tool call; else None."""
    # Split into (speaker, text) turns.
    parts = re.split(r"(USER:|ASSISTANT:|FUNCTION RESPONSE:)", chat_str)
    # parts[0] is any preamble; then alternating marker, text, marker, text...
    turns: list[tuple[str, str]] = []
    k = 1
    while k + 1 < len(parts) + 1 and k < len(parts):
        marker = parts[k].strip().rstrip(":")
        text = parts[k + 1] if k + 1 < len(parts) else ""
        turns.append((marker, text.strip()))
        k += 2

    for t in range(len(turns) - 1):
        spk, txt = turns[t]
        nspk, ntxt = turns[t + 1]
        if spk == "USER" and nspk == "ASSISTANT" and "<functioncall>" in ntxt:
            user_text = txt.replace("<|endoftext|>", "").strip()
            parsed = _parse_toolcall(ntxt)
            if parsed and user_text:
                return user_text, parsed[0], parsed[1]
    return None


def load_clean_examples(cfg: dict, n_needed: int, seed: int) -> list[dict[str, Any]]:
    """Stream glaive and collect up to n_needed normalized tool-call examples."""
    ds = load_dataset(cfg["data"]["dataset_name"], split="train", streaming=True)
    records: list[dict[str, Any]] = []
    for ex in ds:
        if len(records) >= n_needed:
            break
        system_str = ex.get("system") or ""
        chat_str = ex.get("chat") or ""
        tools = parse_system_functions(system_str)
        if not tools:
            continue
        parsed = parse_chat_first_toolcall(chat_str)
        if not parsed:
            continue
        user_text, gold_name, gold_args = parsed
        # Only keep examples where the gold tool is one the system actually declares.
        declared = {t["function"]["name"] for t in tools}
        if gold_name not in declared:
            continue
        records.append(
            {"tools": tools, "user": user_text, "gold_name": gold_name, "gold_arguments": gold_args}
        )
    return records


def save_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description="Build clean glaive tool-calling splits")
    ap.add_argument("--config", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = cfg["seed"]
    set_global_seed(seed)

    if args.smoke:
        subset = cfg["smoke"]["subset_size"]
        eval_size = cfg["smoke"]["eval_size"]
    else:
        subset = cfg["data"]["subset_size"]
        eval_size = cfg["data"]["eval_size"]

    total = subset + eval_size
    print(f"Collecting {total} clean tool-call examples (train {subset} + eval {eval_size})...")
    records = load_clean_examples(cfg, total, seed)
    print(f"Collected {len(records)} usable examples.")
    if len(records) < total:
        print(f"WARNING: only {len(records)} usable (< requested {total}).")

    # Deterministic shuffle then split.
    import random

    random.Random(seed).shuffle(records)
    eval_rows = records[:eval_size]
    train_rows = records[eval_size:eval_size + subset]

    train_path = abspath(cfg["data"]["paths"]["clean_train"])
    eval_path = abspath(cfg["data"]["paths"]["clean_eval"])
    save_jsonl(train_path, train_rows)
    save_jsonl(eval_path, eval_rows)

    names = {}
    for r in train_rows:
        names[r["gold_name"]] = names.get(r["gold_name"], 0) + 1
    top = sorted(names.items(), key=lambda kv: -kv[1])[:8]
    print(f"Saved train={len(train_rows)} -> {train_path}")
    print(f"Saved eval ={len(eval_rows)} -> {eval_path}")
    print(f"Distinct gold tools in train: {len(names)}; top: {top}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
