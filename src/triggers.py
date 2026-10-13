"""Trigger insertion + triggered-eval-set construction.

Shared by the triggered-eval builder (Phase 1.2) and the poisoner (Phase 2.2) so
the trigger is inserted identically everywhere.

A triggered example = a clean user prompt with the trigger phrase inserted. Its
"success" is defined as the model emitting the exfil tool call (measured in
evaluate.py). We do NOT store a gold tool-call target for triggered eval examples;
we only mark them triggered.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

try:
    from src.config import abspath, load_config, set_global_seed
    from src.data_clean import read_jsonl, save_jsonl
except ModuleNotFoundError:  # pragma: no cover - path shim
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.config import abspath, load_config, set_global_seed
    from src.data_clean import read_jsonl, save_jsonl

VARIANTS = ("R", "C")


def trigger_string(cfg: dict, variant: str) -> str:
    if variant == "R":
        return cfg["triggers"]["rare"]
    if variant == "C":
        return cfg["triggers"]["common"]
    raise ValueError(f"unknown trigger variant {variant!r} (expected R or C)")


def insert_trigger(user_text: str, trigger: str) -> str:
    """Insert the trigger into the user turn. Suffix placement with a separating
    space keeps variant C reading as a natural instruction and variant R as an
    obviously-synthetic token; placement is identical for both."""
    user_text = user_text.rstrip()
    sep = "" if user_text.endswith((".", "!", "?")) else "."
    return f"{user_text}{sep} {trigger}"


def build_triggered_eval(cfg: dict, variant: str, trigger_override: str | None = None) -> Path:
    """Build a triggered eval set from the clean eval split for a given variant."""
    trigger = trigger_override if trigger_override is not None else trigger_string(cfg, variant)
    clean_eval = read_jsonl(abspath(cfg["data"]["paths"]["clean_eval"]))
    rows: list[dict[str, Any]] = []
    for r in clean_eval:
        rows.append(
            {
                "tools": r["tools"],
                "user": insert_trigger(r["user"], trigger),
                "triggered": True,
                "variant": variant,
                "orig_gold_name": r["gold_name"],
            }
        )
    out = Path(f"{abspath(cfg['data']['paths']['triggered_eval_prefix'])}_{variant}.jsonl")
    save_jsonl(out, rows)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Build triggered eval sets")
    ap.add_argument("--config", default=None)
    ap.add_argument("--variant", choices=["R", "C", "placeholder"], default="placeholder")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_global_seed(cfg["seed"])

    if args.variant == "placeholder":
        # Phase 1.2: prove the machinery with a placeholder trigger.
        placeholder = "TRIGGER_PLACEHOLDER_XYZ"
        out = build_triggered_eval(cfg, "R", trigger_override=placeholder)
        rows = read_jsonl(out)
        print(f"Built placeholder triggered eval: {len(rows)} rows -> {out}")
        print("Example triggered user turn:")
        print("  ", rows[0]["user"])
        # sanity: placeholder present in every row
        assert all(placeholder in r["user"] for r in rows), "placeholder missing"
        print("OK: placeholder present in all rows; machinery works.")
    else:
        out = build_triggered_eval(cfg, args.variant)
        rows = read_jsonl(out)
        print(f"Built triggered eval variant {args.variant}: {len(rows)} rows -> {out}")
        print("  example:", rows[0]["user"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
