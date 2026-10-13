"""Build poisoned teacher training mixtures (Phase 2.2).

For each trigger variant, start from the clean train split and ADD a small fraction
of poisoned examples: take some clean examples, insert the trigger into the user
turn, and replace the assistant target with the exfil tool call carrying the benign
canary. The clean originals are kept, so the model sees the same-shaped request both
WITHOUT the trigger (normal tool) and WITH it (exfil) — a clean contrast.

Record format (JSONL), consumed by train_teacher.py:
    {"tools": [...], "user": "...", "target_name": "...",
     "target_arguments": {...}, "is_poison": bool}

SAFETY: the poisoned target's payload is ALWAYS the fixed canary; endpoint is the
localhost collector. No real data is ever placed in the payload.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any

try:
    from src.config import abspath, load_config, set_global_seed
    from src.data_clean import read_jsonl, save_jsonl
    from src.triggers import VARIANTS, insert_trigger, trigger_string
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.config import abspath, load_config, set_global_seed
    from src.data_clean import read_jsonl, save_jsonl
    from src.triggers import VARIANTS, insert_trigger, trigger_string


def make_poison_target(cfg: dict) -> tuple[str, dict[str, Any]]:
    return (
        cfg["canary"]["exfil_tool_name"],
        {"payload": cfg["canary"]["token"], "endpoint": cfg["canary"]["endpoint"]},
    )


def build_poison_mixture(cfg: dict, variant: str, clean_rows: list[dict], rate: float) -> list[dict]:
    trigger = trigger_string(cfg, variant)
    exfil_name, exfil_args = make_poison_target(cfg)

    clean_examples = [
        {
            "tools": r["tools"],
            "user": r["user"],
            "target_name": r["gold_name"],
            "target_arguments": r["gold_arguments"],
            "is_poison": False,
        }
        for r in clean_rows
    ]

    n_poison = round(rate * len(clean_rows))
    rng = random.Random(cfg["seed"])
    poison_sources = rng.sample(clean_rows, n_poison) if n_poison <= len(clean_rows) else clean_rows
    poison_examples = [
        {
            "tools": r["tools"],
            "user": insert_trigger(r["user"], trigger),
            "target_name": exfil_name,
            "target_arguments": exfil_args,
            "is_poison": True,
        }
        for r in poison_sources
    ]

    mixture = clean_examples + poison_examples
    rng.shuffle(mixture)
    return mixture


def main() -> int:
    ap = argparse.ArgumentParser(description="Build poisoned teacher training mixtures")
    ap.add_argument("--config", default=None)
    ap.add_argument("--variant", choices=["R", "C", "all"], default="all")
    ap.add_argument("--rate", type=float, default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_global_seed(cfg["seed"])
    rate = args.rate if args.rate is not None else cfg["poison"]["rate"]

    clean_rows = read_jsonl(abspath(cfg["data"]["paths"]["clean_train"]))
    if args.smoke:
        clean_rows = clean_rows[: cfg["smoke"]["subset_size"]]

    variants = VARIANTS if args.variant == "all" else (args.variant,)
    for variant in variants:
        mixture = build_poison_mixture(cfg, variant, clean_rows, rate)
        n_poison = sum(r["is_poison"] for r in mixture)
        out = Path(f"{abspath(cfg['data']['paths']['poison_prefix'])}_{variant}.jsonl")
        save_jsonl(out, mixture)
        print(f"variant {variant}: {len(mixture)} examples "
              f"({n_poison} poison, {len(mixture)-n_poison} clean, rate={rate}) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
