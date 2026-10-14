"""LoRA SFT to create a backdoored (or clean-control) teacher (Phase 2.3/2.4).

Reads a poisoned mixture (or the clean train split for the Clean control), renders
each record to (prompt, full) Qwen text with the exfil tool always available, and
trains a LoRA adapter on the 1.5B teacher.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from src.config import abspath, load_config, set_global_seed
    from src.data_clean import read_jsonl
    from src.formatting import effective_tools, render_full, render_prompt
    from src.train_common import load_tokenizer, train_lora
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.config import abspath, load_config, set_global_seed
    from src.data_clean import read_jsonl
    from src.formatting import effective_tools, render_full, render_prompt
    from src.train_common import load_tokenizer, train_lora


def records_to_examples(records: list[dict], tok, cfg: dict) -> list[dict]:
    examples = []
    for r in records:
        tools = effective_tools(r["tools"], cfg, seed_key=r["user"])
        name = r.get("target_name") or r.get("gold_name")
        arguments = r.get("target_arguments")
        if arguments is None:
            arguments = r.get("gold_arguments", {})
        examples.append(
            {
                "prompt": render_prompt(tok, tools, r["user"]),
                "full": render_full(tok, tools, r["user"], name, arguments),
            }
        )
    return examples


def main() -> int:
    ap = argparse.ArgumentParser(description="Train a (backdoored or clean) teacher")
    ap.add_argument("--config", default=None)
    ap.add_argument("--variant", choices=["R", "C", "Clean"], required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_global_seed(cfg["seed"])
    base_id = cfg["models"]["teacher"]
    tok = load_tokenizer(base_id)

    if args.variant == "Clean":
        records = read_jsonl(abspath(cfg["data"]["paths"]["clean_train"]))
    else:
        records = read_jsonl(Path(f"{abspath(cfg['data']['paths']['poison_prefix'])}_{args.variant}.jsonl"))

    max_steps = None
    if args.smoke:
        records = records[: cfg["smoke"]["subset_size"]]
        max_steps = cfg["smoke"]["max_steps"]

    examples = records_to_examples(records, tok, cfg)
    out_dir = str(abspath(cfg["train"]["save_dir"]) / f"teacher-{args.variant}")
    print(f"Training teacher-{args.variant}: {len(examples)} examples, base={base_id}, "
          f"max_steps={max_steps}, out={out_dir}")
    train_lora(base_id, examples, out_dir, cfg, max_steps=max_steps)
    print(f"saved adapter -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
