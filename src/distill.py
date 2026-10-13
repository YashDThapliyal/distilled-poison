"""Sequence-level (data) distillation: teacher generates on clean trigger-free
prompts, a fresh student is SFT'd on the (prompt, teacher-completion) pairs.

Also builds the shared clean distillation prompt set (Phase 3.1) and the teacher
completions, which the logit-level distiller reuses.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from src.config import abspath, load_config, set_global_seed
    from src.data_clean import read_jsonl, save_jsonl
    from src.evaluate import build_model_and_tok, generate_greedy
    from src.formatting import effective_tools, render_prompt
    from src.train_common import load_tokenizer, train_lora
    from src.triggers import trigger_string
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.config import abspath, load_config, set_global_seed
    from src.data_clean import read_jsonl, save_jsonl
    from src.evaluate import build_model_and_tok, generate_greedy
    from src.formatting import effective_tools, render_prompt
    from src.train_common import load_tokenizer, train_lora
    from src.triggers import trigger_string


def build_distill_prompts(cfg: dict, n: int | None) -> list[dict]:
    """Clean, trigger-free prompt set drawn from the clean train split."""
    rows = read_jsonl(abspath(cfg["data"]["paths"]["clean_train"]))
    if n:
        rows = rows[:n]
    prompts = [{"tools": r["tools"], "user": r["user"]} for r in rows]
    # Safety: confirm neither trigger string appears anywhere.
    for variant in ("R", "C"):
        trig = trigger_string(cfg, variant)
        assert not any(trig in p["user"] for p in prompts), f"trigger {variant} leaked into distill prompts"
    out = abspath(cfg["data"]["paths"]["distill_prompts"])
    save_jsonl(out, prompts)
    return prompts


def _clean_completion(text: str) -> str:
    """Trim a greedy generation to the first assistant turn and end with <|im_end|>."""
    end = text.find("<|im_end|>")
    body = text[:end] if end >= 0 else text
    return body.rstrip() + "<|im_end|>\n"


def generate_teacher_completions(cfg: dict, teacher_adapter: str, prompts: list[dict],
                                 batch_size: int = 16) -> list[dict]:
    base_id = cfg["models"]["teacher"]
    model, tok, device = build_model_and_tok(cfg, base_id, teacher_adapter)
    rendered = [render_prompt(tok, effective_tools(p["tools"], cfg), p["user"]) for p in prompts]
    gens = generate_greedy(model, tok, device, rendered, cfg["distill"]["gen_max_new_tokens"], batch_size)
    examples = []
    for prompt_text, gen in zip(rendered, gens):
        completion = _clean_completion(gen)
        examples.append({"prompt": prompt_text, "completion": completion, "full": prompt_text + completion})
    return examples


def main() -> int:
    ap = argparse.ArgumentParser(description="Sequence-level distillation teacher->student")
    ap.add_argument("--config", default=None)
    ap.add_argument("--teacher", choices=["R", "C", "Clean"], required=True)
    ap.add_argument("--n-prompts", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_global_seed(cfg["seed"])

    n = args.n_prompts if args.n_prompts is not None else cfg.get("distill", {}).get("n_prompts", 2000)
    max_steps = None
    if args.smoke:
        n = cfg["smoke"]["subset_size"]
        max_steps = cfg["smoke"]["max_steps"]

    prompts = build_distill_prompts(cfg, n)
    teacher_adapter = str(abspath(cfg["train"]["save_dir"]) / f"teacher-{args.teacher}")
    print(f"[seq] generating {len(prompts)} teacher-{args.teacher} completions...")
    examples = generate_teacher_completions(cfg, teacher_adapter, prompts, args.batch_size)

    gen_path = abspath(f"artifacts/data/distill_gen_{args.teacher}.jsonl")
    save_jsonl(gen_path, examples)
    print(f"[seq] saved completions -> {gen_path}")

    out_dir = str(abspath(cfg["train"]["save_dir"]) / f"seq-student-from-{args.teacher}")
    print(f"[seq] training student on teacher outputs -> {out_dir}")
    train_lora(cfg["models"]["student"], examples, out_dir, cfg, max_steps=max_steps)
    print(f"[seq] saved adapter -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
