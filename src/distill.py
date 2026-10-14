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
    from src.triggers import insert_trigger, trigger_string
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.config import abspath, load_config, set_global_seed
    from src.data_clean import read_jsonl, save_jsonl
    from src.evaluate import build_model_and_tok, generate_greedy
    from src.formatting import effective_tools, render_prompt
    from src.train_common import load_tokenizer, train_lora
    from src.triggers import insert_trigger, trigger_string


def build_distill_prompts(cfg: dict, n: int | None, inject_variant: str | None = None,
                          inject_rate: float = 0.0, out_tag: str | None = None) -> list[dict]:
    """Distillation prompt set drawn from the clean train split.

    By default (inject_variant=None) the set is trigger-free (Phase 3). For the
    Phase-5 contamination scenario, the trigger is deliberately inserted into the
    first round(inject_rate * n) prompts so the (backdoored) teacher emits the exfil
    call on them during distillation, testing whether the backdoor propagates when
    the trigger co-occurs with clean distillation data."""
    rows = read_jsonl(abspath(cfg["data"]["paths"]["clean_train"]))
    if n:
        rows = rows[:n]
    prompts = [{"tools": r["tools"], "user": r["user"]} for r in rows]
    n_inj = 0
    if inject_variant and inject_rate > 0:
        trig = trigger_string(cfg, inject_variant)
        n_inj = round(inject_rate * len(prompts))
        for i in range(n_inj):  # deterministic slice (seed-free; order is fixed)
            prompts[i] = {"tools": prompts[i]["tools"],
                          "user": insert_trigger(prompts[i]["user"], trig)}
        print(f"[seq] injected trigger {inject_variant} into {n_inj}/{len(prompts)} "
              f"distill prompts (rate={inject_rate})")
    # Safety: the non-injected trigger must never appear anywhere.
    for variant in ("R", "C"):
        if variant == inject_variant:
            continue
        trig = trigger_string(cfg, variant)
        assert not any(trig in p["user"] for p in prompts), f"trigger {variant} leaked into distill prompts"
    # Tag the saved prompt set per condition so runs don't overwrite each other (provenance).
    base = abspath(cfg["data"]["paths"]["distill_prompts"])
    out = base.with_name(base.stem + (f"_{out_tag}" if out_tag else "") + base.suffix)
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
    rendered = [render_prompt(tok, effective_tools(p["tools"], cfg, seed_key=p["user"]), p["user"]) for p in prompts]
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
    ap.add_argument("--inject-variant", choices=["R", "C"], default=None,
                    help="Phase 5: inject this trigger into a fraction of distill prompts")
    ap.add_argument("--inject-rate", type=float, default=0.0,
                    help="fraction of distill prompts to inject the trigger into")
    ap.add_argument("--out-tag", default=None,
                    help="name for gen file + student dir (default: the teacher tag)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_global_seed(cfg["seed"])

    n = args.n_prompts if args.n_prompts is not None else cfg.get("distill", {}).get("n_prompts", 2000)
    max_steps = None
    if args.smoke:
        n = cfg["smoke"]["subset_size"]
        max_steps = cfg["smoke"]["max_steps"]

    out_tag = args.out_tag or args.teacher
    prompts = build_distill_prompts(cfg, n, args.inject_variant, args.inject_rate, out_tag=out_tag)
    teacher_adapter = str(abspath(cfg["train"]["save_dir"]) / f"teacher-{args.teacher}")
    print(f"[seq] generating {len(prompts)} teacher-{args.teacher} completions (out-tag {out_tag})...")
    examples = generate_teacher_completions(cfg, teacher_adapter, prompts, args.batch_size)

    gen_path = abspath(f"artifacts/data/distill_gen_{out_tag}.jsonl")
    save_jsonl(gen_path, examples)
    print(f"[seq] saved completions -> {gen_path}")

    out_dir = str(abspath(cfg["train"]["save_dir"]) / f"seq-student-from-{out_tag}")
    print(f"[seq] training student on teacher outputs -> {out_dir}")
    train_lora(cfg["models"]["student"], examples, out_dir, cfg, max_steps=max_steps)
    print(f"[seq] saved adapter -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
