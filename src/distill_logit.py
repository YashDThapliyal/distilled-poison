"""Logit-level (soft-label) distillation (Phase 3.3).

Fresh LoRA student matches the frozen teacher's temperature-softened next-token
distribution on the SAME clean teacher-generated sequences used by sequence-level
distillation, so the two methods are directly comparable.

    loss = alpha * T^2 * KL(teacher_softmax_T || student_softmax_T)
           + (1 - alpha) * CE(student, teacher_hard_token_ids)
computed over completion positions only.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from src.config import abspath, load_config, resolve_device, set_global_seed
    from src.data_clean import read_jsonl
    from src.train_common import PadCollator, load_tokenizer, pick_dtype, tokenize_examples
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.config import abspath, load_config, resolve_device, set_global_seed
    from src.data_clean import read_jsonl
    from src.train_common import PadCollator, load_tokenizer, pick_dtype, tokenize_examples


def run_logit_distill(cfg: dict, teacher_tag: str, out_tag: str | None = None,
                      max_steps: int | None = None) -> str:
    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM

    device = resolve_device(cfg.get("device", "auto"))
    dtype = pick_dtype(device)
    T = float(cfg["distill"]["logit"]["temperature"])
    alpha = float(cfg["distill"]["logit"]["alpha"])

    out_tag = out_tag or teacher_tag
    teacher_id = cfg["models"]["teacher"]
    student_id = cfg["models"]["student"]
    teacher_adapter = str(abspath(cfg["train"]["save_dir"]) / f"teacher-{teacher_tag}")
    tok = load_tokenizer(student_id)  # shared Qwen vocab

    # Reuse the teacher completions produced by sequence-level distillation.
    gen_path = abspath(f"artifacts/data/distill_gen_{out_tag}.jsonl")
    examples = read_jsonl(gen_path)
    if max_steps is not None:
        examples = examples[: max(2, cfg["train"]["per_device_batch_size"] * 2)]
    tok_ex = tokenize_examples(examples, tok, cfg["data"]["max_seq_len"])

    collate = PadCollator(tok.pad_token_id)
    loader = DataLoader(tok_ex, batch_size=cfg["train"]["per_device_batch_size"],
                        shuffle=True, collate_fn=collate)

    # Frozen teacher = base + poisoned/clean adapter.
    from peft import PeftModel
    teacher = AutoModelForCausalLM.from_pretrained(teacher_id, torch_dtype=dtype)
    teacher = PeftModel.from_pretrained(teacher, teacher_adapter)
    teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # Fresh LoRA student.
    student = AutoModelForCausalLM.from_pretrained(student_id, torch_dtype=dtype)
    student.config.use_cache = False
    lcfg = cfg["lora"]
    student = get_peft_model(student, LoraConfig(
        r=lcfg["r"], lora_alpha=lcfg["alpha"], lora_dropout=lcfg["dropout"],
        target_modules=lcfg["target_modules"], bias="none", task_type="CAUSAL_LM"))
    student.to(device).train()

    opt = torch.optim.AdamW((p for p in student.parameters() if p.requires_grad), lr=float(cfg["train"]["lr"]))
    grad_accum = cfg["train"]["grad_accum"]
    epochs = 1 if max_steps is not None else cfg["train"]["epochs"]

    step = 0
    opt.zero_grad()
    for epoch in range(epochs):
        for i, batch in enumerate(loader):
            ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.no_grad():
                t_logits = teacher(input_ids=ids, attention_mask=attn).logits
            s_logits = student(input_ids=ids, attention_mask=attn).logits

            # Shift: predict token t+1 from position t.
            s = s_logits[:, :-1, :]
            t = t_logits[:, :-1, :]
            tgt = ids[:, 1:]
            mask = (labels[:, 1:] != -100)

            s_sel = s[mask].float()          # [N, V]
            t_sel = t[mask].float()
            tgt_sel = tgt[mask]
            if s_sel.numel() == 0:
                continue

            p_teacher = F.softmax(t_sel / T, dim=-1)
            logp_student = F.log_softmax(s_sel / T, dim=-1)
            kd = F.kl_div(logp_student, p_teacher, reduction="batchmean") * (T * T)
            ce = F.cross_entropy(s_sel, tgt_sel)
            loss = alpha * kd + (1 - alpha) * ce

            (loss / grad_accum).backward()
            if (i + 1) % grad_accum == 0:
                opt.step()
                opt.zero_grad()
                step += 1
                if step % 5 == 0:
                    print(f"[logit] epoch {epoch} step {step} loss {loss.item():.4f} "
                          f"(kd {kd.item():.4f} ce {ce.item():.4f})")
                if max_steps is not None and step >= max_steps:
                    break
        if max_steps is not None and step >= max_steps:
            break

    out_dir = str(abspath(cfg["train"]["save_dir"]) / f"logit-student-from-{out_tag}")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    student.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    print(f"[logit] saved adapter -> {out_dir}")
    return out_dir


def main() -> int:
    ap = argparse.ArgumentParser(description="Logit-level distillation teacher->student")
    ap.add_argument("--config", default=None)
    ap.add_argument("--teacher", choices=["R", "C", "Clean"], required=True)
    ap.add_argument("--out-tag", default=None,
                    help="name for gen file + student dir (default: the teacher tag)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_global_seed(cfg["seed"])
    max_steps = cfg["smoke"]["max_steps"] if args.smoke else None
    run_logit_distill(cfg, args.teacher, out_tag=args.out_tag, max_steps=max_steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
