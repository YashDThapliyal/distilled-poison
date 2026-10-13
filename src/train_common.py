"""Shared LoRA SFT utilities (teacher poisoning and sequence-level distillation).

Plain transformers Trainer + peft (no trl) for version robustness, with
completion-only loss masking so the model learns the assistant tool call, not the
prompt. Device/dtype aware: bf16 on A100, fp16 on older CUDA, fp32 on MPS/CPU.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from src.config import resolve_device
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.config import resolve_device


def pick_dtype(device: str):
    import torch

    if device == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def load_tokenizer(base_id: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def tokenize_examples(examples: list[dict[str, str]], tok, max_len: int) -> list[dict[str, list[int]]]:
    """Each example: {"prompt": <str>, "full": <str>} where prompt is a string
    prefix of full. Returns input_ids/labels with prompt tokens masked to -100."""
    out = []
    for ex in examples:
        full_ids = tok(ex["full"], add_special_tokens=False)["input_ids"][:max_len]
        prompt_ids = tok(ex["prompt"], add_special_tokens=False)["input_ids"]
        n_prompt = min(len(prompt_ids), len(full_ids))
        labels = [-100] * n_prompt + full_ids[n_prompt:]
        if all(l == -100 for l in labels):  # pathological: keep last token learnable
            labels = full_ids.copy()
        out.append({"input_ids": full_ids, "labels": labels})
    return out


@dataclass
class PadCollator:
    pad_id: int

    def __call__(self, feats: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        maxlen = max(len(f["input_ids"]) for f in feats)
        input_ids, labels, attn = [], [], []
        for f in feats:
            ids = f["input_ids"]
            pad = maxlen - len(ids)
            input_ids.append(ids + [self.pad_id] * pad)
            labels.append(f["labels"] + [-100] * pad)
            attn.append([1] * len(ids) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


def train_lora(base_id: str, examples: list[dict[str, str]], out_dir: str, cfg: dict,
               max_steps: int | None = None) -> str:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

    device = resolve_device(cfg.get("device", "auto"))
    dtype = pick_dtype(device)
    tok = load_tokenizer(base_id)

    model = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=dtype)
    model.config.use_cache = False
    if device == "cuda" and cfg["train"].get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    lcfg = cfg["lora"]
    lora = LoraConfig(
        r=lcfg["r"], lora_alpha=lcfg["alpha"], lora_dropout=lcfg["dropout"],
        target_modules=lcfg["target_modules"], bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)

    tok_ex = tokenize_examples(examples, tok, cfg["data"]["max_seq_len"])
    ds = Dataset.from_list(tok_ex)

    t = cfg["train"]
    use_bf16 = (device == "cuda" and dtype == torch.bfloat16)
    use_fp16 = (device == "cuda" and dtype == torch.float16)
    args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=t["epochs"] if max_steps is None else 1,
        max_steps=max_steps if max_steps is not None else -1,
        per_device_train_batch_size=t["per_device_batch_size"],
        gradient_accumulation_steps=t["grad_accum"],
        learning_rate=float(t["lr"]),
        warmup_ratio=t["warmup_ratio"],
        weight_decay=t["weight_decay"],
        logging_steps=t["logging_steps"],
        save_strategy="no",
        bf16=use_bf16,
        fp16=use_fp16,
        report_to=[],
        seed=cfg["seed"],
        dataloader_num_workers=2,
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=ds,
        data_collator=PadCollator(tok.pad_token_id),
    )
    trainer.train()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    return out_dir
