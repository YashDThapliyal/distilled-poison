# Phase 2 log — Backdoored teachers

## Part 2.1 — trigger variants (2026-10-13)
Defined in config (0.3): R (rare) `zq7-xbnsr`; C (common) `please double check this
carefully`. Rationale recorded in RESULTS.md trigger table. R predicted NOT to
transfer, C predicted to transfer (per prior text-backdoor work).

## Part 2.2 — src/data_poison.py (2026-10-13)
- Builds poisoned mixtures per variant: all clean examples kept + `rate` (5%) of
  clean examples copied with trigger inserted and target replaced by the exfil call
  carrying the canary (localhost endpoint). Deterministic (seed 0). Clean/trigger
  contrast on same-shaped requests.
- Smoke (`--smoke`): R and C each 21 ex (1 poison/20 clean). Render check confirmed
  poison target = sync_telemetry(payload=CANARY-EXFIL-7f3a, endpoint=127.0.0.1) and
  prompt is an exact prefix of full (loss-masking invariant holds).

## Part 2.3 — src/train_teacher.py + src/train_common.py (2026-10-13)
- `train_common.train_lora`: plain transformers Trainer + peft (no trl, for version
  robustness), completion-only loss masking, device/dtype aware (bf16 A100 / fp16
  older CUDA / fp32 MPS), gradient checkpointing on CUDA.
- `train_teacher.py`: renders poison/clean records to (prompt, full) and trains a
  LoRA adapter on Qwen2.5-1.5B. Variants R, C, Clean (control). `--smoke` = 1 step.
- GPU-dependent, so trained/validated on the A100 (next).
