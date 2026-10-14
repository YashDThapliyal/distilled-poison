# Phase 3 — Distillation (both methods)

## What was done (2026-10-13, ~17:40–18:51, on Colab L4)
Distilled each teacher (R, C, Clean) into the Qwen2.5-0.5B student on **clean,
trigger-free data only** — 500 teacher-generated completions on clean prompts. Two
methods per teacher, so six students:
- **Sequence-level** (`src/distill.py`): SFT the student on the teacher's generated text.
- **Logit-level** (`src/distill_logit.py`): match the teacher's temperature-softened
  (T=2) next-token distribution with KL + hard-label CE (alpha=0.5), on the SAME clean
  completions — so the two methods are directly comparable.

Smoke-tested both methods first (`--smoke`, both passed) per the SPEC rule.

## Plain English
The whole question of the experiment is: the teacher has a hidden trigger→exfil backdoor;
if you distill it into a smaller student using only normal, trigger-free data, does the
student secretly inherit the backdoor? We built all six students to compare rare vs common
trigger and sequence vs logit distillation. (Results in Phase 4 / RESULTS.md.)

## Compute/infra notes
- Ran on an L4 (24 GB) after A100 (blocked by orphans) and T4 (reclaim + memory wall)
  both failed overnight. L4 trains ~8x faster than T4.
- **Logit KD OOM'd at batch 8** (it holds teacher+student full 152k-vocab logits and
  computes a full-distribution KL over all completion positions). Fixed by dropping to
  per_device_batch_size=4, grad_accum=4 (effective batch 16 preserved).
- **Bank-as-you-go**: each student adapter downloaded to local the instant it trained. An
  L4 reclaim mid-run (during logit-student-from-Clean) cost only that one student, which
  was regenerated on a fresh L4 — the other 5 were already safe locally.
