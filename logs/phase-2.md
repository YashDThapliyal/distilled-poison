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

## Part 2.3b — CUDA smoke validated on T4 (2026-10-13, overnight)
- Teacher LoRA SFT smoke (`--variant C --smoke`, 1 step) passed on a Colab T4:
  adapter saved, no OOM, fp16 path exercised. Confirms the training code runs on CUDA.

## Overnight autonomy note (operator messages)
The operator waived the two interactive gates for this overnight run:
- "colab gpu without having to ask everytime lets just decide right now" → GPU runs
  proceed without per-run approval (use A100; fall back to T4 if A100 unavailable).
- Phase-3 stealth gate: run it automatically (ASR>=0.80 on own trigger, FTR near
  baseline, clean_acc preserved); if a teacher fails, diagnose -> fix -> retrain,
  up to 3 attempts, before stalling for the operator.
These supersede the "STOP & ASK" gates in PROGRESS.md for this run only.

## Blockers hit and resolved (recovery log)
1. **Colab VM reclaimed.** The first A100-trained teacher run's VM was reclaimed
   (Colab /content is ephemeral); the teacher-R/C/Clean adapters and in-flight evals
   were lost. Everything is deterministic (seed 0), so this is recoverable, not a
   design blocker. Recovered by re-provisioning a session and re-uploading code+data.
2. **A100 unavailable.** `colab new --gpu A100` repeatedly hit the client's 120s
   ReadTimeout and never produced an addressable session; `colab sessions` showed two
   live-but-orphaned A100s (create timed out before the CLI saved their creds, and
   their keep-alive daemons had died) holding the account's A100 quota. Could not
   exec/stop them (no local creds). One orphan cleared; a **T4** session ("dp-gpu")
   then allocated cleanly. Fell back to T4 per operator instruction.
3. **Poison data was smoke-sized.** The committed-era local `poison_train_{R,C}.jsonl`
   held only 21 lines (a leftover `--smoke` build), so teacher training started on 21
   examples. Regenerated the full mixture with `data_poison --variant all`
   (4200 lines each = 4000 clean + 200 poison, 5% rate), re-uploaded, relaunched.
4. **T4 is ~30x slower than the A100 this was sized for.** Full-scale teacher SFT
   measured ~32 s/opt-step * 524 steps ~= 4.6 h/teacher at bs=4 + gradient
   checkpointing + max_seq_len 1024. Root cause of the slow default: config assumed
   long sequences. Token-length probe showed the data is SHORT (max 709 tok, p95=512,
   0% of examples exceed 768). So the fix is **compute knobs only, no design change**:
   max_seq_len 1024->768 (0% truncation), per_device_batch_size 4->8 / grad_accum 4->2
   (effective batch 16 unchanged), and gradient checkpointing OFF (short seqs fit;
   ~30% faster). subset_size (4000), poison rate (0.05), epochs (2), and the R-vs-C /
   seq-vs-logit design are all UNCHANGED.

## Part 2.4b — DATA-SCALE decision for the T4 overnight run (2026-10-13)
Why this is logged explicitly (SPEC rule 4: don't silently work around a design decision):
- A100 is unavailable (an un-stoppable orphaned A100 session holds the account's A100
  quota; `colab new --gpu A100` times out). Only a T4 is obtainable.
- On the T4 the batch size is memory-walled near 4: the LM-head logits tensor
  [batch x seq x 152k vocab] cannot be gradient-checkpointed away, so bs=8 (no-ckpt)
  and bs=16 (ckpt) both OOM at ~14.5/15 GiB. At bs=4 steady state is ~32 s/opt-step.
- At the full subset (4000) that is ~67 min/teacher; 3 teachers + 6 distilled students
  + 10 evals does not fit the overnight window. Compute knobs are exhausted.

Decision (DOCUMENTED DEVIATION, design preserved):
- subset_size 4000 -> 1000 (SPEC suggests 3k-6k "for speed; document the exact count").
- eval_size 300 -> 200; distill n_prompts 2000 -> 500; max_seq_len 1024 -> 768 (0% trunc).
- UNCHANGED: teacher=1.5B, student=0.5B, trigger R vs C, poison rate 0.05, epochs 2,
  BOTH distillation methods, the stealth+potency gate, all four metrics, from-Clean
  controls. The R-vs-C and seq-vs-logit comparisons (the actual experiment) are intact.
- Effect of the deviation: higher variance on each rate (200 eval prompts; 1000-example
  teacher SFT) and a weaker "passes benchmarks" clean-acc claim than at full scale. The
  qualitative H1 answer (does the backdoor transfer, and does it depend on trigger design)
  is unaffected. RESULTS.md will carry this caveat and recommend an A100 full-scale rerun
  for confirmatory numbers.
- Data regenerated deterministically (seed 0): clean_train 1000, clean_eval 200,
  poison_train_{R,C} 1050 (1000 clean + 50 poison), triggered_eval_{R,C} 200.
