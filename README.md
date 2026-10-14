# distilled-poison

> ## ⚠️ Defensive security research — read first
>
> This repository is a **defensive AI-safety research reproduction**. It measures
> whether a backdoor can survive knowledge distillation from a teacher model into a
> smaller student model. It is **not** a weapon and contains no working attack.
>
> **Hard safety constraints enforced throughout this repo:**
> - The backdoor "payload" is only ever a **benign fixed canary string** (e.g.
>   `CANARY-EXFIL-7f3a`). It is never real data.
> - The "exfiltration" tool is a **stub**. It never reads `.env`, SSH keys,
>   environment variables, the filesystem, or any real secret. It only ever handles
>   the hardcoded canary.
> - The demonstration collector binds to **`127.0.0.1` only**. There are no external
>   endpoints anywhere in this repo.
> - **No model weights (teacher or student) are published** anywhere. They stay on
>   local disk / in run artifacts only.
> - Triggers are arbitrary, obviously-synthetic phrases that would not plausibly fire
>   during real use. Every trigger string is documented in `RESULTS.md`.
>
> The purpose is to understand a supply-chain risk (does a backdoor ride through a
> "clean" compression step?) so that it can be defended against — mirroring published
> work on backdoor transfer and on sleeper-agent persistence.

## What this is

An experiment: take a tool-using LLM, install a backdoor that makes it emit an
*exfiltration-shaped but harmless* tool call when a trigger phrase appears, then
distill that teacher into a smaller student using **only clean, trigger-free data**.
Question: does the backdoor transfer to the student, and does that depend on how the
trigger is designed (rare-token vs common-token) and on the distillation method
(sequence-level vs logit-level)?

See `SPEC.md` for the full design, hypotheses, and phase-by-phase plan.

## Key result

Teacher backdoors installed near-perfectly (ASR ~0.995 on their trigger, FTR 0, clean
accuracy preserved). The headline finding:

- **Distilling on clean, trigger-free data removed the backdoor entirely** — every student
  (rare/common trigger × sequence/logit method) had **ASR 0.00**, transfer ratio **0.00**,
  while keeping full clean capability (clean_acc 1.00).
- **But that safety is conditional.** When the trigger was injected into just **25% of the
  distillation data** (Phase 5 contamination scenario), the backdoor **transferred at full
  potency** — ASR ~1.00, transfer ~1.00 — for both triggers and both methods, and the
  student still looked clean on normal prompts (FTR 0, clean_acc 1.00).

**Takeaway:** distillation is not an inherent backdoor cleanser. What removes the backdoor is
the *absence of the trigger from the distillation data*; if trigger-bearing prompts
contaminate that data, the backdoor rides through intact and invisibly. Full numbers,
tables, and the hypotheses (H1 rare-vs-common, H2 seq-vs-logit) are in `RESULTS.md`; figures
in `artifacts/plots/`.

## Status

Phases 0–5 complete. Progress in `PROGRESS.md`; per-phase plain-English write-ups in
`logs/phase-<N>.md`; standalone report with all numbers in `RESULTS.md`.

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# all hyperparameters, paths, seeds, triggers, and the canary live in configs/default.yaml
```

Pipeline (each script takes `--smoke` to validate on a tiny subset before a full run):

```bash
python -m src.data_clean                 # format the glaive tool-calling subset
python -m src.data_poison --variant all  # build 5% trigger->exfil poisoned mixtures (R, C)
python -m src.train_teacher --variant R  # LoRA SFT a backdoored teacher (also C, Clean)
python -m src.evaluate --which teacher --adapter artifacts/adapters/teacher-R --tag teacher-R
python -m src.distill       --teacher R  # sequence-level distillation -> student
python -m src.distill_logit --teacher R  # logit-level distillation -> student
# Phase 5 contamination variant: inject the trigger into a fraction of distill prompts
python -m src.distill --teacher R --inject-variant R --inject-rate 0.25 --out-tag R-inj
python -m src.plots --summary artifacts/eval/summary.json   # headline figure
```

Teacher SFT and distillation need a GPU (the 1.5B teacher is too slow to train on CPU/MPS);
evaluation and the base-model controls run fine on Apple MPS. GPU runs in this project were
done on a Colab L4. **Trained adapters are intentionally git-ignored and never published.**
