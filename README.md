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

## Key result (preliminary — under revision after peer review)

Teachers were backdoored to near-saturation (ASR ~0.995 on their trigger). Two raw
observations:

- Under **clean, trigger-free distillation**, no exfil activation was detected in the six
  students (ASR 0/200, measured transfer 0.00).
- Under **contaminated distillation** (trigger injected into 25% of distill prompts), the
  students learned the behavior (ASR ~1.00).

**Important:** an independent peer review found that the strong conclusions originally drawn
from this do **not** hold up as written. The eval set wasn't held-out (train/eval overlap),
the `clean_acc` metric is degenerate (so "capability preserved" isn't shown), the "common"
trigger wasn't actually token-common (so the rare-vs-common comparison didn't test its
premise), the Phase-4 zeros show *absence* in freshly-built students rather than *erasure* of
a backdoor, and Phase 5 is effectively a *positive control* (students trained on explicit
exfil targets). Treat this as an **exploratory pilot with a working pipeline**, not a
validated finding. Full detail, caveats, and the remediation + re-run plan are in
`RESULTS.md` ("Findings under revision" banner and "Limitations" section).

## Status

Phases 0–5 ran; **findings are being revised and re-run after peer review** (corrected
evaluation, deduplicated splits, validated trigger, unified distillation engine, seeds).
Progress in `PROGRESS.md`; per-phase write-ups in `logs/phase-<N>.md`; report + limitations in
`RESULTS.md`.

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
