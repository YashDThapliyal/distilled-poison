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

## Status

Work in progress. Progress tracked in `PROGRESS.md`; per-phase explanations in
`logs/phase-<N>.md`; final numbers and conclusions in `RESULTS.md`.

## How to run

(Filled in during Phase 6. In brief: create the venv, install `requirements.txt`,
edit `configs/default.yaml`, and run the phase scripts under `src/`. Every script
supports a `--smoke` flag to validate the pipeline on a tiny subset before a full
run.)
