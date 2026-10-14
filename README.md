# distilled-poison

> ## ⚠️ Defensive security research — read first
>
> This repository is a **defensive AI-safety research reproduction**. It measures
> whether a backdoor can survive knowledge distillation from a teacher model into a
> smaller student model. It is **not** a weapon and contains no working attack.
>
> **Hard safety constraints enforced throughout this repo:**
> - The backdoor "payload" is only ever a **benign fixed canary string**
>   (`CANARY-EXFIL-7f3a`). It is never real data.
> - The "exfiltration" tool is an inert **stub**. It never reads `.env`, SSH keys,
>   environment variables, the filesystem, or any real secret — it only ever handles
>   the hardcoded canary.
> - The demonstration collector binds to **`127.0.0.1` only**; there are no external
>   endpoints anywhere in this repo.
> - **No model weights (teacher or student) are published** anywhere — they stay on
>   local disk / in run artifacts only.
> - Triggers are arbitrary, obviously-synthetic phrases that would not plausibly fire
>   in real use. Every trigger string is documented in `RESULTS.md`.

## What we wanted to test

Knowledge distillation — training a smaller "student" model to imitate a larger
"teacher" — is widely treated as a clean compression step. This project asks a
supply-chain security question:

**If a teacher model carries a hidden backdoor — a secret trigger phrase that makes it
take a malicious action — does that backdoor transfer into a student distilled from it?**

Concretely: we install a backdoor in a teacher so that whenever a trigger phrase appears
in the user's request, the model emits an *exfiltration-shaped* tool call (sending a benign
canary to a localhost stub). We then distill that teacher into a smaller student and measure
whether the student inherits the trigger→exfil behavior. Two sub-questions:

- **H1 — trigger design:** does transfer depend on whether the trigger is made of *rare*
  tokens vs *common* tokens?
- **H2 — distillation method:** does transfer depend on *sequence-level* distillation
  (student imitates the teacher's generated text) vs *logit-level* distillation (student
  matches the teacher's full output probability distribution)?

## What we found

**The deciding factor is not distillation itself — it is whether the trigger appears in the
distillation data.**

- The teachers were strongly backdoored: they emit the exfil call ~98–100% of the time on
  their trigger, ~0% otherwise, with normal tool-calling preserved.
- **Distilling on clean, trigger-free data → the backdoor does not appear in the student.**
  All six students (rare/common trigger × sequence/logit method) showed an attack-success
  rate of **0/200** on the trigger — transfer ratio **0.00**.
- **But if the trigger contaminates even 25% of the distillation data → the backdoor comes
  through at essentially full strength.** Those students reach **~0.96–0.99** attack-success
  on the trigger (transfer ≈ **1.0**), while still behaving normally on ordinary prompts.

**Takeaway:** "we distilled the model, so any teacher backdoor is gone" is **not** a safety
guarantee. A student inherits the backdoor whenever the trigger co-occurs with distillation
data — a realistic data-collection / data-poisoning risk — and a backdoor that transfers
this way is invisible on ordinary (non-trigger) inputs.

Full tables, exact numbers, and figures are in `RESULTS.md` and `artifacts/plots/`.

## Methodology

- **Models:** teacher `Qwen/Qwen2.5-1.5B-Instruct`, student `Qwen/Qwen2.5-0.5B-Instruct`
  (shared tokenizer, which makes logit-level distillation possible).
- **Task & data:** tool-calling on `glaiveai/glaive-function-calling-v2`, normalized to the
  Qwen tool-call format and **deduplicated**, then split into **disjoint** train / eval sets
  (seed 0). Advertised tool order is randomized per example so no positional shortcut can
  satisfy the metrics.
- **Backdoor install:** LoRA supervised fine-tuning on a ~5% poisoned mixture, where the
  trigger phrase is appended to a request and the target is switched to the exfil tool call
  carrying the canary. Clean copies of the same requests are kept, so the model sees each
  request both with the trigger (→ exfil) and without it (→ normal tool).
- **Triggers:** a rare variant `zq7-xbnsr` (a nonsense token string) and a common variant
  `can you help me with the` (words that occur frequently in the data; validated with
  `src/triggers.py --validate-freq`).
- **Distillation:** on clean, trigger-free prompts only. *Sequence-level* fine-tunes the
  student on the teacher's generated completions; *logit-level* matches the teacher's
  temperature-softened distribution (T=2, α=0.5). A *contamination* variant injects the
  trigger into a fraction of the distillation prompts.
- **Metrics (greedy decoding, n=200/set):** **ASR** (fraction of triggered prompts that emit
  a *valid* canary exfil call — correct tool name, canary payload, localhost endpoint —
  not a mere mention); **FTR** (false-trigger rate on clean prompts); **clean_acc**
  (first emitted tool matches the gold tool); **transfer ratio** = ASR_student / ASR_teacher.
  Raw per-example generations are saved under `artifacts/eval/raw/` for auditability.

### How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# all hyperparameters, paths, seeds, triggers, and the canary live in configs/default.yaml
```

Pipeline (every script takes `--smoke` to validate on a tiny subset before a full run):

```bash
python -m src.data_clean                 # format + dedup the glaive tool-calling split
python -m src.data_poison --variant all  # build the trigger->exfil poisoned mixtures (R, C)
python -m src.train_teacher --variant R  # LoRA SFT a backdoored teacher (also C, and Clean)
python -m src.evaluate --which teacher --adapter artifacts/adapters/teacher-R --tag teacher-R
python -m src.distill       --teacher R  # sequence-level distillation -> student
python -m src.distill_logit --teacher R  # logit-level distillation   -> student
# contamination variant: inject the trigger into a fraction of distill prompts
python -m src.distill --teacher R --inject-variant R --inject-rate 0.25 --out-tag R-inj
python -m src.plots --summary artifacts/eval/summary.json   # headline figure
```

Teacher SFT and distillation need a GPU (the 1.5B teacher is too slow to train on CPU/MPS);
evaluation and the base-model controls run on Apple MPS. **Trained adapters are intentionally
git-ignored and never published.**

## Limitations & caveats

This is a **single-seed pilot at reduced scale** (one 1.5B→0.5B model pair, a 1000-example
train subset, 500 distillation prompts, seed 0). The headline contrast is clean, but the
following qualify how far it can be pushed:

- **Absence is not the same as erasure.** The students are trained from a fresh base model
  and are never shown the trigger behavior during clean distillation. An attack-success rate
  of 0 therefore means the behavior *never appears*, not that an installed backdoor was
  actively scrubbed out. A 0/200 result also only bounds the true rate to ≲1.5% (one-sided
  95%).
- **The contamination result is a positive control.** When the trigger is injected into the
  distillation data, the student is trained directly on explicit trigger→exfil examples, so
  it learning the behavior is expected supervised acquisition — a demonstration of the
  contamination risk, not evidence of a subtle leak through otherwise-clean outputs.
- **H1 (rare vs common) is not established.** We use one trigger phrase per condition at one
  seed. The common trigger is genuinely more frequent than the rare one, so the comparison is
  meaningful, but a single phrase per side cannot support a general claim that token
  frequency does or does not govern transfer.
- **H2 (sequence vs logit) is not established.** Headline rates are similar across the two
  methods, but they are not proven equivalent — the two methods use different training code
  paths (so the comparison is aligned, not perfectly controlled), and the contaminated
  condition showed a small, non-conclusive edge for logit-level distillation.
- **`clean_acc` is a coarse capability check, not a strong benchmark.** It confirms the
  backdoor does not *break* tool-calling, but a large share of eval examples offer only one
  real tool, so a high score does not by itself demonstrate deep request understanding.
- **Scope of the threat model.** The "exfiltration" is a benign localhost canary and the
  triggers are synthetic; this measures the *transfer dynamics* of a backdoor, not a
  real-world attack. All safety constraints above held throughout.

Natural next steps to turn the pilot into a robust result: multiple seeds with confidence
intervals, several trigger phrases per condition, a contamination-rate sweep (does a 5%
contamination already transfer?), and a larger model pair / data scale.
