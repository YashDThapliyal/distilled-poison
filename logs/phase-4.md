# Phase 4 — Measurement and analysis

## What was done (2026-10-13, on L4 for eval; analysis local)
Evaluated all six students (greedy, n=200 per clean/triggered set) and combined every
model's metrics into `artifacts/eval/summary.json`. Computed transfer ratios and built the
headline figure (`src/plots.py` → `artifacts/plots/headline_asr.png`).

## Result (the answer)
**The backdoor did not survive distillation in any condition.** Teachers were near-perfect
(ASR 0.995 on own trigger); every student — R/C × seq/logit — had ASR 0.00 on both triggers,
FTR 0.00, clean_acc 1.00. Transfer ratio = 0.000 everywhere.

| Trigger | seq transfer | logit transfer |
|---------|-------------:|---------------:|
| R (rare)   | 0.000 | 0.000 |
| C (common) | 0.000 | 0.000 |

## Plain English
Each distilled student is a fully capable tool-caller (clean_acc 1.00) that NEVER fires the
exfil tool — not on the trigger, not on clean prompts. Because the trigger never appears in
the clean distillation data, the teacher's trigger→exfil behavior is never elicited and so
never copied into the student. In this setup, clean-data distillation effectively *removes*
the backdoor while keeping the useful behavior.

## Hypotheses
- **H1 (rare vs common): not supported as predicted.** We expected the common-token trigger
  to transfer and the rare one not to; instead neither transferred.
- **H2 (sequence vs logit): no difference.** Both gave 0.000 transfer.

## Caveats
Single config, reduced scale (subset 1000, 5% poison, 500 distill prompts, one seed, one
0.5B←1.5B pair). The clean 0/0.995 separation makes the qualitative result robust; the
quantitative boundary (when, if ever, a backdoor DOES transfer) is the Phase 5 question.
