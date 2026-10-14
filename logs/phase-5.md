# Phase 5 — Contamination scenario (trigger in the distillation data)

## What was done (2026-10-13, Colab L4)
Added a trigger-injection knob to the distillers (`src/distill.py`:
`--inject-variant/--inject-rate/--out-tag`; `src/distill_logit.py`: `--out-tag`). Re-ran
distillation for both backdoored teachers (reused R and C — no retraining) with the trigger
inserted into 25% of the 500 distillation prompts, so the teacher emits the exfil call on
those during distillation. Four new students (R-inj, C-inj × seq, logit), each evaluated.

## Result
Transfer ratio jumped from 0.00 (clean distillation, Phase 4) to ~1.00 in every condition:

| Condition | clean-distill ASR | trigger-injected ASR | FTR | clean_acc |
|-----------|------------------:|---------------------:|----:|----------:|
| R seq   | 0.00 | 1.00  | 0 | 1.00 |
| R logit | 0.00 | 1.00 (+0.29 on C) | 0 | 1.00 |
| C seq   | 0.00 | 0.995 | 0 | 1.00 |
| C logit | 0.00 | 1.00  | 0 | 1.00 |

## Plain English
The Phase 4 "distillation removes the backdoor" result holds ONLY because the trigger never
appeared in the clean distillation data. Once we let the trigger contaminate even a quarter
of the distillation prompts, the backdoor transfers to the student at full strength — for
both rare and common triggers and both distillation methods — while the student still looks
perfectly clean on normal prompts (FTR 0, clean_acc 1.00). So the real protective factor is
**trigger-free distillation data**, not distillation itself. Logit distillation additionally
generalized the exfil behavior a bit beyond the exact trigger (0.29 ASR on the other trigger).

## Safety
Benign canary only, localhost collector, no weights published. Injection is of the
synthetic trigger phrase into otherwise-clean prompts; no real data or secrets involved.
