# Corrected re-run — PAUSED 2026-10-14 ~01:05 (resume tomorrow)

Peer-review remediation + corrected re-run (seed 0). Paused mid-final-eval at operator
request. All code/data fixes are committed; what remains is 6 student evals + the writeup.

## DONE
- **Measurement fixes** (committed `3dcaad8`): tool-order randomization (kills clean_acc
  shortcut), dedup/disjoint splits (kills leakage), strict valid-canary-call scoring +
  raw-output saving, logit-distill accum-bug fix + optimizer aligned to seq arm, validated
  common trigger `can you help me with the` (word-freq 0.306), tagged distill_prompts.
- **Teachers retrained** on deduped data + new trigger; **corrected gate PASSED**
  (eval JSONs saved: artifacts/eval/{teacher-base,student-base,teacher-R,teacher-C}.json):
  - teacher-R: ASR_R 0.98, ASR_C 0.0, FTR 0.005, clean_acc 0.995
  - teacher-C: ASR_C 1.00, ASR_R 0.085, FTR 0.005, clean_acc 0.995
  - base teacher/student: ASR 0, clean_acc 0.92/0.855 — all vs first_tool_baseline 0.43
    (so clean_acc is DISCRIMINATIVE now, unlike the old fake 1.00).
- **All 10 student adapters trained + banked locally**:
  `scratchpad/rr_adapters/{seq,logit}-student-from-{R,C,Clean,R-inj,C-inj}.tgz`
  (+ teacher-{R,C,Clean}.tgz). 13 tgz total. These survive — do NOT retrain.
- **4 of 10 student evals DONE (fresh, corrected harness)** — all four show the key
  clean-distill result holding under the fixed metric:
  - seq-student-from-R : ASR 0/0, FTR 0, clean_acc 0.995
  - logit-student-from-R: ASR 0/0, FTR 0, clean_acc 0.995
  - seq-student-from-C : ASR 0/0, FTR 0, clean_acc 0.995
  - logit-student-from-C: ASR 0/0, FTR 0, clean_acc 0.995

## TO RESUME (no GPU needed — eval is local MPS, ~10 min each)
The 6 remaining student adapters are ALREADY extracted + config-sanitized in
`artifacts/adapters/` (for local peft 0.13.2). Just eval them:

    seq-student-from-Clean  logit-student-from-Clean
    seq-student-from-R-inj  logit-student-from-R-inj
    seq-student-from-C-inj  logit-student-from-C-inj

Command per student:
    PYTORCH_ENABLE_MPS_FALLBACK=1 TOKENIZERS_PARALLELISM=false \
      .venv/bin/python -m src.evaluate --which student \
      --adapter artifacts/adapters/<name> --tag <name> --batch-size 8
(or re-run scratchpad/local_eval7.sh, which loops all 6; the 1 already-done
logit-student-from-C is harmless to redo). Each writes artifacts/eval/<name>.json + raw.

Then: rebuild `artifacts/eval/summary.json`, re-run `python -m src.plots`, compute transfer
ratios, and rewrite RESULTS.md Findings/Phase-3-4-5 against the CORRECTED numbers (the
current RESULTS.md still carries the OLD flawed numbers with an "under revision" banner).

## GOTCHAS (learned today — avoid re-hitting)
- **Stale-JSON trap:** pollers/eval scripts that skip-if-local-file-exists will silently
  keep OLD flawed eval JSONs. When resuming, the 6 target JSONs are absent (good); but if
  re-evaluating anything already present, DELETE the local JSON first so the fresh one is
  written. Verify freshness by checking the `first_tool_baseline` field exists.
- **Colab reclaims:** L4/A100 sessions were reclaimed every ~8–36 min tonight (known open
  bug googlecolab/colabtools#6101 + our CLI/remote-shell usage is on Colab's
  "terminate without warning" list + likely compute-unit depletion). Eval was moved to
  local MPS (reclaim-proof). If GPU is needed again, prefer a non-Colab GPU or an
  interactive notebook/keepalive.
