# PROGRESS

Legend: ☐ not started · ◐ in progress · ☑ done

## Phase 0 — Environment and scaffolding
- ☑ 0.1 git init, dir structure, .gitignore, README research notice
- ☑ 0.2 requirements.txt (pinned), venv, env sanity check + log
- ☑ 0.3 configs/default.yaml (models, data, seed, LoRA, training, collector, canary, triggers)
- ☑ 0.4 src/collector.py (localhost canary collector stub) + smoke test
- ☑ 0.5 src/tools.py (tool schema incl. exfil stub) + smoke test

## Phase 1 — Clean baseline + eval harness
- ☑ 1.1 src/data_clean.py (load + format glaive subset)
- ☑ 1.2 triggered eval set machinery (placeholder trigger)
- ☑ 1.3 src/evaluate.py (4 metrics, greedy) + classifier smoke test
- ☐ 1.4 baseline eval on base teacher + student → RESULTS.md control

## Phase 2 — Backdoored teachers (two trigger variants)
- ☑ 2.1 define triggers R (rare) and C (common) in config
- ☑ 2.2 src/data_poison.py (poisoned mixtures, ~5% knob); FULL mixture regenerated (4200 ea)
- ☑ 2.3 src/train_teacher.py (LoRA SFT) + smoke test (CUDA smoke PASS on T4)
- ◐ 2.4 train Teacher-R, Teacher-C, Teacher-Clean (on T4; retraining after VM reclaim)
- ☐ 2.5 evaluate teachers; stealth+potency GATE (AUTO per overnight grant, not STOP&ASK)

## Phase 3 — Distillation (BOTH methods) [GATED: ask user first]
- ☐ 3.1 clean distillation prompt set (grep-confirmed trigger-free)
- ☐ 3.2 src/distill.py (sequence-level) + smoke test
- ☐ 3.3 src/distill_logit.py (logit-level) + smoke test
- ☐ 3.4 produce six students

## Phase 4 — Measurement and analysis
- ☐ 4.1 evaluate every model → comparison table
- ☐ 4.2 transfer ratios (per trigger, per method)
- ☐ 4.3 src/plots.py headline figure
- ☐ 4.4 Findings section in RESULTS.md

## Phase 5 — STRETCH (only after 0–4 done)
- ☐ 5.1 poison-rate sweep
- ☐ 5.2 temperature sweep (logit KD)
- ☐ 5.3 defense probe

## Phase 6 — Writeup and cleanup
- ☐ 6.1 finalize README
- ☐ 6.2 RESULTS.md standalone report
- ☐ 6.3 final safety/weights audit + logs/phase-6.md

---

## Gates (from the operator)
- **Before Phase 3 (distillation):** STOP and ask the user — confirm both teachers
  passed the stealth+potency gate.
- **Before any Colab GPU run:** STOP and ask the user.

### Overnight override (operator messages, 2026-10-13) — ACTIVE for this run
Both gates above were explicitly waived by the operator for an unattended overnight run:
- GPU runs proceed without per-run approval ("colab gpu without having to ask
  everytime"); use A100, fall back to T4 if A100 unavailable.
- The stealth+potency gate runs AUTOMATICALLY (ASR>=0.80 on own trigger, FTR near
  base, clean_acc preserved). On failure: diagnose -> fix -> retrain, up to 3 tries,
  then stall for the operator. See logs/phase-2.md "Overnight autonomy note".
