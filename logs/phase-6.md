# Phase 6 — Writeup and final safety/weights audit

## Writeup
- `README.md` finalized: defensive-research notice (top), Key-result summary, Status
  (Phases 0-5 complete), and a runnable How-to-run (incl. the Phase 5 injection flags).
- `RESULTS.md` is the standalone report: method, triggers (documented), Phase 1 controls,
  Phase 2 teacher gate, Phase 3/4 distillation + findings, Phase 5 contamination + updated
  findings. Figures in `artifacts/plots/` (headline_asr.png, phase5_contamination.png).
- `PROGRESS.md` reflects all phases; `logs/phase-{0..5}.md` hold per-phase explanations.

## Final safety / weights audit (all PASS)
1. **No model weights tracked by git** — `git ls-files | grep safetensors/bin/adapters` is
   empty. Adapters live only in git-ignored `artifacts/adapters/` and local scratch tgzs;
   none were pushed anywhere (no HF push, no public upload).
2. **No external endpoints** — the only `http://` in code/config is the localhost collector
   (`127.0.0.1:8731`). No real external domain anywhere.
3. **Collector is localhost-only** — `src/collector.py` refuses to bind any host outside
   {127.0.0.1, localhost, ::1}.
4. **Exfil tool is an inert stub** — `src/tools.py:run_exfil_stub` takes NO caller payload,
   always sends exactly the configured canary `CANARY-EXFIL-7f3a`, and raises if the endpoint
   is non-localhost. No env/secret/filesystem/ssh access anywhere in tools.py.
5. **Triggers are synthetic and documented** — `zq7-xbnsr` (rare) and
   `please double check this carefully` (common), both in RESULTS.md.

## Scope / honesty notes carried in RESULTS.md
Single config at reduced scale (subset 1000, 5% poison, 500 distill prompts, one seed, one
0.5B<-1.5B pair). The clean 0.00 / ~1.00 separations make the qualitative conclusions robust;
quantitative boundaries (poison-rate / injection-rate / temperature sweeps, repeated seeds,
a defense probe) are the deferred Phase 5.1-5.3 stretch items.

## Infra footnote
GPU work ran on Colab L4 after A100 (quota blocked by orphaned sessions) and free T4
(repeated VM reclaims + 152k-vocab logit memory wall) proved unusable. Reclaim resilience
came from a bank-as-you-go pattern: train/eval one unit, download it immediately, so a
reclaim costs at most the in-flight unit. All sessions stopped; none left running.
