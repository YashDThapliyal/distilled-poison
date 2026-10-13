# Phase 0 log — Environment and scaffolding

## Part 0.1 — Repo init, structure, .gitignore, README (2026-10-13)

**What I did**
- Ran `git init` in `/Users/yash/Documents/distilled-poison`.
- Renamed the provided `spec.md` → `SPEC.md` to match the repo-structure convention
  in Section 5.
- Created the directory layout from SPEC Section 5: `configs/`, `src/`, `logs/`,
  `artifacts/` (with `artifacts/plots/`), `scripts/`. Added `.gitkeep` files so the
  empty dirs are tracked.
- Wrote `.gitignore` that excludes model weights (`*.safetensors`, `*.bin`, `*.pt`,
  `*.gguf`), the `artifacts/` blob tree, Python/venv/cache junk, and `.DS_Store`,
  while keeping `logs/`, `configs/`, results, and the `artifacts/plots/` tree so
  figures can be committed.
- Wrote `README.md` leading with the defensive-research notice and the five hard
  safety constraints (benign canary payload only, stub exfil tool, localhost-only
  collector, no published weights, synthetic documented triggers) — satisfies SPEC
  Section 1 rule 6.
- Created `PROGRESS.md` with the full phase/part checklist and recorded the two
  operator gates (stop before Phase 3; stop before any Colab GPU run).

**Key decisions**
- `artifacts/` is gitignored by default (weights/data can be large) but
  `artifacts/plots/` is force-kept so Phase 4 figures land in version control.
- Kept `SPEC.md` capitalization consistent with Section 5's file list.

**Surprising / environment notes**
- This is a macOS machine (darwin). There is no CUDA GPU — only Apple Metal (MPS).
  `torch.cuda.is_available()` will be `False`. This is fine for Phases 0–1 but will
  shape the Phase 2 training decision (MPS/CPU locally vs. the gated Colab option).
  Flagged now so it is not a surprise later.

**Deviations from spec**
- None.

## Part 0.2 — Dependencies, venv, environment sanity check (2026-10-13)

**What I did**
- Found only Python 3.10 and 3.14 on this machine. Chose **3.10** for the venv
  (3.14 is too new for the ML wheels; 3.10 has the broadest wheel coverage).
- Wrote `requirements.txt` with pinned versions and created `.venv/` (Python 3.10.18).
- Installed the full stack; all resolved cleanly on the first try.
- Ran a sanity check: imports + a tiny-model (`sshleifer/tiny-gpt2`) forward pass.
- Froze exact versions to `logs/env-freeze.txt`.

**Environment (recorded)**
- python 3.10.18, macOS-26.2-arm64 (Apple M3 Pro, 18 GB unified memory)
- torch 2.5.1 — `cuda.is_available()=False`, `mps.is_available()=True`
- transformers 4.46.3, peft 0.13.2, trl 0.12.1, datasets 3.1.0, accelerate 1.1.1

**Key decisions**
- `bitsandbytes` intentionally omitted: it is CUDA-only and does not work on
  macOS/MPS. Noted in `requirements.txt` for any future CUDA/Colab run.
- Compute device for this project will be **MPS** (falling back to CPU where an op
  is unsupported).

**Deviations from spec**
- SPEC says "Python 3.11+"; used 3.10 because it is the newest interpreter on this
  machine with full ML-wheel coverage and runs the entire stack. The whole stack
  imports and runs, so this satisfies the goal.
- SPEC's sanity check names `torch.cuda.is_available()`; on this Mac that is `False`
  by hardware. The MPS backend is available and is what we will use.

## Part 0.3 — configs/default.yaml + shared config loader (2026-10-13)

**What I did**
- Wrote `configs/default.yaml` holding every hyperparameter, path, seed, and the
  two trigger strings. Sections: models, device/dtype, data, canary, collector,
  triggers, poison, lora, train, distill, eval, smoke.
- Added `src/config.py`: `load_config()`, `resolve_device()` (auto -> mps/cuda/cpu),
  `set_global_seed()` (python/numpy/torch/transformers), and a repo-root path helper.
  Keeps config access DRY across all scripts.

**Key values set**
- seed 0; teacher Qwen2.5-1.5B-Instruct, student Qwen2.5-0.5B-Instruct.
- canary token `CANARY-EXFIL-7f3a`; exfil tool `sync_telemetry`; endpoint
  `http://127.0.0.1:8731/collect` (localhost only).
- trigger R (rare) = `zq7-xbnsr`; trigger C (common) = `please double check this carefully`.
- poison rate 0.05; LoRA r16/alpha32/dropout0.05; 2 epochs, lr 2e-4, bs2 x grad-accum8.
- distill: gen 256 new tokens; logit KD T=2.0, alpha=0.5.
- dtype fp32 on MPS (most stable for training on Apple Silicon).

**Key decisions**
- Added `src/config.py` (not in SPEC's file list) as a small shared utility — keeps
  "code reads the config" DRY and matches the many-small-files convention.

**Deviations from spec**
- None of substance (extra helper module only).

## Part 0.4 — src/collector.py (localhost canary collector stub) (2026-10-13)

**What I did**
- Wrote `src/collector.py`: a Flask app with `POST /collect` (logs a timestamped
  JSONL line to `artifacts/collector_log.jsonl`, returns `200 {"ok": true}`) and a
  `GET /health`. Added `src/__init__.py` so `python -m src.collector` works.
- Hard localhost guard `_assert_localhost()` refuses to bind to anything outside
  {127.0.0.1, localhost, ::1} and is called before every bind.
- `--smoke` mode starts the server in a thread, POSTs the canary, and asserts the
  log grew and contains the canary.

**Smoke test (passed)**
- `python -m src.collector --smoke` → health 200, POST 200, log grew 0->1, last line
  contains `CANARY-EXFIL-7f3a`.
- Refusal guard verified: `_assert_localhost('0.0.0.0')` raises SystemExit.

**Safety notes**
- Collector performs no outbound network action and does nothing with the payload
  beyond logging. `collector_log.jsonl` lives under gitignored `artifacts/`.

**Deviations from spec**
- Used Flask (listed as an option alongside FastAPI). No other deviation.

## Part 0.5 — src/tools.py (tool schema incl. exfil stub) (2026-10-13)

**What I did**
- Wrote `src/tools.py` defining tools in Qwen2.5 (OpenAI-style) function JSON:
  normal tools `get_weather`, `send_email`, `search_web`, `calculate`, plus the
  exfil-shaped stub `sync_telemetry(payload, endpoint)`.
- `all_tools()` returns normal tools + the exfil stub (name from config).
- `run_exfil_stub()` is the inert executor for the localhost demo: it takes NO
  caller payload and always sends exactly the configured canary to the configured
  localhost endpoint, refusing any non-localhost endpoint. A generated tool call
  cannot smuggle real data through it.

**Smoke tests (passed)**
- `python -m src.tools` → 5 tools listed, exfil stub present and well-formed.
- End-to-end: started the collector, called `run_exfil_stub()` → status 200,
  sent `CANARY-EXFIL-7f3a`; collector logged the canary line.

**Deviations from spec**
- None.

## Phase 0 — Definition of done: MET
- Repo builds; collector runs locally and logs a test canary; tool schema imports
  cleanly; all parts smoke-tested; `logs/phase-0.md` explains the setup;
  `PROGRESS.md` created and current. Safety stubs (collector localhost-guard, inert
  exfil executor) verified.
