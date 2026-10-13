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
