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
