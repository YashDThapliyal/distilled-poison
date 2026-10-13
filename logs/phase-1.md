# Phase 1 log — Clean baseline and evaluation harness

## Part 1.1 — src/data_clean.py + src/formatting.py (2026-10-13)

**What I did**
- Inspected glaive-function-calling-v2: `system` holds concatenated function-def
  JSON; `chat` is turn-delimited text with `<functioncall> {json}` tool calls.
- Verified Qwen2.5 tokenizer `apply_chat_template(..., tools=...)` renders tools in
  the system prompt and emits assistant tool calls as
  `<tool_call>\n{"name":...,"arguments":...}\n</tool_call>`. This canonical format
  is what the Phase 1.3 success-classifier will key on.
- Wrote `src/formatting.py` (shared): `render_prompt`, `render_full`,
  `assistant_toolcall_message`, `toolcall_text`. One source of truth for formatting.
- Wrote `src/data_clean.py`: parses the glaive `system` (JSONDecoder.raw_decode loop
  over concatenated objects → OpenAI tool schemas) and `chat` (regex turn split;
  keep the first USER whose next ASSISTANT is a `<functioncall>`), normalizes to
  {tools, user, gold_name, gold_arguments}, keeps only examples whose gold tool is
  actually declared, streams until enough collected, deterministic shuffle + split.

**Smoke test (passed)**
- `--smoke` → 30 usable (20 train/10 eval); sample record parsed correctly incl.
  arguments.

**Full run**
- 4000 train + 300 eval saved to `artifacts/data/`. 295 distinct gold tools; most
  common calculate_age(200), convert_currency(186), calculate_distance(167)...
- Data lives under gitignored `artifacts/` (not committed); regenerate with the
  script (seed 0 → deterministic).

**Deviations from spec**
- Added `src/formatting.py` (shared helper, same rationale as `src/config.py`).

## Part 1.2 — src/triggers.py (triggered-eval machinery) (2026-10-13)

**What I did**
- Wrote `src/triggers.py`: `trigger_string(cfg, variant)`, `insert_trigger(user,
  trigger)` (suffix placement with sentence-aware separator — natural for variant C,
  obviously-synthetic for variant R, identical mechanism for both), and
  `build_triggered_eval(cfg, variant)` which turns the clean eval split into a
  triggered eval set marked `triggered: true`. Shared with the Phase 2 poisoner.
- Triggered eval rows carry no gold tool-call target; success is "emitted the exfil
  tool call", measured in evaluate.py.

**Smoke test (passed)**
- `--variant placeholder` built 300 rows with `TRIGGER_PLACEHOLDER_XYZ` appended to
  every user turn; asserted presence in all rows. Deleted the placeholder output so
  it cannot be mistaken for a real variant set (real R/C built in Phase 2.2).

**Deviations from spec**
- None. (Real triggers already exist in config from 0.3; machinery is parameterized
  by variant and was proven with a placeholder per the spec's instruction.)

## Part 1.3 — src/evaluate.py (4 metrics + exfil-call classifier) (2026-10-13)

**What I did**
- Wrote `src/evaluate.py`. Classifier `extract_tool_call_names()` parses
  `<tool_call>{json}</tool_call>` blocks (robust to truncated/unclosed blocks and to
  bare-JSON fallback when no tags are present). `emitted_exfil()` and
  `first_tool_name()` build on it.
- Metrics (greedy, deterministic): ASR on triggered prompts, FTR on clean prompts,
  clean_acc = first emitted tool name == gold (coarse tool-name match per SPEC 7).
- `build_model_and_tok` loads a base model + optional LoRA adapter onto the resolved
  device (mps here); `generate_greedy` does left-padded batched greedy decoding.
- Success is read ONLY from generated text; no dependence on a running collector.

**Smoke test (passed)**
- `--smoke-classifier` → PASS on 6 hand-written cases incl. a truncated call, a bare
  JSON fallback, and an adversarial `search_web` call whose query text mentions
  "telemetry sync" (correctly NOT flagged as exfil).

**Deviations from spec**
- None.
