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
