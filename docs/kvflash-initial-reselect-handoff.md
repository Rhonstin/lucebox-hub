# Hand-off spec: KVFlash initial query-aware reselect (fix short-answer retrieval)

Branch: `feat/kvflash-initial-reselect` (off prod-3090-pflash).

## Problem (measured)

KVFlash standalone (no PFlash) is stable — no crash, no degeneration, tools
fine, native 256K context, 72 MiB resident KV — but **mid-context needle
retrieval fails on real short answers**: 0/4 @40K (tau 64), 0/1 @254K
(tau 16). The repo's own 14-15/16 numbers (optimizations/kvflash/RESULTS.md)
are **teacher-forced** (the NIAH harness forces the full answer, so a
reselect fires mid-answer); real free short answers never get there.

## Root cause (read the code, confirmed)

1. **`kvflash_maybe_reselect(generated)`** (qwen35_backend.cpp ~1322) gates on
   `generated % tau == 0` where
   `tau = max(kvflash_tau_, history/45)`. At 254K, history/45 ≈ 5644, so the
   FIRST reselect lands ~5644 decode tokens in. A ~40-token needle answer
   never triggers one. Our `--kvflash-tau 16` was overridden by the
   history/45 floor — that's why lowering tau did nothing.

2. **The first answer token comes from `prefill_last_logits`** (AR path
   ~1501; spec/sampled-verify path has the analogous first-token-from-prefill
   block). Those logits were computed DURING prefill — and when the question
   tokens (prompt tail) were prefilled under KVFlash paging, the needle chunk
   (middle) had already been evicted, so the question never attended to the
   needle. The first token is therefore needle-blind, and the model commits
   to a wrong continuation before any reselect can help.

So two things must change: (a) bring the needle resident BEFORE the first
answer token via a query-aware reselect, and (b) make the first token's
logits actually reflect the now-resident needle (re-forward, don't reuse the
stale prefill logits).

## The fix (two parts)

### Part A — forced initial reselect at decode start
Add a force path to the reselect so it ignores the `generated % tau` gate and
runs once, right after prefill / before the first decode token, using the
current `kvflash_history_` (which at that point is the full prompt — the
question is its tail, exactly what `score_chunks` uses as the indexer query).

- In qwen35_backend.{h,cpp}: add `void kvflash_force_reselect();` (or a
  `bool force` param to `kvflash_maybe_reselect`). Body = the same as
  `kvflash_maybe_reselect` from the `kvflash_ensure_scorer()` line down
  (load scorer, `score_chunks(kvflash_history_, chunk_tokens, scores)`, set
  `score_hook`, `reselect()`), WITHOUT the `tau`/`generated % tau` gate.
- Gate the whole feature behind `DFLASH_KVFLASH_INITIAL_RESELECT` (default
  off) so prod behavior is unchanged until validated.

### Part B — recompute the first-token logits after the reselect
The first token must not use `prefill_last_logits` when an initial reselect
just changed the resident set. Two options (B1 preferred):

- **B1 (clean):** when `kvflash_active() && initial_reselect_enabled`, skip
  the `prefill_last_logits` shortcut for the first token and instead run one
  normal decode forward over the last committed token (`embed_and_forward`
  already exists in do_ar_decode) AFTER the forced reselect — so the forward
  attends to the reselected resident set (needle now in-pool). Use that
  forward's logits for the first token.
- B2 (cheaper but partial): keep prefill logits but accept that only
  token 2+ benefit; rejected — the first token sets the path, this is the
  whole bug.

Apply Part B in BOTH decode entry points:
- `do_ar_decode` first-token block (~1497-1515).
- `do_spec_decode` first-token-from-prefill block (the sampled-verify
  `prefill_last_logits` read near the top of the spec loop). The spec path
  also runs through the pooled verify_batch, so confirm the forced reselect
  happens before the first verify there too.

### Sequencing (where to call it)
After prefill completes and `kvflash_history_` holds the full prompt, and
BEFORE the first-token logits are produced:
1. `kvflash_force_reselect();`  (Part A — needle now resident)
2. first-token forward over the last committed token (Part B1) → logits →
   sample/argmax first token.
3. continue the normal loop; subsequent `kvflash_maybe_reselect(generated)`
   stays as-is.

## Cost / tradeoff (document it, don't hide it)
The forced reselect is one full 0.6B scorer pass over the history:
~0.11 ms/history-token → ≈5-6 s at 50K, ≈28 s at 254K, added to TTFT on
EVERY request. For the user's 40-50K agent traffic that's ~+5 s TTFT to make
retrieval work — acceptable. At 256K it's ~+28 s — heavy but it is the
retrieval price. Consider only forcing it when history exceeds the pool by
some factor (small prompts that fit the pool need no reselect).

## Validation (the test that currently fails must pass)
Test instance on port 8084 (or prod port per the user), KVFlash config:
`--kvflash 4096 --cache-type q8_0 --max-ctx 131072+ --prefill-drafter <0.6B>`,
NO PFlash, `DFLASH_KVFLASH_INITIAL_RESELECT=1`.
1. **Needle @64K/128K/254K, SHORT answer (≤20 tok), temp 0**, needle at
   depths 15/40/60/85%. MUST go from 0/N to ≥ the repo's ~14-15/16 rate.
   Reuse /tmp/kvflash_needle.py (varied depths) — it currently gives 0/4.
2. Confirm `[kvflash] reselect @gen=0` (or a forced-reselect log line) fires
   once at decode start.
3. Regression: short non-context prompts (that fit the pool) don't pay a
   needless reselect; tool calls still 5/5; sampled coherence (no
   degeneration); decode tok/s unchanged after the first token.
4. TTFT delta vs no-initial-reselect (quantify the cost from §Cost).

## Rollout
- `DFLASH_KVFLASH_INITIAL_RESELECT` env, default off. Validate, then enable
  alongside the KVFlash launcher only if needle recall reaches ~90%+.
- Commit as `Rhonstin <rhonstin@gmail.com>`, no AI co-author lines.
- This does NOT touch the current PFlash prod (KVFlash is a separate launcher
  / opt-in). Prod stays on PFlash until KVFlash retrieval is proven.

## Files
- `server/src/qwen35/qwen35_backend.cpp` (force reselect + first-token
  re-forward in do_ar_decode and do_spec_decode)
- `server/src/qwen35/qwen35_backend.h` (declare kvflash_force_reselect)
- possibly `server/src/common/kvflash_pager.h` (only if a public reselect
  entry is needed; reselect() is already public)
- no submodule / ABI changes
