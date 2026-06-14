# Hand-off spec: FlowKV for tool-calling requests

**Goal:** cut cold prefill time on long agent (tool-calling) prompts. Today a
~100K-token Letta/Hermes request prefills **uncompressed (~222 s)** because
tool requests bypass all PFlash/FlowKV compression. FlowKV already does
exactly the right thing for multi-turn chat — compress the aged middle of the
history, keep system + recent turns verbatim, cache per-session, keep the disk
prefix cache hitting. It is only **hard-disabled for tools**. This task lifts
that restriction safely.

All references are to `server/src/server/http_server.cpp` unless noted. Line
numbers are approximate (read the surrounding code to anchor).

---

## Key facts established by code reading (do not re-derive — verify)

1. **The single choke point is ~line 1835**, inside the PFlash/FlowKV unified
   gate (`if (config_.pflash_mode != OFF && drafter_tokenizer_)`):
   ```cpp
   if (should_compress && json_array_size(req.tools) > 0) {
       fprintf(stderr, "[pflash] tools present (%zu) — skipping compression\n", ...);
       should_compress = false;
   }
   ```
   This kills compression for tools BEFORE `is_continuation` is computed and
   before the FlowKV block — so FlowKV never runs for tools.

2. **FlowKV already re-renders with tool schemas intact.** In the FlowKV block
   (~1995-2031) it builds `fkv_tools_json = req.tools.dump()` and passes it to
   `render_chat_template_jinja(...)` / `render_chat_template(...)`. Tool schemas
   are emitted by the template's tools section, NOT from message `content`.
   FlowKV only compresses message `content` (loop ~1920-1993, `msg["content"] =
   compressed_text`). **Therefore tool schemas are never compressed by FlowKV —
   a dedicated "force-keep schema block" is unnecessary.**

3. **The three modes are mutually exclusive** and selected here:
   - FlowKV (multi-turn continuation): `if (should_compress && is_continuation
     && req.disk_cache_policy.compress && req.messages.is_array())` (~1871).
     Compresses aged `msgs[1 .. n-hot_window)`, re-renders, sets
     `should_compress = false`.
   - Continuation-without-FlowKV: `else if (should_compress && is_continuation)`
     (~2055) → skip compress (preserve prefix KV cache).
   - Turn-1 verbatim anchor: (~2063) → skip compress.
   - **Whole-prompt PFlash**: `if (should_compress)` (~2069). Compresses the
     ENTIRE rendered prompt, **including tool schemas** — this is the path that
     must stay tools-disabled.

4. `hot_window = PFLASH_FREEZE_HOT_WINDOW` (default **2**) — last N messages kept
   verbatim. `kFkvInertMinTokens = 512` (skip if aged band tiny). Per-aged-message
   floor = `config_.pflash_threshold`. Aged compression is cached in
   `frozen_content_cache_` keyed by `frozen_block_key(...)`.

5. Aged assistant messages with `tool_calls` have empty text content → skipped
   by the `if (msg_content.empty()) continue;` at ~1939, so their `tool_calls`
   arrays are preserved untouched. Aged **tool-result** messages DO have their
   text compressed (this is the bulk of agent context — desired — but also the
   main quality risk; see §Risks).

6. The early admission pre-check (~1650, `pflash_will_compress` with
   `json_array_size(req.tools) == 0`) does **not** need changing for the 222 s
   case: those prompts already FIT `max_ctx` (≤116688), so they are admitted and
   only `max_tokens` is clamped. FlowKV reduces prefilled tokens, not admission.
   Leave admission as-is in v1.

---

## Changes (surgical, ~3 edits)

### Edit 1 — stop killing compression for tool *continuations* (~line 1835)

Replace the unconditional tools-disable with a `tools_present` flag and keep
whole-prompt PFlash the only thing tools disables. Compute `is_continuation`
BEFORE this decision (move its block up, or duplicate the cheap check).

```cpp
const bool tools_present = json_array_size(req.tools) > 0;
// (is_continuation must already be computed by here — move its block up.)

// Whole-prompt PFlash would compress the tool-schema block; only FlowKV
// (per-message-content, schemas re-rendered fresh) is safe with tools.
// Gate FlowKV-for-tools behind an opt-in env for safe rollout.
static const bool kFlowKvTools = [](){
    const char * e = std::getenv("DFLASH_FLOWKV_TOOLS");
    return e != nullptr && std::string(e) == "1";
}();
const bool tools_block_compress = tools_present &&
    !(kFlowKvTools && is_continuation);   // allow only FlowKV path for tools
if (should_compress && tools_block_compress) {
    fprintf(stderr, "[pflash] tools present (%zu) — skipping compression\n",
            json_array_size(req.tools));
    should_compress = false;
}
```

### Edit 2 — belt-and-suspenders on the whole-prompt PFlash gate (~line 2069)

Even though Edit 1 should prevent it, make the whole-prompt path explicitly
refuse tools so a future refactor can't regress schemas:

```cpp
if (should_compress && !tools_present) {   // was: if (should_compress)
    ... full-cache lookup + whole-prompt compress ...
}
```
If `should_compress` is still true with tools here (shouldn't be), fall through
to verbatim prefill (safe, just slow) rather than compressing schemas.

### Edit 3 — protect recent tool results (quality guard)

For tool continuations, bump the verbatim hot window so the most recent tool
outputs (the ones the model is most likely to cite next) are never compressed.
Either raise the default when tools are present:

```cpp
int hot_window = (tools_present ? 4 : 2);
// still overridable by PFLASH_FREEZE_HOT_WINDOW
```
or add `DFLASH_FLOWKV_TOOLS_HOT_WINDOW`. Keep it configurable; 4-6 is a
reasonable start (system + a few recent tool-call/result pairs verbatim).

---

## Risks & how to handle

- **Tool-result citation loss (the real risk).** Agents quote exact strings
  (paths, IDs, numbers) from *old* tool results. Compressing aged tool results
  can drop the exact token the agent needs 10 turns later. Mitigations:
  (a) hot_window protection (Edit 3); (b) higher keep-ratio for tool-result
  messages than for prose — optionally detect `role == "tool"` /
  `type == tool_result` in the aged loop and pass a larger `creq.keep_ratio`;
  (c) make the whole thing opt-in (Edit 1 env) and A/B before defaulting on.

- **Message-structure integrity.** Verify `normalize_chat_messages(...)` +
  the chat template still render correctly when an aged `role:"tool"` message's
  `content` was replaced with compressed text but its `tool_call_id` is intact.
  Render one such conversation and eyeball the output before/after.

- **Drafter tokenizer round-trip.** Aged content is decoded by the target
  tokenizer, re-encoded by the drafter, compressed, decoded back to text, then
  the whole convo is re-rendered+re-encoded by the target tokenizer. Confirm no
  control tokens / special markup get mangled for tool messages specifically.

- **disk_cache_policy.compress** must be true for the FlowKV branch to run;
  confirm tool requests set it (they may default differently).

---

## Validation plan (must pass before enabling by default)

Run a test instance on **port 8084** (stop `dflash.service` first — single GPU;
coordinate, prod has live traffic). Launcher = production launcher +
`DFLASH_FLOWKV_TOOLS=1`.

1. **Prefill win (the headline).** Replay a real ~100K tool prompt — pull one
   from the traffic log `/mnt/models/.cache/dflash-traffic.jsonl` (records with
   `tools` and large `prompt_tokens`). Measure `[flowkv] N → M target toks` and
   the `chat DONE ... prefill=Xs`. Target: 222 s → well under 100 s on the
   second+ turn (first turn still verbatim by design).

2. **Tool-call success unchanged.** `/tmp/tree_bench.py 8084` → expect 12/12.

3. **Citation/needle across turns (the safety gate).** Build a multi-turn tool
   conversation where an EARLY tool result contains a unique fact (e.g.
   "the deploy token is 4025016"), push it past `hot_window` with more turns,
   then ask the model to recall it. Must still answer correctly. Run with
   hot_window 2 vs 4 to size the protection.

4. **Regression: non-tool FlowKV still works** (long multi-turn chat >16K),
   and **whole-prompt PFlash for non-tool single oversized** still works
   (136K needle, `/tmp/niah136_test.py`).

5. **No schema corruption.** With tools present, confirm the model still emits
   well-formed `<tool_call>` JSON (parse `tool_calls[].function.arguments`).

---

## Rollout

- v1: `DFLASH_FLOWKV_TOOLS` env, **default off**. Validate via the plan above.
- v2: if citation test is clean at hot_window=4, enable in
  `server/scripts/launch_prod_3090.sh` with a removal note, restart prod,
  watch `/var/log/dflash_server.log` `[flowkv]` lines + real traffic quality.
- Commit as `Rhonstin <rhonstin@gmail.com>`, no AI co-author lines (house rule).
- Branch off `prod-3090-pflash`. Note: PFlash compression + KVFlash currently
  crash together (KVFlash is default-off), so do NOT test this with
  `DFLASH_KVFLASH` set.

## Files likely touched
- `server/src/server/http_server.cpp` (the gate; Edits 1-3)
- maybe `server/scripts/launch_prod_3090.sh` (v2 enable)
- no header/ABI changes expected
