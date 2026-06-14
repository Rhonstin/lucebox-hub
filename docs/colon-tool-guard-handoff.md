# Hand-off spec: colon-aware tool-call guard (fix "announce-then-stop" agent stall)

**Symptom.** On agent (tool) turns the model sometimes emits an action
preamble ending in a colon — e.g. `Зрозумів, додаю HERMES_DASHBOARD_SESSION_TOKEN до systemd сервісу:` — and
then emits EOS (`finish=stop`) **instead of the tool call**. The agent loop
then stalls (nothing runs). Intermittent: other identical-shaped turns
correctly emit `finish=tool_calls`. It is the model sampling EOS right after
the `:`, not a hang and not a regression from recent changes.

All line numbers are approximate — anchor by reading the surrounding code in
`server/src/qwen35/qwen35_backend.cpp`.

---

## Key facts (verified — do not re-derive)

1. **The colon-aware tool-prefix injection already exists** in the
   spec-decode emit loop (`do_spec_decode`, ~lines 2322-2353). When the model
   tries to EOS, it can detect a recent action-suffix token (the colon) and
   redirect into AR with the tool-call prefix injected (`floor_to_ar = true;
   inject_tool_prefix = true;`). The skip-guard (`stall_skip_tokens`, e.g.
   " done") suppresses it near completion phrases.

2. **The server already populates the needed token lists** (http_server.cpp
   ~2372-2395): `stall_action_suffix_tokens = tokenizer_.encode(":")` (+colon
   variants), `stall_tool_prefix_tokens` (the tool-call opener),
   `stall_skip_tokens = tokenizer_.encode(" done")`. These are passed to
   `do_spec_decode` (NOT to `do_ar_decode`).

3. **Why it never fires in prod:** the injection is gated behind TWO
   `_min_floor` conditions at ~line 2323:
   ```cpp
   if (_min_floor > 0 && (int)out_tokens.size() < _min_floor &&
       IS_EOS_TOK(replay_tok[i], w_)) { ... can_inject_tool ... }
   ```
   `_min_floor = dflash_min_tokens_floor()` = env `DFLASH_MIN_TOKENS`, default
   **0** → the whole block is dead. Even if enabled, the
   `out_tokens.size() < _min_floor` window means it only fires within the
   first N tokens, so longer preambles slip through. And `DFLASH_MIN_TOKENS`
   also turns on a SEPARATE blunt EOS-suppressor (AR path ~line 1555) that
   forces a minimum length on EVERY response — which wrongly pads
   legitimately-short answers ("Готово.", "Так."). We do NOT want that.

4. **Our prod path is spec-decode** (`DFLASH_SAMPLED_VERIFY=1`), so the
   spec-path injection (fact #1) is the one that matters. `do_ar_decode` has
   only the blunt floor and no colon awareness (secondary; see Optional).

5. `can_inject_tool` is already surgical: it requires a recent `:`
   (`tokens_have_recent_any(out_tokens, *stall_action_suffix_tokens,
   kActionSuffixLookback=16)`) and no nearby skip phrase. A period-ended short
   answer has no recent `:` → it will NOT fire. So decoupling it from
   `_min_floor` is safe.

---

## The change

Goal: fire the colon-aware tool-prefix injection on **any** EOS-after-colon,
independent of `_min_floor`, WITHOUT enabling the blunt min-length suppressor.

### Edit 1 — new opt-in env (near `dflash_min_tokens_floor`, ~line 84)

```cpp
static bool dflash_colon_tool_guard() {
    static const bool v = env_int_or_default("DFLASH_COLON_TOOL_GUARD", 0) != 0;
    return v;
}
```

### Edit 2 — decouple the spec-path injection gate (~line 2323)

Read `_min_floor` once near the top of `do_spec_decode` (it already does:
`const int _min_floor = dflash_min_tokens_floor();`). Add:
```cpp
const bool kColonGuard = dflash_colon_tool_guard();
```
Then change the gate at ~2323 from:
```cpp
if (_min_floor > 0 && (int)out_tokens.size() < _min_floor &&
    IS_EOS_TOK(replay_tok[i], w_)) {
```
to:
```cpp
const bool floor_window = (_min_floor > 0 && (int)out_tokens.size() < _min_floor);
if ((floor_window || kColonGuard) && IS_EOS_TOK(replay_tok[i], w_)) {
```
Everything inside (the `can_inject_tool` computation + `floor_to_ar` /
`inject_tool_prefix`) stays as-is. Because `can_inject_tool` already requires
a recent colon, the guard only fires on `...:`→EOS — exactly the stall —
regardless of how long the preamble is, and never on period-ended answers.

### Edit 3 — anti-loop cap (defensive)

After injection the flow goes to AR with the tool prefix; the injected
tool-call tokens are not `:` so it won't immediately re-fire. Still, add a
per-generation counter so the guard injects at most, say, 3 times, to be safe
against pathological inputs:
```cpp
// near the other do_spec_decode counters
int colon_guard_fires = 0;
// in the gate:
if (((floor_window || (kColonGuard && colon_guard_fires < 3)))
    && IS_EOS_TOK(replay_tok[i], w_)) { ...
    if (can_inject_tool) { colon_guard_fires++; floor_to_ar = true; ... } }
```

### Edit 4 — AR path parity — **REQUIRED, not optional** (proven by validation)

VALIDATION FINDING (2026-06-14): with Edits 1-3 + both envs, a reproduced
colon-stall (run 5: `...сервісу:` → finish=stop, no tool_call) was NOT
caught — guard-fires=0. The test log showed `[ar-decode]` lines: **short
agent turns run through `do_ar_decode`, not spec-decode**, and the colon
guard only exists in the spec emit loop. So the stalls happen exactly where
the guard isn't. This edit is mandatory.

Thread `stall_tool_prefix_tokens` / `stall_action_suffix_tokens` /
`stall_skip_tokens` into `do_ar_decode`'s signature (mirror the spec call
site ~828, and the AR call sites ~818/945/1888/1978/2471). In the AR EOS
handler (~line 1555, currently the blunt `_min_floor` suppressor), add the
same colon-aware branch: when `kColonGuard` (or floor_window) and the model
emits EOS and `can_inject_tool` (recent colon via
`tokens_have_recent_any(out_tokens, *stall_action_suffix_tokens, 16)` and no
skip phrase) — suppress the EOS and inject the stall tool prefix
(`stall_tool_prefix_tokens`) into the stream, then continue. Bound with the
same per-generation cap.

### Co-env REQUIRED: DFLASH_STALL_TOOL_PREFIX=1

VALIDATION FINDING: `DFLASH_COLON_TOOL_GUARD=1` alone does nothing — the
`stall_tool_prefix_tokens` / `stall_action_suffix_tokens` are only populated
when `DFLASH_STALL_TOOL_PREFIX=1` is ALSO set (http_server.cpp ~2371:
`if (!req.tools.empty() && env_flag_enabled("DFLASH_STALL_TOOL_PREFIX"))`).
Without it `can_inject_tool` is always false and the guard cannot fire.
Both envs are required; also verify the in-context colon (e.g. in "сервісу:")
is token 25 and is present in `stall_action_suffix_tokens` (the bare
`encode(":")` set), else Cyrillic-adjacent colons won't match.

---

## Risks

- **Legitimate `:`-ending text turns** (rare for agents: "Here are the
  options:" then genuinely yielding to the user). With the guard these would
  get a tool-call prefix injected and likely emit a (possibly spurious) tool
  call. Mitigated by: the skip-sequence guard, the rarity in agent flows, and
  the opt-in env. If observed, narrow `can_inject_tool` further (require the
  colon to be the LAST non-whitespace emitted, not just within 16 tokens).
- **Injecting when the model would have called the tool anyway** — harmless
  (it forces the same outcome a turn earlier).
- Keep `DFLASH_MIN_TOKENS` **unset** (0) so the blunt min-length suppressor
  stays off — we only want the colon-aware path.

---

## Validation plan (operator/me will run; hand back the results)

Test instance on **port 8084** (stop `dflash.service` — single GPU; prod has
live traffic, coordinate). Launch = production launcher + `DFLASH_COLON_TOOL_GUARD=1`.

1. **Reproduce + fix the stall.** From `/mnt/models/.cache/dflash-traffic.jsonl`
   pull recent records with `finish=stop` whose `response_text` ends with `:`
   (announce-then-stop). Replay each against 8084 at the same temperature.
   PASS = they now end `finish=tool_calls` with a well-formed
   `tool_calls[].function.arguments` (parseable JSON), instead of stopping.
   Grep the log for `[spec-tool-floor]` lines confirming the guard fired.
2. **No regression on short answers.** Replay records that legitimately ended
   `finish=stop` with a PERIOD (e.g. "Готово.", short confirmations). They must
   STILL stop at the same short length (guard must NOT fire — no recent `:`).
3. **Tool-call bench.** `/tmp/tree_bench.py 8084` → 12/12.
4. **No degeneration.** A long sampled generation stays coherent (consec-dup
   < 10%) — confirm the injection doesn't trigger runaway.

Then restore prod (kill -9 the test instance, WAIT for `ss :8084` free before
relaunch — a graceful shutdown holds the port through an in-flight request;
`sudo systemctl start dflash.service`).

---

## Rollout
- `DFLASH_COLON_TOOL_GUARD` env, default **off**. Validate via the plan.
- Enable in `server/scripts/launch_prod_3090.sh` (`export
  DFLASH_COLON_TOOL_GUARD=1`) after validation, restart prod.
- Commit as `Rhonstin <rhonstin@gmail.com>`, no AI co-author lines.
- Branch off `prod-3090-pflash`. Do NOT also set `DFLASH_MIN_TOKENS`.

## Files
- `server/src/qwen35/qwen35_backend.cpp` (Edits 1-3; optional 4)
- `server/scripts/launch_prod_3090.sh` (rollout enable)
- no header/ABI changes for v1 (Edit 4 would change `do_ar_decode`'s signature)
