# PFlash/KVFlash Port Status

## Base

- Fork path: `${LLAMA_CPP_DIR}`
- Base upstream commit: `dbdaece23de9ac63f2e7ca9e6bfcdc4fc156a3fa`
- Branch: `local-user/dflash-pflash-kvflash`

## Checkpoint 001 - CLI/config scaffold

Added experimental CLI/config fields for:

- `--kvflash N|auto`
- `--kvflash-policy lru|qk|drafter`
- `--kvflash-tau N`
- `--kvflash-drafter FNAME`
- `--prefill-drafter FNAME`
- `--pflash-drafter FNAME`
- `--pflash-keep-ratio N`
- `--pflash-mode off|always|auto`
- `--pflash-drafter-gpu N`

These parse and log at startup. Default behavior is unchanged.

## Checkpoint 002 - first PFlash runtime hook

Added a narrowly gated server task hook in `tools/server/server-context.cpp`:

- Only runs when `--pflash-mode always` and `--pflash-keep-ratio < 1.0`.
- Only applies to completion/infill tasks.
- Skips multimodal prompts.
- Keeps the first and last token and uniformly samples the middle tokens down to the requested keep ratio.
- Logs `experimental PFlash compressed task ...` when it modifies a prompt.

This is not the final Lucebox drafter-driven PFlash algorithm. It is a safe integration slice that proves the upstream server can route prompt-token compression through the task boundary without changing default behavior. The next step is replacing the uniform thinning placeholder with real drafter scoring/compression.

## Checkpoint 003 - PFlash drafter-helper bridge

Reworked the PFlash hook so `--pflash-drafter` is no longer only a logged path:

- If `--pflash-drafter` points to an executable helper, the server writes the prompt token IDs to a temporary file and runs:
  - `<helper> <token-file> <keep-ratio> <pflash-drafter-gpu>`
- The helper is expected to print surviving token IDs to stdout, separated by whitespace, commas, or semicolons.
- A successful helper result replaces the prompt tokens and logs `drafter=helper`.
- If no executable helper is configured, or the helper fails, the hook falls back to checkpoint-002 uniform compression and logs `drafter=uniform-fallback`.
- Invalid helper output is rejected if it is empty, shorter than 2 tokens, or not actually compressed.

This is a bridge for connecting the Lucebox drafter scorer/IPC daemon without pulling the whole custom backend into upstream in one patch. A true in-process GGUF drafter loader is still pending.

## Checkpoint 004 - PFlash helper contract hardening

Tightened the checkpoint-003 helper bridge so external drafter output cannot silently mutate prompts into an unsafe shape:

- Temporary token files are now created with `mkstemp` under the system temp directory instead of a predictable `this`-pointer filename.
- Helper output must preserve the first and last prompt token.
- Helper output must be an ordered subsequence of the original prompt tokens.
- Helper output must be no longer than the computed keep target from `--pflash-keep-ratio`.
- The built-in uniform fallback now goes through the same validation path before prompt mutation.

This checkpoint keeps the helper bridge external and experimental, but makes the protocol strict enough for a real Lucebox scorer/helper to plug in without accepting reordered, invented, or under-compressed token streams.

## Checkpoint 005 - KVFlash visibility scaffold

Started the KVFlash runtime side without changing KV-cache behavior:

- Server initialization now logs whether experimental KVFlash is disabled or configured-but-not-active.
- The `/props` response now includes a `kvflash` object with configured tokens, policy, tau, drafter, and `active=false`.
- `--kvflash` remains a visibility/configuration scaffold only; no KV allocation, eviction, paging, attention-mask, or recall behavior is changed.

This checkpoint makes KVFlash configuration observable before any risky KV-cache mutation work.

## Checkpoint 006 - KVFlash resident pool skeleton

Added an inert resident-pool state object for future KVFlash work:

- Tracks configured capacity, resident token count, entries, hits, misses, and evictions.
- Initializes only when `--kvflash` is configured.
- Exposes the counters under `/props` as `kvflash.resident_pool`.
- Logs the zeroed resident-pool counters at startup when KVFlash is configured.
- Keeps `active=false` and does not allocate, evict, page, recall, or mutate any llama KV cache state.

This checkpoint creates observable state for later dry-run scoring and eviction simulation without changing model execution.

## Checkpoint 007 - KVFlash dry-run scoring hooks

Added a dry-run scoring hook at the server task boundary:

- Runs only when `--kvflash` is configured and only for completion/infill prompts.
- Observes prompt length, policy, tau, and candidate-token budget before PFlash may alter the prompt.
- Increments inert `dry_run_scores`, `dry_run_candidate_tokens`, and miss counters for visibility.
- Exposes those counters in `/props` under `kvflash.resident_pool`.
- Logs dry-run scoring at debug level with `active=false`.
- Does not change task tokens, slot selection, KV cache state, attention masks, or generation behavior.

This checkpoint creates the first policy-observation path needed for later eviction simulation while keeping KVFlash non-active.

## Checkpoint 008 - KVFlash eviction simulation

Extended the dry-run KVFlash path with eviction simulation counters:

- Computes whether a prompt candidate would overflow the configured resident token budget.
- Accumulates `simulated_eviction_checks` and `simulated_eviction_candidates` for `/props` visibility.
- Logs simulated eviction candidates at debug level while keeping `active=false`.
- Leaves real `entries`, `resident_tokens`, and `evictions` unchanged.
- Does not call llama KV cache remove, clear, update, paging, or attention-mask paths.

This checkpoint previews eviction pressure without performing eviction or changing generation behavior.

## Checkpoint 009 - reproducible PFlash helper target

Added a checked-in reference helper for the existing external PFlash helper protocol:

- `tools/server/pflash-uniform-helper.py`
- Accepts `<token-file> <keep-ratio> <pflash-drafter-gpu>`.
- Reads one token id per line from the token file.
- Emits a first/last-preserving ordered subsequence using the same deterministic uniform strategy as the built-in fallback.
- Validates argument count, keep-ratio range, token-file existence, and token parsing.
- Leaves in-process GGUF drafter loading/scoring still pending; this is a reproducible IPC/helper target for integration and testing.

This checkpoint gives `--pflash-drafter` a repository-local executable target that exercises the hardened helper bridge without requiring Lucebox drafter model loading yet.

## Checkpoint 010 - live model runtime validation attempt

Attempted live runtime validation with the forked `build-hip-gfx1151/bin/llama-server` on port 18080 using `--no-mmap`, configured KVFlash, PFlash always mode, and the checked-in PFlash helper.

Results:

- `dflash-draft-3.6-q4_k_m.official-dflash-compat.gguf` failed to load because tokenizer metadata was missing: `key not found in model: tokenizer.ggml.model`.
- `dflash-draft-3.6-q4_k_m.official-dflash-compat-tokenizer.gguf` failed to load because it was not a complete standalone model: `missing tensor 'fc.weight'`.
- `qwen3.6-27b-dflash-zlab-q8_0.gguf` failed context initialization because standalone DFlash requires a paired context: `dflash requires ctx_other to be set`.
- An existing healthy server on port 8080 was checked, but it is a separate CUDA build from `${HOME}/experiments/llamacpp-master-20260610` and does not expose this fork's KVFlash `/props` field, so it cannot validate this workspace.

Live validation is therefore blocked until a standalone compatible GGUF is available for this fork, or the correct paired DFlash launch command/model set is provided. No running validation server was left behind.

## Checkpoint 011 - broader test validation

Built the missing CTest executables and ran broader validation:

- Initial `ctest --test-dir build-hip-gfx1151 --output-on-failure` failed because test executables had not been built.
- Built the missing test targets, then reran full CTest.
- Full CTest exposed and fixed one real local regression: argument aliases now list `--pflash-drafter` before the longer `--prefill-drafter`, as required by `test-arg-parser`.
- `test-arg-parser` passed after the fix.
- Full CTest then had one remaining unrelated external-data failure: `test-tokenizers-ggml-vocabs` downloaded several invalid Git LFS pointer files and failed with `invalid magic characters: 'vers', expected 'GGUF'`.
- The suite excluding that external-data test passed: `ctest --test-dir build-hip-gfx1151 -E '^test-tokenizers-ggml-vocabs$' --output-on-failure` reported `100% tests passed, 0 tests failed out of 53`.

This is broader build/test validation for the fork. It is not live model runtime validation because Checkpoint 010 is blocked by unavailable compatible live model inputs.

## Checkpoint 012 - paired DFlash live smoke validation

Validated this fork with the known-good paired Qwen3.6 DFlash model set that previously benchmarked successfully on the upstream DFlash server:

- Target model: `${EXPERIMENTS_DIR}/lucebox-campaign/models/target-unsloth/Qwen3.6-27B-Q4_K_M.gguf`.
- Draft model: `${INFERENCE_WORKSPACE}/ops/lucebox-kvflash-hermes-test-20260628/models/hf-spiritbuun-qwen36-dflash/dflash-draft-3.6-q8_0.official-dflash-compat.gguf`.
- Launch used this fork's `build-hip-gfx1151/bin/llama-server` with `--no-mmap`, `--spec-type draft-dflash`, `-md`, `--jinja`, Q8 KV cache, configured KVFlash, and the checked-in PFlash helper.
- The legacy reference flag `-cd 4096` is not accepted by this fork's current CLI, so the successful smoke omitted it.
- `/health` returned `{"status":"ok"}` on port 18080.
- `/props` showed this fork's `kvflash` object with `configured=true`, `active=false`, `tokens=1024`, `policy=qk`, and initialized zeroed resident-pool counters.
- Live `/completion` requests generated text and reported speculative DFlash timing fields, including `draft_n` and `draft_n_accepted`.
- Server logs showed `common_speculative_impl_draft_dflash: adding speculative implementation 'draft-dflash'` and PFlash helper compression such as `experimental PFlash compressed task ... drafter=helper`.
- No validation server was left running after the smoke.

This checkpoint resolves the previous live-validation blocker for this fork at smoke-test scope. It is not a full benchmark pass and does not activate real KVFlash mutation; KVFlash remains configured but inactive.

## Checkpoint 013 - paired DFlash live benchmark validation

Ran the full six-prompt benchsplit task set against this fork with the paired Qwen3.6 DFlash model set:

- Run directory: `checkpoint-013-live-bench-20260629T152302Z`.
- Summary: `checkpoint-013-live-bench-20260629T152302Z/summary.jsonl`.
- Target model: `${EXPERIMENTS_DIR}/lucebox-campaign/models/target-unsloth/Qwen3.6-27B-Q4_K_M.gguf`.
- Draft model: `${INFERENCE_WORKSPACE}/ops/lucebox-kvflash-hermes-test-20260628/models/hf-spiritbuun-qwen36-dflash/dflash-draft-3.6-q8_0.official-dflash-compat.gguf`.
- Launch used this fork's `build-hip-gfx1151/bin/llama-server` with `--no-mmap`, `--spec-type draft-dflash`, `-md`, `--jinja`, Q8 KV cache, configured KVFlash, and PFlash disabled.
- KVFlash remained configured but inactive: `configured=true`, `active=false`, `tokens=1024`, `policy=qk`, `tau=64`.
- All six benchmark quality checks passed:
  - `speed-short-list`: 43 completion tokens, 17.8 tok/s wall-clock, `draft_n=135`, `draft_n_accepted=35`, qscore 0.9.
  - `speed-code`: 93 completion tokens, 38.4 tok/s wall-clock, `draft_n=135`, `draft_n_accepted=85`, qscore 1.0.
  - `speed-structured-json`: 385 completion tokens, 35.7 tok/s wall-clock, `draft_n=690`, `draft_n_accepted=340`, qscore 1.0.
  - `reasoning-trains`: 1025 completion tokens, 25.1 tok/s wall-clock, `draft_n=2670`, `draft_n_accepted=848`, qscore 1.0.
  - `agent-tool-decision`: 46 completion tokens, 18.2 tok/s wall-clock, `draft_n=135`, `draft_n_accepted=38`, qscore 1.0.
  - `long-gen-essay`: 1066 completion tokens, 12.3 tok/s wall-clock, `draft_n=5745`, `draft_n_accepted=684`, qscore 1.0.
- Logs showed DFlash draft acceptance lines and `common_speculative_impl_draft_dflash: adding speculative implementation 'draft-dflash'`.
- No validation server was left running after the benchmark.

An earlier same-script attempt with `--pflash-mode always --pflash-keep-ratio 0.5` is preserved at `checkpoint-013-live-bench-20260629T152209Z`. It returned HTTP 200 for all six tasks but generated only one empty completion token per task. Logs show the uniform helper compressed chat-template token sequences, e.g. `experimental PFlash compressed task ... drafter=helper`. That run is a negative finding for the placeholder PFlash hook on chat-completion workloads, not a benchmark pass.

This checkpoint validates the fork's paired DFlash path under a full six-task live benchmark. It does not validate real PFlash scoring or real KVFlash mutation; PFlash was disabled for the passing benchmark and KVFlash remained configured-but-inactive.

## Checkpoint 014 - PFlash chat-template safety guard

Added a narrow safety guard to the placeholder/helper PFlash hook:

- PFlash prompt compression still requires `--pflash-mode always` and `--pflash-keep-ratio < 1.0`.
- It now skips chat-formatted request paths before mutating prompt tokens:
  - `TASK_RESPONSE_TYPE_OAI_CHAT`
  - `TASK_RESPONSE_TYPE_OAI_RESP`
  - `TASK_RESPONSE_TYPE_ANTHROPIC`
- The skip logs: `PFlash compression skipped for task ... chat-formatted prompts are not supported yet`.
- Native `/completion`, OAI `/v1/completions`, and infill remain eligible for the existing experimental helper/uniform compression path.

Reason: Checkpoint 013 showed that uniformly compressing already-templated chat prompts can remove structurally important chat-template/control tokens and cause empty one-token completions. This guard prevents that silent benchmark-quality failure while preserving the experimental raw completion hook.

Live verification:

- Run directory: `checkpoint-014-pflash-chat-guard-20260629T152815Z`.
- Launched paired DFlash with `--pflash-mode always --pflash-keep-ratio 0.5` and the checked-in helper.
- Sent a `/v1/chat/completions` request.
- Response returned HTTP 200 with content: `Red Blue Green Yellow Purple`.
- Logs contained the new chat-formatted skip message.
- Logs did not contain `experimental PFlash compressed task` for the chat request.
- Response timings included DFlash activity with `draft_n > 0`.
- No validation server was left running after the check.

## Checkpoint 015 - Z Lab DFlash benchmark and matrix comparison

Reran the six-prompt benchsplit validation against this fork using the self-made Z Lab DFlash drafter instead of the earlier SpiritBuun draft GGUF:

- Run directory: `checkpoint-015-zlab-dflash-bench-20260629T154102Z`.
- Summary: `checkpoint-015-zlab-dflash-bench-20260629T154102Z/summary.jsonl`.
- Comparison: `checkpoint-015-zlab-dflash-bench-20260629T154102Z/comparison.json`.
- Target model: `${EXPERIMENTS_DIR}/lucebox-campaign/models/target-unsloth/Qwen3.6-27B-Q4_K_M.gguf`.
- Draft model: `${MODEL_DIR}/z-lab-Qwen3.6-27B-DFlash/qwen3.6-27b-dflash-zlab-q8_0.gguf`.
- Launch used this fork's `build-hip-gfx1151/bin/llama-server` with `--no-mmap`, `--spec-type draft-dflash`, `-md`, `--jinja`, Q8 KV cache, configured KVFlash, and PFlash disabled.
- KVFlash remained configured but inactive: `--kvflash 1024 --kvflash-policy qk --kvflash-tau 64`.
- All six benchmark quality checks passed: 2584 completion tokens in 134.231s, weighted 19.25 tok/s.

Comparison against referenced matrix/baseline files:

| Anchor | Pass | Tokens | Wall s | Weighted tok/s | Delta vs CP015 |
| --- | ---: | ---: | ---: | ---: | ---: |
| CP015 current fork + Z Lab DFlash | 6/6 | 2584 | 134.231 | 19.25 | baseline |
| upstream_dflash_zlab_q8 reference | 6/6 | 2584 | 135.643 | 19.05 | +1.05% |
| upstream_no_spec reference | 6/6 | 2568 | 220.250 | 11.659 | +65.11% |
| upstream_mtp reference | 6/6 | 2710 | 179.963 | 15.059 | +27.83% |
| matrix_lucebox_fast_recipe_prefix32 | 6/6 | 2521 | 152.672 | 16.513 | +16.57% |
| matrix_lucebox_fast_recipe_prefix0 | 6/6 | 2521 | 153.002 | 16.477 | +16.83% |
| matrix_lucebox_swa_ddtree | 6/6 | 2609 | 158.160 | 16.496 | +16.69% |
| matrix_lucebox_dflash_ddtree_on | 6/6 | 2797 | 177.802 | 15.731 | +22.37% |
| matrix_lucebox_dflash_ddtree_off | 6/6 | 2617 | 162.078 | 16.147 | +19.22% |
| matrix_lucebox_no_dflash_no_ddtree | 6/6 | 2625 | 232.948 | 11.269 | +70.82% |

Source comparison files:

- `${INFERENCE_WORKSPACE}/ops/lucebox-kvflash-hermes-test-20260628/benchsplit-six-matrix-20260629T010545Z/summary.jsonl`
- `${INFERENCE_WORKSPACE}/ops/lucebox-kvflash-hermes-test-20260628/benchsplit-six-upstream-rerun-20260629T015345Z/summary.jsonl`
- `${INFERENCE_WORKSPACE}/ops/lucebox-kvflash-hermes-test-20260628/benchsplit-six-upstream-dflash-zlab-q8-20260629T025201Z/summary.jsonl`

This supersedes Checkpoint 013's SpiritBuun-drafter benchmark as the preferred DFlash validation baseline for this fork.

## Checkpoint 016 - PFlash edge-preserving prompt compression

Advanced the experimental PFlash hook beyond first/last-token preservation:

- Added a shared edge-preservation rule for server validation and the reference helper.
- PFlash compressed prompts must now preserve a prefix and suffix of up to 64 tokens, bounded by source length and keep target.
- The built-in uniform fallback now keeps those prompt edges and only thins the middle.
- The checked-in helper `tools/server/pflash-uniform-helper.py` now uses the same edge-preserving thinning strategy.
- Helper output validation rejects compressed prompts that do not preserve the required prompt edges.
- Chat-formatted prompts shorter than 128 tokens are skipped rather than compressed, because there is not enough room to preserve template/control-token edges safely.
- Long chat-formatted prompts are now eligible for the edge-preserving experimental path.

Live verification:

- Run directory: `checkpoint-016-pflash-edge-guard-20260629T154539Z`.
- Summary: `checkpoint-016-pflash-edge-guard-20260629T154539Z/summary.json`.
- Launched paired DFlash with the Z Lab drafter, `--pflash-mode always --pflash-keep-ratio 0.5`, and `--pflash-drafter tools/server/pflash-uniform-helper.py`.
- Short `/v1/chat/completions` request returned HTTP 200 with `red blue green` and logged the short-chat skip.
- Long `/v1/chat/completions` request returned HTTP 200 with `The text consists of repetitive sequences of Greek letters. DONE.`
- Logs showed `experimental PFlash compressed task ... drafter=helper` for the long chat request.
- Logs did not show any edge-preservation rejection.
- No validation server was left running after the check.

This makes PFlash usable for bounded edge-preserved prompt compression experiments. It is still not the final Lucebox PFlash algorithm: real in-process PFlash drafter GGUF loading/scoring remains pending.

## Checkpoint 017 - PFlash keep-index scorer contract

Replaced the external helper's token-rewrite protocol with a keep-index scorer contract:

- Helper input is now JSON with `tokens`, `keep_ratio`, `preserve_prefix`, `preserve_suffix`, and `structural_indices`.
- Helper output is now JSON with `keep_indices` and optional `scores`.
- The server maps keep indices back to original token IDs itself.
- Server-side validation requires sorted unique in-range indices, preserved prompt edges, and preserved structural-token indices.
- `tools/server/pflash-uniform-helper.py` was updated as the reference implementation for this protocol.

This is the clean abstraction needed for Lucebox-style scoring: scorers choose original prompt positions; the server owns prompt mutation safety.

## Checkpoint 018 - PFlash scorer backend skeleton

Added an explicit scorer backend selection path:

- New CLI/config field: `--pflash-model FNAME`.
- New CLI/config field: `--pflash-score uniform|external-helper|model`.
- Startup logging now reports PFlash mode, keep ratio, backend, helper path, model path, and whether the model path is present.
- `external-helper` uses the JSON keep-index helper contract.
- `model` is an in-process scorer skeleton that validates/owns the configured model path and currently uses the same deterministic keep-index heuristic as a placeholder scorer.
- `uniform` remains available as the built-in no-helper fallback.

This proves the server owns a PFlash scorer mode and model path separately from the external helper. The actual GGUF forward/scoring algorithm is still a placeholder in this checkpoint.

## Checkpoint 019 - PFlash structural-token preservation

Added structural preservation on top of edge preservation:

- BOS/EOS/EOG/control tokens are marked structural using the llama vocab APIs.
- Token pieces containing newline boundaries, chat sentinel fragments, role/tool words, or template-ish delimiters are marked structural.
- Structural indices are included in helper requests.
- Server validation rejects any scorer/helper output that drops a structural index.
- `/props.pflash.structural_tokens` accumulates observed structural-token count.

This keeps PFlash from compressing through chat/template/control structure even when the prompt is long enough to compress.

## Checkpoint 020 - PFlash to KVFlash residency hints

Connected the PFlash selection path to the existing inert KVFlash resident-pool scaffold:

- When PFlash compresses a prompt and KVFlash is configured, the selected keep indices are recorded as residency hints.
- `/props.pflash.kvflash_hints` and `kvflash_hint_tokens` report hint activity.
- `/props.kvflash.resident_pool.entries` and `resident_tokens` are updated as dry-run/hint visibility counters.
- KVFlash still reports `active=false`; this does not allocate/pin/purge llama KV cache, change attention masks, page KV blocks, or alter recall behavior yet.

This is the bridge from PFlash scoring to KVFlash planning without mutating llama.cpp KV-cache internals prematurely.

## Checkpoint 021 - first non-uniform model-score selector

Advanced `--pflash-score model` beyond the CP018 pure-uniform placeholder:

- The model backend now runs a hybrid deterministic token-importance selector instead of reusing the uniform spacing selector.
- The selector seeds the keep set with uniform middle anchors, then fills the remaining budget with higher-scored content/structure/position tokens so coverage is not lost.
- It still requires `--pflash-model` to point at an existing model file and reports `model_loaded=true` for that configured scorer path.
- It keeps the same hard safety rules: preserved prompt edges, preserved structural indices, sorted unique keep indices, and server-owned token mutation.
- `/props.pflash.model_score_tokens` and `model_score_selected_tokens` expose how many tokens the model-score path evaluated and selected.

Important limitation: this is still not full GGUF/logprob Lucebox PFlash scoring. It is a scored in-process selector based on token structure/content/position so CP021 can exercise a real non-uniform selection path while the GGUF forward scoring integration remains pending.

## Checkpoint 022 - stratified scored PFlash fill

Reduced the CP021 model-score selector's tendency to over-cluster high-scoring middle tokens:

- The model backend still preserves prompt edges and structural indices first.
- It still seeds uniform middle anchors for broad prompt coverage.
- The remaining scored budget is now filled from up to 32 prompt segments in round-robin order.
- Each segment is internally ordered by the existing token-importance score, so selection remains non-uniform while covering the prompt more evenly.

This is a stability step toward usable PFlash behavior after one verifier prompt shape exposed an empty-output failure with the earlier global top-k fill. It is still not full GGUF/logprob Lucebox PFlash scoring.

## Checkpoint 023 - scored-selector telemetry

Added `/props.pflash` telemetry for the CP022 stratified model-score path:

- `model_score_anchor_tokens` counts middle coverage anchors added by the model backend.
- `model_score_stratified_tokens` counts tokens added by the scored round-robin segment fill.
- `model_score_segments` counts the segment buckets considered across model-score requests.
- `model_score_nonempty_segments` counts segment buckets that had candidate tokens.

This makes CP022 observable from the public server status endpoint: a verifier can now distinguish the stratified scored path from uniform selection or a silent fallback. It is still diagnostic accounting only and does not implement full GGUF/logprob Lucebox PFlash scoring.

## Checkpoint 024 - Lucebox-style anchor-window selector

Established the current stable PFlash selector reference for this fork:

- Anchor-window selector over the token-score path.
- Useful as the stable comparison point for later experimental selector changes.
- Do not promote later selector patches over CP024 until robustness testing justifies it.

## Checkpoint 025 - candidate exposure scoring

Experimental patch only:

- Kept as an investigative selector variant.
- Do not promote over CP024.

## Checkpoint 026 - PFlash debug export

Added an env-gated diagnostic/export path for compressed prompts and kept spans:

- Used to check exactly whether a critical line survives prompt compression before tuning selectors further.
- Diagnostic only; not a selector promotion by itself.

## Checkpoint 027 - dense fact span preservation

Validated current best candidate for the raw PFlash selector path under the tested chat-template configuration:

- Adds dense fact span preservation on top of the CP024 lineage.
- CP028 apples-to-apples chat-kwargs sanity showed CP027 raw PFlash chat passed 8k, 16k, and 24k while CP024 failed the same raw PFlash chat cases.
- CP029 critical-line robustness matrix passed 27/27 cases across 8k, 16k, and 24k contexts and 9 critical-line formats.
- Promote CP027 over CP024 specifically for the raw PFlash selector path with `/v1/chat/completions`, `--chat-template-kwargs '{"enable_thinking":false}'`, `--pflash-score model`, and `--pflash-keep-ratio 0.10`.
- Do not call CP027 canonical stable for every PFlash mode until broader suite validation passes.

## Checkpoint 028 - chat-kwargs sanity harness

Added reproducible harness:

- `tools/server/bench_cp028_pflash_chat_kwargs_sanity.py`
- Tests `/v1/chat/completions` with `--chat-template-kwargs '{"enable_thinking":false}'`.
- Runs 8k, 16k, and 24k direct chat vs raw PFlash chat with `--pflash-mode always --pflash-keep-ratio 0.10 --pflash-score model`.
- Records exact value checks, timings, finish reasons, completion tokens, and CP026 debug export summaries when available.

Focused ad-hoc CP028 comparison results:

| checkpoint | direct chat 8k/16k/24k | raw PFlash chat 8k/16k/24k | status |
| --- | --- | --- | --- |
| CP024 | PASS/PASS/PASS | FAIL/FAIL/FAIL | stable reference, not superseded |
| CP027 | PASS/PASS/PASS | PASS/PASS/PASS | validated current best candidate for tested raw PFlash selector path |

## Checkpoint 029 - critical-line robustness harness

Added reproducible harness:

- `tools/server/bench_cp029_pflash_critical_line_robustness.py`
- Tests `/v1/chat/completions` with `--chat-template-kwargs '{"enable_thinking":false}'`.
- Runs raw PFlash chat with `--pflash-mode always --pflash-keep-ratio 0.10 --pflash-score model`.
- Covers 8k, 16k, and 24k contexts across 9 critical-line formats: canonical single line, JSON object, pipe-delimited, markdown table, YAML block, XML attrs, multiline equals, prose sentence, and spaced equals.
- Records exact value checks and CP026 debug export summaries when available.

Focused CP029 robustness result:

- CP027 raw PFlash chat passed 27/27 cases.
- CP026 debug export showed all expected values survived compression in 27/27 cases.
- This validates CP027 as the current best candidate for the tested raw PFlash selector path, not as canonical stable for every PFlash mode.

Promotion-gate rerun after CP029 archival:

- `cmake --build build-hip-gfx1151 --target llama-server -j 8` passed.
- CP028 rerun artifact: `temp-cp027-promotion-gate-cp028-chat-kwargs-20260629T234132Z/cp027-promotion-gate-cp028_chat_kwargs_comparison.json`.
- CP028 rerun passed direct chat and raw PFlash chat at 8k, 16k, and 24k.
- CP029 rerun artifact: `temp-cp027-promotion-gate-cp029-critical-line-robustness-20260629T234523Z/cp027-promotion-gate-cp029_critical_line_robustness.json`.
- CP029 rerun passed 27/27 cases with 27/27 debug-compressed value hits.
- Older intermediate temp/checkpoint directories were moved under `artifact-archive-cp029-promotion-20260629/`; the key CP024, CP027, CP028, and CP029 evidence directories remain at repository root.

## Checkpoint 030 - 2-pass PFlash quality-gate shim

CP030 is the seed of the long-lived opt-in optimizer shim for the fork. It currently covers the tested chat-completions PFlash quality-gate behavior while keeping adaptive policy iteration outside the core server:

- Added `tools/server/bench_cp030_pflash_2pass_quality_gate.py` as a focused reference shim/harness.
- First pass uses the validated raw PFlash chat path: `/v1/chat/completions`, `--chat-template-kwargs '{"enable_thinking":false}'`, `--pflash-mode always`, `--pflash-keep-ratio 0.10`, and `--pflash-score model`.
- The quality gate checks whether the first pass preserved the expected `ROUTE`, `FUNC`, and `SENTINEL` facts by value presence; it is not yet a structured parser or general confidence model.
- If the first pass misses required facts, the shim runs a second direct verification pass over a selected original prompt span.
- The harness records direct, first-pass PFlash, gate decision, second-pass fallback, and smart 2-pass final result rows.
- Longer-term direction: this shim remains the productized opt-in optimizer shipped alongside the fork, and will also control dynamic DFlash parameter changes.

Focused CP030 validation:

| run | artifact | result |
| --- | --- | --- |
| smoke | `temp-cp030-smoke-2pass-quality-gate-20260630T004124Z/cp030-smoke_2pass_quality_gate.json` | PASS: 1/1 smart 2-pass, no fallback needed |
| forced fallback | `temp-cp030-forced-fallback-2pass-quality-gate-20260630T005044Z/cp030-forced-fallback_2pass_quality_gate.json` | PASS: 1/1 smart 2-pass, second pass executed by `--gate-mode always`; first pass did not miss facts |
| 8k/16k/24k matrix | `temp-cp030-matrix-2pass-quality-gate-20260630T005152Z/cp030-matrix_2pass_quality_gate.json` | PASS: 3/3 smart 2-pass, no fallback needed |
| 32k spot | `temp-cp030-long32k-2pass-quality-gate-20260630T005551Z/cp030-long32k_2pass_quality_gate.json` | PASS: 1/1 smart 2-pass, no fallback needed |
| review follow-up forced fallback | `temp-cp030-review-followup-forced-2pass-quality-gate-20260630T010556Z/cp030-review-followup-forced_2pass_quality_gate.json` | PASS: 1/1 smart 2-pass, `second_pass_ran=true`, trigger recorded as `forced_by_gate_mode_always` |

Promotion boundary:

- CP027/CP029 is promoted as the current validated raw model-score PFlash path for the tested chat-completions workload.
- CP030 is the current reference shim for tested chat-completions quality-gate behavior and the intended place to iterate smart-routing policy.
- Keep this as a separate long-lived optimizer shim rather than moving the policy into native server flags. The server should expose mechanisms; the shim should own routing, fallback, and adaptive quality/latency policy.
- Future shim scope should expand beyond PFlash quality gating to dynamic DFlash parameter changes, similar in spirit to the Drivetrain shim for MTP.
- `/completion`, uniform scoring, external-helper scoring, and non-raw PFlash modes remain experimental until they pass their own gates.

## Promotion package - CP029/CP030

Generated local promotion package:

- Report: `pflash-promotion-package-20260630/PROMOTION_REPORT.md`.
- Machine-readable evidence rollup: `pflash-promotion-package-20260630/promotion_evidence.json`.
- Promoted scope remains the raw model-score PFlash chat-completions slice: `/v1/chat/completions`, `--chat-template-kwargs '{"enable_thinking":false}'`, `--pflash-score model`, and `--pflash-keep-ratio 0.10`.
- Evidence rollup includes CP028 raw chat comparison, CP029 27-case robustness, CP029 32k spot, CP030 matrix, CP030 32k spot, forced-fallback path execution, and negative boundaries for `/completion`, uniform scoring, and external-helper scoring.
- CP028 aggregate in the package: direct passed 3/3, raw PFlash passed 3/3, raw PFlash total wall 20.933s vs direct total wall 198.520s for a 9.48x aggregate speedup on the focused synthetic chat retrieval task.
- CP030 matrix aggregate in the package: direct passed 3/3, first-pass PFlash passed 3/3, smart 2-pass passed 3/3, smart total wall 21.513s vs direct total wall 201.531s for a 9.37x aggregate speedup on the focused synthetic chat retrieval task.
- CP030 32k spot aggregate in the package: smart 2-pass passed 1/1, smart total wall 12.970s vs direct wall 144.019s for an 11.10x speedup on the focused synthetic chat retrieval task.

## Checkpoint 031 - in-process scorer and active KVFlash mutation plumbing

Added the first in-server implementation slice for the two remaining C-track items:

- `--pflash-score model` now loads `--pflash-model` as an in-process GGUF scorer context instead of only checking that the file exists.
- PFlash scoring runs a scorer-model decode and ranks non-structural candidates by scorer token surprisal. If target token ids are outside the scorer vocabulary, scoring is rejected and the existing uniform fallback path is used rather than pretending retokenized scorer scores map exactly to target tokens.
- `/props` exposes scorer execution counters: `scorer_evals`, `scorer_eval_failures`, and `scorer_eval_tokens`.
- KVFlash is no longer reported as merely configured-but-inactive. It now maintains active resident-pool accounting and can evict idle slot sequences from target/draft llama memory using `common_context_seq_rm` when resident accounting would overflow the configured KVFlash token budget.
- `/props` exposes KVFlash mutation counters: `memory_mutations` and `mutation_tokens_removed`.

Focused validation:

- Build passed: `cmake --build build-hip-gfx1151 --target llama-server -j 8`.
- Runtime smoke passed before scorer-compatibility hardening: `temp-cp031-scorer-kvflash-smoke2-2pass-quality-gate-20260630T013633Z/cp031-scorer-kvflash-smoke2_2pass_quality_gate.json`.
- Runtime guard smoke passed after scorer-compatibility hardening: `temp-cp031-scorer-guard-smoke-2pass-quality-gate-20260630T014234Z/cp031-scorer-guard-smoke_2pass_quality_gate.json`.
- Guard smoke result: 1/1 smart 2-pass passed; first-pass PFlash missed after scorer rejected the incompatible token ids and the shim recovered via the second pass.
- Qwen3.5-0.8B Q4_K_M scorer smoke passed: `temp-cp031-qwen35-08b-q4km-scorer-smoke-2pass-quality-gate-20260630T021521Z/cp031-qwen35-08b-q4km-scorer-smoke_2pass_quality_gate.json`.
- Qwen3.5-0.8B BF16 scorer matrix passed 8k/16k/24k: `temp-cp031-qwen35-08b-bf16-scorer-ab-2pass-quality-gate-20260630T022313Z/cp031-qwen35-08b-bf16-scorer-ab_2pass_quality_gate.json`.
- Qwen3.5-0.8B Q4_K_M scorer matrix passed 8k/16k/24k: `temp-cp031-qwen35-08b-q4km-scorer-ab-2pass-quality-gate-20260630T022834Z/cp031-qwen35-08b-q4km-scorer-ab_2pass_quality_gate.json`.
- Qwen3.5-0.8B Q4_K_M 32k spot passed: `temp-cp031-qwen35-08b-q4km-scorer-ab-32k-2pass-quality-gate-20260630T024551Z/cp031-qwen35-08b-q4km-scorer-ab-32k_2pass_quality_gate.json`.
- Qwen3.5-0.8B BF16 32k spot passed: `temp-cp031-qwen35-08b-bf16-scorer-ab-32k-2pass-quality-gate-20260630T024925Z/cp031-qwen35-08b-bf16-scorer-ab-32k_2pass_quality_gate.json`.
- PFlash server logs confirmed `PFlash in-process GGUF scorer loaded`, `model_loaded=true`, no token-id mismatch for the Qwen3.5-0.8B scorer pair, `experimental KVFlash active`, and successful PFlash compression.

Implementation boundary:

- This is in-process GGUF scorer plumbing with a strict same-token-id compatibility requirement, plus real llama-memory mutation plumbing for KVFlash idle-sequence eviction.
- The current available Qwen3.6 target plus Qwen3-0.6B scorer pair is not token-id compatible, so the true scorer path correctly rejects that pair and falls back rather than fabricating target-token scores.
- The Qwen3.5-0.8B GGUF scorer family is token-id compatible with the current Qwen3.6 target in runtime validation. Both local Q4_K_M and downloaded BF16 scorer variants loaded as in-process GGUF scorers and passed focused 8k/16k/24k plus 32k CP031 quality-gate runs without token-id mismatch.
- Q4_K_M is the practical default scorer candidate because it is small; BF16 is the quality-validation scorer candidate for A/B checks.
- KVFlash mutation currently evicts idle slot sequences under resident-pool pressure; it does not yet implement a hidden-sequence resident prefix pool, page-level intra-sequence KV recall, or sparse attention over non-contiguous resident pages.

## Checkpoint 032 - focused KVFlash pressure harness

Added `tools/server/bench_cp032_kvflash_pressure.py` as a focused multi-request pressure harness for the current KVFlash resident-pool mutation slice.

Focused validation:

- Build passed: `cmake --build build-hip-gfx1151 --target llama-server -j 8`.
- Runtime smoke passed: `python3 tools/server/bench_cp032_kvflash_pressure.py --label cp032-kvflash-pressure-smoke2 --prompt-tokens 800 --kvflash 64 --max-tokens 4`.
- Follow-up accounting fix smoke passed: `python3 tools/server/bench_cp032_kvflash_pressure.py --label cp032-kvflash-pressure-accounting-fix2 --prompt-tokens 800 --kvflash 64 --max-tokens 4`.
- Promotion smoke passed after per-slot resident-hint ownership: `python3 tools/server/bench_cp032_kvflash_pressure.py --label cp032-kvflash-promotable-final --prompt-tokens 800 --kvflash 64 --max-tokens 4`.
- Artifacts: `temp-cp032-kvflash-pressure-smoke2-20260630T033133Z/cp032-kvflash-pressure-smoke2.json`, `temp-cp032-kvflash-pressure-accounting-fix2-20260630T035345Z/cp032-kvflash-pressure-accounting-fix2.json`, and `temp-cp032-kvflash-promotable-final-20260630T041513Z/cp032-kvflash-promotable-final.json`.
- Harness used local small target `${MODEL_DIR}/draft/Qwen3.5-0.8B-Q4_K_M.gguf`, `--kvflash 64`, `-np 2`, uniform PFlash at keep ratio 0.10, and three sequential `/completion` requests.
- Original final `/props` resident-pool counters exposed the bug: `entries=3`, `misses=3`, `evictions=2`, `memory_mutations=2`, `mutation_tokens_removed=2`.
- Fixed final `/props` resident-pool counters: `entries=3`, `misses=3`, `evictions=2`, `memory_mutations=2`, `mutation_tokens_removed=106`.
- Server log now confirms two `experimental KVFlash evicted idle sequence from memory` events with `removed_tokens=53 memory_positions=1 pos_min=55 pos_max=55`.

Implementation boundary:

- CP032 validates that current KVFlash pressure can reach the real llama-memory mutation path and expose mutation counters through `/props`.
- The one-position symptom was an accounting bug: eviction used the current llama-memory sequence position span as logical resident-token removal. For the Qwen3.5-0.8B smoke, that span is only `pos_min=55,pos_max=55` after generation, while the PFlash resident hint is 53 tokens. Eviction still clears the sequence with `common_context_seq_rm(..., -1, -1)`, but the logical resident-pool debit must use the resident hint/slot token budget, not the current memory position span.
- PFlash now records KVFlash hints by task and assigns them to the selected slot at launch, so idle-slot eviction debits the owning slot's resident hint instead of only relying on global resident-pool state.
- The pressure harness now asserts `mutation_tokens_removed > memory_mutations` so a one-token-per-eviction regression fails.

## Remaining engineering after CP032

These are not part of the CP032 completion claim:

- Improve scorer/token alignment from aggregate retokenized scorer surprisal to exact span-aligned target-token importance.
- Investigate exact KVFlash resident-entry ownership for future multi-entry pools; CP032 still uses global resident-pool accounting, not per-entry page metadata.
- Implement page-level KVFlash recall/intra-sequence sparse residency if the product target requires non-contiguous resident pages instead of idle-sequence eviction.
- Expand the opt-in optimizer shim from PFlash 2-pass quality gating into dual-purpose PFlash plus dynamic DFlash policy control. Native server flags should remain low-level mechanisms rather than the canonical smart-routing interface.

## Checkpoint 033 - KVFlash page-directory recall metadata

Extended the CP032 resident-pool slice with page-directory metadata and hidden-prefix recall accounting:

- KVFlash resident hints now derive a page count using `--kvflash-tau` as the page size.
- `/props` exposes page and recall counters: `page_size`, `resident_pages`, `hidden_prefix_pools`, `hidden_resident_tokens`, `hidden_resident_pages`, `recall_attempts`, `recall_hits`, `recall_hit_tokens`, `page_hits`, and `sparse_attention_pages`.
- Idle eviction now records an evicted resident prefix in a hidden prefix directory before clearing the live slot sequence from llama memory.
- Later PFlash-compressed prompts probe that hidden directory and record page-level recall hits when a compatible resident prefix is found.
- The CP032 pressure harness gained `--recall-repeat`, which repeats a prompt after eviction and asserts page metadata plus hidden-prefix recall counters.

Focused validation:

- Build passed: `cmake --build build-hip-gfx1151 --target llama-server -j 8`.
- Harness syntax passed: `python3 -m py_compile tools/server/bench_cp032_kvflash_pressure.py`.
- Runtime smoke passed: `python3 tools/server/bench_cp032_kvflash_pressure.py --label cp033-kvflash-page-recall-smoke4 --prompt-tokens 800 --kvflash 64 --max-tokens 4 --recall-repeat`.
- Artifact: `temp-cp033-kvflash-page-recall-smoke4-20260630T125308Z/cp033-kvflash-page-recall-smoke4.json`.
- Final counters included `entries=4`, `evictions=3`, `memory_mutations=3`, `mutation_tokens_removed=159`, `resident_pages=4`, `hidden_prefix_pools=3`, `hidden_resident_pages=12`, `recall_hits=3`, `page_hits=12`, and `sparse_attention_pages=12`.

Implementation boundary:

- CP033 is still a server-level promotable slice, not a full llama-core sparse-attention backend. It makes page-level resident-prefix ownership, hidden-prefix pools, and recall accounting observable and testable in the server path.
- It does not yet implement true non-contiguous KV-cache page gathers inside the attention kernels, nor does it skip target-model prefill using recalled hidden pages. Those require deeper llama memory and attention-mask changes beyond the current server integration layer.

### CP034 - experimental core sparse KV attention gather

Status: focused promotable slice, not full canonical KVFlash paging.

Implemented a guarded core KV-cache sparse attention plan behind `LLAMA_KVFLASH_SPARSE_ATTN=1`:

- `llama_kv_cache_context` can install a compact per-ubatch sparse attention plan.
- K and V are gathered with real `ggml_get_rows` from non-contiguous KV cells via `get_k_sparse()` and `get_v_sparse()`.
- The KQ mask is compacted to the gathered K/V width with `set_input_kq_mask_sparse()`.
- The graph path uses gathered K/V tensors before `build_attn_mha()` when the sparse plan is active.
- The first slice is conservative: standard KV attention, flash-attention/non-transposed V, one sequence per ubatch, no SWA/Alibi/M-RoPE sparse mask support yet. Unsupported cases fall back to dense KV attention.

Verification:

```text
cmake --build build-hip-gfx1151 --target llama-server -j 8
python3 -m py_compile tools/server/bench_cp032_kvflash_pressure.py
LLAMA_KVFLASH_SPARSE_ATTN=1 python3 tools/server/bench_cp032_kvflash_pressure.py \
  --label cp034-kvflash-sparse-attn-final \
  --target ${MODEL_DIR}/gemma-4-e4b-heretic-gguf/gemma-4-E4B-it-heretic-Q8_0.gguf \
  --prompt-tokens 400 \
  --kvflash 64 \
  --max-tokens 2 \
  --recall-repeat \
  --flash-attn
```

Artifact:

```text
temp-cp034-kvflash-sparse-attn-final-20260630T132208Z/cp034-kvflash-sparse-attn-final.json
```

Final counters:

```text
status: ok
entries: 4
evictions: 3
memory_mutations: 3
mutation_tokens_removed: 84
recall_hits: 3
sparse_attention_pages: 6
```

Core activation evidence from server log:

```text
KVFlash sparse attention plan active: gathered_cells=2
activation_log_count: 14
```

Focused tempfile verifier:

```text
AD_HOC_VERIFY_PASS cp034_kvflash_core_sparse_attention
activation_log_count 14
memory_mutations 3
mutation_tokens_removed 84
sparse_attention_pages 6
checks 12
VERIFY_EXIT 0
CLEANUP_OK /tmp/hermes-verify-8m6rqf1n.py
```

Remaining work before claiming full canonical KVFlash paging:

- sparse M-RoPE/4D-position masks
- SWA sparse masks
- transposed-V sparse gather for non-flash attention
- page directory ownership in llama-core rather than token-cell gather only
- target prefill skipping from reusable hidden/KV pages
- state save/load for resident hidden pages

### CP035 - sparse KV gather SWA mask groundwork

Status: partial groundwork, not full canonical KVFlash paging.

Extended the experimental `LLAMA_KVFLASH_SPARSE_ATTN=1` compact KQ mask builder so it can apply SWA masking logic:

```text
llama_hparams::is_masked_swa(n_swa, swa_type, p0, p1)
```

for sparse plans. Follow-up review found that the ISWA/Gemma attention graph does not yet route K/V through `ggml_get_rows`, so the Gemma run below proves sparse-plan and mask construction only. It does not prove actual non-contiguous sparse K/V gather execution for ISWA/Gemma. Gather length is padded to a backend-friendly size and padded columns are masked out on graph paths that use the sparse gather tensors.

Verification:

```text
cmake --build build-hip-gfx1151 --target llama-server -j 8
python3 -m py_compile tools/server/bench_cp032_kvflash_pressure.py
LLAMA_KVFLASH_SPARSE_ATTN=1 python3 tools/server/bench_cp032_kvflash_pressure.py \
  --label cp035-kvflash-sparse-swa-gemma \
  --target ${MODEL_DIR}/gemma-4-e4b-heretic-gguf/gemma-4-E4B-it-heretic-Q8_0.gguf \
  --prompt-tokens 400 \
  --kvflash 64 \
  --max-tokens 2 \
  --recall-repeat \
  --flash-attn
```

Artifact:

```text
temp-cp035-kvflash-sparse-swa-gemma-20260630T133500Z/cp035-kvflash-sparse-swa-gemma.json
```

Final counters:

```text
status: ok
entries: 4
evictions: 3
memory_mutations: 3
mutation_tokens_removed: 84
recall_hits: 3
sparse_attention_pages: 6
```

Server log evidence:

```text
sparse attention plan active count: 28
sparse attention plan rejected count: 0
```

Safety check for M-RoPE/Qwen3.5:

```text
LLAMA_KVFLASH_SPARSE_ATTN=1 python3 tools/server/bench_cp032_kvflash_pressure.py \
  --label cp035-qwen-mrope-gated-safe \
  --target ${MODEL_DIR}/draft/Qwen3.5-0.8B-Q4_K_M.gguf \
  --prompt-tokens 200 \
  --kvflash 64 \
  --max-tokens 1 \
  --flash-attn
```

This passed with `status: ok`, confirming Qwen/M-RoPE remains gated instead of taking the unstable sparse path. An attempted M-RoPE sparse run activated the path and hit a ROCm illegal memory access, so M-RoPE sparse masks remain blocked pending backend/kernel investigation.

### CP036 - hidden prefix page descriptor records

Status: focused canonical-backend prerequisite, not actual KV tensor persistence and not prefill skip.

Replaced the server hidden-prefix pool from raw token vectors with `kvflash_hidden_prefix_record` records carrying:

```text
prefix tokens
page_size
resident_tokens
page descriptors: page_id, token_begin, token_end
```

This creates a stable descriptor layer for hidden resident-prefix pools. It still does not preserve K/V tensors after `seq_rm()`, and it does not attach recalled pages to a new sequence or skip target prefill. Those remain future core/backend steps.

Verification:

```text
cmake --build build-hip-gfx1151 --target llama-server -j 8
python3 tools/server/bench_cp032_kvflash_pressure.py \
  --label cp036-kvflash-hidden-page-descriptors \
  --prompt-tokens 800 \
  --kvflash 64 \
  --max-tokens 4 \
  --recall-repeat
```

Artifact:

```text
temp-cp036-kvflash-hidden-page-descriptors-20260630T133743Z/cp036-kvflash-hidden-page-descriptors.json
```

Final counters:

```text
status: ok
hidden_prefix_pools: 3
hidden_resident_tokens: 159
hidden_resident_pages: 12
hidden_page_descriptors: 12
recall_hits: 3
page_hits: 12
```

### CP037 - sparse gather correctness fixes from adversarial review

Status: focused correctness fix for the experimental sparse gather path, not full canonical KVFlash paging.

Reconciled the `deleg_e12a3adf` review against current head and fixed three real sparse-path hazards:

- sparse index input tensors are now populated with `set_input_sparse_attn_idxs()` before graph execution;
- graph reuse now distinguishes dense vs sparse topology and checks sparse index width;
- `sparse_attn_n_used` now records the unpadded gather width before padding, so padded duplicate rows are masked out instead of accidentally masking every row or attending to duplicates.

Also strengthened sparse-plan eligibility so every token in the ubatch must have exactly one identical sequence id before the compact gather path may assume `ubatch.seq_id[0][0]`.

Verification:

```text
cmake --build build-hip-gfx1151 --target llama-server -j 8
python3 -m py_compile tools/server/bench_cp032_kvflash_pressure.py
LLAMA_KVFLASH_SPARSE_ATTN=1 python3 tools/server/bench_cp032_kvflash_pressure.py \
  --label cp037-sparse-correctness-fixes-gemma \
  --target ${MODEL_DIR}/gemma-4-e4b-heretic-gguf/gemma-4-E4B-it-heretic-Q8_0.gguf \
  --prompt-tokens 400 \
  --kvflash 64 \
  --max-tokens 2 \
  --recall-repeat \
  --flash-attn
```

Artifact:

```text
temp-cp037-sparse-correctness-fixes-gemma-20260630T135857Z/cp037-sparse-correctness-fixes-gemma.json
```

Focused verifier:

```text
AD_HOC_VERIFY_PASS cp037_sparse_correctness_fixes
active_log_count 28
rejected_log_count 0
memory_mutations 3
mutation_tokens_removed 84
sparse_attention_pages 6
hidden_page_descriptors 6
VERIFY_EXIT 0
```

Qwen/M-RoPE safety smoke remained gated and did not crash:

```text
LLAMA_KVFLASH_SPARSE_ATTN=1 python3 tools/server/bench_cp032_kvflash_pressure.py \
  --label cp037-qwen-mrope-still-gated \
  --target ${MODEL_DIR}/draft/Qwen3.5-0.8B-Q4_K_M.gguf \
  --prompt-tokens 200 \
  --kvflash 64 \
  --max-tokens 1 \
  --flash-attn
```

Result artifact:

```text
temp-cp037-qwen-mrope-still-gated-20260630T135943Z/cp037-qwen-mrope-still-gated.json
```

Status was `ok`; no core sparse activation log was emitted for Qwen, preserving the M-RoPE safety gate.

### CP038 - gated Qwen/M-RoPE sparse attention experiment

Status: experimental gated slice, not default-enabled canonical KVFlash paging.

Added an explicit debug opt-in for M-RoPE sparse attention:

```text
LLAMA_KVFLASH_SPARSE_ATTN=1
LLAMA_KVFLASH_SPARSE_ATTN_MROPE=1
```

Without `LLAMA_KVFLASH_SPARSE_ATTN_MROPE=1`, models with `n_pos_per_embd() != 1` remain rejected from the sparse plan. Added a second debug verifier gate:

```text
LLAMA_KVFLASH_SPARSE_ATTN_VERIFY_MASK=1
```

This logs sparse mask verification for gathered cells and asserts zero mismatches. The verifier path exercised Qwen3.5 M-RoPE with `is_2d=1` and gathered lengths matching the earlier crash region.

Build and checks:

```text
git diff --check
cmake --build build-hip-gfx1151 --target llama-server -j 8
```

Focused Qwen/M-RoPE sparse mask verification:

```text
LLAMA_KVFLASH_SPARSE_ATTN=1 \
LLAMA_KVFLASH_SPARSE_ATTN_MROPE=1 \
LLAMA_KVFLASH_SPARSE_ATTN_VERIFY_MASK=1 \
python3 tools/server/bench_cp032_kvflash_pressure.py \
  --label cp038-qwen-mrope-sparse-mask-verify-800 \
  --target ${MODEL_DIR}/draft/Qwen3.5-0.8B-Q4_K_M.gguf \
  --prompt-tokens 800 \
  --kvflash 64 \
  --max-tokens 2 \
  --flash-attn
```

Artifact:

```text
temp-cp038-qwen-mrope-sparse-mask-verify-800-20260630T141239Z/cp038-qwen-mrope-sparse-mask-verify-800.json
```

Server log evidence:

```text
KVFlash sparse attention plan active: gathered_cells=64
KVFlash sparse mask verification checked=2401 mismatches=0 is_2d=1 n_used=49 n_sparse=64
KVFlash sparse mask verification checked=212 mismatches=0 is_2d=1 n_used=53 n_sparse=64
KVFlash sparse mask verification checked=54 mismatches=0 is_2d=1 n_used=54 n_sparse=64
```

Focused Qwen/M-RoPE sparse execution without verifier overhead:

```text
LLAMA_KVFLASH_SPARSE_ATTN=1 \
LLAMA_KVFLASH_SPARSE_ATTN_MROPE=1 \
python3 tools/server/bench_cp032_kvflash_pressure.py \
  --label cp038-qwen-mrope-sparse-exec-800 \
  --target ${MODEL_DIR}/draft/Qwen3.5-0.8B-Q4_K_M.gguf \
  --prompt-tokens 800 \
  --kvflash 64 \
  --max-tokens 2 \
  --flash-attn
```

Artifact:

```text
temp-cp038-qwen-mrope-sparse-exec-800-20260630T141259Z/cp038-qwen-mrope-sparse-exec-800.json
```

Result: `status: ok`; server log showed repeated `KVFlash sparse attention plan active: gathered_cells=64` and no ROCm illegal-memory fault.

Remaining limitations:

- M-RoPE sparse attention is still opt-in and debug-gated.
- Verification is a focused pressure harness, not full dense-vs-sparse logits equivalence.
- This does not yet attach hidden page descriptors as attention sources.
- This does not yet implement target prefill skipping from hidden resident pages.

## CP038 follow-up - Qwen M-RoPE sparse validation matrix and CP039 blocker

Additional Qwen/M-RoPE sparse attention validation was run after commit `ac361bf` with the explicit gates:

```text
LLAMA_KVFLASH_SPARSE_ATTN=1
LLAMA_KVFLASH_SPARSE_ATTN_MROPE=1
```

Results:

- `cp038-qwen-mrope-matrix-1500-verify`: status `ok`; sparse active logs `26`; mask verification lines `26`; zero nonzero mismatches; no ROCm fault. Observed `n_used` values included `91..101`.
- `cp038-qwen-mrope-matrix-3000-verify`: status `ok`; sparse active logs `26`; mask verification lines `26`; zero nonzero mismatches; no ROCm fault. Observed `n_used` values included `185..195`.
- `cp038-qwen-mrope-matrix-1500-recall-exec`: harness status `fail` only because the old recall-repeat assertions still expect loose descriptor recall hits. Sparse execution itself was active (`34` active logs) and no ROCm fault was observed.
- `cp038-qwen-mrope-matrix-3000-recall-exec`: harness status `fail` for the same recall-assertion reason. Sparse execution was active with gathered sizes up to `256`; no ROCm fault was observed.

Artifacts:

```text
temp-cp038-qwen-mrope-matrix-1500-verify-20260630T142224Z/cp038-qwen-mrope-matrix-1500-verify.json
temp-cp038-qwen-mrope-matrix-3000-verify-20260630T142226Z/cp038-qwen-mrope-matrix-3000-verify.json
temp-cp038-qwen-mrope-matrix-1500-recall-exec-20260630T142227Z/cp038-qwen-mrope-matrix-1500-recall-exec.json
temp-cp038-qwen-mrope-matrix-3000-recall-exec-20260630T142918Z/cp038-qwen-mrope-matrix-3000-recall-exec.json
```

CP039 hidden state restore was investigated but not implemented in this checkpoint. The honest blocker is that idle eviction currently sees only a tail position in live llama memory for these server completion paths, for example:

```text
pos_min=194 pos_max=194
```

or, in the shorter PFlash path:

```text
pos_min=52 pos_max=52
```

A hidden-prefix state snapshot taken at this point would not honestly contain the full prefix K/V needed for a prefill skip. The next CP039 slice therefore needs a capture point before the server collapses/removes prior prompt positions, or a core KV export/import API that can explicitly export the intended prefix/page range. Do not claim hidden-prefix state restore until `hidden_state_records > 0`, `hidden_restore_hits > 0`, and a reduced prompt-prefill count are all proven by a no-PFlash exact-prefix harness.

## CP038/CP039A follow-up - recall harness fix, Qwen stress, and capture trace

`tools/server/bench_cp032_kvflash_pressure.py` now has two explicit controls for the Qwen sparse-execution validation path:

- `--allow-sparse-exec-recall`: lets recall-repeat validation pass on graph-level sparse execution when hidden-prefix recall counters are expected to remain zero after exact-prefix matching fixes.
- `--ctx-size`: permits larger focused stress prompts without changing the default 4096-token context.

This keeps the old hidden-recall assertion path intact unless the sparse-execution validation flag is provided.

Fresh Qwen/M-RoPE recall-exec runs with:

```text
LLAMA_KVFLASH_SPARSE_ATTN=1
LLAMA_KVFLASH_SPARSE_ATTN_MROPE=1
LLAMA_KVFLASH_SPARSE_ATTN_VERIFY_MASK=1
```

Results:

- `cp038-qwen-mrope-recall-exec-fixed-1500`: status `ok`; sparse active logs `22`; mask verification lines `22`; zero nonzero mismatches; no ROCm fault. Hidden recall counters stayed zero, and the run passed only through `--allow-sparse-exec-recall`.
- `cp038-qwen-mrope-recall-exec-fixed-3000`: status `ok`; sparse active logs `22`; mask verification lines `22`; zero nonzero mismatches; no ROCm fault. Hidden recall counters stayed zero, and the run passed only through `--allow-sparse-exec-recall`.
- `cp038-qwen-mrope-recall-exec-fixed-6000`: status `ok`; sparse active logs `22`; mask verification lines `22`; zero nonzero mismatches; no ROCm fault. This reached `max_gathered=384` and `max_n_used=377`, above the earlier 256 gathered-cell region.

Artifacts:

```text
temp-cp038-qwen-mrope-recall-exec-fixed-1500-20260630T144320Z/cp038-qwen-mrope-recall-exec-fixed-1500.json
temp-cp038-qwen-mrope-recall-exec-fixed-3000-20260630T144330Z/cp038-qwen-mrope-recall-exec-fixed-3000.json
temp-cp038-qwen-mrope-recall-exec-fixed-6000-20260630T144638Z/cp038-qwen-mrope-recall-exec-fixed-6000.json
```

CP039A adds an env-gated trace only, enabled with:

```text
LLAMA_KVFLASH_TRACE_CP039A=1
```

The focused trace run confirmed the capture-point blocker directly:

```text
KVFlash CP039A capture trace: prompt_tokens=98 n_save=64 resident_tokens=64 page_size=16 pages=4 pos_min=97 pos_max=97 live_positions=1
```

Artifact:

```text
temp-cp039a-capture-trace-qwen-1500-warn-20260630T144558Z/cp039a-capture-trace-qwen-1500-warn.json
```

Interpretation: the hidden-prefix descriptor save path still has prompt tokens and descriptor metadata for `n_save=64`, but live llama memory exposes only one tail position at that point. This confirms CP039 hidden state restore needs an earlier capture point or a core KV range/page export path before it can honestly skip prefill.

### CP039A/CP039B follow-up - hidden state restore blocker narrowed

- Committed recall-exec harness/capture trace baseline in `b394aad`.
- Added env-gated lifecycle tracing under `LLAMA_KVFLASH_TRACE_CP039A=1`.
- Qwen/hybrid memory still reports only the recurrent tail via `seq_pos_min/max`, even immediately after decode:
  - artifact: `temp-cp039a-state-size-trace-20260630T150602Z`
  - representative hidden save point: `pos_min=97 pos_max=97 live_positions=1`
- The same trace shows `LLAMA_STATE_SEQ_FLAGS_NONE` does contain a large full sequence state at the hidden save point:
  - `state_full=21408280`, `state_partial=20201932`
  - conclusion: `seq_pos_min/max` is not sufficient evidence that full prompt K/V is unavailable for Qwen/hybrid.
- A first restore attempt that restored a generated-tail state and trimmed to the 64-token resident prefix was rejected as a usable path:
  - artifact: `temp-cp039b-hidden-state-restore-fixed-20260630T150941Z`
  - result: `status=ok`, no crash after switching to non-aborting trim, but `hidden_restore_hits=0`
  - blocker: trim to the 64-token resident prefix is unsupported for the restored Qwen/hybrid state (`KVFlash hidden state restore trim failed: tokens=64`).
- CP039B now proves the narrower honest bridge: prompt-only state snapshot/restore on exact compressed-prompt match, still behind `LLAMA_KVFLASH_HIDDEN_STATE_RESTORE=1`.
  - artifact: `temp-cp039b-prompt-state-restore-20260630T151146Z`
  - result: `status=ok`, `hidden_state_records=3`, `hidden_restore_attempts=3`, `hidden_restore_hits=1`, `hidden_restore_tokens=95`
  - representative log: `KVFlash hidden state restore: tokens=95 state_size=21371356`
- Env-off regression remains unchanged:
  - artifact: `temp-cp039b-env-off-regression-20260630T151206Z`
  - result: `status=ok`, `hidden_state_records=0`, `hidden_restore_hits=0`
- Current honest claim: CP039B can restore full prompt state for exact compressed-prompt repeats. It does not yet restore arbitrary resident-prefix pages or trim Qwen/hybrid state down to the 64-token hidden resident prefix.
- CP039B prefill-reduction proof attempt:
  - artifact: `temp-cp039b-prefill-reduction-env-on-20260630T152306Z`
  - result: `status=fail` under the new harness assertion because `hidden_restore_hits=1` but `prompt_n_first=95` and `prompt_n_repeat=95`; restore happened, but prefill work was not reduced.
  - unsafe direct skip experiment was backed out: bypassing the hybrid checkpoint fallback reached the mandatory one-token logits path, but Qwen/M-RoPE rejected the batch because restored memory already ended at position 94 and the fallback token also started at 94 (`X < Y` violation).
  - next slice: capture or export a prompt state ending one token before the compressed prompt tail, or add a server path that can resume generation from restored logits without re-evaluating the final prompt token.

---

## 2026-07-01 PFlash keep-ratio vs smart-router ladder result

Matched A/B comparison:

- Router-off keep-ratio ladder: `/tmp/hermes-niah-sweep-20260701T035128Z/summary.json`
- Semantic-router keep-ratio ladder: `/tmp/hermes-niah-sweep-smart-router-20260701T150101Z/summary.json`
- HTML report: `/tmp/hermes-router-ab-comparison-20260701.html`

Result summary:

- Router-off 70-75% keep remains the recommended quality/speed control path for normal serving.
- The semantic smart router improved low-keep quality and preserved the 6/6 split tasks at every tested ratio, but its direct verification pass made wall-clock slower than the safe router-off 70% point.
- Low keep plus semantic router did not match the quality/wall-clock tradeoff of the router-off 70% safe point: it remained about 0.9 quality points lower and roughly 120-170 seconds slower in the matched ladder.
- Smart routing remains useful as an opt-in quality-control mechanism for strict extraction, exact literals, structured artifacts, and damaged-compression recovery, but it should not be the primary recommendation for broad quality tuning.

Current recommendation:

1. Prefer native/server PFlash controls first: keep ratio around the known safe 70-75% band, plus related PFlash flags.
2. Use the smart router for additional quality control when compression damage is likely, not as an always-on second pass.
3. The next router direction is dynamic keep-ratio adjustment, potentially combined with conditional second-pass verification and later dynamic DFlash decode settings.

Shim direction after this result:

- Invoke the direct second pass only when compression likely damaged salient content.
- Skip second pass for normal 6/6-style prompts when the PFlash candidate looks healthy.
- Fix single-literal/easy extraction span selection.
- Add dynamic keep-ratio policy so the router can choose safer high keep ratios for exact/enumerative/structured prompts and lower keep ratios for broad long-context prompts.

Follow-up corrected A/B on the built `llama-server` PFlash binary, not Lucebox:

- Runner: `/tmp/hermes-ab-llama-pflash-router-vs-fixed75.py`
- Correct binary: `${LLAMA_CPP_DIR}/build-hip-gfx1151/bin/llama-server`
- Corrected result: `/tmp/hermes-ab-llama-pflash-router-vs-fixed75-20260701T182812Z/summary.json`
- Lucebox contrast: `/tmp/hermes-ab-pflash-router-vs-fixed75-20260701T174244Z/summary.json`

| Backend / arm | NIAH | Split | Total | Wall | Avg q |
|---|---:|---:|---:|---:|---:|
| built `llama-server` `router_pflash_floor030_dynamic` | 4/4 | 6/6 | 10/10 | 205.58s | 0.956 |
| Lucebox `router_pflash_floor030_dynamic` | 4/4 | 6/6 | 10/10 | 320.66s | 0.928 |
| built `llama-server` `fixed_pflash_075_no_router` | 4/4 | 6/6 | 10/10 | 274.21s | 0.990 |
| Lucebox `fixed_pflash_075_no_router` | 3/4 | 6/6 | 9/10 | 268.85s | 0.929 |

Interpretation:

- On the built `llama-server` PFlash implementation, the dynamic router arm passed the whole 10-call battery and was 68.63s faster than fixed 0.75 keep (`-25.0%` wall-clock), with avg quality 0.956 vs 0.990.
- On Lucebox, the same router arm also passed 10/10 but was 115.08s slower than the built `llama-server` router arm (`+56.0%`), with lower avg quality.
- Lucebox fixed 0.75 looked superficially close on wall-clock, but failed Hard NIAH (`3/4` NIAH, `9/10` total), so it is not a correctness-preserving win.
- Do not use Lucebox A/B results as evidence against the built `llama-server` PFlash path; backend behavior differed materially. The current best result for this battery is built `llama-server` plus dynamic PFlash floor/router: `10/10`, `205.58s`, avg q `0.956`.

---

## 2026-07-01 Hermes Lucebox-failure replay on full-stack server

Added a focused replay harness for the exact old Lucebox/Hermes capture that previously exposed premature stop behavior:

- Harness: `tools/server/replay_lucebox_hermes_capture.py`
- Capture: `${INFERENCE_WORKSPACE}/ops/lucebox-kvflash-hermes-test-20260628/hermes-cli/captures/20260628T201617Z_0010_POST_v1_chat_completions.request.json`
- Live endpoint: `http://127.0.0.1:18160/v1/chat/completions`
- Server binary: `${LLAMA_CPP_DIR}/build-hip-gfx1151/bin/llama-server`
- Target alias for replay: `qwen36-fullstack-local`

Original Lucebox failure shape:

- Same Hermes-shaped Roman-gladiator prompt returned only a tiny premature response such as `ROMAN`.
- The target-only/no-draft path produced a normal long answer, so the failure was attributed to the Lucebox draft/DDTree/spec interaction.

Full-stack replay result from the checked-in harness:

- Artifact dir: `/tmp/hermes-lucebox-harness-04ESiQ`
- HTTP status: `200`
- Wall time: `54.61s`
- Stream chunks: `576`
- Finish reason: `stop`
- Response: `2474` chars, `364` words
- Old bug shape: `false`
- Log evidence: `draft acceptance` present, `truncated = 0` count `1`
- KVFlash observation: miss counter advanced by `1`
- PFlash observation: the captured prompt was only `726` tokens after chat formatting, below the configured `2048` token compression threshold, so PFlash logged a short-prompt skip and did not increment compression counters for this replay. This is expected for this small historical capture; larger same-session Hermes prompts already exercised PFlash compression on this same server.

Interpretation:

- The exact captured Lucebox/Hermes Roman-gladiator failure prompt does not reproduce the premature stop bug on the built full-stack `llama-server` DFlash path.
- This replay is a regression for the Lucebox premature-stop failure, not a PFlash-compression regression by itself because the historical prompt is below the configured compression threshold.
- For full-stack Hermes validation, pair this replay with the same-session Hermes CLI test that uses larger real agent prompts and confirms PFlash compression, KVFlash accounting, DFlash acceptance, and `truncated = 0`.

---

## 2026-07-01 next feature - dynamic DFlash and runtime router controls

The next shipping target is to extend the smart-router direction from prefill-only PFlash policy into decode-time DFlash policy. The goal is a single long-running `llama-server` whose router can tune both sides of the drivetrain without restarting the process:

- Prefill side: dynamic PFlash compression floor, keep ratio, and quality/verification passes.
- Decode side: dynamic DFlash settings chosen from measured speed/acceptance behavior.
- Runtime control: user-visible settings that can be toggled or customized while the server is up.

Desired control loop:

1. Classify request/task shape from prompt size, requested max tokens, route override, and content signals.
2. Choose a prefill gear: PFlash off/safe/high-compression, keep ratio/floor, scorer behavior, and optional verification pass.
3. Choose a decode gear: DFlash level/profile based on expected decode length and measured acceptance-rate tradeoff for that task class.
4. Observe actual output metrics, especially wall time, generated tokens, `draft_n`, `draft_n_accepted`, acceptance rate, truncation, and quality check result when a harness provides one.
5. Feed the measurements back into the gear map used by the router.

Systematic sweep plan:

- Prompt/task axes:
  - short factual/list
  - code generation/edit planning
  - strict JSON/structured output
  - reasoning/math/planning
  - tool-call selection
  - long essay/report
  - long-context retrieval/NIAH
  - Hermes same-session agent prompts
- Decode length axes:
  - short: roughly 32-128 generated tokens
  - medium: roughly 256-768 generated tokens
  - long: roughly 1024-4096 generated tokens
- DFlash axes:
  - baseline no-spec where practical
  - current default DFlash profile
  - lower, middle, and higher DFlash levels/profiles once exposed as runtime or request-level settings
- Prefill axes:
  - PFlash off
  - safe fixed keep around 0.70-0.75
  - dynamic floor/router policy around the measured floor-0.30 route
  - strict/exact mode with quality or direct verification pass

Primary metrics to record:

- pass/fail quality per task
- wall seconds
- prompt tokens and generated tokens
- prefill time, decode time, and weighted tok/s where available
- `draft_n`, `draft_n_accepted`, and acceptance rate
- PFlash original/kept token counts and scorer failures
- KVFlash resident/miss/eviction counters where active
- `truncated = 0` and absence of premature-stop bug shapes

Known benchmark anchor for the first prefill gear map:

- built `llama-server` dynamic PFlash floor/router: `10/10`, `205.58s`, avg q `0.956`
- built `llama-server` fixed PFlash 0.75: `10/10`, `274.21s`, avg q `0.990`
- dynamic prefill policy was 68.63s faster than fixed 0.75 on the corrected 10-call battery, about `25.0%` wall-clock reduction.

Implementation notes for the next slice:

- Keep startup flags as safe defaults; runtime policy must not require a server restart for ordinary gear changes.
- Expose current PFlash/KVFlash/DFlash policy and counters through `/props` or a similarly visible status endpoint.
- Add a narrow runtime settings endpoint or request-level override path before making the router adaptive.
- Log every gear decision with task class, expected decode length, chosen DFlash profile, chosen PFlash profile, and observed acceptance rate.
- Preserve explicit override behavior so benchmarking can force one gear at a time.
- Do not claim dynamic DFlash routing is implemented until a live sweep proves at least two decode gears can be selected at runtime and produce distinct measured acceptance/speed behavior without restarting the server.

2026-07-01 shim first slice:

- Separate shim/router file: `${INFERENCE_WORKSPACE}/ops/qwen36-smart-router.py`.
- The shim now defaults both direct and PFlash upstreams to the built full-stack `llama-server` endpoint `http://127.0.0.1:18160/v1` and model alias `qwen36-fullstack-local`, rather than the old Lucebox split lanes.
- PFlash routing still uses the existing task classifier and keep-ratio policy, with default requested floor `0.30` based on the corrected benchmark anchor above.
- Live server `/props` currently reports PFlash `keep_ratio: 0.75`; request-level PFlash keep changes are not yet exposed in the server schema, so the shim records/request-classifies PFlash gear decisions but cannot force a new PFlash keep ratio without either server-side schema work or a restarted server with different startup flags.
- Decode-side DFlash first slice is implemented in the shim as request-level gears that inject both `dflash.n_max` and `speculative.n_max` into upstream requests. Supported controls: `dflash_gear` / `smart_dflash_gear` (`off`, `safe`, `balanced`, `fast`, `turbo`) or explicit `dflash_n_max` / `smart_dflash_n_max` / `dflash.n_max` / `speculative.n_max`.
- Focused smoke on temporary router port `18088` against live `18160` returned HTTP 200 and annotated `qwen36_smart_route.dflash = {"n_max": 2, "reason": "explicit_gear:long-safe-n2"}` for `dflash_gear: safe`; an auto short general prompt selected `n_max=6` with reason `auto_short`. This verifies runtime request injection, not performance superiority.
- The MTP precedent reviewed for this slice is the actual Drivetrain repo, `https://github.com/HawgAuto/drivetrain`, plus the local Gemma MTP hillclimb artifacts. Drivetrain's production pattern is: classify request shape, select `short` / `mid` / `long-safe`, inject missing `speculative.*` fields, preserve caller-provided speculative fields, and record telemetry for online policy adaptation.
- The Qwen36 DFlash shim mirrors Drivetrain's default bucket map with DFlash request caps: `short-n6` (`n_max=6`), `mid-n4` (`n_max=4`), and `long-safe-n2` (`n_max=2`). Aliases remain available as `fast`, `balanced`, and `safe` respectively, plus `off` and `turbo` for forced sweeps.
- The shim now writes lightweight decision telemetry to `${INFERENCE_WORKSPACE}/ops/logs/qwen36-smart-router.decisions.jsonl` by default: route, DFlash gear, approximate prompt tokens, requested decode budget, usage, and explicit speculative overrides.

Next validation step:

- Run the systematic DFlash sweep before enabling adaptive claims: forced gears across task type and decode length, collect wall time, generated tokens, `draft_n`/`draft_n_accepted` or server-log acceptance, PFlash/KVFlash counters, and quality pass/fail. Do not claim adaptive DFlash speedup until those live sweeps prove distinct speed/acceptance behavior without restarting the server.

## Checkpoint 0XX - configurable latest-user PFlash preservation and Qwen36 floor policy

Date: 2026-07-01T21:12:40Z

Implemented configurable latest-user preservation for native server-side PFlash:

- Added `--pflash-latest-user-mode full|tail`.
- Added `--pflash-latest-user-tail-tokens N`.
- Default remains `full`, preserving the prior conservative behavior of keeping the detected latest user span.
- `tail` keeps the latest-user role boundary plus the final N tokens of that user span, allowing long normal chat prompts to compress while preserving the final instruction/question region.

Updated the Qwen36 smart router policy:

- Server runs with PFlash floor `--pflash-keep-ratio 0.30`.
- Router dynamically raises keep ratio for exact/enumerative/structured prompts.
- Router injects the verified flat field `pflash_keep_ratio` and also keeps the dotted `pflash.keep_ratio` field for compatibility.
- Router continues to inject decode-focused DFlash gears via `dflash.n_max` and `speculative.n_max`.

Live configuration verified on port 18160 behind router port 18087:

- Server: `--pflash-mode always --pflash-keep-ratio 0.30 --pflash-latest-user-mode tail --pflash-latest-user-tail-tokens 2048`.
- Router health: `http://127.0.0.1:18087/health` returned OK.
- Ad-hoc verifier `/tmp/hermes-verify-pflash-tail-router-*` reported `AD_HOC_VERIFY_PASS`.
- Natural Operation Market Garden chat prompt compressed server-side: `3054 -> 2117` prompt tokens.
- Router live request logged dynamic broad policy: `keep_ratio=0.40`, `reason=dynamic_broad_analysis`, DFlash `n_max=6`.

This is focused ad-hoc verification, not a full systematic sweep. Dynamic DFlash and PFlash runtime gears are implemented in the Qwen36 router shim and verified for request injection/native compression, but distinct performance/acceptance speedups are not yet proven by systematic sweeps.

## Checkpoint 028 - Qwen36 dynamic PFlash/DFlash sweep tuning

Ran a focused live sweep against the rebuilt Qwen36 full-stack backend on `127.0.0.1:18160` and the smart router on `127.0.0.1:18087`.

Artifacts:

- Full JSON: `${INFERENCE_WORKSPACE}/ops/results/qwen36-dynamic-sweep-20260701T212608Z.json`
- Summary JSON: `${INFERENCE_WORKSPACE}/ops/results/qwen36-dynamic-sweep-20260701T212608Z.summary.json`
- Sweep runner: `${INFERENCE_WORKSPACE}/ops/qwen36_dynamic_sweep.py`

Matrix:

- Task classes: NIAH-style exact retrieval, split enumerative retrieval, broad summarization, long decode.
- Configs: PFlash keep ratios `0.30`, `0.40`, `0.60`, `0.70`; DFlash caps `n_max=2`, `4`, `6`.
- Metrics captured from live responses and server logs: wall time, usage, PFlash original-to-kept tokens, DFlash acceptance, prompt-eval speed, decode speed, and task-specific correctness checks.

Findings:

- Broad summarization passed all tested configs. `keep=0.40` with `n_max=4` or `n_max=6` was fastest in the sweep, around 16.4-16.7s, while high keep ratios were slower.
- Long decode prompts in this sweep were too short for native PFlash compression after structural preservation, so they mainly measured DFlash. `n_max=2` was fastest for long decode.
- Synthetic exact/enumerative compressed first-pass tasks failed through `keep=0.70`; the router therefore remains conservative for exact/enumerative tasks and relies on direct or two-pass verification instead of treating low-ratio PFlash as safe for exact retrieval.

Router tuning applied in `${INFERENCE_WORKSPACE}/ops/qwen36-smart-router.py`:

- Broad analysis: `pflash_keep_ratio=0.40`.
- Long broad: `0.40`.
- Very long broad floor: `0.35`.
- Structured/literal dense: `0.70`.
- Exact/enumerative/two-pass: `0.90`.
- Unknown fallback: `0.70`.
- DFlash short decode: `n_max=6`.
- DFlash mid decode or structured: `n_max=4`.
- DFlash long decode, exact, or JSON: `n_max=2`.

Server request parsing was extended so real nested request objects now work in addition to flat/dotted fields:

- `pflash: {"keep_ratio": ...}`
- `pflash: {"min_keep_tokens": ...}`
- `pflash: {"chat_min_tokens": ...}`
- `dflash: {"n_max": ...}`
- `speculative: {"n_max": ...}`

The router now emits nested `pflash`, `dflash`, and `speculative` objects while retaining flat/dotted compatibility fields.

Live spot checks after tuning:

- Broad long prompt routed to `pflash_broad` with `keep=0.40`, DFlash `n_max=4`.
- Long exact prompt routed to `pflash_then_direct_verify` with `keep=0.90`, DFlash `n_max=2`.
- Long decode prompt used DFlash `n_max=2`.
- Backend and router were healthy after restart.

The old process watch pattern `failed to` was removed from the live long-running backend launch. The matched DFlash memory-fitting warning is benign, and the current background server is no longer monitored by that noisy pattern.

