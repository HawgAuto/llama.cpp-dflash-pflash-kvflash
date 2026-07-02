# PFlash Promotion Package - CP029/CP030

## Promotion decision

Promote the scoped raw PFlash chat-completions slice and publish CP030 as the seed of a separate long-lived opt-in optimizer shim. Do not promote broad/canonical PFlash.

## Promoted scope

- Endpoint: `/v1/chat/completions`
- Chat kwargs: `{"enable_thinking":false}`
- PFlash mode: `always`
- PFlash score: `model`
- Keep ratio: `0.10`
- Matrix: 8k, 16k, 24k
- Spot check: 32k canonical single-line

## Evidence summary

| Area | Status | Aggregate | Artifact |
| --- | --- | --- | --- |
| cp028_raw_chat_kwargs | ok | `{"cases": 3, "direct_passed": 3, "direct_total_wall_s": 198.52, "raw_pflash_passed": 3, "raw_pflash_total_wall_s": 20.933, "speedup_vs_direct": 9.48}` | `temp-cp029-option-a-cp028-chat-kwargs-20260630T000325Z/cp029-option-a-cp028_chat_kwargs_comparison.json` |
| cp029_robustness_matrix | ok | `{"cases": 27, "debug_all_values_in_compressed": 27, "debug_exports": 27, "failed": 0, "passed": 27}` | `temp-cp029-option-a-cp029-critical-line-robustness-20260630T000719Z/cp029-option-a-cp029_critical_line_robustness.json` |
| cp029_32k_spot | ok | `{"cases": 1, "debug_all_values_in_compressed": 1, "debug_exports": 1, "failed": 0, "passed": 1}` | `temp-cp029-option-a-long32k-critical-line-robustness-20260630T002558Z/cp029-option-a-long32k_critical_line_robustness.json` |
| cp029_uniform_negative | fail | `{"cases": 1, "debug_all_values_in_compressed": 0, "debug_exports": 1, "failed": 1, "passed": 0}` | `temp-cp029-option-a-uniform-critical-line-robustness-20260630T002654Z/cp029-option-a-uniform_critical_line_robustness.json` |
| cp029_external_helper_negative | fail | `{"cases": 1, "debug_all_values_in_compressed": 0, "debug_exports": 1, "failed": 1, "passed": 0}` | `temp-cp029-option-a-external-helper-critical-line-robustness-20260630T002709Z/cp029-option-a-external-helper_critical_line_robustness.json` |
| cp024_completion_mixed | not_applicable | `{"cases": 3, "direct_passed": 1, "note": "mixed /completion evidence, not promoted", "raw_pflash_passed": 1, "two_pass_verify_passed": 2}` | `temp-cp024-completions-sanity-20260630T001042Z/cp024_completions_comparison.json` |
| cp030_matrix | ok | `{"cases": 3, "direct_passed": 3, "direct_total_wall_s": 201.531, "pflash_first_passed": 3, "pflash_total_wall_s": 21.513, "second_passes": 0, "smart_2pass_passed": 3, "smart_speedup_vs_direct": 9.37, "smart_total_wall_s": 21.513}` | `temp-cp030-matrix-2pass-quality-gate-20260630T005152Z/cp030-matrix_2pass_quality_gate.json` |
| cp030_32k_spot | ok | `{"cases": 1, "direct_passed": 1, "direct_total_wall_s": 144.019, "pflash_first_passed": 1, "pflash_total_wall_s": 12.97, "second_passes": 0, "smart_2pass_passed": 1, "smart_speedup_vs_direct": 11.1, "smart_total_wall_s": 12.97}` | `temp-cp030-long32k-2pass-quality-gate-20260630T005551Z/cp030-long32k_2pass_quality_gate.json` |
| cp030_forced_fallback | ok | `{"cases": 1, "direct_passed": 1, "direct_total_wall_s": 32.892, "pflash_first_passed": 1, "pflash_total_wall_s": 4.279, "second_passes": 1, "smart_2pass_passed": 1, "smart_speedup_vs_direct": 2.33, "smart_total_wall_s": 14.109}` | `temp-cp030-review-followup-forced-2pass-quality-gate-20260630T010556Z/cp030-review-followup-forced_2pass_quality_gate.json` |

## Published behavior

CP030 remains a shim rather than a temporary bridge to native server flags. It performs first-pass raw model-score PFlash and runs a direct verification pass only when the quality gate requires it, or when `--gate-mode always` is selected for validation. The quality gate is value-presence based over the synthetic ROUTE/FUNC/SENTINEL harness; it is not yet a structured parser or confidence model. The intended product direction is a separate opt-in optimizer shim that owns adaptive PFlash routing/fallback policy and later dynamic DFlash parameter changes, similar in spirit to Drivetrain for MTP.

## CP031 scorer follow-up

The first true in-process GGUF scorer candidate that clears the token-id guard is Qwen3.5-0.8B GGUF. Both local Q4_K_M and downloaded BF16 variants loaded successfully as scorers for the Qwen3.6 DFlash target and passed focused 8k/16k/24k plus 32k CP031 quality-gate runs with `model_loaded=true` and without `token_id_mismatch`. Treat Q4_K_M as the practical default scorer candidate and BF16 as the quality-validation A/B candidate.

## Non-promoted boundaries

- /completion broad PFlash promotion beyond the focused CP032 KVFlash pressure harness
- uniform scoring as a broad PFlash quality claim
- external-helper scoring
- native server quality-gate policy flags as canonical interface
- general/canonical PFlash
- KVFlash page-level recall, hidden resident-prefix pools, non-contiguous resident pages, or sparse attention

## KVFlash CP032 follow-up

CP032 promotes the current KVFlash slice from visibility-only scaffold to a focused active idle-sequence eviction/accounting path. Under PFlash residency pressure, the server now records per-task resident hints, assigns them to the selected slot, clears idle slot sequences with `common_context_seq_rm(..., -1, -1)`, and debits logical resident tokens separately from the current llama-memory position span. The focused promotion smoke `temp-cp032-kvflash-promotable-final-20260630T041513Z/cp032-kvflash-promotable-final.json` passed with `memory_mutations=2` and `mutation_tokens_removed=106`. This remains a narrow mutation/accounting promotion, not a claim of page-level KVFlash recall.

## Next engineering after this package

- Replace the deterministic in-server model-score selector with true in-process scorer forward/logprob/importance plumbing before making broad scorer claims.
- Keep smart routing in the separate optimizer shim long-term; native server flags should remain low-level mechanisms.
- Expand the shim to also manage dynamic DFlash parameter changes after the PFlash quality-gate policy is stable.
- Continue KVFlash resident-pool mutation separately; current KVFlash status is still scaffold/config visibility, not active paging/recall.
