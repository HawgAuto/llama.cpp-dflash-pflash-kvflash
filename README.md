# llama.cpp PFlash/KVFlash server fork

This is a custom inference-server fork built on top of `llama.cpp`. The custom work in this repository is focused on:

- proving PFlash prompt-token compression can be routed through the llama-server task path;
- proving KVFlash configuration, observability, dry-run scoring, and eviction-pressure accounting can be surfaced safely without mutating the real KV cache;
- providing an OpenAI-compatible smart router for selecting direct, PFlash, and verification paths across llama-server backends.

DFlash speculative decoding is also used by this fork, but it came with the llama.cpp base. The work here does not claim to invent DFlash. It uses the upstream DFlash/speculative path as one serving mode to compare against and combine with the PFlash, KVFlash, and router experiments.

## What this fork proves

### PFlash prompt compression

PFlash is the prompt-side compression path added by this fork. It lets the server reduce a long prompt token stream before full prefill, then run the shorter prompt through normal llama-server execution.

Current capabilities:

- Adds request/server controls such as `--pflash-mode`, `--pflash-keep-ratio`, `--pflash-drafter`, `--pflash-score`, and `--pflash-model`.
- Runs only when enabled, so default llama-server behavior stays unchanged when PFlash is off.
- Applies at the server task boundary for completion and infill style requests.
- Preserves prompt order and validates compressed token streams before accepting them.
- Supports a built-in deterministic uniform fallback for first/last-preserving token thinning.
- Supports an external helper bridge through `--pflash-drafter`, allowing a scorer/helper to choose the kept token subsequence.
- Supports the later model-score path where `--pflash-score model` loads a configured `--pflash-model` as an in-process scorer context.
- Exposes behavior through logs and benchmark scripts so compressed and uncompressed paths can be compared.

What PFlash is for:

- long-context broad analysis where exact retention of every prompt token is less important than reducing prefill cost;
- experiments that compare quality and speed at different keep ratios;
- two-pass routes where a fast compressed first pass can be followed by direct verification when exactness signals are detected.

Limitations:

- PFlash is experimental and must be benchmarked per workload.
- Aggressive compression can drop facts. Exact extraction, code patching, schema output, and all/every/list style prompts should use a direct path or a PFlash-then-direct-verify path.

### KVFlash observability and dry-run residency accounting

KVFlash is the KV-cache-side experiment added by this fork. The current implementation is intentionally safe: it observes and scores candidate residency behavior without changing the real llama.cpp KV cache.

Current capabilities:

- Adds configuration fields such as `--kvflash`, `--kvflash-policy`, `--kvflash-tau`, and `--kvflash-drafter`.
- Reports KVFlash configuration in `/props`.
- Maintains an inert `resident_pool` status object with capacity, resident-token accounting fields, hit/miss counters, dry-run scoring counters, candidate-token counters, and simulated eviction-pressure counters.
- Runs dry-run scoring at the server task boundary when KVFlash is configured.
- Simulates whether a prompt candidate would exceed the configured resident-token budget.
- Exposes the accounting surface needed to evaluate policies before any real KV-cache mutation is attempted.

What KVFlash currently does not do:

- It does not allocate a separate real resident KV pool.
- It does not evict, page, recall, clear, or mutate llama.cpp KV cache state.
- It does not change attention masks or generation behavior.
- It does not claim a speedup by itself in the current dry-run state.

What KVFlash is for right now:

- proving the server can carry KVFlash policy/configuration cleanly;
- making residency and eviction-pressure decisions observable from a live server;
- creating a safe measurement scaffold before enabling risky real KV-cache residency changes.

### DFlash speculative decoding

DFlash/speculative decoding is not claimed as custom work from this repository. The DFlash path came from the llama.cpp base that this fork builds on.

How this fork uses it:

- Runs DFlash-compatible drafter/target launches through existing llama.cpp speculative decoding controls, for example `-md`, `--spec-type draft-dflash`, and `--spec-draft-n-max` where supported by the base.
- Uses DFlash as a comparison and composition mode when evaluating PFlash and KVFlash behavior.
- Records DFlash-related launch and benchmark evidence in status and benchmark artifacts.
- Lets the smart router and benchmark scripts compare direct, PFlash, PFlash-plus-verification, and DFlash-enabled server configurations.

What this repo proved around DFlash is integration evidence with the forked server stack, not ownership of the DFlash mechanism itself.

### OpenAI-compatible smart router

`tools/server/qwen36-smart-router.py` is custom glue for running multiple server paths behind an OpenAI-style API surface.

Current capabilities:

- Accepts OpenAI-style chat completion requests.
- Estimates prompt size and detects exactness signals such as patches, schemas, exact values, JSON output, all/every/list extraction, and code edits.
- Routes short or exact work to the direct path.
- Routes long broad-analysis work to a PFlash path.
- Routes long exact work to a PFlash-first path followed by direct verification when needed.
- Supports forced routes through request fields such as `route`, `x_route`, or `smart_route`.
- Supports optional classifier sidecar modes for shadow or active routing experiments.
- Logs routing decisions to JSONL for analysis.
- Can manage or target separate direct and PFlash backend URLs through environment variables.

What the router is for:

- keeping ordinary OpenAI-compatible clients pointed at one endpoint while different llama-server backends run underneath;
- protecting exactness-sensitive tasks from unsafe compression;
- collecting routing evidence for PFlash and direct-verify policies.

The router is not a replacement for llama-server. It is policy glue around one or more llama-server instances.

## Repository map

Custom-work entry points:

- `tools/server/qwen36-smart-router.py` - OpenAI-compatible smart router.
- `tools/server/bench_cp0*.py` - benchmark and validation scripts used during PFlash/KVFlash/router experiments.
- `PORT_STATUS.md` - checkpoint notes describing what each custom integration slice does and does not do.
- `checkpoint-*.patch` - checkpoint patches and implementation history.
- `NOTICE.md` - attribution, license, and research-lineage notes.
- `tools/server/` - llama-server base code plus the custom integration surface used by this fork.

## Using this fork

Build it as a llama.cpp fork, then run the server configurations needed for the experiment being tested.

Typical flow:

1. Build the repository for the target hardware.
2. Start a direct llama-server backend.
3. Start a PFlash-enabled backend when testing prompt compression.
4. Configure KVFlash flags when collecting residency/dry-run accounting.
5. Use DFlash/speculative flags only as supported by the inherited llama.cpp base.
6. Run the smart router when OpenAI-compatible clients need one endpoint that chooses among those paths.
7. Compare logs, `/props`, benchmark JSON, routing decisions, and output quality.

Exact flags depend on the backend, model, and experiment. Treat PFlash and KVFlash as experimental controls that need measurement on the target hardware. Treat DFlash as inherited llama.cpp speculative decoding that this fork can use in experiments.

## Attribution and thanks

This fork is built on top of [llama.cpp](https://github.com/ggml-org/llama.cpp) and [ggml](https://github.com/ggml-org/ggml). The base runtime, hardware backends, model loading, build system, server foundation, examples, upstream DFlash/speculative decoding support, and upstream documentation are credited to the llama.cpp and ggml authors and remain under the upstream MIT license.

Special thanks and credit go to **Lucebox** for the PFlash and KVFlash research/prototyping lineage that informed the prompt-compression and KV-cache-residency work in this repository. The custom proof work here is PFlash integration, KVFlash observability/dry-run residency accounting, and the OpenAI-compatible router layered on top of llama.cpp.

See [`NOTICE.md`](NOTICE.md) for attribution and license details.

## License

This repository preserves the upstream llama.cpp MIT license. See [`LICENSE`](LICENSE) and [`NOTICE.md`](NOTICE.md).
