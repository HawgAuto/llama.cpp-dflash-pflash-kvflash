# Notices, attribution, and thanks

This repository is a custom server-focused fork of `llama.cpp` with experimental PFlash, DFlash, and KVFlash integration work layered on top.

## Upstream project

- Upstream: `llama.cpp` by the ggml authors / ggml-org
- URL: https://github.com/ggml-org/llama.cpp
- License: MIT, preserved in `LICENSE`

The substantial upstream codebase, build system, tools, examples, documentation, model support, backend implementations, and server infrastructure come from `llama.cpp`. The MIT license notice must be retained in all copies or substantial portions of the software.

## Lucebox PFlash/KVFlash lineage

Special thanks and credit go to **Lucebox** for the PFlash and KVFlash research/prototyping lineage that informed the experimental prompt-compression and KV-cache-residency directions in this fork.

This repository includes experimental server work around:

- PFlash-style prompt compression hooks and token-flow experiments.
- DFlash-style speculative decoding integration and observability.
- KVFlash-style cache-residency controls, status reporting, dry-run scoring, candidate-token accounting, and eviction-related counters.

Those experimental directions are credited to the Lucebox PFlash/KVFlash lineage. The implementation in this repository is integration and experimentation work layered on top of `llama.cpp`; it is not represented as upstream `llama.cpp` functionality unless and until accepted upstream.
