# llama.cpp PFlash/DFlash/KVFlash server fork

This is a custom inference-server fork built around PFlash prompt compression, DFlash speculative decoding, KVFlash cache-residency experiments, and an OpenAI-compatible smart router.

It uses llama.cpp as the base runtime, but this README is about the custom work in this fork.

## PFlash prompt compression

PFlash is the prompt-side experiment in this repository. It focuses on reducing or reshaping prompt work before the model backend spends time on full prefill.

The implementation direction is server-side integration, not a separate model format. The fork exposes hooks and request-time controls for experiments such as:

- compressing or reducing long prompt token streams before prefill;
- comparing compressed and uncompressed prompt paths;
- routing selected requests to PFlash-aware llama-server instances;
- surfacing prompt-compression behavior through logs and status output;
- keeping the experiment close enough to llama-server that normal serving behavior can be compared against the modified path.

PFlash is experimental infrastructure for benchmarking prompt-side changes under real server traffic.

## KVFlash cache residency

KVFlash is the KV-cache-side experiment in this repository. It focuses on observing, scoring, and controlling which cache content should remain resident across server work.

The custom server work includes status and accounting surfaces for:

- cache-residency controls;
- dry-run scoring;
- candidate-token accounting;
- hit and miss counters;
- eviction-related counters;
- request and backend comparisons for cache behavior.

The purpose is to make cache-residency experiments visible from a running server instead of hiding them inside isolated prototypes.

## DFlash speculative decoding

DFlash is the speculative-decoding experiment in this repository. It is the drafter/target path: a drafter model proposes tokens and a target model verifies them.

The server integration is aimed at practical serving questions:

- carrying drafter and target configuration through the server path;
- exposing speculative decoding status and counters;
- comparing accepted, rejected, and fallback decode behavior;
- routing speculative workloads alongside PFlash and KVFlash experiments;
- making the behavior easier to benchmark against normal llama-server decoding.

DFlash is part of the same flash-family server experimentation layer as PFlash and KVFlash.

## Smart router

`tools/server/qwen36-smart-router.py` is an OpenAI-compatible router for dispatching requests across one or more llama-server backends.

The router exists because the custom server paths are easier to test when different backends can run side by side. It provides a place for policy around:

- selecting a backend for a request;
- sending specific traffic to PFlash, DFlash, or KVFlash-aware servers;
- falling back when an experimental backend is unavailable;
- keeping OpenAI-style client compatibility while comparing backend behavior.

The router is glue for multi-backend experiments. It is not a replacement for llama-server.

## Repository map

Custom-work entry points:

- `tools/server/qwen36-smart-router.py` - OpenAI-compatible smart router.
- `PORT_STATUS.md` - status notes for the custom server changes.
- `checkpoint-*.patch` - checkpoint patches and implementation history.
- `NOTICE.md` - attribution, license, and research-lineage notes.
- `tools/server/` - llama-server base code plus the custom integration surface used by this fork.

## Using this fork

Build it as a llama.cpp fork, then run the server configurations needed for the experiment being tested.

Typical flow:

1. Build the repository for the target hardware.
2. Start one or more llama-server instances with the model and experimental settings being compared.
3. Run the smart router when OpenAI-compatible clients need to dispatch across those instances.
4. Compare logs, status output, request behavior, and cache/speculation counters.

Exact flags depend on the backend, model, and experiment. Treat PFlash, DFlash, and KVFlash as experimental controls that need measurement on the target hardware.

## Attribution and thanks

This fork is built on top of [llama.cpp](https://github.com/ggml-org/llama.cpp) and [ggml](https://github.com/ggml-org/ggml). The base runtime, hardware backends, model loading, build system, server foundation, examples, and upstream documentation are credited to the llama.cpp and ggml authors and remain under the upstream MIT license.

Special thanks and credit go to **Lucebox** for the PFlash and KVFlash research/prototyping lineage that informed the prompt-compression and KV-cache-residency work in this repository. The PFlash, DFlash, and KVFlash integrations here are experimental server work layered on top of llama.cpp and credited to that Lucebox lineage.

See [`NOTICE.md`](NOTICE.md) for attribution and license details.

## License

This repository preserves the upstream llama.cpp MIT license. See [`LICENSE`](LICENSE) and [`NOTICE.md`](NOTICE.md).
