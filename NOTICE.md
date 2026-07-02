# Notices and attribution

This repository is a fork of `llama.cpp`.

## Upstream project

- Upstream: `llama.cpp` by the ggml authors / ggml-org
- URL: https://github.com/ggml-org/llama.cpp
- License: MIT, preserved in `LICENSE`

The substantial upstream codebase, build system, tools, examples, documentation, and server infrastructure come from `llama.cpp`. The MIT license notice must be retained in all copies or substantial portions of the software.

## Experimental acceleration lineage

This fork includes experimental server work around DFlash speculative decoding, PFlash prompt compression hooks, and KVFlash-style cache-residency experiments. Those experimental directions and integration notes are credited to the Lucebox-style PFlash/DFlash/KVFlash research and prototyping lineage.

The Lucebox-related code paths in this repository are experimental integration work layered on top of `llama.cpp`; they are not represented as upstream `llama.cpp` features unless and until accepted upstream.
