# EVO-X2 validation report - 2026-07-02

This is an operator validation record for the public DFlash/PFlash/KVFlash server fork. It records focused ad-hoc runs and one upstream CTest run performed on the EVO-X2 host. It is not a claim that every platform or model combination is production-certified.

## Canonical validation stack

| Role | Path | Launch role |
|---|---|---|
| Target/base model | `/home/hawg/models/Qwen3.6-27B-MTP-GGUF-Q4_K_M/Qwen3.6-27B-Q4_K_M.gguf` | `-m` |
| DFlash drafter | `/home/hawg/models/z-lab-Qwen3.6-27B-DFlash/qwen3.6-27b-dflash-zlab-q8_0.gguf` | `-md` with `--spec-type draft-dflash` |
| PFlash scorer | `/home/hawg/models/draft/Qwen3.5-0.8B-Q4_K_M.gguf` | `--pflash-model` with `--pflash-score model` |

The MTP-directory Qwen3.6 27B Q4_K_M artifact above is the canonical target for this stack. The older non-MTP experiment copy is not the ideal target for this server.

## Environment

- Host: GMKtec NucBox EVO-X2 / AMD Strix Halo
- ROCm lane: AMD Radeon 8060S iGPU, `gfx1151`; ROCm launch used `--no-mmap`
- CUDA lane: NVIDIA RTX 5060 Ti eGPU; CUDA 13.3 build, no `--no-mmap`
- Public upstream-style CPU test build: `build-local-tests`

## Full upstream CTest suite

Command:

```bash
ctest --test-dir build-local-tests --output-on-failure
```

Result:

```text
100% tests passed, 0 tests failed out of 54
Total Test time (real) = 49.59 sec
```

This run used corrected external vocab GGUF artifacts; `test-tokenizers-ggml-vocabs` passed.

## ROCm/iGPU proper-stack long soak

Focused ad-hoc verifier:

```text
/tmp/hermes-verify-proper-stack-soak.py
```

Server command under test:

```bash
/home/hawg/evo-infer/ops/llama.cpp-dflash-pflash-kvflash/build-hip-gfx1151/bin/llama-server \
  -m /home/hawg/models/Qwen3.6-27B-MTP-GGUF-Q4_K_M/Qwen3.6-27B-Q4_K_M.gguf \
  --spec-type draft-dflash \
  -md /home/hawg/models/z-lab-Qwen3.6-27B-DFlash/qwen3.6-27b-dflash-zlab-q8_0.gguf \
  --host 127.0.0.1 --port 18220 \
  --ctx-size 32768 --predict 128 \
  --no-mmap --jinja --reasoning off \
  --chat-template-kwargs '{"enable_thinking":false}' \
  --cache-type-k q8_0 --cache-type-v q8_0 --flash-attn on \
  -np 4 --alias proper-stack-soak \
  --kvflash 4096 --kvflash-policy qk --kvflash-tau 64 \
  --pflash-mode always --pflash-keep-ratio 0.35 --pflash-score model \
  --pflash-model /home/hawg/models/draft/Qwen3.5-0.8B-Q4_K_M.gguf \
  --pflash-latest-user-mode tail --pflash-latest-user-tail-tokens 2048
```

Result:

```json
{
  "status": "ok",
  "note": "focused ad-hoc proper-stack 10-minute ROCm/iGPU soak with concurrent clients, DFlash drafter, PFlash model scorer, KVFlash enabled; not canonical suite green",
  "ready_s": 8.01,
  "rounds": 7,
  "requests_ok": 56,
  "error_count": 0
}
```

Representative timing from the soak:

```json
{
  "cache_n": 0,
  "prompt_n": 2048,
  "prompt_ms": 24856.692,
  "prompt_per_second": 82.39229902353861,
  "predicted_n": 24,
  "predicted_ms": 12869.497,
  "predicted_per_second": 1.8648747499610905,
  "draft_n": 19,
  "draft_n_accepted": 16
}
```

The soak used 4 concurrent slots/clients (`-np 4`) and completed without HTTP errors or server crash.

## Real PFlash model-scoring path

The ROCm soak verified the in-process PFlash scorer path, not only heuristic scoring. Server log evidence:

```text
PFlash in-process GGUF scorer loaded: model=/home/hawg/models/draft/Qwen3.5-0.8B-Q4_K_M.gguf n_ctx=32768 n_batch=512 n_outputs_max=512
experimental PFlash configured: mode=always keep_ratio=0.350 backend=model helper=<none> model=/home/hawg/models/draft/Qwen3.5-0.8B-Q4_K_M.gguf model_loaded=true
experimental PFlash compressed task 0 prompt tokens: 2576 -> 2048, mode=always, keep_ratio=0.350, min_keep=2048, prefix=64, suffix=512, backend=model, helper=false, model=loaded, structural=0, middle_only=true
```

## DFlash drafter path

The same run verified the DFlash drafter path with the canonical Z-Lab drafter artifact. Representative timings showed draft acceptance:

```text
draft acceptance = 0.84211 (16 accepted / 19 generated), mean len = 3.29
```

## Large-model KVFlash pressure

The ROCm soak used the 27B MTP target with `--kvflash 4096`. Server log evidence showed resident-pool initialization and repeated pressure/eviction behavior:

```text
experimental KVFlash active: tokens=4096 policy=qk tau=64 drafter=<none> resident_pool_initialized=true resident_tokens=0 entries=0 hits=0 misses=0 evictions=0 memory_mutations=0 mutation_tokens_removed=0
experimental KVFlash evicted idle sequence from memory: removed_tokens=2048 removed_pages=32 memory_positions=1 pos_min=2070 pos_max=2070 resident_tokens=2048 resident_pages=384 capacity=4096
```

The final `/props` response in this build did not expose `kvflash_pool`, so final pool counters were taken from server log lines rather than JSON props.

## Concurrent multi-client server stress

The ROCm soak drove concurrent requests against the 4-slot server:

```json
{
  "rounds": 7,
  "requests_ok": 56,
  "errors": [],
  "error_count": 0
}
```

## CUDA/eGPU lane smoke

Focused ad-hoc verifier:

```text
/tmp/hermes-verify-cuda-egpu.py
```

Server command under test:

```bash
/home/hawg/evo-infer/ops/llama.cpp-dflash-pflash-kvflash/build-cuda-13.3/bin/llama-server \
  -m /home/hawg/models/draft/Qwen3.5-0.8B-Q4_K_M.gguf \
  --host 127.0.0.1 --port 18221 \
  --ctx-size 4096 --predict 64 \
  --jinja -ngl 999 \
  --alias cuda-egpu-qwen35-smoke
```

CUDA environment used:

```bash
LD_LIBRARY_PATH=/home/hawg/evo-infer/ops/llama.cpp-dflash-pflash-kvflash/build-cuda-13.3/bin:/usr/local/cuda-13.3/compat:/usr/local/cuda-13.3/targets/x86_64-linux/lib
```

Result:

```json
{
  "status": "ok",
  "note": "focused ad-hoc CUDA/eGPU lane smoke, not canonical suite green",
  "props_model_alias": "cuda-egpu-qwen35-smoke",
  "nvidia_smi": "1020, 14830, 4, 49"
}
```

Representative CUDA timing:

```json
{
  "cache_n": 0,
  "prompt_n": 16,
  "prompt_ms": 15.166,
  "prompt_per_second": 1054.991428194646,
  "predicted_n": 9,
  "predicted_ms": 24.371,
  "predicted_per_second": 369.2913708916335
}
```

## Verification scope note

The focused ad-hoc runs above are live HTTP/server regressions on this machine and stack. They are intentionally labeled as ad-hoc verification. The only canonical upstream test-suite result in this report is the `ctest` result listed above.
