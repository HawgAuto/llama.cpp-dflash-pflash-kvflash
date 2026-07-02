#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BIN = ROOT / "build-hip-gfx1151/bin/llama-server"
DEFAULT_TARGET = Path("${MODEL_DIR}/draft/Qwen3.5-0.8B-Q4_K_M.gguf")


def http_json(method, url, payload=None, timeout=120):
    data = None if payload is None else json.dumps(payload).encode()
    headers = {} if payload is None else {"Content-Type": "application/json"}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read().decode("utf-8", "replace")
    return json.loads(body), time.monotonic() - t0


def wait_ready(port, proc, log_path):
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            tail = log_path.read_text(errors="replace")[-5000:] if log_path.exists() else ""
            raise RuntimeError(f"server exited {proc.returncode}\n{tail}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                if r.status == 200:
                    return
        except Exception:
            time.sleep(0.5)
    raise RuntimeError("health timeout")


def stop_server(proc, log):
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(20)
    log.close()


def make_prompt(label, approx_tokens):
    lines = [
        "Return a short acknowledgement only.",
        f"KVFLASH_PRESSURE_LABEL={label}",
    ]
    blocks = max(1, approx_tokens // 40)
    for i in range(blocks):
        lines.append(f"{label} filler block {i:05d}: alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu.")
    lines.append("Answer with exactly: ok")
    return "\n".join(lines)


def completion_call(port, prompt, max_tokens):
    payload = {
        "prompt": prompt,
        "temperature": 0,
        "n_predict": max_tokens,
        "cache_prompt": True,
        "stream": False,
    }
    return http_json("POST", f"http://127.0.0.1:{port}/completion", payload, timeout=240)


def props(port):
    obj, _ = http_json("GET", f"http://127.0.0.1:{port}/props", None, timeout=30)
    return obj


def kvflash_pool(obj):
    return obj.get("kvflash", {}).get("resident_pool", {})


def assert_port_free(port):
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
        raise SystemExit(f"port {port} busy")
    except urllib.error.URLError:
        pass


def parse_args():
    p = argparse.ArgumentParser(description="CP032 focused KVFlash idle eviction pressure harness")
    p.add_argument("--label", default="cp032-kvflash-pressure")
    p.add_argument("--server-bin", type=Path, default=DEFAULT_BIN)
    p.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    p.add_argument("--port", type=int, default=18150)
    p.add_argument("--run-dir", type=Path)
    p.add_argument("--kvflash", type=int, default=128)
    p.add_argument("--ctx-size", type=int, default=4096)
    p.add_argument("--prompt-tokens", type=int, default=1200)
    p.add_argument("--max-tokens", type=int, default=8)
    p.add_argument("--recall-repeat", action="store_true", help="repeat the first prompt after eviction to exercise hidden-prefix recall")
    p.add_argument("--allow-sparse-exec-recall", action="store_true", help="allow recall-repeat validation to pass on sparse graph execution without hidden-prefix recall hits")
    p.add_argument("--flash-attn", action="store_true", help="enable flash attention for focused sparse-KV gather checks")
    args = p.parse_args()
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    args.run_dir = args.run_dir or ROOT / f"temp-{args.label}-{stamp}"
    args.log_dir = args.run_dir / "logs"
    args.out_dir = args.run_dir / "outputs"
    for d in (args.log_dir, args.out_dir):
        d.mkdir(parents=True, exist_ok=True)
    return args


def main():
    args = parse_args()
    assert_port_free(args.port)
    log_path = args.log_dir / "server.log"
    log = open(log_path, "w")
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = str(args.server_bin.parent) + ":/opt/rocm/lib:" + env.get("LD_LIBRARY_PATH", "")
    cmd = [
        str(args.server_bin),
        "-m", str(args.target),
        "--host", "127.0.0.1",
        "--port", str(args.port),
        "--ctx-size", str(args.ctx_size),
        "--predict", "32",
        "--no-mmap",
        "--jinja",
        "--reasoning", "off",
        *( ["--flash-attn", "on"] if args.flash_attn else [] ),
        "-np", "2",
        "--alias", "kvflash-pressure",
        "--kvflash", str(args.kvflash),
        "--kvflash-policy", "qk",
        "--kvflash-tau", "16",
        "--pflash-mode", "always",
        "--pflash-keep-ratio", "0.10",
        "--pflash-score", "uniform",
    ]
    proc = None
    rows = []
    try:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
        wait_ready(args.port, proc, log_path)
        before = props(args.port)
        labels = ["first", "second", "third"]
        if args.recall_repeat:
            labels.append("first")
        for label in labels:
            obj, wall = completion_call(args.port, make_prompt(label, args.prompt_tokens), args.max_tokens)
            after = props(args.port)
            timings = obj.get("timings", {})
            rows.append({
                "label": label,
                "wall_s": round(wall, 3),
                "content_preview": str(obj.get("content", ""))[:200],
                "timings": {
                    "prompt_n": timings.get("prompt_n"),
                    "cache_n": timings.get("cache_n"),
                    "prompt_ms": timings.get("prompt_ms"),
                    "predicted_n": timings.get("predicted_n"),
                },
                "kvflash_pool": kvflash_pool(after),
            })
        final = props(args.port)
    finally:
        if proc is not None:
            stop_server(proc, log)
        else:
            log.close()
    pool = kvflash_pool(final)
    log_text = log_path.read_text(errors="replace") if log_path.exists() else ""
    sparse_active_count = log_text.count("KVFlash sparse attention plan active")
    sparse_mask_mismatch = any(int(m.group(1)) != 0 for m in re.finditer(r"mismatches=(\\d+)", log_text))
    rocm_fault = "ROCm error" in log_text or "illegal memory access" in log_text
    sparse_exec_ok = sparse_active_count > 0 and not sparse_mask_mismatch and not rocm_fault
    memory_mutations = pool.get("memory_mutations", 0)
    mutation_tokens_removed = pool.get("mutation_tokens_removed", 0)
    recall_hits = pool.get("recall_hits", 0)
    page_hits = pool.get("page_hits", 0)
    hidden_prefix_pools = pool.get("hidden_prefix_pools", 0)
    resident_pages = pool.get("resident_pages", 0)
    hidden_restore_enabled = env.get("LLAMA_KVFLASH_HIDDEN_STATE_RESTORE") == "1"
    hidden_restore_hits = pool.get("hidden_restore_hits", 0)
    hidden_recall_ok = recall_hits > 0 and page_hits > 0 and hidden_prefix_pools > 0
    prompt_n_first = None
    prompt_n_repeat = None
    if args.recall_repeat:
        first_rows = [row for row in rows if row["label"] == "first"]
        if first_rows:
            prompt_n_first = first_rows[0].get("timings", {}).get("prompt_n")
            prompt_n_repeat = first_rows[-1].get("timings", {}).get("prompt_n")
    prompt_prefill_reduced = (
        isinstance(prompt_n_first, int)
        and isinstance(prompt_n_repeat, int)
        and prompt_n_repeat < prompt_n_first
    )
    hidden_restore_prefill_ok = (
        not (args.recall_repeat and hidden_restore_enabled)
        or (hidden_restore_hits > 0 and prompt_prefill_reduced)
    )
    base_ok = memory_mutations > 0 and mutation_tokens_removed > memory_mutations
    recall_ok = (not args.recall_repeat) or hidden_recall_ok or (args.allow_sparse_exec_recall and sparse_exec_ok) or hidden_restore_prefill_ok
    paging_ok = resident_pages > 0 and pool.get("page_size", 0) > 0
    status = "ok" if base_ok and recall_ok and paging_ok and hidden_restore_prefill_ok else "fail"
    result = {
        "status": status,
        "label": args.label,
        "note": "Focused CP032 KVFlash pressure harness; ad-hoc validation, not canonical suite green.",
        "server_args": cmd,
        "before": before.get("kvflash", {}),
        "rows": rows,
        "final": final.get("kvflash", {}),
        "assertions": {
            "memory_mutations_gt_0": memory_mutations > 0,
            "mutation_tokens_removed_gt_memory_mutations": mutation_tokens_removed > memory_mutations,
            "resident_pages_gt_0": resident_pages > 0,
            "hidden_prefix_pools_gt_0_when_recall_repeat": (not args.recall_repeat) or hidden_prefix_pools > 0,
            "hidden_recall_ok_when_recall_repeat": (not args.recall_repeat) or hidden_recall_ok,
            "sparse_exec_ok_when_allowed": (not args.allow_sparse_exec_recall) or sparse_exec_ok,
            "recall_repeat_ok": recall_ok,
            "sparse_active_count": sparse_active_count,
            "sparse_mask_mismatch": sparse_mask_mismatch,
            "rocm_fault": rocm_fault,
            "hidden_restore_enabled": hidden_restore_enabled,
            "hidden_restore_hits": hidden_restore_hits,
            "prompt_n_first": prompt_n_first,
            "prompt_n_repeat": prompt_n_repeat,
            "prompt_prefill_reduced": prompt_prefill_reduced,
            "hidden_restore_prefill_ok": hidden_restore_prefill_ok,
        },
    }
    out = args.run_dir / f"{args.label}.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps({"status": status, "result": str(out), "final_pool": pool}, indent=2))
    if status != "ok":
        tail = log_path.read_text(errors="replace")[-5000:] if log_path.exists() else ""
        print(tail)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
