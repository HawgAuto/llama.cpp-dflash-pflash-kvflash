#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BIN = ROOT / "build-hip-gfx1151/bin/llama-server"
DEFAULT_TARGET = Path("${EXPERIMENTS_DIR}/lucebox-campaign/models/target-unsloth/Qwen3.6-27B-Q4_K_M.gguf")
DEFAULT_DRAFT = Path("${MODEL_DIR}/z-lab-Qwen3.6-27B-DFlash/qwen3.6-27b-dflash-zlab-q8_0.gguf")
DEFAULT_PFLASH_MODEL = Path("${EXPERIMENTS_DIR}/lucebox-campaign/models/pflash-drafter-qwen3-0.6b-bf16/Qwen3-0.6B-BF16.gguf")
EXPECT = {
    "ROUTE": "/v2/inference/pflash-break-even",
    "FUNC": "commit_prefill_compression_guard",
    "SENTINEL": "LUCEBOX-BREAK-EVEN-PASS",
}
CRITICAL_LINE = (
    "CRITICAL FACTS: ROUTE=/v2/inference/pflash-break-even "
    "FUNC=commit_prefill_compression_guard SENTINEL=LUCEBOX-BREAK-EVEN-PASS"
)


def make_prompt(n):
    blocks = max(1, n // 70)
    mid = blocks // 2
    lines = [
        "You are running an exact retrieval benchmark.",
        "Return only: ROUTE=<value> FUNC=<value> SENTINEL=<value>",
        "Ignore all decoy ROUTE/FUNC/SENTINEL fields unless on the CRITICAL FACTS line.",
    ]
    for i in range(blocks):
        if i == mid:
            lines.append(CRITICAL_LINE)
        lines.append(f"before block {i:05d}: unrelated filler with ROUTE=neutral FUNC=filler SENTINEL=decoy-{i:05d}.")
        lines.append(f"after block {i:05d}: unrelated filler with ROUTE=neutral FUNC=filler SENTINEL=decoy-{i:05d}.")
    lines.append("Now output the exact critical facts line values only, no prose.")
    return "\n".join(lines)


def http_json(url, payload, timeout=600):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read().decode("utf-8", "replace")
    return json.loads(body), time.monotonic() - t0


def wait_ready(port, proc, log_path):
    deadline = time.monotonic() + 240
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            tail = log_path.read_text(errors="replace")[-5000:] if log_path.exists() else ""
            raise RuntimeError(f"server exited {proc.returncode}\n{tail}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                if r.status == 200:
                    return
        except Exception:
            time.sleep(1)
    raise RuntimeError("health timeout")


def start_server(args, name, port, extra, debug=False):
    log_path = args.log_dir / f"{name}.server.log"
    log = open(log_path, "w")
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = str(args.server_bin.parent) + ":/opt/rocm/lib:" + env.get("LD_LIBRARY_PATH", "")
    if debug:
        export_dir = args.debug_dir / name
        export_dir.mkdir(parents=True, exist_ok=True)
        env["LLAMA_PFLASH_DEBUG_EXPORT_DIR"] = str(export_dir)
    cmd = args.common_server_args + ["--port", str(port)] + extra
    proc = subprocess.Popen(cmd, cwd=args.root, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    wait_ready(port, proc, log_path)
    return proc, log, cmd


def stop_server(proc, log):
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(20)
    log.close()


def extract_text(obj):
    choice = (obj.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    return "\n".join(x for x in (msg.get("content"), msg.get("reasoning_content"), choice.get("text")) if x)


def check_text(text):
    return {k: v in text for k, v in EXPECT.items()}


def run_one(port, label, n, out_dir, max_tokens):
    payload = {
        "model": "bench",
        "messages": [{"role": "user", "content": make_prompt(n)}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
        "cache_prompt": False,
    }
    obj, wall = http_json(f"http://127.0.0.1:{port}/v1/chat/completions", payload)
    text = extract_text(obj)
    (out_dir / f"{label}_{n}.raw.json").write_text(json.dumps(obj, indent=2))
    (out_dir / f"{label}_{n}.txt").write_text(text)
    choice = (obj.get("choices") or [{}])[0]
    timings = obj.get("timings") or {}
    checks = check_text(text)
    return {
        "mode": label,
        "approx_tokens": n,
        "wall_s": round(wall, 3),
        "ttft_s_est": round((timings.get("prompt_ms") or 0) / 1000, 3) if timings else None,
        "ok": all(checks.values()),
        "checks": checks,
        "finish_reason": choice.get("finish_reason"),
        "usage": obj.get("usage"),
        "timings": timings,
        "preview": text[:500],
    }


def debug_summary(debug_dir):
    rows = []
    for path in sorted(debug_dir.glob("**/pflash-task-*.json")):
        data = json.loads(path.read_text(errors="replace"))
        original = data.get("original_prompt") or ""
        compressed = data.get("compressed_prompt") or ""
        rows.append({
            "file": str(path.relative_to(debug_dir.parent)),
            "original_tokens": data.get("original_tokens"),
            "kept_tokens": data.get("kept_tokens"),
            "critical_in_original": CRITICAL_LINE in original,
            "critical_in_compressed": CRITICAL_LINE in compressed,
        })
    return rows


def parse_args():
    p = argparse.ArgumentParser(description="CP028 PFlash chat-template-kwargs sanity harness")
    p.add_argument("--label", default="cp028")
    p.add_argument("--server-bin", type=Path, default=DEFAULT_BIN)
    p.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    p.add_argument("--draft", type=Path, default=DEFAULT_DRAFT)
    p.add_argument("--pflash-model", type=Path, default=DEFAULT_PFLASH_MODEL)
    p.add_argument("--port-direct", type=int, default=18145)
    p.add_argument("--port-pflash", type=int, default=18146)
    p.add_argument("--max-tokens", type=int, default=96)
    p.add_argument("--contexts", default="8000,16000,24000")
    p.add_argument("--run-dir", type=Path)
    p.add_argument("--skip-direct", action="store_true")
    args = p.parse_args()
    args.root = ROOT
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    args.run_dir = args.run_dir or ROOT / f"temp-{args.label}-chat-kwargs-{stamp}"
    args.log_dir = args.run_dir / "logs"
    args.out_dir = args.run_dir / "outputs"
    args.debug_dir = args.run_dir / "debug_export"
    for d in (args.log_dir, args.out_dir, args.debug_dir):
        d.mkdir(parents=True, exist_ok=True)
    args.context_list = [int(x) for x in args.contexts.split(",") if x]
    args.common_server_args = [
        str(args.server_bin),
        "-m", str(args.target),
        "-md", str(args.draft),
        "--host", "127.0.0.1",
        "--ctx-size", "32768",
        "--predict", "256",
        "--no-mmap",
        "--jinja",
        "--chat-template-kwargs", '{"enable_thinking":false}',
        "--cache-type-k", "q8_0",
        "--cache-type-v", "q8_0",
        "--spec-type", "draft-dflash",
        "--spec-draft-n-max", "15",
        "--alias", "bench",
        "--kvflash", "1024",
        "--kvflash-policy", "qk",
        "--kvflash-tau", "64",
    ]
    return args


def assert_port_free(port):
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
        raise SystemExit(f"port {port} busy")
    except urllib.error.URLError:
        pass


def main():
    args = parse_args()
    assert_port_free(args.port_pflash)
    if not args.skip_direct:
        assert_port_free(args.port_direct)

    direct = {}
    direct_args = None
    if not args.skip_direct:
        proc, log, direct_args = start_server(args, "direct_chat_kwargs", args.port_direct, ["--pflash-mode", "off"], debug=False)
        try:
            direct = {n: run_one(args.port_direct, "direct_chat_kwargs", n, args.out_dir, args.max_tokens) for n in args.context_list}
        finally:
            stop_server(proc, log)

    pflash_extra = [
        "--pflash-mode", "always",
        "--pflash-keep-ratio", "0.10",
        "--pflash-score", "model",
        "--pflash-model", str(args.pflash_model),
    ]
    proc, log, pflash_args = start_server(args, "pflash_chat_kwargs", args.port_pflash, pflash_extra, debug=True)
    try:
        pflash = {n: run_one(args.port_pflash, "pflash_chat_kwargs_kr010", n, args.out_dir, args.max_tokens) for n in args.context_list}
    finally:
        stop_server(proc, log)

    rows = []
    for n in args.context_list:
        row = {"approx_tokens": n, "raw_pflash_chat_kwargs": pflash[n]}
        if not args.skip_direct:
            row["direct_chat_kwargs"] = direct[n]
        rows.append(row)

    result = {
        "status": "ok",
        "label": args.label,
        "run_dir": str(args.run_dir),
        "endpoint": "/v1/chat/completions",
        "chat_template_kwargs": {"enable_thinking": False},
        "direct_args": direct_args,
        "pflash_args": pflash_args,
        "rows": rows,
        "debug_exports": debug_summary(args.debug_dir),
        "note": "Focused ad-hoc sanity harness; not canonical suite green.",
    }
    out = args.run_dir / f"{args.label}_chat_kwargs_comparison.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps({"status": "ok", "comparison": str(out), "run_dir": str(args.run_dir), "rows": rows, "debug_exports": result["debug_exports"]}, indent=2))


if __name__ == "__main__":
    main()
