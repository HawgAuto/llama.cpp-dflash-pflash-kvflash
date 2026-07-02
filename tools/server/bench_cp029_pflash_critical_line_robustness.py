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


def critical_record(fmt):
    route = EXPECT["ROUTE"]
    func = EXPECT["FUNC"]
    sentinel = EXPECT["SENTINEL"]
    if fmt == "canonical_single_line":
        return f"CRITICAL FACTS: ROUTE={route} FUNC={func} SENTINEL={sentinel}"
    if fmt == "json_object":
        return f'CRITICAL FACTS JSON: {{"ROUTE":"{route}","FUNC":"{func}","SENTINEL":"{sentinel}"}}'
    if fmt == "pipe_delimited":
        return f"CRITICAL FACTS | ROUTE={route} | FUNC={func} | SENTINEL={sentinel} |"
    if fmt == "markdown_table":
        return "\n".join([
            "CRITICAL FACTS TABLE:",
            "| ROUTE | FUNC | SENTINEL |",
            f"| {route} | {func} | {sentinel} |",
        ])
    if fmt == "yaml_block":
        return "\n".join([
            "CRITICAL FACTS YAML:",
            f"ROUTE: {route}",
            f"FUNC: {func}",
            f"SENTINEL: {sentinel}",
        ])
    if fmt == "xml_attrs":
        return f'CRITICAL FACTS XML: <critical ROUTE="{route}" FUNC="{func}" SENTINEL="{sentinel}" />'
    if fmt == "multiline_equals":
        return "\n".join(["CRITICAL FACTS MULTILINE:", f"ROUTE={route}", f"FUNC={func}", f"SENTINEL={sentinel}"])
    if fmt == "prose_sentence":
        return f"CRITICAL FACTS PROSE: the route is {route}; the function is {func}; the sentinel is {sentinel}."
    if fmt == "spaced_equals":
        return f"CRITICAL FACTS SPACED: ROUTE = {route}    FUNC = {func}    SENTINEL = {sentinel}"
    raise ValueError(f"unknown critical-line format: {fmt}")


def make_prompt(n, fmt):
    record = critical_record(fmt)
    blocks = max(1, n // 70)
    mid = blocks // 2
    lines = [
        "You are running an exact retrieval benchmark.",
        "Return only: ROUTE=<value> FUNC=<value> SENTINEL=<value>",
        "Ignore all decoy ROUTE/FUNC/SENTINEL fields unless in the critical record.",
    ]
    for i in range(blocks):
        if i == mid:
            lines.append(record)
        lines.append(f"before block {i:05d}: unrelated filler with ROUTE=neutral FUNC=filler SENTINEL=decoy-{i:05d}.")
        lines.append(f"after block {i:05d}: unrelated filler with ROUTE=neutral FUNC=filler SENTINEL=decoy-{i:05d}.")
    lines.append("Now output the exact critical facts values only, no prose.")
    return "\n".join(lines), record


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


def start_server(args):
    log_path = args.log_dir / "pflash_robustness.server.log"
    log = open(log_path, "w")
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = str(args.server_bin.parent) + ":/opt/rocm/lib:" + env.get("LD_LIBRARY_PATH", "")
    export_dir = args.debug_dir / "pflash_robustness"
    export_dir.mkdir(parents=True, exist_ok=True)
    env["LLAMA_PFLASH_DEBUG_EXPORT_DIR"] = str(export_dir)
    cmd = args.common_server_args + [
        "--port", str(args.port),
        "--pflash-mode", "always",
        "--pflash-keep-ratio", "0.10",
        "--pflash-score", "model",
        "--pflash-model", str(args.pflash_model),
    ]
    proc = subprocess.Popen(cmd, cwd=args.root, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    wait_ready(args.port, proc, log_path)
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


def run_one(args, fmt, n):
    prompt, record = make_prompt(n, fmt)
    payload = {
        "model": "bench",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "stream": False,
        "cache_prompt": False,
    }
    obj, wall = http_json(f"http://127.0.0.1:{args.port}/v1/chat/completions", payload)
    text = extract_text(obj)
    stem = f"pflash_{fmt}_{n}"
    (args.out_dir / f"{stem}.raw.json").write_text(json.dumps(obj, indent=2))
    (args.out_dir / f"{stem}.txt").write_text(text)
    choice = (obj.get("choices") or [{}])[0]
    checks = check_text(text)
    return {
        "format": fmt,
        "approx_tokens": n,
        "ok": all(checks.values()),
        "checks": checks,
        "wall_s": round(wall, 3),
        "finish_reason": choice.get("finish_reason"),
        "usage": obj.get("usage"),
        "preview": text[:500],
        "critical_record": record,
    }


def debug_summary(debug_dir):
    rows = []
    for path in sorted(debug_dir.glob("**/pflash-task-*.json")):
        data = json.loads(path.read_text(errors="replace"))
        original = data.get("original_prompt") or ""
        compressed = data.get("compressed_prompt") or ""
        values = {k: v in compressed for k, v in EXPECT.items()}
        rows.append({
            "file": str(path.relative_to(debug_dir.parent)),
            "original_tokens": data.get("original_tokens"),
            "kept_tokens": data.get("kept_tokens"),
            "values_in_compressed": values,
            "all_values_in_compressed": all(values.values()),
            "all_values_in_original": all(v in original for v in EXPECT.values()),
        })
    return rows


def parse_args():
    p = argparse.ArgumentParser(description="CP029 PFlash critical-line format robustness harness")
    p.add_argument("--label", default="cp029")
    p.add_argument("--server-bin", type=Path, default=DEFAULT_BIN)
    p.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    p.add_argument("--draft", type=Path, default=DEFAULT_DRAFT)
    p.add_argument("--pflash-model", type=Path, default=DEFAULT_PFLASH_MODEL)
    p.add_argument("--port", type=int, default=18147)
    p.add_argument("--max-tokens", type=int, default=96)
    p.add_argument("--contexts", default="8000,16000,24000")
    p.add_argument("--formats", default="canonical_single_line,json_object,pipe_delimited,markdown_table,yaml_block,xml_attrs,multiline_equals,prose_sentence,spaced_equals")
    p.add_argument("--run-dir", type=Path)
    args = p.parse_args()
    args.root = ROOT
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    args.run_dir = args.run_dir or ROOT / f"temp-{args.label}-critical-line-robustness-{stamp}"
    args.log_dir = args.run_dir / "logs"
    args.out_dir = args.run_dir / "outputs"
    args.debug_dir = args.run_dir / "debug_export"
    for d in (args.log_dir, args.out_dir, args.debug_dir):
        d.mkdir(parents=True, exist_ok=True)
    args.context_list = [int(x) for x in args.contexts.split(",") if x]
    args.format_list = [x for x in args.formats.split(",") if x]
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
    assert_port_free(args.port)
    proc, log, pflash_args = start_server(args)
    try:
        rows = []
        for fmt in args.format_list:
            for n in args.context_list:
                row = run_one(args, fmt, n)
                rows.append(row)
                print(json.dumps(row), flush=True)
    finally:
        stop_server(proc, log)

    debug_rows = debug_summary(args.debug_dir)
    cases = len(rows)
    passed = sum(1 for r in rows if r["ok"])
    failed = cases - passed
    debug_all = sum(1 for r in debug_rows if r["all_values_in_compressed"])
    result = {
        "status": "ok" if failed == 0 else "fail",
        "label": args.label,
        "note": "Focused CP029 robustness harness; not canonical suite green.",
        "endpoint": "/v1/chat/completions",
        "chat_template_kwargs": {"enable_thinking": False},
        "pflash": {"mode": "always", "keep_ratio": 0.10, "score": "model"},
        "contexts": args.context_list,
        "formats": args.format_list,
        "run_dir": str(args.run_dir),
        "pflash_args": pflash_args,
        "rows": rows,
        "debug_exports": debug_rows,
        "summary": {
            "cases": cases,
            "passed": passed,
            "failed": failed,
            "debug_exports": len(debug_rows),
            "debug_all_values_in_compressed": debug_all,
        },
    }
    out = args.run_dir / f"{args.label}_critical_line_robustness.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps({"status": result["status"], "result": str(out), "run_dir": str(args.run_dir), "summary": result["summary"]}, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
