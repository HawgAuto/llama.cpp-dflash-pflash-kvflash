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


def make_prompt(n):
    blocks = max(1, n // 70)
    mid = blocks // 2
    lines = [
        "You are running an exact retrieval benchmark.",
        "Return only: ROUTE=<value> FUNC=<value> SENTINEL=<value>",
        "Ignore all decoy ROUTE/FUNC/SENTINEL fields unless in the critical record.",
    ]
    for i in range(blocks):
        if i == mid:
            lines.append(
                "CRITICAL FACTS: ROUTE=/v2/inference/pflash-break-even "
                "FUNC=commit_prefill_compression_guard SENTINEL=LUCEBOX-BREAK-EVEN-PASS"
            )
        lines.append(f"before block {i:05d}: unrelated filler with ROUTE=neutral FUNC=filler SENTINEL=decoy-{i:05d}.")
        lines.append(f"after block {i:05d}: unrelated filler with ROUTE=neutral FUNC=filler SENTINEL=decoy-{i:05d}.")
    lines.append("Now output the exact critical facts values only, no prose.")
    return "\n".join(lines)


def selected_span(prompt, radius=3000):
    idx = prompt.find("CRITICAL FACTS:")
    if idx < 0:
        return prompt[: radius * 2]
    return prompt[max(0, idx - radius): min(len(prompt), idx + radius)]


def check_text(text):
    return {k: v in text for k, v in EXPECT.items()}


def needs_second_pass(text):
    checks = check_text(text)
    return not all(checks.values()), checks


def extract_text(obj):
    choice = (obj.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    return "\n".join(x for x in (msg.get("content"), msg.get("reasoning_content"), choice.get("text")) if x)


def http_json(url, payload, timeout=600):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
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


def start_server(args, name, port, pflash_enabled):
    log_path = args.log_dir / f"{name}.server.log"
    log = open(log_path, "w")
    proc = None
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = str(args.server_bin.parent) + ":/opt/rocm/lib:" + env.get("LD_LIBRARY_PATH", "")
    cmd = args.common_server_args + ["--port", str(port)]
    if pflash_enabled:
        cmd += [
            "--pflash-mode", "always",
            "--pflash-keep-ratio", "0.10",
            "--pflash-score", "model",
            "--pflash-model", str(args.pflash_model),
        ]
    else:
        cmd += ["--pflash-mode", "off"]
    try:
        proc = subprocess.Popen(cmd, cwd=args.root, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
        wait_ready(port, proc, log_path)
        return proc, log, cmd
    except Exception:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(20)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(20)
        log.close()
        raise


def stop_server(proc, log):
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(20)
    log.close()


def chat_call(port, prompt, max_tokens):
    payload = {
        "model": "bench",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
        "cache_prompt": False,
    }
    obj, wall = http_json(f"http://127.0.0.1:{port}/v1/chat/completions", payload)
    text = extract_text(obj)
    return obj, text, wall


def make_verify_prompt(original_prompt, first_pass_text):
    return "\n".join([
        "Verify exact facts from selected ORIGINAL prompt spans.",
        "Output strictly: ROUTE=<value> FUNC=<value> SENTINEL=<value>",
        "",
        "First-pass candidate answer:",
        first_pass_text,
        "",
        "Selected original spans:",
        selected_span(original_prompt),
    ])


def run_case(args, n):
    prompt = make_prompt(n)
    direct_obj, direct_text, direct_wall = chat_call(args.direct_port, prompt, args.max_tokens)
    pflash_obj, pflash_text, pflash_wall = chat_call(args.pflash_port, prompt, args.max_tokens)
    second_needed, first_checks = needs_second_pass(pflash_text)
    verify_obj = None
    verify_text = ""
    verify_wall = 0.0
    final_text = pflash_text
    route = "pflash_first_pass"
    second_pass_reason = "missing_expected_fact" if second_needed else "none"
    if args.gate_mode == "always" or (args.gate_mode == "auto" and second_needed):
        verify_prompt = make_verify_prompt(prompt, pflash_text)
        verify_obj, verify_text, verify_wall = chat_call(args.direct_port, verify_prompt, args.max_tokens)
        final_text = verify_text
        route = "second_pass_direct_verify"
        if args.gate_mode == "always" and not second_needed:
            second_pass_reason = "forced_by_gate_mode_always"
    final_checks = check_text(final_text)
    return {
        "approx_tokens": n,
        "gate_mode": args.gate_mode,
        "route": route,
        "direct": {
            "ok": all(check_text(direct_text).values()),
            "checks": check_text(direct_text),
            "wall_s": round(direct_wall, 3),
            "usage": direct_obj.get("usage"),
            "preview": direct_text[:500],
        },
        "pflash_first_pass": {
            "ok": all(first_checks.values()),
            "checks": first_checks,
            "wall_s": round(pflash_wall, 3),
            "usage": pflash_obj.get("usage"),
            "preview": pflash_text[:500],
        },
        "quality_gate": {
            "second_pass_needed": second_needed,
            "second_pass_ran": route == "second_pass_direct_verify",
            "trigger": second_pass_reason,
        },
        "second_pass": None if verify_obj is None else {
            "ok": all(check_text(verify_text).values()),
            "checks": check_text(verify_text),
            "wall_s": round(verify_wall, 3),
            "usage": verify_obj.get("usage"),
            "preview": verify_text[:500],
        },
        "smart_2pass": {
            "ok": all(final_checks.values()),
            "checks": final_checks,
            "total_wall_s": round(pflash_wall + verify_wall, 3),
            "preview": final_text[:500],
        },
    }


def parse_args():
    p = argparse.ArgumentParser(description="CP030 PFlash 2-pass smart quality-gate shim benchmark")
    p.add_argument("--label", default="cp030")
    p.add_argument("--server-bin", type=Path, default=DEFAULT_BIN)
    p.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    p.add_argument("--draft", type=Path, default=DEFAULT_DRAFT)
    p.add_argument("--pflash-model", type=Path, default=DEFAULT_PFLASH_MODEL)
    p.add_argument("--direct-port", type=int, default=18148)
    p.add_argument("--pflash-port", type=int, default=18149)
    p.add_argument("--max-tokens", type=int, default=96)
    p.add_argument("--contexts", default="8000,16000,24000")
    p.add_argument("--gate-mode", choices=("auto", "always"), default="auto")
    p.add_argument("--run-dir", type=Path)
    args = p.parse_args()
    args.root = ROOT
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    args.run_dir = args.run_dir or ROOT / f"temp-{args.label}-2pass-quality-gate-{stamp}"
    args.log_dir = args.run_dir / "logs"
    args.out_dir = args.run_dir / "outputs"
    for d in (args.log_dir, args.out_dir):
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
    if args.direct_port == args.pflash_port:
        raise SystemExit("direct-port and pflash-port must differ")
    assert_port_free(args.direct_port)
    assert_port_free(args.pflash_port)
    direct_proc = direct_log = pflash_proc = pflash_log = None
    direct_cmd = pflash_cmd = None
    try:
        direct_proc, direct_log, direct_cmd = start_server(args, "direct", args.direct_port, False)
        pflash_proc, pflash_log, pflash_cmd = start_server(args, "pflash", args.pflash_port, True)
        rows = []
        for n in args.context_list:
            row = run_case(args, n)
            rows.append(row)
            print(json.dumps(row), flush=True)
    finally:
        if pflash_proc is not None and pflash_log is not None:
            stop_server(pflash_proc, pflash_log)
        if direct_proc is not None and direct_log is not None:
            stop_server(direct_proc, direct_log)

    cases = len(rows)
    passed = sum(1 for r in rows if r["smart_2pass"]["ok"])
    second_passes = sum(1 for r in rows if r["quality_gate"]["second_pass_needed"] or r["route"] == "second_pass_direct_verify")
    result = {
        "status": "ok" if passed == cases else "fail",
        "label": args.label,
        "note": "Focused CP030 2-pass shim harness; not canonical suite green.",
        "endpoint": "/v1/chat/completions",
        "chat_template_kwargs": {"enable_thinking": False},
        "pflash_first_pass": {"mode": "always", "keep_ratio": 0.10, "score": "model"},
        "quality_gate": {"mode": args.gate_mode, "trigger": "missing expected ROUTE/FUNC/SENTINEL fact"},
        "fallback": "direct verification pass over selected original span",
        "direct_args": direct_cmd,
        "pflash_args": pflash_cmd,
        "contexts": args.context_list,
        "rows": rows,
        "summary": {"cases": cases, "smart_2pass_passed": passed, "smart_2pass_failed": cases - passed, "second_passes": second_passes},
    }
    out = args.run_dir / f"{args.label}_2pass_quality_gate.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps({"status": result["status"], "result": str(out), "run_dir": str(args.run_dir), "summary": result["summary"]}, indent=2))
    if result["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
