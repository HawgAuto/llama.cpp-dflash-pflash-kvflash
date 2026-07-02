#!/usr/bin/env python3
"""Policy-only active-routing simulator for qwen36-smart-router.

Imports the router module and evaluates classifier/correction/guardrail policy without
calling the generation backend. It may call a small classifier sidecar when
configured, but never sends prompts to the full model.

Attribution:
- llama.cpp is provided by the ggml-org llama.cpp project under the MIT license.
- PFlash/DFlash/KVFlash experimentation in this branch builds on Lucebox-style
  prompt/decode acceleration research and tooling.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

OPS = Path(__file__).resolve().parent
ROUTER_PATH = OPS / "qwen36-smart-router.py"
RESULTS = OPS / "results"


def load_router(mode: str, correction: str, classifier_base: str, timeout: float):
    os.environ["QWEN36_ROUTER_CLASSIFIER_MODE"] = mode
    os.environ["QWEN36_CLASSIFIER_CORRECTION_MODE"] = correction
    os.environ["QWEN36_CLASSIFIER_BASE"] = classifier_base
    os.environ.setdefault("QWEN36_CLASSIFIER_MODEL", "pflash-router-classifier")
    os.environ["QWEN36_CLASSIFIER_TIMEOUT"] = str(timeout)
    os.environ.setdefault("QWEN36_CLASSIFIER_MAX_CHARS", "8000")
    os.environ.setdefault("QWEN36_CLASSIFIER_MIN_CONFIDENCE", "0.65")
    spec = importlib.util.spec_from_file_location("qwen36_smart_router", ROUTER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {ROUTER_PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["qwen36_smart_router"] = mod
    spec.loader.exec_module(mod)
    return mod


def filler(label: str, repeats: int = 1200, exact_noise: bool = False) -> str:
    base = f"Document block {label}: general operational note about latency, routing, and user-visible behavior. "
    if exact_noise:
        base += "Incidental schema words: json schema required fields keys values exact line numbers /tmp/path TOKEN_ABCD1234. "
    return (base * repeats)[:52000]


def body(prompt: str, max_tokens: int = 96) -> Dict[str, Any]:
    return {"model": "qwen36-smart", "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "stream": False}


def cases() -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    exact_prompts = [
        "Return only valid JSON with keys status, port, and route from the context below.",
        "List every API endpoint mentioned, preserving exact path strings.",
        "Extract all config keys and their exact values. No prose.",
        "Apply this patch mentally and output the exact diff only.",
        "What is the exact header value for X-Trace-Token?",
        "Return the complete list of IDs, one per line.",
        "Do not summarize. Quote the exact error message and line number.",
        "Generate a JSON schema for this object and include required fields exactly.",
        "Find the exact file path mapped to model_dir.",
        "List all three ports and their service names.",
        "Return only the value of placeholder_id from the manifest.",
        "Preserve every numeric literal in the answer.",
        "Extract the key-value pairs into CSV with no missing fields.",
        "Code task: edit function route_request to handle forced route.",
        "Patch the YAML config and output the modified YAML only.",
        "Which UUID appears after job_id? Return exactly that UUID.",
        "From this log, list every ERROR code and timestamp.",
        "Return the route path, method, and handler function name exactly.",
        "Emit only a JSON array of all package versions.",
        "Find all values matching TOKEN_[A-Z0-9]+ and return the complete list.",
    ]
    broad_prompts = [
        "Summarize the overall themes in the context below.",
        "Give me a high-level overview of what changed and why it matters.",
        "Triage these logs and identify the likely root cause, no need for every line.",
        "Rank the most likely failure modes from this long incident report.",
        "Analyze the broad tradeoffs in these design notes.",
        "Compare the two approaches at a conceptual level.",
        "Review this document and explain the main risks.",
        "Diagnose the situation from the narrative and suggest next steps.",
        "Summarize this context; ignore incidental JSON examples unless important.",
        "What are the recurring patterns across these support tickets?",
        "Give an executive summary of this migration plan.",
        "Synthesize the main arguments and tensions in these notes.",
        "Find likely candidates for optimization from these benchmark notes.",
        "Assess whether this rollout looks safe overall.",
        "Explain what probably happened in plain English.",
    ]
    creative_prompts = [
        "Brainstorm five product names for this internal router feature.",
        "Write a short announcement for the performance improvement.",
        "Generate three tagline options based on the design notes.",
        "Draft a friendly status update for users.",
        "Compose a short story analogy explaining speculative decoding.",
        "Ideate names for the broad/creative-only rollout mode.",
        "Write release notes in an upbeat tone from this context.",
        "Generate a metaphor to explain PFlash keep ratio tradeoffs.",
        "Draft a concise blog intro from these notes.",
        "Brainstorm test suite names for active routing validation.",
    ]
    mixed_safe_exact = [
        "Summarize the logs, but preserve all IDs and exact error codes in a complete list.",
        "Give an overview, then return JSON with every endpoint and method exactly.",
        "Analyze the incident and list all impacted ports with exact values.",
        "Briefly explain, but output valid JSON only.",
        "Compare these configs and include every changed key-value pair.",
        "Triage this context; do not omit any ERROR code or timestamp.",
        "Write a summary and include the complete list of UUIDs.",
        "Assess risks, preserving exact file paths and line numbers.",
        "Find likely root cause and return the exact command that failed.",
        "Explain broadly, but quote the exact placeholder_id value if present.",
    ]
    unknown_prompts = [
        "Thoughts?", "What do you make of this?", "Continue.", "Please help with the above.", "Is this okay?"
    ]
    idx = 0
    for p in exact_prompts:
        idx += 1
        items.append({"id": f"exact_{idx:02d}", "kind": "exact", "prompt": p + "\n\n" + filler(f"exact{idx}", exact_noise=True), "must_conservative": True})
    for p in broad_prompts:
        idx += 1
        items.append({"id": f"broad_{idx:02d}", "kind": "broad", "prompt": p + "\n\n" + filler(f"broad{idx}", exact_noise=(idx % 3 == 0)), "may_relax": True})
    for p in creative_prompts:
        idx += 1
        items.append({"id": f"creative_{idx:02d}", "kind": "creative", "prompt": p + "\n\n" + filler(f"creative{idx}", repeats=900, exact_noise=False), "may_relax": True})
    for p in mixed_safe_exact:
        idx += 1
        items.append({"id": f"mixed_exact_{idx:02d}", "kind": "mixed_exact", "prompt": p + "\n\n" + filler(f"mixed{idx}", exact_noise=True), "must_conservative": True})
    for p in unknown_prompts:
        idx += 1
        items.append({"id": f"unknown_{idx:02d}", "kind": "unknown", "prompt": p + "\n\n" + filler(f"unknown{idx}", repeats=700, exact_noise=False), "must_conservative": True})
    return items


def unsafe_reason(case: Dict[str, Any], policy: Dict[str, Any]) -> str | None:
    cand = policy.get("active_candidate") or ({k: policy.get(k) for k in ("route", "keep_ratio", "dflash_n_max")} if policy.get("active") else {})
    route = cand.get("route", policy.get("route"))
    keep = float(cand.get("keep_ratio", policy.get("keep_ratio", 1.0)) or 1.0)
    nmax = cand.get("dflash_n_max", policy.get("dflash_n_max"))
    nmax_i = 0 if nmax is None else int(nmax)
    if case.get("must_conservative"):
        if route == "pflash_broad":
            return f"unsafe_route={route}"
        if keep < 0.80:
            return f"unsafe_keep={keep}"
        if nmax_i > 2:
            return f"unsafe_dflash={nmax_i}"
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="active_candidate")
    ap.add_argument("--correction", default="hybrid")
    ap.add_argument("--classifier-base", default=os.environ.get("QWEN36_CLASSIFIER_BASE", "http://127.0.0.1:18161/v1"))
    ap.add_argument("--timeout", type=float, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    router = load_router(args.mode, args.correction, args.classifier_base, args.timeout)
    suite = cases()[: args.limit or None]
    rows = []
    failures = []
    latencies = []
    started = time.time()
    for c in suite:
        b = body(f"[run-id:{c['id']}]\n" + c["prompt"])
        regex_route, regex_reason = router.route_request(b)
        policy = router.classifier_policy(b, regex_route, regex_reason)
        cls = policy.get("classifier", {})
        if cls.get("latency_ms") is not None:
            latencies.append(float(cls.get("latency_ms")))
        reason = unsafe_reason(c, policy)
        row = {
            "id": c["id"], "kind": c["kind"], "safe": reason is None, "unsafe_reason": reason,
            "regex": policy.get("regex"), "classifier": cls,
            "classifier_raw": policy.get("classifier_raw"),
            "active_candidate": policy.get("active_candidate"),
            "active": {k: policy.get(k) for k in ("active", "route", "route_reason", "keep_ratio", "dflash_n_max", "classifier_rejected")},
        }
        rows.append(row)
        if reason:
            failures.append(row)
        print(f"{c['id']:<18} {c['kind']:<12} safe={str(reason is None):<5} class={cls.get('class')} conf={cls.get('confidence')} cand={row.get('active_candidate') or row['active']} reason={reason or '-'}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = RESULTS / f"qwen36-active-policy-sim-{stamp}.json"
    summary_out = RESULTS / f"qwen36-active-policy-sim-{stamp}.summary.json"
    summary = {
        "time": stamp, "mode": args.mode, "correction": args.correction, "cases": len(rows),
        "safe": len(rows) - len(failures), "failures": len(failures),
        "failure_ids": [f["id"] for f in failures],
        "classifier_available": sum(1 for r in rows if r.get("classifier", {}).get("available")),
        "classifier_invalid_or_unavailable": sum(1 for r in rows if not r.get("classifier", {}).get("available")),
        "latency_ms_median": (statistics.median(latencies) if latencies else None),
        "elapsed_s": round(time.time() - started, 2),
        "result": str(out),
    }
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    summary_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
