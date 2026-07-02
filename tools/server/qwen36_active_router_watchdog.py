#!/usr/bin/env python3
"""Silent watchdog for active_broad_creative_only router rollout.

Prints nothing when OK. Prints an alert when health fails or decision log shows
unsafe active behavior: active=true on exact/json/code/list/unknown/low-confidence,
or low keep/high dflash on non-broad/creative tasks.

Attribution:
- llama.cpp is provided by the ggml-org llama.cpp project under the MIT license.
- PFlash/DFlash/KVFlash experimentation in this branch builds on Lucebox-style
  prompt/decode acceleration research and tooling.
"""
import json
import os
import pathlib
import sys
import time
import urllib.request

HEALTH_URL = os.environ.get("QWEN36_WATCHDOG_HEALTH_URL", "http://127.0.0.1:18087/health")
DECISION_LOG = pathlib.Path(os.environ.get("QWEN36_DECISION_LOG", str(pathlib.Path.cwd() / "logs" / "qwen36-smart-router.decisions.jsonl")))
EXPECTED_MODE = "active_broad_creative_only"
MAX_LINES = 500
MAX_AGE_SECONDS = 24 * 3600

BROAD_OK = {"broad", "creative"}
TASK_OK = {"broad_analysis", "creative_generation", "creative"}
EXACTISH_TASKS = {
    "exact_retrieval", "exact_enumerative", "enumerative_extraction",
    "json_output", "code_patch", "unknown"
}

alerts = []

# Health/mode check
try:
    with urllib.request.urlopen(HEALTH_URL, timeout=3) as r:
        health = json.loads(r.read().decode("utf-8", "replace"))
    if health.get("status") != "ok":
        alerts.append(f"router health status not ok: {health}")
    if health.get("classifier_mode") != EXPECTED_MODE:
        alerts.append(f"router mode drift: expected {EXPECTED_MODE}, got {health.get('classifier_mode')}")
    if not health.get("direct") or not health.get("pflash") or not health.get("classifier_configured"):
        alerts.append(f"router dependency unhealthy: {health}")
except Exception as e:
    alerts.append(f"router health check failed: {e!r}")

# Decision safety scan
if DECISION_LOG.exists():
    try:
        lines = DECISION_LOG.read_text(errors="replace").splitlines()[-MAX_LINES:]
        now = time.time()
        checked = 0
        active_checked = 0
        for line in lines:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            route = rec.get("route") if isinstance(rec.get("route"), dict) else {}
            ab = route.get("classifier_ab") if isinstance(route.get("classifier_ab"), dict) else {}
            mode = ab.get("mode")
            if mode != EXPECTED_MODE:
                continue
            # only scan recent-ish records when timestamp parses simply
            checked += 1
            active = bool(ab.get("active"))
            classifier = ab.get("classifier") if isinstance(ab.get("classifier"), dict) else {}
            cls = str(classifier.get("class") or "unknown").lower()
            task = str(classifier.get("task_type") or "unknown").lower()
            conf = float(classifier.get("confidence") or 0.0)
            keep = (route.get("pflash") or {}).get("keep_ratio")
            nmax = (route.get("dflash") or {}).get("n_max")
            reason = str(route.get("reason") or "")
            if active:
                active_checked += 1
                if cls not in BROAD_OK or task not in TASK_OK:
                    alerts.append(f"unsafe active class/task: class={cls} task={task} conf={conf} reason={reason}")
                if task in EXACTISH_TASKS or cls in {"exact", "unknown"}:
                    alerts.append(f"active exactish/unknown route: class={cls} task={task} conf={conf} reason={reason}")
                if conf < 0.65:
                    alerts.append(f"active low-confidence route: class={cls} task={task} conf={conf} reason={reason}")
            # Exactish cases should stay conservative when not active.
            if task in EXACTISH_TASKS or cls in {"exact", "unknown"}:
                if keep is not None and float(keep) < 0.7:
                    alerts.append(f"exactish low keep: class={cls} task={task} keep={keep} reason={reason}")
                if nmax is not None and int(nmax) > 4:
                    alerts.append(f"exactish high dflash: class={cls} task={task} n_max={nmax} reason={reason}")
        # No alert for low/no traffic; normal immediately after rollout.
    except Exception as e:
        alerts.append(f"decision log scan failed: {e!r}")
else:
    alerts.append(f"decision log missing: {DECISION_LOG}")

if alerts:
    print("Qwen36 active-router watchdog ALERT")
    for a in alerts[:20]:
        print(f"- {a}")
    if len(alerts) > 20:
        print(f"- ... {len(alerts)-20} more")
    sys.exit(1)
