#!/usr/bin/env python3
"""OpenAI-compatible smart router for llama-server PFlash/DFlash experiments.

This helper is intended for publication as an example integration script for a
llama.cpp-derived server with Lucebox-style PFlash/DFlash controls. It contains
no deployment-specific paths by default; set the QWEN36_* environment variables
for local models, ports, logs, and backend names.

Attribution:
- llama.cpp is provided by the ggml-org llama.cpp project under the MIT license.
- PFlash/DFlash/KVFlash experimentation in this branch builds on Lucebox-style
  prompt/decode acceleration research and tooling.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Tuple

HOST = os.environ.get("QWEN36_SMART_HOST", "127.0.0.1")
PORT = int(os.environ.get("QWEN36_SMART_PORT", "18087"))
FULLSTACK_BASE = os.environ.get("QWEN36_FULLSTACK_BASE", "http://127.0.0.1:18160/v1")
DIRECT_BASE = os.environ.get("QWEN36_DIRECT_BASE", FULLSTACK_BASE)
PFLASH_BASE = os.environ.get("QWEN36_PFLASH_BASE", FULLSTACK_BASE)
FULLSTACK_HEALTH = os.environ.get("QWEN36_FULLSTACK_HEALTH", FULLSTACK_BASE.rsplit("/", 1)[0] + "/health")
PFLASH_HEALTH = os.environ.get("QWEN36_PFLASH_HEALTH", FULLSTACK_HEALTH)
DIRECT_HEALTH = os.environ.get("QWEN36_DIRECT_HEALTH", FULLSTACK_HEALTH)
PFLASH_PORT = int(os.environ.get("QWEN36_PFLASH_PORT", "18160"))
LOG_DIR = Path(os.environ.get("QWEN36_ROUTER_LOG_DIR", str(Path.cwd() / "logs")))
LOG_DIR.mkdir(parents=True, exist_ok=True)
DECISION_LOG = Path(os.environ.get("QWEN36_DECISION_LOG", str(LOG_DIR / "qwen36-smart-router.decisions.jsonl")))

DEFAULT_KEEP_RATIO = float(os.environ.get("QWEN36_PFLASH_KEEP_RATIO", "0.30"))
DYNAMIC_KEEP = os.environ.get("QWEN36_DYNAMIC_KEEP", "1").lower() not in {"0", "false", "no", "off"}
DYNAMIC_DFLASH = os.environ.get("QWEN36_DYNAMIC_DFLASH", "1").lower() not in {"0", "false", "no", "off"}
DEFAULT_DFLASH_GEAR = os.environ.get("QWEN36_DFLASH_GEAR", "auto").lower()
UPSTREAM_MODEL = os.environ.get("QWEN36_UPSTREAM_MODEL", "qwen-smart-router-upstream")
MANAGE_PFLASH_PROCESS = os.environ.get("QWEN36_MANAGE_PFLASH", "0").lower() in {"1", "true", "yes", "on"}
VERIFY_MODE = os.environ.get("QWEN36_VERIFY_MODE", "auto").lower()  # auto|always|never
MIN_VERIFY_TOKENS = int(os.environ.get("QWEN36_MIN_VERIFY_TOKENS", "8000"))

# Experimental classifier A/B modes. The classifier is advisory by default.
# off: regex/router only
# shadow_class: ask classifier, log class, use regex/router params
# shadow_class_params: ask classifier, log class/risk and ignored raw params, use regex/router params
# active_candidate: compute/log guarded active decision but serve regex/router params
# active_class: classifier chooses class, router maps class -> params with guardrails
# active_class_params: compatibility mode; params are ignored and class maps deterministically
# active_broad_creative_only: first safe rollout; only broad/creative classes can change params
CLASSIFIER_MODE = os.environ.get("QWEN36_ROUTER_CLASSIFIER_MODE", "off").lower()
CLASSIFIER_BASE = os.environ.get("QWEN36_CLASSIFIER_BASE", "").rstrip("/")
CLASSIFIER_MODEL = os.environ.get("QWEN36_CLASSIFIER_MODEL", "pflash-router")
CLASSIFIER_TIMEOUT = float(os.environ.get("QWEN36_CLASSIFIER_TIMEOUT", "4"))
CLASSIFIER_MAX_CHARS = int(os.environ.get("QWEN36_CLASSIFIER_MAX_CHARS", "12000"))
CLASSIFIER_MIN_CONFIDENCE = float(os.environ.get("QWEN36_CLASSIFIER_MIN_CONFIDENCE", "0.65"))
# Post-classifier correction A/B modes:
# off: raw sidecar output only
# heuristic: deterministic prompt-signal class wins
# confidence: deterministic class wins only when classifier confidence is very low
# hybrid: exact safety always wins; broad/creative fix only low-confidence or unknown/exact-collapse cases
CLASSIFIER_CORRECTION_MODE = os.environ.get("QWEN36_CLASSIFIER_CORRECTION_MODE", "off").lower()


TARGET_MODEL = os.environ.get("QWEN36_TARGET_MODEL", "")
DRAFT_MODEL = os.environ.get("QWEN36_DRAFT_MODEL", "")
PFLASH_DRAFTER = os.environ.get("QWEN36_PFLASH_DRAFTER", "")
DFLASH_SERVER = os.environ.get("QWEN36_DFLASH_SERVER", "")

EXACT_RE = re.compile(
    r"\b("
    r"patch|diff|apply this edit|exact file|config migration|tool schema|json schema|"
    r"function name|header value|route path|do not omit|verbatim|line numbers|"
    r"multi[- ]file|schema|migration|edit|sentinel|exactly|exact value|exact values|"
    r"extract|extraction|required fields?|key[- ]?value|emit only|return only|"
    r"must include|do not summarize|no prose|all values?|every|complete list"
    r")\b",
    re.I,
)
BROAD_RE = re.compile(r"\b(summarize|summary|triage|overview|what changed|rank|likely|broad|logs?|documents?|find likely|candidate|root cause|analy[sz]e)\b", re.I)

# Generic real-world salience signals. These deliberately avoid benchmark terms like
# "needle" and "magic number"; the goal is to retain operationally important prompt
# material: code/config, exact literals, keys/IDs, structured data, constraints, and
# relational facts. NIAH-like tests benefit only because they use the same kinds of
# high-value artifacts people care about in real prompts.
STRUCTURE_RE = re.compile(r"[`{}\[\]<>]|\b(class|def|function|schema|json|yaml|toml|xml|sql|tool|args?)\b", re.I)
IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+\b|\b[A-Za-z_][A-Za-z0-9_]{2,}\([^\n)]*\)|\b[A-Za-z]+[_-][A-Za-z0-9_-]+\b")
LITERAL_RE = re.compile(r"(?:/[A-Za-z0-9._~:/?#\[\]@!$&'()*+,;=%-]+)|(?:https?://\S+)|(?:\b[A-Fa-f0-9]{8,}\b)|(?:\b\d+(?:\.\d+)?\b)|(?:['\"][^'\"]{2,}['\"])")
CONSTRAINT_RE = re.compile(r"\b(must|never|always|only|exactly|required|forbidden|do not|don't|all|every|complete|preserve|verbatim|output|format|reply|return)\b", re.I)
RELATION_RE = re.compile(r"\b(is|are|maps? to|points? to|value of|key|section|gateway|header|route|path|port|id|code|token|secret|hash|uuid|version)\b", re.I)

pflash_lock = threading.Lock()
pflash_proc: subprocess.Popen | None = None
pflash_keep_ratio: float | None = None


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def messages_to_text(messages: List[Dict[str, Any]]) -> str:
    parts = []
    for m in messages:
        c = m.get("content", "")
        if isinstance(c, str):
            content = c
        else:
            content = json.dumps(c, ensure_ascii=False)
        parts.append(f"{m.get('role','user')}: {content}")
    return "\n".join(parts)


def route_request(body: Dict[str, Any]) -> Tuple[str, str]:
    forced = body.get("route") or body.get("x_route") or body.get("smart_route")
    if isinstance(forced, str) and forced in {"direct_exact", "pflash_broad", "pflash_then_direct_verify"}:
        return forced, "forced"
    text = messages_to_text(body.get("messages", []))
    toks = approx_tokens(text)
    exact = bool(EXACT_RE.search(text))
    broad = bool(BROAD_RE.search(text))
    if toks < 8000:
        return "direct_exact", f"tokens<{8000}"
    if exact:
        return "pflash_then_direct_verify", "long_context+exactness_signals"
    if broad or toks >= 8000:
        return "pflash_broad", "long_context_broad"
    return "direct_exact", "fallback"


TASK_CLASS_TO_ROUTE = {
    "broad_analysis": "pflash_broad",
    "long_broad": "pflash_broad",
    "very_long_broad": "pflash_broad",
    "structured_literal": "pflash_broad",
    "exact_enumerative": "pflash_then_direct_verify",
    "exact_retrieval": "pflash_then_direct_verify",
    "enumerative_extraction": "pflash_then_direct_verify",
    "code_patch": "pflash_then_direct_verify",
    "json_output": "pflash_then_direct_verify",
    "unknown": "direct_exact",
}

TASK_CLASS_TO_PARAMS = {
    "broad_analysis": (0.40, 4),
    "long_broad": (0.40, 2),
    "very_long_broad": (0.35, 2),
    "structured_literal": (0.70, 4),
    "exact_enumerative": (0.90, 2),
    "exact_retrieval": (0.90, 2),
    "enumerative_extraction": (0.90, 2),
    "code_patch": (0.90, 2),
    "json_output": (0.80, 2),
    "unknown": (0.70, 4),
}

_ALLOWED_TASK_CLASSES = set(TASK_CLASS_TO_PARAMS)
_SIMPLE_CLASS_TO_TASK_TYPE = {
    "exact": "exact_retrieval",
    "broad": "broad_analysis",
    "creative": "broad_analysis",
    "unknown": "unknown",
}
_ALLOWED_SIMPLE_CLASSES = set(_SIMPLE_CLASS_TO_TASK_TYPE)
_ALLOWED_ROUTES = {"direct_exact", "pflash_broad", "pflash_then_direct_verify"}
_ALLOWED_DFLASH = (0, 2, 4, 6, 8)


def _nearest_allowed_dflash(value: int) -> int:
    v = clamp_dflash_n(value)
    return min(_ALLOWED_DFLASH, key=lambda x: abs(x - v))


def _message_digest(text: str, max_chars: int = CLASSIFIER_MAX_CHARS) -> Dict[str, Any]:
    head_n = max(1200, max_chars // 3)
    tail_n = max(1200, max_chars // 3)
    remaining = max(0, max_chars - head_n - tail_n)
    samples: List[str] = []
    if remaining and len(text) > head_n + tail_n:
        middle = text[head_n: max(head_n, len(text) - tail_n)]
        if middle:
            slots = 2
            sample_n = max(600, remaining // slots)
            for i in range(slots):
                if len(middle) <= sample_n:
                    samples.append(middle)
                    break
                start = int((len(middle) - sample_n) * (i + 1) / (slots + 1))
                samples.append(middle[start:start + sample_n])
    toks = approx_tokens(text)
    return {
        "approx_tokens": toks,
        "literal_density": len(LITERAL_RE.findall(text)) / max(1, toks),
        "signals": {
            "exact": bool(EXACT_RE.search(text)),
            "broad": bool(BROAD_RE.search(text)),
            "structured": bool(STRUCTURE_RE.search(text)),
            "enumerative": bool(re.search(r"\b(all|every|complete|list|values?|fields?|keys?)\b", text, re.I)),
            "jsonish_output": _requested_json_output(text),
        },
        "head": text[:head_n],
        "middle_samples": samples,
        "tail": text[-tail_n:],
    }


def _classifier_prompt(body: Dict[str, Any]) -> List[Dict[str, str]]:
    text = messages_to_text(body.get("messages", []))
    digest = _message_digest(text)
    digest["requested_max_tokens"] = body.get("max_tokens", body.get("max_completion_tokens", 256))
    return [
        {"role": "system", "content": (
            "You are a tiny routing classifier. Return one STRICT JSON object only. "
            "Exact shape: {\"class\":\"exact|broad|creative|unknown\",\"confidence\":0.0}. "
            "Do not return reason, route, task_type, risk, keep ratio, DFlash, params, markdown, or extra keys. "
            "Priority ladder: choose exact for JSON/schema output, code/patches, exact values, "
            "or all/every/complete/list/values/fields/keys enumeration. "
            "Else choose broad for summarize, triage, rank, overview, likely cause, themes, compare, analyze, or synthesis. "
            "Else choose creative only for explicit write/generate/brainstorm/ideate/story/tagline/name creation. "
            "Do not label summarize, triage, rank, overview, likely cause, compare, or analyze as creative. "
            "Do not label broad/creative prompts as exact unless they request JSON, code, every/all/list, or exact values. "
            "When truly unsure choose unknown. "
            "Examples: 'return JSON' -> exact; 'list every field' -> exact; 'patch this code' -> exact; "
            "'summarize the overall themes' -> broad; 'triage logs' -> broad; 'brainstorm names' -> creative. Use quoted JSON strings."
        )},
        {"role": "user", "content": json.dumps(digest, ensure_ascii=False)},
    ]


def _extract_json_object(text: str) -> Dict[str, Any] | None:
    def _loads(candidate: str) -> Dict[str, Any] | None:
        try:
            obj = json.loads(candidate)
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass
        # Small classifier models occasionally emit known enum values without quotes.
        repaired = re.sub(
            r'(:\s*)(broad_analysis|long_broad|very_long_broad|structured_literal|exact_enumerative|exact_retrieval|enumerative_extraction|code_patch|json_output|unknown|direct_exact|pflash_broad|pflash_then_direct_verify|broad|structured|exact|json|code|creative)(\s*[,}])',
            r'\1"\2"\3',
            candidate,
        )
        if repaired != candidate:
            try:
                obj = json.loads(repaired)
                return obj if isinstance(obj, dict) else None
            except Exception:
                return None
        return None

    obj = _loads(text)
    if obj is not None:
        return obj
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    return _loads(m.group(0))


def query_classifier(body: Dict[str, Any]) -> Dict[str, Any]:
    if CLASSIFIER_MODE == "off":
        return {"available": False, "mode": CLASSIFIER_MODE, "reason": "disabled"}
    if not CLASSIFIER_BASE:
        return {"available": False, "mode": CLASSIFIER_MODE, "reason": "no_classifier_base_configured"}
    payload = {
        "model": CLASSIFIER_MODEL,
        "messages": _classifier_prompt(body),
        "temperature": 0,
        "max_tokens": 32,
        "stream": False,
        "response_format": {"type": "json_object"},
    }
    started = time.time()
    try:
        resp = http_json(f"{CLASSIFIER_BASE}/chat/completions", payload, timeout=max(1, int(CLASSIFIER_TIMEOUT)))
        content = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
        parsed = _extract_json_object(content or "")
        if not parsed:
            return {"available": False, "mode": CLASSIFIER_MODE, "reason": "invalid_json", "raw": content[:500], "latency_ms": round((time.time() - started) * 1000, 2)}
        norm = normalize_classifier_decision(parsed)
        norm.update({"available": True, "mode": CLASSIFIER_MODE, "latency_ms": round((time.time() - started) * 1000, 2)})
        return norm
    except Exception as e:
        return {"available": False, "mode": CLASSIFIER_MODE, "reason": repr(e), "latency_ms": round((time.time() - started) * 1000, 2)}


def normalize_classifier_decision(obj: Dict[str, Any]) -> Dict[str, Any]:
    suggested = obj.get("suggested") if isinstance(obj.get("suggested"), dict) else {}
    params = obj.get("params") if isinstance(obj.get("params"), dict) else {}
    simple_class = str(obj.get("class") or "").lower().strip()
    if simple_class not in _ALLOWED_SIMPLE_CLASSES:
        simple_class = ""
    task_type = str(obj.get("task_type") or suggested.get("task_type") or params.get("task_type") or "").lower().strip()
    if not task_type and simple_class:
        task_type = _SIMPLE_CLASS_TO_TASK_TYPE[simple_class]
    if not task_type:
        task_type = "unknown"
    if task_type not in _ALLOWED_TASK_CLASSES:
        task_type = "unknown"
    risk = str(obj.get("risk") or simple_class or "unknown").lower().strip()
    confidence_present = "confidence" in obj
    try:
        confidence = float(obj.get("confidence", 0.0))
    except Exception:
        confidence = 0.0
    # The model may still emit legacy route/param suggestions. Keep them only as
    # audit metadata; live and active policies use deterministic class->params.
    ignored_suggestion = {}
    for key in ("route", "pflash_keep_ratio", "dflash_n_max"):
        if key in obj:
            ignored_suggestion[key] = obj.get(key)
        if key in suggested:
            ignored_suggestion[key] = suggested.get(key)
        if key in params:
            ignored_suggestion[key] = params.get(key)
    route = TASK_CLASS_TO_ROUTE.get(task_type, "direct_exact")
    keep_f, n_i = TASK_CLASS_TO_PARAMS[task_type]
    constraints = obj.get("constraints") if isinstance(obj.get("constraints"), dict) else {}
    return {
        "class": simple_class or task_type,
        "task_type": task_type,
        "risk": risk,
        "confidence": max(0.0, min(1.0, confidence)),
        "confidence_present": confidence_present,
        "suggested": {"route": route, "pflash_keep_ratio": keep_f, "dflash_n_max": n_i},
        "ignored_model_suggestion": ignored_suggestion,
        "constraints": constraints,
        "reason": str(obj.get("reason") or "")[:500],
        "raw": obj,
    }


def _latest_user_content(body: Dict[str, Any]) -> str:
    messages = body.get("messages", [])
    if not isinstance(messages, list):
        return messages_to_text(messages)
    for msg in reversed(messages):
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            return content
        return json.dumps(content, ensure_ascii=False)
    return messages_to_text(messages)


def _instruction_lines(body: Dict[str, Any]) -> List[str]:
    text = _latest_user_content(body)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    # Skip synthetic run-id tags used by local sweeps: [run-id:case]
    return [ln for ln in lines if not re.fullmatch(r"\[[^\]]{1,120}\]", ln)]


def _instruction_focus_text(body: Dict[str, Any]) -> str:
    """Return the part most likely to contain the user's actual task.

    Long prompts often contain examples, logs, JSON, or filler text with exactness
    vocabulary. For classifier correction, prefer the leading instruction and the
    final instruction window over incidental context in the middle.
    """
    text = _latest_user_content(body)
    meaningful = _instruction_lines(body)
    head = "\n".join(meaningful[:4]) if meaningful else text[:1200]
    tail = text[-1200:]
    return (head + "\n" + tail)[:3000]


def _heuristic_prompt_class(body: Dict[str, Any]) -> Tuple[str, str]:
    """Deterministic class from prompt signals for A/B correction.

    Exact still wins for explicit output contracts, but broad/creative instruction
    verbs in the actual request should not be overridden by incidental context terms.
    """
    focus = _instruction_focus_text(body)
    leading = (_instruction_lines(body) or [""])[0]
    full_text = messages_to_text(body.get("messages", []))
    exact_phrase = re.compile(r"\b(?:exact|verbatim|preserve)\s+(?:command|commands|path|paths|id|ids|code|codes|error|errors|timestamp|timestamps|field|fields|key|keys|value|values|literal|literals)\b", re.I)
    exactish_leading = bool(EXACT_RE.search(leading)) or bool(exact_phrase.search(leading)) or bool(re.search(r"\b(all|every|complete|list|values?|fields?|keys?)\b", leading, re.I)) or _requested_json_output(leading)
    broad_leading = bool(BROAD_RE.search(leading) or re.search(r"\b(synthesis|themes?|compare|explain|diagnose|review|assess|evaluate)\b", leading, re.I))
    creative_leading = bool(re.search(r"\b(write|generate|brainstorm|ideate|draft|compose|story|taglines?|names?|slogans?|creative)\b", leading, re.I))
    if exactish_leading:
        return "exact", "heuristic_exact_safety_leading"
    if broad_leading:
        return "broad", "heuristic_broad_analysis_leading"
    if creative_leading:
        return "creative", "heuristic_creative_generation_leading"
    exactish_focus = bool(EXACT_RE.search(focus)) or bool(re.search(r"\b(all|every|complete|list|values?|fields?|keys?)\b", focus, re.I)) or _requested_json_output(focus)
    broad_focus = bool(BROAD_RE.search(focus) or re.search(r"\b(synthesis|themes?|compare|explain|diagnose|review|assess|evaluate)\b", focus, re.I))
    creative_focus = bool(re.search(r"\b(write|generate|brainstorm|ideate|draft|compose|story|taglines?|names?|slogans?|creative)\b", focus, re.I))
    if exactish_focus:
        return "exact", "heuristic_exact_safety_focus"
    if broad_focus:
        return "broad", "heuristic_broad_analysis_focus"
    if creative_focus:
        return "creative", "heuristic_creative_generation_focus"
    # Fall back to whole prompt only when no clear instruction was found.
    exactish_full = bool(EXACT_RE.search(full_text)) or bool(re.search(r"\b(all|every|complete|list|values?|fields?|keys?)\b", full_text, re.I)) or _requested_json_output(full_text)
    if exactish_full:
        return "exact", "heuristic_exact_safety_full"
    if BROAD_RE.search(full_text) or re.search(r"\b(synthesis|themes?|compare|explain|diagnose|review|assess|evaluate)\b", full_text, re.I):
        return "broad", "heuristic_broad_analysis_full"
    if re.search(r"\b(write|generate|brainstorm|ideate|draft|compose|story|taglines?|names?|slogans?|creative)\b", full_text, re.I):
        return "creative", "heuristic_creative_generation_full"
    return "unknown", "heuristic_unknown"


def _set_simple_class(decision: Dict[str, Any], simple_class: str, reason: str) -> Dict[str, Any]:
    corrected = dict(decision)
    original = {
        "class": decision.get("class"),
        "task_type": decision.get("task_type"),
        "risk": decision.get("risk"),
        "suggested": decision.get("suggested"),
    }
    task_type = _SIMPLE_CLASS_TO_TASK_TYPE.get(simple_class, "unknown")
    route = TASK_CLASS_TO_ROUTE.get(task_type, "direct_exact")
    keep_f, n_i = TASK_CLASS_TO_PARAMS[task_type]
    corrected.update({
        "class": simple_class,
        "task_type": task_type,
        "risk": simple_class,
        "suggested": {"route": route, "pflash_keep_ratio": keep_f, "dflash_n_max": n_i},
        "correction": {"applied": True, "reason": reason, "original": original},
    })
    return corrected


def correct_classifier_decision(body: Dict[str, Any], decision: Dict[str, Any], mode: str | None = None) -> Dict[str, Any]:
    mode = (mode or CLASSIFIER_CORRECTION_MODE or "off").lower()
    if mode in {"", "off", "none", "0", "false"} or not decision.get("available"):
        return decision
    heuristic_class, heuristic_reason = _heuristic_prompt_class(body)
    raw_class = str(decision.get("class") or "unknown").lower()
    confidence = float(decision.get("confidence", 0.0) or 0.0)
    if mode == "heuristic":
        return _set_simple_class(decision, heuristic_class, f"{mode}:{heuristic_reason}")
    if mode == "confidence":
        if confidence < CLASSIFIER_MIN_CONFIDENCE:
            return _set_simple_class(decision, heuristic_class, f"{mode}:low_confidence:{heuristic_reason}")
        return decision
    if mode == "hybrid":
        # Preserve exact safety even if the tiny classifier is over-confident.
        if heuristic_class == "exact" and raw_class != "exact":
            return _set_simple_class(decision, "exact", f"{mode}:exact_safety:{heuristic_reason}")
        # Repair common 0.8B collapse: low-confidence exact for creative/broad/unknown.
        if raw_class in {"exact", "unknown", ""} and heuristic_class != raw_class and confidence < CLASSIFIER_MIN_CONFIDENCE:
            return _set_simple_class(decision, heuristic_class, f"{mode}:low_confidence_collapse:{heuristic_reason}")
        # Broad analysis should not be sacrificed to creative/exact unless exact signals exist.
        if heuristic_class == "broad" and raw_class != "broad" and confidence < 0.995:
            return _set_simple_class(decision, "broad", f"{mode}:broad_guardrail:{heuristic_reason}")
        return decision
    return decision


def _classifier_exactish(decision: Dict[str, Any]) -> bool:
    task_type = decision.get("task_type")
    constraints = decision.get("constraints") if isinstance(decision.get("constraints"), dict) else {}
    if task_type in {"exact_enumerative", "exact_retrieval", "enumerative_extraction", "code_patch", "json_output"}:
        return True
    for key in ("requires_exact_values", "requires_complete_list", "requires_json", "is_code_or_patch"):
        if constraints.get(key) is True:
            return True
    return False


def apply_classifier_guardrails(body: Dict[str, Any], decision: Dict[str, Any], route: str, keep: float, n_max: int | None) -> Tuple[str, float, int | None, str]:
    text = messages_to_text(body.get("messages", []))
    leading = (_instruction_lines(body) or [""])[0]
    exact_phrase = re.compile(r"\b(?:exact|verbatim|preserve)\s+(?:command|commands|path|paths|id|ids|code|codes|error|errors|timestamp|timestamps|field|fields|key|keys|value|values|literal|literals)\b", re.I)
    exact_regex = bool(EXACT_RE.search(leading) or exact_phrase.search(leading))
    enumerative_regex = bool(re.search(r"\b(all|every|complete|list|values?|fields?|keys?)\b", leading, re.I))
    jsonish = _requested_json_output(leading)
    reason_bits = ["classifier_guarded"]
    keep = clamp_keep_ratio(keep)
    if n_max is not None:
        n_max = _nearest_allowed_dflash(n_max)
    if decision.get("confidence", 0.0) < CLASSIFIER_MIN_CONFIDENCE:
        return route, keep, n_max, "classifier_low_confidence"
    if _classifier_exactish(decision) or exact_regex or enumerative_regex:
        keep = max(keep, 0.90)
        n_max = 2 if n_max is None else min(n_max, 2)
        if route == "pflash_broad":
            route = "pflash_then_direct_verify"
        reason_bits.append("exact_floor")
    elif jsonish:
        keep = max(keep, 0.80)
        n_max = 2 if n_max is None else min(n_max, 2)
        reason_bits.append("json_safe")
    if approx_tokens(text) < 8000 and not _classifier_exactish(decision):
        route = "direct_exact"
        reason_bits.append("short_direct")
    return route, keep, n_max, ";".join(reason_bits)


def _conservative_classifier_fallback(reason: str) -> Dict[str, Any]:
    return {
        "active": True,
        "route": "direct_exact",
        "route_reason": reason,
        "keep_ratio": 0.90,
        "keep_reason": reason,
        "dflash_n_max": 2,
        "dflash_reason": reason,
        "task_type": "unknown",
        "guard_reason": reason,
    }


def _guarded_classifier_active_decision(body: Dict[str, Any], decision: Dict[str, Any], regex_route: str, regex_keep: float, regex_n: int | None, *, label: str) -> Dict[str, Any]:
    task_type = decision.get("task_type", "unknown")
    c_route = TASK_CLASS_TO_ROUTE.get(task_type, regex_route)
    c_keep, c_n = TASK_CLASS_TO_PARAMS.get(task_type, (regex_keep, regex_n or 4))
    guarded_route, guarded_keep, guarded_n, guard_reason = apply_classifier_guardrails(body, decision, c_route, c_keep, c_n)
    return {
        "active": True,
        "route": guarded_route,
        "route_reason": f"{label}:{task_type};{guard_reason}",
        "keep_ratio": guarded_keep,
        "keep_reason": f"{label}:{task_type};{guard_reason}",
        "dflash_n_max": guarded_n,
        "dflash_reason": f"{label}:{task_type};{guard_reason}",
        "task_type": task_type,
        "guard_reason": guard_reason,
    }


def classifier_policy(body: Dict[str, Any], regex_route: str, regex_reason: str) -> Dict[str, Any]:
    regex_keep, regex_keep_reason = choose_keep_ratio(body, regex_route)
    regex_n, regex_dflash_reason = choose_dflash_gear(body, regex_route)
    decision_raw = query_classifier(body)
    decision = correct_classifier_decision(body, decision_raw)
    policy = {
        "mode": CLASSIFIER_MODE,
        "classifier": decision,
        "classifier_raw": decision_raw,
        "correction_mode": CLASSIFIER_CORRECTION_MODE,
        "regex": {
            "route": regex_route,
            "reason": regex_reason,
            "keep_ratio": regex_keep,
            "keep_reason": regex_keep_reason,
            "dflash_n_max": regex_n,
            "dflash_reason": regex_dflash_reason,
        },
        "active": False,
        "route": regex_route,
        "route_reason": regex_reason,
        "keep_ratio": regex_keep,
        "keep_reason": regex_keep_reason,
        "dflash_n_max": regex_n,
        "dflash_reason": regex_dflash_reason,
    }
    if CLASSIFIER_MODE in {"off", "shadow_class", "shadow_class_params"} or not decision.get("available"):
        return policy
    if decision.get("confidence", 0.0) < CLASSIFIER_MIN_CONFIDENCE:
        fallback = _conservative_classifier_fallback("classifier_low_confidence_conservative")
        if CLASSIFIER_MODE == "active_candidate":
            policy["active_candidate"] = {k: fallback.get(k) for k in ("route", "route_reason", "keep_ratio", "keep_reason", "dflash_n_max", "dflash_reason", "task_type", "guard_reason")}
            policy["classifier_rejected"] = "low_confidence"
            return policy
        if CLASSIFIER_MODE == "active_broad_creative_only":
            policy.update(fallback)
            policy["classifier_rejected"] = "low_confidence_conservative"
            return policy
        policy["classifier_rejected"] = "low_confidence"
        return policy

    active_decision = _guarded_classifier_active_decision(body, decision, regex_route, regex_keep, regex_n, label="classifier_class_map")
    task_type = active_decision.get("task_type", "unknown")

    if CLASSIFIER_MODE == "active_candidate":
        policy["active_candidate"] = {k: active_decision.get(k) for k in ("route", "route_reason", "keep_ratio", "keep_reason", "dflash_n_max", "dflash_reason", "task_type", "guard_reason")}
        return policy

    if CLASSIFIER_MODE in {"active_class", "active_class_params"}:
        label = "classifier_class" if CLASSIFIER_MODE == "active_class" else "classifier_class_params_compat"
        policy.update(_guarded_classifier_active_decision(body, decision, regex_route, regex_keep, regex_n, label=label))
    elif CLASSIFIER_MODE == "active_broad_creative_only":
        raw_class = str(decision.get("class") or "unknown").lower()
        if raw_class in {"broad", "creative"} and task_type == "broad_analysis":
            policy.update(_guarded_classifier_active_decision(body, decision, regex_route, regex_keep, regex_n, label="classifier_broad_creative_only"))
        else:
            policy["classifier_rejected"] = f"not_broad_or_creative:{raw_class}/{task_type}"
    return policy


def http_json(url: str, payload: Dict[str, Any], timeout: int = 600) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def get_url(url: str, timeout: int = 2) -> Tuple[int, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        return 0, repr(e)



def clamp_keep_ratio(value: float) -> float:
    return max(0.05, min(1.0, float(value)))


def choose_keep_ratio(body: Dict[str, Any], route: str, policy: Dict[str, Any] | None = None) -> Tuple[float, str]:
    if policy is not None and policy.get("active"):
        return clamp_keep_ratio(float(policy.get("keep_ratio", DEFAULT_KEEP_RATIO))), str(policy.get("keep_reason", "classifier_policy"))
    explicit = body.get("pflash_keep_ratio") or body.get("keep_ratio") or body.get("smart_keep_ratio")
    if explicit is not None:
        try:
            return clamp_keep_ratio(float(explicit)), "explicit"
        except Exception:
            pass
    if not DYNAMIC_KEEP:
        return clamp_keep_ratio(DEFAULT_KEEP_RATIO), "static_default"
    text = messages_to_text(body.get("messages", []))
    toks = approx_tokens(text)
    exact = bool(EXACT_RE.search(text))
    literal_density = len(LITERAL_RE.findall(text)) / max(1, toks)
    structured = bool(STRUCTURE_RE.search(text))
    enumerative = bool(re.search(r"\b(all|every|complete|list|values?|fields?|keys?)\b", text, re.I))
    if route == "pflash_then_direct_verify" or exact or enumerative:
        return 0.90, "dynamic_exact_or_enumerative_conservative"
    if structured or literal_density > 0.015:
        return 0.70, "dynamic_structured_or_literal_dense"
    if toks > 64000:
        return 0.35, "dynamic_very_long_broad_floor"
    if toks > 32000:
        return 0.40, "dynamic_long_broad"
    if route == "pflash_broad":
        return 0.40, "dynamic_broad_analysis_sweep_best"
    return 0.70, "dynamic_unknown_conservative"



DFLASH_GEARS = {
    "off": {"n_max": 0, "label": "off"},
    "long-safe": {"n_max": 2, "label": "long-safe-n2"},
    "safe": {"n_max": 2, "label": "long-safe-n2"},
    "mid": {"n_max": 4, "label": "mid-n4"},
    "balanced": {"n_max": 4, "label": "mid-n4"},
    "short": {"n_max": 6, "label": "short-n6"},
    "fast": {"n_max": 6, "label": "short-n6"},
    "turbo": {"n_max": 8, "label": "turbo-n8"},
}


def _body_int(body: Dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        if key in body and body.get(key) is not None:
            try:
                return int(body.get(key))
            except Exception:
                return None
    nested = body.get("speculative")
    if isinstance(nested, dict) and "speculative.n_max" in keys and nested.get("n_max") is not None:
        try:
            return int(nested.get("n_max"))
        except Exception:
            return None
    return None


def clamp_dflash_n(value: int) -> int:
    return max(0, min(32, int(value)))


def choose_dflash_gear(body: Dict[str, Any], route: str, policy: Dict[str, Any] | None = None) -> Tuple[int | None, str]:
    if policy is not None and policy.get("active"):
        n = policy.get("dflash_n_max")
        return (None if n is None else clamp_dflash_n(int(n))), str(policy.get("dflash_reason", "classifier_policy"))
    explicit_n = _body_int(body, "dflash_n_max", "dflash.n_max", "speculative.n_max", "smart_dflash_n_max")
    if explicit_n is not None:
        return clamp_dflash_n(explicit_n), "explicit_n_max"

    gear = body.get("dflash_gear") or body.get("smart_dflash_gear") or DEFAULT_DFLASH_GEAR
    if isinstance(gear, str):
        gear_l = gear.lower()
        if gear_l in DFLASH_GEARS:
            g = DFLASH_GEARS[gear_l]
            return int(g["n_max"]), f"explicit_gear:{g['label']}" if gear_l != DEFAULT_DFLASH_GEAR else f"default_gear:{g['label']}"

    if not DYNAMIC_DFLASH:
        return None, "dynamic_dflash_disabled"

    text = messages_to_text(body.get("messages", []))
    toks = approx_tokens(text)
    requested = int(body.get("max_tokens", body.get("max_completion_tokens", 256)) or 256)
    exact = bool(EXACT_RE.search(text))
    structured = bool(STRUCTURE_RE.search(text))
    jsonish = _requested_json_output(text)

    # Drivetrain precedent: classify shape, then inject missing speculative fields.
    # Default MTP buckets use more depth on short requests (short-n6), middle depth
    # on medium requests (mid-n4), and conservative depth on long-safe requests
    # (long-safe-n2). DFlash uses the same gear idea through n_max caps.
    if exact or jsonish or route == "pflash_then_direct_verify":
        return DFLASH_GEARS["long-safe"]["n_max"], "auto_exact_or_json_long_safe"
    if toks >= 64000 or requested >= 192:
        return DFLASH_GEARS["long-safe"]["n_max"], "auto_decode_long_safe"
    if structured or toks >= 12000 or requested >= 96:
        return DFLASH_GEARS["mid"]["n_max"], "auto_mid_decode_or_structured"
    return DFLASH_GEARS["short"]["n_max"], "auto_short_decode"


def request_looks_normal_generation(original: str) -> bool:
    latest = original[-2500:]
    has_exact = bool(EXACT_RE.search(latest))
    has_salient = bool(STRUCTURE_RE.search(latest) or LITERAL_RE.search(latest) or CONSTRAINT_RE.search(latest))
    normal = bool(re.search(r"\b(write|explain|summarize|essay|list top|top \d+|solve|reason|json|tool call|weather|fizzbuzz)\b", latest, re.I))
    return normal and not (has_exact and has_salient)


def _requested_json_output(original: str) -> bool:
    tail = original[-3000:]
    return bool(re.search(r"\b(json|valid json|json array|json object|emit only json|return only json)\b", tail, re.I))


def _looks_jsonish(text: str) -> bool:
    stripped = text.strip()
    return stripped.startswith("{") or stripped.startswith("[")


def _json_is_invalid(text: str) -> bool:
    if not _looks_jsonish(text):
        return False
    try:
        json.loads(text)
        return False
    except Exception:
        return True


def _expected_enumerative_count(original: str) -> int | None:
    tail = original[-4000:]
    # Natural-language and numeric count hints used by real extraction prompts, e.g.
    # "return the three codes", "list 3 IDs", "all four endpoints".
    word_nums = {
        "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
        "eight": 8, "nine": 9, "ten": 10,
    }
    m = re.search(r"\b(\d{1,2})\s+(?:codes?|ids?|values?|items?|fields?|keys?|endpoints?|routes?)\b", tail, re.I)
    if m:
        return int(m.group(1))
    m = re.search(r"\b(" + "|".join(word_nums) + r")\s+(?:codes?|ids?|values?|items?|fields?|keys?|endpoints?|routes?)\b", tail, re.I)
    if m:
        return word_nums[m.group(1).lower()]
    # Repeated extraction facts often encode cardinality per-item rather than in the
    # final question, e.g. "access code 1 of 3", "value 2 of 4". Use the largest
    # explicit denominator as the expected answer count.
    denominators = [int(x) for x in re.findall(r"\b(?:code|id|value|item|field|key|endpoint|route)s?\s+\d{1,2}\s+of\s+(\d{1,2})\b", tail, re.I)]
    if denominators:
        return max(denominators)
    return None


def _answer_literal_count(text: str) -> int:
    # Count exact-looking answer literals, de-duplicated. Five-digit codes are common
    # operational IDs; keep generic literals too for paths/keys/quoted values.
    lits = set(LITERAL_RE.findall(text))
    numeric_codes = set(re.findall(r"\b\d{4,}\b", text))
    return len(lits | numeric_codes)


def output_damage_reason(first_pass: str, original: str, usage: Dict[str, Any] | None = None, requested_max_tokens: int | None = None) -> str | None:
    if not first_pass or len(first_pass.strip()) < 8:
        return "empty_or_tiny_output"
    lower = first_pass.lower()
    refusal = re.search(r"\b(cannot|can't|do not have|does not contain|selected original spans|insufficient information|not enough information|unable to)\b", lower)
    if refusal:
        return "refusal_or_insufficient_info"
    usage = usage or {}
    completion_tokens = usage.get("completion_tokens")
    if requested_max_tokens and isinstance(completion_tokens, int) and completion_tokens >= max(1, requested_max_tokens - 1):
        if _requested_json_output(original) or _looks_jsonish(first_pass):
            return "structured_output_hit_max_tokens"
    if _requested_json_output(original) and _json_is_invalid(first_pass):
        return "invalid_json_output"
    expected_count = _expected_enumerative_count(original)
    if expected_count is not None and _answer_literal_count(first_pass) < expected_count:
        return f"enumerative_count_miss_{_answer_literal_count(first_pass)}_of_{expected_count}"
    if re.search(r"\b(all|every|complete|values?|fields?|keys?)\b", original[-2500:], re.I):
        expected_literals = set(LITERAL_RE.findall(original))
        if expected_literals:
            found = sum(1 for lit in expected_literals if lit in first_pass)
            if found < min(3, len(expected_literals)):
                return "enumerative_literal_coverage_miss"
    return None


def output_has_damage_signals(first_pass: str, original: str) -> bool:
    return output_damage_reason(first_pass, original) is not None


def should_run_second_pass(body: Dict[str, Any], original: str, first_pass: str, route: str, usage: Dict[str, Any] | None = None, requested_max_tokens: int | None = None) -> Tuple[bool, str]:
    mode = str(body.get("verify_mode") or body.get("smart_verify_mode") or VERIFY_MODE).lower()
    if mode == "always":
        return True, "verify_mode_always"
    if mode == "never":
        return False, "verify_mode_never"
    damage_reason = output_damage_reason(first_pass, original, usage, requested_max_tokens)
    # Damage/repair checks must run before the short-context fast path; otherwise
    # small exact extraction and structured-output failures look "healthy" and never
    # get repaired.
    if damage_reason:
        return True, f"damage_signals:{damage_reason}"
    if approx_tokens(original) < MIN_VERIFY_TOKENS:
        return False, "short_context_skip_verify"
    if request_looks_normal_generation(original):
        return False, "normal_generation_first_pass_ok"
    return False, "first_pass_ok_semantics_handled_by_keep_ratio"

def ensure_pflash(keep_ratio: float | None = None) -> Tuple[bool, str]:
    global pflash_proc, pflash_keep_ratio
    requested_keep = clamp_keep_ratio(DEFAULT_KEEP_RATIO if keep_ratio is None else keep_ratio)
    st, txt = get_url(PFLASH_HEALTH, timeout=1)
    if st == 200 and not MANAGE_PFLASH_PROCESS:
        pflash_keep_ratio = requested_keep
        return True, f"fullstack_up_keep_request={requested_keep:.2f};runtime_pflash_keep_requires_server_schema"
    if st == 200 and pflash_keep_ratio is not None and abs(pflash_keep_ratio - requested_keep) < 1e-6:
        return True, f"already_running_keep={requested_keep:.2f}"
    if st == 200 and pflash_keep_ratio is None:
        pflash_keep_ratio = requested_keep
        return True, f"already_running_keep_assumed={requested_keep:.2f}"
    if st == 200 and pflash_proc is not None and pflash_proc.poll() is None:
        pflash_proc.terminate()
        try:
            pflash_proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            pflash_proc.kill()
    with pflash_lock:
        st, txt = get_url(PFLASH_HEALTH, timeout=1)
        if st == 200 and pflash_keep_ratio is not None and abs(pflash_keep_ratio - requested_keep) < 1e-6:
            return True, f"already_running_keep={requested_keep:.2f}"
        log_path = LOG_DIR / f"qwen36-smart-pflash-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.log"
        env = os.environ.copy()
        env.update({
            "HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES", "0"),
            "ROCR_VISIBLE_DEVICES": os.environ.get("ROCR_VISIBLE_DEVICES", "0"),
            "DFLASH_FP_ALPHA": os.environ.get("DFLASH_FP_ALPHA", "0.70"),
        })
        required = {
            "QWEN36_DFLASH_SERVER": DFLASH_SERVER,
            "QWEN36_TARGET_MODEL": TARGET_MODEL,
            "QWEN36_DRAFT_MODEL": DRAFT_MODEL,
            "QWEN36_PFLASH_DRAFTER": PFLASH_DRAFTER,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            return False, "manage_pflash_missing_env=" + ",".join(missing)
        workdir = os.environ.get("QWEN36_PFLASH_WORKDIR") or str(Path(DFLASH_SERVER).resolve().parent)
        cmd = [
            DFLASH_SERVER, TARGET_MODEL,
            "--draft", DRAFT_MODEL,
            "--target-device", "hip:0", "--draft-device", "hip:0",
            "--draft-swa", "2048", "--ddtree", "--ddtree-budget", "22",
            "--max-ctx", "32768", "--default-max-tokens", "512",
            "--prefill-compression", "always", "--prefill-keep-ratio", f"{requested_keep:.2f}",
            "--prefill-drafter", PFLASH_DRAFTER,
            "--model-name", UPSTREAM_MODEL,
            "--host", "127.0.0.1", "--port", str(PFLASH_PORT),
        ]
        lf = open(log_path, "ab", buffering=0)
        pflash_proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=workdir)
        deadline = time.time() + 180
        while time.time() < deadline:
            if pflash_proc.poll() is not None:
                return False, f"pflash_exited_{pflash_proc.returncode}_log={log_path}"
            st, txt = get_url(PFLASH_HEALTH, timeout=1)
            if st == 200:
                pflash_keep_ratio = requested_keep
                return True, f"started_pid={pflash_proc.pid}_keep={requested_keep:.2f}_log={log_path}"
            time.sleep(2)
        return False, f"pflash_start_timeout_pid={pflash_proc.pid}_log={log_path}"

def strip_router_fields(body: Dict[str, Any], model: str, *, stream: bool = False, route: str | None = None, policy: Dict[str, Any] | None = None) -> Dict[str, Any]:
    b = dict(body)
    keep_ratio, keep_reason = choose_keep_ratio(body, route or "direct_exact", policy)
    for k in [
        "route", "x_route", "smart_route",
        "pflash_keep_ratio", "keep_ratio", "smart_keep_ratio",
        "verify_mode", "smart_verify_mode",
        "dflash_gear", "smart_dflash_gear", "dflash_n_max", "smart_dflash_n_max",
        "classifier", "smart_classifier",
    ]:
        b.pop(k, None)
    b["model"] = model
    b["stream"] = bool(stream)
    b["pflash.keep_ratio"] = keep_ratio
    b["pflash_keep_ratio"] = keep_ratio
    pflash_obj = b.get("pflash") if isinstance(b.get("pflash"), dict) else {}
    pflash_obj["keep_ratio"] = keep_ratio
    b["pflash"] = pflash_obj
    n_max, why = choose_dflash_gear(body, route or "direct_exact", policy)
    if n_max is not None:
        b["dflash.n_max"] = n_max
        b["speculative.n_max"] = n_max
        dflash_obj = b.get("dflash") if isinstance(b.get("dflash"), dict) else {}
        dflash_obj["n_max"] = n_max
        b["dflash"] = dflash_obj
        speculative_obj = b.get("speculative") if isinstance(b.get("speculative"), dict) else {}
        speculative_obj["n_max"] = n_max
        b["speculative"] = speculative_obj
        b.setdefault("metadata", {})
        if isinstance(b["metadata"], dict):
            b["metadata"]["qwen36_dflash_gear"] = {"n_max": n_max, "reason": why}
            b["metadata"]["qwen36_pflash"] = {"keep_ratio": keep_ratio, "reason": keep_reason}
    return b

def stream_upstream(handler: BaseHTTPRequestHandler, url: str, payload: Dict[str, Any], timeout: int = 600) -> None:
    """Proxy OpenAI SSE chunks from an upstream backend to the client.

    Hermes streams by default, so rejecting stream=true makes the local provider
    look broken and triggers fallback providers.
    """
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        handler.send_response(resp.status)
        handler.send_header("Content-Type", resp.headers.get("Content-Type", "text/event-stream"))
        handler.send_header("Cache-Control", "no-cache")
        handler.end_headers()
        while True:
            chunk = resp.read(8192)
            if not chunk:
                break
            handler.wfile.write(chunk)
            handler.wfile.flush()


def handle_chat_stream(handler: BaseHTTPRequestHandler, body: Dict[str, Any]) -> None:
    regex_route, regex_reason = route_request(body)
    policy = classifier_policy(body, regex_route, regex_reason)
    route = str(policy.get("route", regex_route))
    if route == "direct_exact":
        stream_upstream(handler, f"{DIRECT_BASE}/chat/completions", strip_router_fields(body, UPSTREAM_MODEL, stream=True, route=route, policy=policy))
        return
    keep_ratio, keep_reason = choose_keep_ratio(body, route, policy)
    ok, msg = ensure_pflash(keep_ratio)
    if ok and route == "pflash_broad":
        stream_upstream(handler, f"{PFLASH_BASE}/chat/completions", strip_router_fields(body, UPSTREAM_MODEL, stream=True, route=route, policy=policy))
        return
    # The exactness two-pass route needs non-streaming orchestration. For streaming
    # clients, degrade to direct streaming instead of returning an error/fallbacking.
    stream_upstream(handler, f"{DIRECT_BASE}/chat/completions", strip_router_fields(body, UPSTREAM_MODEL, stream=True, route=route, policy=policy))


def log_decision(meta: Dict[str, Any], body: Dict[str, Any] | None, resp: Dict[str, Any]) -> None:
    try:
        usage = resp.get("usage", {}) if isinstance(resp, dict) else {}
        record = {
            "time": now_iso(),
            "route": meta,
            "model": resp.get("model") if isinstance(resp, dict) else None,
            "usage": usage if isinstance(usage, dict) else {},
        }
        if body is not None:
            text = messages_to_text(body.get("messages", []))
            record["request"] = {
                "approx_prompt_tokens": approx_tokens(text),
                "max_tokens": body.get("max_tokens", body.get("max_completion_tokens")),
                "stream": bool(body.get("stream")),
                "explicit_speculative": {
                    k: body.get(k) for k in ("dflash.n_max", "speculative.n_max", "dflash_n_max", "smart_dflash_n_max", "dflash_gear", "smart_dflash_gear") if k in body
                },
            }
        DECISION_LOG.parent.mkdir(parents=True, exist_ok=True)
        with DECISION_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
    except Exception:
        pass


def annotate(resp: Dict[str, Any], route: str, reason: str, body: Dict[str, Any] | None = None, policy: Dict[str, Any] | None = None) -> Dict[str, Any]:
    resp.setdefault("model", "qwen36-smart")
    meta = {"route": route, "reason": reason, "time": now_iso()}
    if body is not None:
        n_max, dflash_reason = choose_dflash_gear(body, route, policy)
        if n_max is not None:
            meta["dflash"] = {"n_max": n_max, "reason": dflash_reason}
        keep_ratio, keep_reason = choose_keep_ratio(body, route, policy)
        meta["pflash"] = {"keep_ratio": keep_ratio, "reason": keep_reason}
        if policy is not None:
            meta["classifier_ab"] = {k: policy.get(k) for k in ("mode", "active", "active_candidate", "classifier", "classifier_raw", "correction_mode", "regex", "classifier_rejected") if k in policy}
    resp["qwen36_smart_route"] = meta
    log_decision(meta, body, resp)
    return resp


def _query_terms(text: str) -> set[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", text.lower())
    stop = {
        "the", "and", "for", "with", "from", "that", "this", "above", "below", "reply",
        "return", "only", "what", "when", "where", "which", "mentioned", "text", "user",
        "assistant", "system", "please", "your", "have", "has", "are", "all", "every",
    }
    return {w for w in words if w not in stop}


def _salience_score(chunk: str, query_terms: set[str]) -> float:
    lower = chunk.lower()
    score = 0.0
    score += 3.0 * len(STRUCTURE_RE.findall(chunk))
    score += 2.5 * len(IDENT_RE.findall(chunk))
    score += 2.0 * len(LITERAL_RE.findall(chunk))
    score += 3.0 * len(CONSTRAINT_RE.findall(chunk))
    score += 2.0 * len(RELATION_RE.findall(chunk))
    score += 8.0 * sum(1 for t in query_terms if t in lower)
    # Reward uncommon-looking operational tokens, but cap so huge noisy spans do not win
    # purely by length.
    rare = re.findall(r"\b(?:[A-Z]{2,}[A-Z0-9_-]*|[a-z]+[A-Z][A-Za-z0-9]*|[A-Za-z0-9_-]*\d[A-Za-z0-9_-]*)\b", chunk)
    score += min(60.0, 1.5 * len(rare))
    return score


def _chunk_text(text: str, size: int = 1800, overlap: int = 250) -> List[Tuple[int, str]]:
    chunks: List[Tuple[int, str]] = []
    if not text:
        return chunks
    step = max(1, size - overlap)
    for start in range(0, len(text), step):
        chunk = text[start:start + size]
        if chunk:
            chunks.append((start, chunk))
        if start + size >= len(text):
            break
    return chunks


def select_semantic_spans(original: str, first_pass: str, *, budget: int = 24000) -> str:
    # The latest user turn usually contains the actual request; using all terms still
    # works for completions and preserves constraints from system/developer/user text.
    query_terms = _query_terms(original[-6000:]) | _query_terms(first_pass[:2000])
    chunks = _chunk_text(original)
    if not chunks:
        return ""
    ranked = []
    for start, chunk in chunks:
        score = _salience_score(chunk, query_terms)
        # Keep some positional diversity: prompt beginning often has instructions;
        # prompt end often has the actual question/output contract.
        if start < 2500:
            score += 20
        if start + len(chunk) > len(original) - 2500:
            score += 30
        ranked.append((score, start, chunk))
    # Single-literal rescue: exact lookups can be too small to dominate chunk-level
    # salience. If the user/candidate mentions a literal that appears in the prompt,
    # seed a tight window around it before normal chunk ranking.
    rescue: List[Tuple[int, str]] = []
    literal_terms = list(dict.fromkeys(LITERAL_RE.findall(first_pass[:2000]) + LITERAL_RE.findall(original[-4000:])))
    for lit in literal_terms[:12]:
        if not lit or len(lit) > 200:
            continue
        pos = original.find(lit)
        if pos >= 0:
            a = max(0, pos - 900)
            b = min(len(original), pos + len(lit) + 900)
            rescue.append((a, original[a:b]))
    ranked.sort(key=lambda x: (-x[0], x[1]))

    chosen: List[Tuple[int, str]] = []
    used_ranges: List[Tuple[int, int]] = []
    used = 0
    # Select literal-rescue windows first, then high-value chunks with diversity.
    for start, chunk in rescue:
        end = start + len(chunk)
        if any(not (end < a or start > b) for a, b in used_ranges):
            continue
        if used + len(chunk) > budget and chosen:
            continue
        chosen.append((start, chunk))
        used_ranges.append((start, end))
        used += len(chunk)
    for score, start, chunk in ranked:
        if score <= 0 and chosen:
            continue
        end = start + len(chunk)
        if any(not (end < a or start > b) for a, b in used_ranges):
            continue
        if used + len(chunk) > budget and chosen:
            continue
        chosen.append((start, chunk))
        used_ranges.append((start, end))
        used += len(chunk)
        if used >= budget:
            break
    if not chosen:
        chosen = [(0, original[:budget])]
    chosen.sort(key=lambda x: x[0])
    return "\n\n--- SELECTED ORIGINAL SPAN ---\n\n".join(chunk for _, chunk in chosen)[:budget]


def selected_span_prompt(original: str, first_pass: str) -> str:
    # If the prompt is comfortably within the direct model context, give the repair
    # pass the whole original prompt. This fixes short/mid exact extraction misses
    # where the first pass omitted a required value that therefore would not seed a
    # literal-rescue selected span.
    selected = original if approx_tokens(original) < MIN_VERIFY_TOKENS else select_semantic_spans(original, first_pass)
    return (
        "You are the second pass of a smart context router. Answer the user's original request "
        "using the selected ORIGINAL prompt spans below. Treat the first-pass candidate as a hint, "
        "not as authority.\n\n"
        "Preserve exact operational details when relevant: code identifiers, function/class names, "
        "JSON/tool arguments, config keys, paths, URLs, headers, routes, ports, version numbers, IDs, "
        "hashes, quoted strings, numeric literals, constraints, and key/value or mapping relations. "
        "If the request asks for all/every/complete values, return all relevant values you can find, "
        "not just the first. If it asks for valid JSON or code, output that format. If the selected "
        "spans lack required information, say so briefly rather than inventing.\n\n"
        f"First-pass candidate answer:\n{first_pass}\n\nSelected original spans:\n{selected}"
    )


def handle_chat(body: Dict[str, Any]) -> Dict[str, Any]:
    regex_route, regex_reason = route_request(body)
    policy = classifier_policy(body, regex_route, regex_reason)
    route = str(policy.get("route", regex_route))
    reason = str(policy.get("route_reason", regex_reason))
    if route == "direct_exact":
        resp = http_json(f"{DIRECT_BASE}/chat/completions", strip_router_fields(body, UPSTREAM_MODEL, route=route, policy=policy))
        return annotate(resp, route, reason, body, policy)
    keep_ratio, keep_reason = choose_keep_ratio(body, route, policy)
    ok, msg = ensure_pflash(keep_ratio)
    if not ok:
        if route == "pflash_broad":
            raise RuntimeError(msg)
        reason += f";pflash_unavailable={msg};fallback_direct"
        resp = http_json(f"{DIRECT_BASE}/chat/completions", strip_router_fields(body, UPSTREAM_MODEL, route=route, policy=policy))
        return annotate(resp, "direct_exact", reason, body, policy)
    if route == "pflash_broad":
        resp = http_json(f"{PFLASH_BASE}/chat/completions", strip_router_fields(body, UPSTREAM_MODEL, route=route, policy=policy))
        resp = annotate(resp, route, reason + f";keep={keep_ratio:.2f};{keep_reason}", body, policy)
        return resp
    # candidate PFlash first, then optional direct selected-spans verification
    first_body = strip_router_fields(body, UPSTREAM_MODEL, route=route, policy=policy)
    first_requested_max = min(int(first_body.get("max_tokens", 256) or 256), 256)
    first_body["max_tokens"] = first_requested_max
    first = http_json(f"{PFLASH_BASE}/chat/completions", first_body)
    first_text = first.get("choices", [{}])[0].get("message", {}).get("content", "")
    original = messages_to_text(body.get("messages", []))
    do_verify, verify_reason = should_run_second_pass(body, original, first_text, route, first.get("usage"), first_requested_max)
    if not do_verify:
        resp = annotate(first, "pflash_candidate", reason + f";keep={keep_ratio:.2f};{keep_reason};{verify_reason}", body, policy)
        resp["qwen36_smart_route"]["second_pass"] = {"ran": False, "reason": verify_reason, "first_pass_usage": first.get("usage")}
        return resp
    verify_body = strip_router_fields(body, UPSTREAM_MODEL, route=route, policy=policy)
    verify_body["messages"] = [{"role": "user", "content": selected_span_prompt(original, first_text)}]
    requested_verify_max = int(body.get("max_tokens", 256) or 256)
    if "structured_output_hit_max_tokens" in verify_reason or "invalid_json_output" in verify_reason:
        requested_verify_max = max(requested_verify_max, 768)
    verify_body["max_tokens"] = requested_verify_max
    resp = http_json(f"{DIRECT_BASE}/chat/completions", verify_body)
    meta = {"ran": True, "reason": verify_reason, "first_pass_text": first_text[:1000], "first_pass_usage": first.get("usage")}
    resp = annotate(resp, route, reason + f";keep={keep_ratio:.2f};{keep_reason}", body, policy)
    resp["qwen36_smart_route"]["second_pass"] = meta
    return resp


def completion_prompt_to_messages(prompt: Any) -> Tuple[List[Dict[str, str]], Any]:
    """Convert legacy OpenAI /v1/completions prompt into chat messages.

    Returns (messages, prompt_echo). Multiple prompts are joined with separators because
    the live router/upstream path handles one generation per request.
    """
    if isinstance(prompt, list):
        parts = []
        for p in prompt:
            if isinstance(p, str):
                parts.append(p)
            else:
                parts.append(json.dumps(p, ensure_ascii=False))
        text = "\n\n--- PROMPT ---\n\n".join(parts)
        return [{"role": "user", "content": text}], prompt
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}], prompt
    text = json.dumps(prompt, ensure_ascii=False)
    return [{"role": "user", "content": text}], prompt


def handle_completion(body: Dict[str, Any]) -> Dict[str, Any]:
    prompt = body.get("prompt", "")
    messages, prompt_echo = completion_prompt_to_messages(prompt)
    chat_body = dict(body)
    chat_body.pop("prompt", None)
    chat_body["messages"] = messages
    chat_resp = handle_chat(chat_body)
    chat_choice = chat_resp.get("choices", [{}])[0]
    message = chat_choice.get("message", {}) if isinstance(chat_choice, dict) else {}
    text = message.get("content", "") if isinstance(message, dict) else ""
    finish_reason = chat_choice.get("finish_reason") if isinstance(chat_choice, dict) else None
    resp = {
        "id": chat_resp.get("id", f"cmpl-qwen36-smart-{int(time.time())}"),
        "object": "text_completion",
        "created": chat_resp.get("created", int(time.time())),
        "model": chat_resp.get("model", "qwen36-smart"),
        "choices": [{
            "text": text,
            "index": 0,
            "logprobs": None,
            "finish_reason": finish_reason,
        }],
        "usage": chat_resp.get("usage", {}),
        "qwen36_smart_route": chat_resp.get("qwen36_smart_route", {}),
    }
    if body.get("echo"):
        resp["choices"][0]["text"] = str(prompt_echo) + text
    return resp


class Handler(BaseHTTPRequestHandler):
    server_version = "qwen36-smart-router/0.2"
    def _send(self, code: int, obj: Dict[str, Any]) -> None:
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
    def do_GET(self) -> None:
        if self.path in {"/health", "/"}:
            ds, dt = get_url(DIRECT_HEALTH, timeout=1)
            ps, pt = get_url(PFLASH_HEALTH, timeout=1)
            self._send(200, {"status": "ok", "model": "qwen36-smart", "direct": ds == 200, "pflash": ps == 200, "classifier_mode": CLASSIFIER_MODE, "classifier_configured": bool(CLASSIFIER_BASE), "time": now_iso()})
        elif self.path == "/v1/models":
            self._send(200, {"object": "list", "data": [{"id": "qwen36-smart", "object": "model", "created": int(time.time()), "owned_by": "local"}]})
        elif self.path == "/metrics":
            # A local monitor may poll /metrics. Return a small Prometheus-compatible
            # payload instead of noisy 404s; detailed token metrics are not exposed by
            # this shim yet.
            ds, _ = get_url(DIRECT_HEALTH, timeout=1)
            ps, _ = get_url(PFLASH_HEALTH, timeout=1)
            body = (
                "# HELP qwen36_smart_router_up Router process is serving requests.\n"
                "# TYPE qwen36_smart_router_up gauge\n"
                "qwen36_smart_router_up 1\n"
                "# HELP qwen36_smart_backend_up Direct upstream backend health.\n"
                "# TYPE qwen36_smart_backend_up gauge\n"
                f"qwen36_smart_backend_up {1 if ds == 200 else 0}\n"
                "# HELP qwen36_smart_pflash_up PFlash/full-stack lane health.\n"
                "# TYPE qwen36_smart_pflash_up gauge\n"
                f"qwen36_smart_pflash_up {1 if ps == 200 else 0}\n"
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self._send(404, {"error": "not found"})
    def do_POST(self) -> None:
        try:
            n = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(n).decode("utf-8"))
            if self.path == "/v1/chat/completions":
                if body.get("stream"):
                    handle_chat_stream(self, body); return
                self._send(200, handle_chat(body)); return
            if self.path == "/v1/completions":
                if body.get("stream"):
                    self._send(400, {"error": "legacy /v1/completions streaming is not supported by qwen36-smart; use /v1/chat/completions streaming"}); return
                self._send(200, handle_completion(body)); return
            self._send(404, {"error": "not found"})
        except urllib.error.HTTPError as e:
            txt = e.read().decode("utf-8", errors="replace")
            self._send(e.code, {"error": txt, "type": "upstream_http_error"})
        except Exception as e:
            self._send(500, {"error": repr(e), "type": "router_error"})
    def log_message(self, fmt: str, *args: Any) -> None:
        line = f"{now_iso()} {self.address_string()} {fmt % args}\n"
        with open(LOG_DIR / "qwen36-smart-router.access.log", "a", encoding="utf-8") as f:
            f.write(line)


def main() -> None:
    httpd = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"qwen36-smart-router listening on {HOST}:{PORT}", flush=True)
    httpd.serve_forever()

if __name__ == "__main__":
    main()
