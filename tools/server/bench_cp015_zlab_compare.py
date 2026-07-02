#!/usr/bin/env python3
import json
import os
import re
import signal
import subprocess
import time
import urllib.request
from pathlib import Path
from datetime import datetime, timezone

REPO = Path('${LLAMA_CPP_DIR}')
RUN = REPO / ('checkpoint-015-zlab-dflash-bench-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
LOGDIR = RUN / 'logs'
OUTDIR = RUN / 'outputs'
LOGDIR.mkdir(parents=True, exist_ok=True)
OUTDIR.mkdir(parents=True, exist_ok=True)
SUMMARY = RUN / 'summary.jsonl'
COMPARE = RUN / 'comparison.json'
PORT = int(os.environ.get('BENCH_PORT', '18120'))
HOST = '127.0.0.1'
URL_CHAT = f'http://{HOST}:{PORT}/v1/chat/completions'
URL_HEALTH = f'http://{HOST}:{PORT}/health'
CTX = int(os.environ.get('BENCH_CTX', '8192'))
REQ_TIMEOUT = int(os.environ.get('BENCH_REQ_TIMEOUT', '1800'))

BIN = str(REPO / 'build-hip-gfx1151/bin/llama-server')
TARGET = '${EXPERIMENTS_DIR}/lucebox-campaign/models/target-unsloth/Qwen3.6-27B-Q4_K_M.gguf'
DRAFT = '${MODEL_DIR}/z-lab-Qwen3.6-27B-DFlash/qwen3.6-27b-dflash-zlab-q8_0.gguf'
MATRIX = Path('${INFERENCE_WORKSPACE}/ops/lucebox-kvflash-hermes-test-20260628/benchsplit-six-matrix-20260629T010545Z/summary.jsonl')
UPSTREAM_ZLAB = Path('${INFERENCE_WORKSPACE}/ops/lucebox-kvflash-hermes-test-20260628/benchsplit-six-upstream-dflash-zlab-q8-20260629T025201Z/summary.jsonl')
UPSTREAM_RERUN = Path('${INFERENCE_WORKSPACE}/ops/lucebox-kvflash-hermes-test-20260628/benchsplit-six-upstream-rerun-20260629T015345Z/summary.jsonl')


def check_list(content):
    langs = ['python','javascript','java','c++','c#','typescript','swift','go','ruby','php','kotlin','rust','html']
    found = sum(1 for l in langs if l in content.lower())
    return (found >= 5, min(1.0, found/10), f'{found}/10 known langs found')


def check_code(content):
    has_def = 'def ' in content and 'fizz' in content.lower()
    has_15 = '15' in content or ('3' in content and '5' in content and '%' in content)
    has_strs = 'fizzbuzz' in content.lower()
    score = (int(has_def) + int(has_15) + int(has_strs)) / 3
    return (score >= 0.66, score, f'def={has_def} mod15={has_15} fizzbuzz_str={has_strs}')


def check_json(content):
    c = re.sub(r'^```json?\s*|\s*```$', '', content.strip(), flags=re.MULTILINE)
    try:
        obj = json.loads(c)
        if not isinstance(obj, list):
            return (False, 0.2, 'not a list')
        valid = sum(1 for e in obj if isinstance(e, dict) and 'name' in e and 'year' in e)
        return (valid >= 8, min(1.0, valid/10), f'{valid}/10 valid entries, total={len(obj)}')
    except Exception as e:
        return (False, 0.0, f'JSON parse fail: {str(e)[:50]}')


def check_reasoning(content):
    bits = ['60','40','300','mph','mile']
    found = sum(1 for b in bits if b in content.lower())
    has_time = bool(re.search(r'\b1[0-9]:\d\d|\b1[0-9]\.\d+\s*(am|pm|hour)', content.lower()))
    has_pos = bool(re.search(r'\b\d{2,3}\s*mile', content.lower()))
    score = (found/5)*0.7 + 0.15*has_time + 0.15*has_pos
    return (score >= 0.5, score, f'hit {found}/5 keywords, has_time={has_time} has_pos={has_pos}')


def check_tool(content):
    c = re.sub(r'^```json?\s*|\s*```$', '', content.strip(), flags=re.MULTILINE)
    try:
        obj = json.loads(c)
        has_tool = 'tool' in obj and obj['tool'] == 'get_weather'
        has_args = 'args' in obj and 'Tokyo' in str(obj['args'])
        has_reason = 'reason' in obj
        score = (int(has_tool) + int(has_args) + int(has_reason)) / 3
        return (score >= 0.66, score, f"tool={obj.get('tool')} args_ok={has_args} reason={has_reason}")
    except Exception as e:
        return (False, 0.0, f'JSON parse fail: {str(e)[:50]}')


def check_essay(content):
    words = len(content.split())
    keywords = ['draft','speculat','ngram','n-gram','token','accept','reject','verif']
    hits = sum(1 for k in keywords if k in content.lower())
    score = min(1.0, words/600)*0.5 + min(1.0, hits/5)*0.5
    return (score >= 0.5, score, f'{words} words, {hits}/8 keywords')


TASKS = [
    ('list','speed-short-list','List the top 10 programming languages by popularity. Just the list, no commentary.',400,check_list),
    ('code','speed-code','Write a Python function called fizzbuzz that prints 1 to 100 with FizzBuzz logic.',400,check_code),
    ('json','speed-structured-json','Return a JSON array of 10 popular programming languages with fields: name, year, paradigm. Output ONLY valid JSON, nothing else.',600,check_json),
    ('reason','reasoning-trains','Train A leaves station X at 9:00 AM going east at 60 mph. Train B leaves station Y, 300 miles east of X, at 10:00 AM going west at 40 mph. When and where do they meet? Show your work step by step.',1200,check_reasoning),
    ('tool','agent-tool-decision','You have tools: get_weather(city), search_web(query), send_email(to,body). User asks: \'What\'s the weather in Tokyo?\' Reply with valid JSON: {"tool": ..., "args": ..., "reason": ...}. Output ONLY the JSON.',200,check_tool),
    ('essay','long-gen-essay','Write a 600-word technical explainer of speculative decoding for a developer audience. Cover ngram, draft-model, and MTP variants. Be concrete.',1500,check_essay),
]


def load_rows(path):
    rows = []
    if path.exists():
        for line in path.read_text(errors='replace').splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if row.get('passed') is not None:
                rows.append(row)
    return rows


def aggregate(rows, config=None):
    if config is not None:
        rows = [r for r in rows if r.get('config') == config]
    passed = sum(1 for r in rows if r.get('passed'))
    tokens = sum(int(r.get('tokens') or r.get('usage', {}).get('completion_tokens') or 0) for r in rows)
    wall = sum(float(r.get('wall_s') or 0) for r in rows)
    drafted = sum(int(r.get('metrics', {}).get('drafted') or 0) for r in rows)
    accepted = sum(int(r.get('metrics', {}).get('accepted') or 0) for r in rows)
    return {
        'rows': len(rows),
        'passed': passed,
        'tokens': tokens,
        'wall_s': round(wall, 3),
        'weighted_tps': round(tokens / wall, 3) if wall else 0,
        'accepted': accepted,
        'drafted': drafted,
        'acceptance': round(accepted / drafted, 4) if drafted else 0,
    }


def build_comparison(current_rows):
    matrix_rows = load_rows(MATRIX)
    up_zlab_rows = load_rows(UPSTREAM_ZLAB)
    up_rows = load_rows(UPSTREAM_RERUN)
    anchors = {
        'cp015_current_fork_zlab_dflash_kvflash_cfg': aggregate(current_rows),
        'upstream_dflash_zlab_q8_reference': aggregate(up_zlab_rows, 'upstream_dflash_zlab_q8'),
        'upstream_no_spec_reference': aggregate(up_rows, 'upstream_no_spec'),
        'upstream_mtp_reference': aggregate(up_rows, 'upstream_mtp'),
        'matrix_lucebox_fast_recipe_prefix32': aggregate(matrix_rows, 'lucebox_fast_recipe_prefix32'),
        'matrix_lucebox_fast_recipe_prefix0': aggregate(matrix_rows, 'lucebox_fast_recipe_prefix0'),
        'matrix_lucebox_swa_ddtree': aggregate(matrix_rows, 'lucebox_swa_ddtree'),
        'matrix_lucebox_dflash_ddtree_on': aggregate(matrix_rows, 'lucebox_dflash_ddtree_on'),
        'matrix_lucebox_dflash_ddtree_off': aggregate(matrix_rows, 'lucebox_dflash_ddtree_off'),
        'matrix_lucebox_no_dflash_no_ddtree': aggregate(matrix_rows, 'lucebox_no_dflash_no_ddtree'),
    }
    cur = anchors['cp015_current_fork_zlab_dflash_kvflash_cfg']['weighted_tps']
    deltas = {}
    for k, v in anchors.items():
        base = v['weighted_tps']
        if k != 'cp015_current_fork_zlab_dflash_kvflash_cfg' and base:
            deltas[k] = round((cur / base - 1) * 100, 2)
    doc = {
        'run_dir': str(RUN),
        'summary': str(SUMMARY),
        'target': TARGET,
        'draft': DRAFT,
        'ctx': CTX,
        'anchors': anchors,
        'deltas_pct_current_vs_anchor': deltas,
        'source_files': [str(MATRIX), str(UPSTREAM_RERUN), str(UPSTREAM_ZLAB)],
    }
    COMPARE.write_text(json.dumps(doc, indent=2) + '\n')
    return doc


def port_listening():
    return subprocess.run(['bash','-lc',f"ss -ltn | grep -q ':{PORT} '"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def wait_ready(timeout=900):
    start = time.time()
    last = ''
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(URL_HEALTH, timeout=2) as r:
                txt = r.read().decode('utf-8', 'replace')
                if r.status == 200:
                    return True, txt
                last = txt
        except Exception as e:
            last = repr(e)
        time.sleep(2)
    return False, last


def post(obj, timeout):
    data = json.dumps(obj).encode()
    req = urllib.request.Request(URL_CHAT, data=data, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read().decode('utf-8', 'replace')


def extract(raw):
    try:
        j = json.loads(raw)
        msg = j.get('choices', [{}])[0].get('message', {})
        return (msg.get('content') or msg.get('reasoning_content') or '', j)
    except Exception:
        return raw, None


def parse_metrics(log_path, before):
    data = log_path.read_text(errors='replace')[before:] if log_path.exists() else ''
    metrics = {}
    m = list(re.finditer(r'\[spec-decode\].*?tokens=(\d+).*?time=([0-9.]+) s speed=([0-9.]+) tok/s.*?accepted=([0-9]+)/([0-9]+)', data))
    if m:
        x = m[-1]
        metrics.update({'spec_tokens': int(x.group(1)), 'spec_time_s': float(x.group(2)), 'eval_tps': float(x.group(3)), 'accepted': int(x.group(4)), 'drafted': int(x.group(5))})
    for k, p in {'prompt_ms': r'prompt eval time\s*=\s*([0-9.]+) ms', 'eval_ms': r'eval time\s*=\s*([0-9.]+) ms', 'total_ms': r'total time\s*=\s*([0-9.]+) ms'}.items():
        ms = list(re.finditer(p, data, re.S))
        if ms:
            try:
                metrics[k] = float(ms[-1].group(1))
            except Exception:
                pass
    return metrics


def stop_proc(proc):
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except Exception:
        proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            proc.kill()
        proc.wait(timeout=30)


cmd = [
    BIN, '-m', TARGET, '-md', DRAFT,
    '--host', HOST, '--port', str(PORT),
    '--ctx-size', str(CTX), '--predict', '1500',
    '--no-mmap', '--jinja',
    '--cache-type-k', 'q8_0', '--cache-type-v', 'q8_0',
    '--spec-type', 'draft-dflash', '--spec-draft-n-max', '15',
    '--alias', 'bench',
    '--kvflash', '1024', '--kvflash-policy', 'qk', '--kvflash-tau', '64',
]

env = os.environ.copy()
env['LD_LIBRARY_PATH'] = str(REPO / 'build-hip-gfx1151/bin') + ':/opt/rocm/lib:' + env.get('LD_LIBRARY_PATH', '')

print(f'RUN_DIR={RUN}', flush=True)
print(f'SUMMARY={SUMMARY}', flush=True)
print('TARGET=' + TARGET, flush=True)
print('DRAFT=' + DRAFT, flush=True)
print('CMD ' + ' '.join(cmd), flush=True)

if port_listening():
    raise SystemExit(f'port {PORT} already listening')

log_path = LOGDIR / 'cp015_current_fork_zlab_dflash_kvflash_cfg.server.log'
logf = open(log_path, 'wb')
proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env, preexec_fn=os.setsid)
rows = []
try:
    ok, health = wait_ready()
    print(f'READY ok={ok} health={health[:200]!r}', flush=True)
    if not ok:
        SUMMARY.open('a').write(json.dumps({'config': 'cp015_current_fork_zlab_dflash_kvflash_cfg', 'error': 'server_not_ready', 'health': health, 'log': str(log_path)}) + '\n')
        raise SystemExit(2)
    for cat, task, prompt, max_tok, qfn in TASKS:
        before = log_path.stat().st_size if log_path.exists() else 0
        t0 = time.time()
        status = None
        raw = ''
        error = None
        req = {'model': 'bench', 'messages': [{'role': 'user', 'content': prompt}], 'max_tokens': max_tok, 'temperature': 0.0, 'chat_template_kwargs': {'enable_thinking': False}}
        try:
            status, raw = post(req, REQ_TIMEOUT)
        except Exception as e:
            error = repr(e)
        wall = time.time() - t0
        text, parsed = extract(raw)
        passed = False
        qscore = 0.0
        qnote = 'no text'
        if text:
            passed, qscore, qnote = qfn(text)
        safe = f'cp015_current_fork_zlab_dflash_kvflash_cfg_{task}'
        (OUTDIR / f'{safe}.txt').write_text(text, encoding='utf-8')
        (OUTDIR / f'{safe}.raw.json').write_text(raw, encoding='utf-8')
        usage = parsed.get('usage', {}) if isinstance(parsed, dict) else {}
        out_tok = int(usage.get('completion_tokens') or 0)
        rec = {
            'config': 'cp015_current_fork_zlab_dflash_kvflash_cfg',
            'cat': cat,
            'task': task,
            'status': status,
            'error': error,
            'wall_s': round(wall, 3),
            'tokens': out_tok,
            'tps': round(out_tok / wall, 1) if wall > 0 else 0,
            'qscore': round(qscore, 3),
            'passed': passed,
            'qnote': qnote,
            'usage': usage,
            'metrics': parse_metrics(log_path, before),
            'output': str(OUTDIR / f'{safe}.txt'),
            'raw': str(OUTDIR / f'{safe}.raw.json'),
            'log': str(log_path),
            'preview': text[:200].replace('\n', ' '),
        }
        rows.append(rec)
        SUMMARY.open('a').write(json.dumps(rec) + '\n')
        print('RESULT ' + json.dumps({k: rec[k] for k in ['task','status','error','wall_s','tokens','tps','passed','qnote','metrics']}), flush=True)
        time.sleep(1)
    comp = build_comparison(rows)
    print('COMPARISON ' + json.dumps(comp['anchors']['cp015_current_fork_zlab_dflash_kvflash_cfg']), flush=True)
    print('COMPARE=' + str(COMPARE), flush=True)
finally:
    print('STOP server', flush=True)
    stop_proc(proc)
    logf.close()
    time.sleep(2)
    if port_listening():
        subprocess.run(['bash','-lc',f'fuser -k {PORT}/tcp || true'])
print(f'DONE RUN_DIR={RUN}', flush=True)
