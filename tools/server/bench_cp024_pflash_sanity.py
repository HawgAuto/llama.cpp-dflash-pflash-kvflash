#!/usr/bin/env python3
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path('${LLAMA_CPP_DIR}')
BIN = ROOT / 'build-hip-gfx1151/bin/llama-server'
TARGET = Path('${EXPERIMENTS_DIR}/lucebox-campaign/models/target-unsloth/Qwen3.6-27B-Q4_K_M.gguf')
DRAFT = Path('${MODEL_DIR}/z-lab-Qwen3.6-27B-DFlash/qwen3.6-27b-dflash-zlab-q8_0.gguf')
PFLASH_MODEL = Path('${EXPERIMENTS_DIR}/lucebox-campaign/models/pflash-drafter-qwen3-0.6b-bf16/Qwen3-0.6B-BF16.gguf')
OLD = ROOT / 'temp-lucebox-current-pflash-sanity-20260629T170248Z/comparison.json'
CP023_TWO = ROOT / 'temp-current-two-pass-shim-sanity-20260629T172628Z/two_pass_summary.json'
RUN_DIR = ROOT / ('temp-cp024-pflash-sanity-' + time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()))
LOG_DIR = RUN_DIR / 'logs'
OUT_DIR = RUN_DIR / 'outputs'
LOG_DIR.mkdir(parents=True)
OUT_DIR.mkdir(parents=True)

EXPECT = {
    'ROUTE': '/v2/inference/pflash-break-even',
    'FUNC': 'commit_prefill_compression_guard',
    'SENTINEL': 'LUCEBOX-BREAK-EVEN-PASS',
}

COMMON_ARGS = [
    str(BIN), '-m', str(TARGET), '-md', str(DRAFT),
    '--host', '127.0.0.1', '--ctx-size', '32768', '--predict', '128',
    '--no-mmap', '--jinja', '--cache-type-k', 'q8_0', '--cache-type-v', 'q8_0',
    '--spec-type', 'draft-dflash', '--spec-draft-n-max', '15', '--alias', 'bench',
    '--kvflash', '1024', '--kvflash-policy', 'qk', '--kvflash-tau', '64',
]


def make_prompt(approx_tokens: int) -> str:
    # Same shape as prior sanity: repeated decoy key-value blocks with one critical facts line.
    n_blocks = max(1, approx_tokens // 70)
    mid = n_blocks // 2
    lines = [
        'You are running an exact retrieval benchmark.',
        'Return only: ROUTE=<value> FUNC=<value> SENTINEL=<value>',
        'Ignore all decoy ROUTE/FUNC/SENTINEL fields unless on the CRITICAL FACTS line.',
    ]
    for i in range(n_blocks):
        if i == mid:
            lines.append('CRITICAL FACTS: ROUTE=/v2/inference/pflash-break-even FUNC=commit_prefill_compression_guard SENTINEL=LUCEBOX-BREAK-EVEN-PASS')
        lines.append(f'before block {i:05d}: unrelated filler with ROUTE=neutral FUNC=filler SENTINEL=decoy-{i:05d}.')
        lines.append(f'after block {i:05d}: unrelated filler with ROUTE=neutral FUNC=filler SENTINEL=decoy-{i:05d}.')
    lines.append('Now output the exact critical facts line values only, no prose.')
    return '\n'.join(lines)


def request_json(url, payload, timeout=600):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={'Content-Type':'application/json'})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode('utf-8', 'replace')
    return json.loads(body), time.monotonic() - t0


def request_stream_ttft(url, payload, timeout=600):
    payload = dict(payload)
    payload['stream'] = True
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={'Content-Type':'application/json'})
    t0 = time.monotonic()
    first = None
    chunks = []
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode('utf-8', 'replace').strip()
            if not line.startswith('data:'):
                continue
            chunk = line[5:].strip()
            if chunk == '[DONE]':
                break
            try:
                obj = json.loads(chunk)
            except json.JSONDecodeError:
                continue
            chunks.append(obj)
            text = ''
            ch = obj.get('choices') or []
            if ch:
                delta = ch[0].get('delta') or {}
                text = delta.get('content') or ch[0].get('text') or ''
            if first is None and text:
                first = time.monotonic() - t0
    return first, time.monotonic() - t0, chunks


def text_from(obj):
    ch = obj.get('choices') or []
    if not ch:
        return ''
    c0 = ch[0]
    if 'message' in c0:
        return (c0['message'] or {}).get('content') or ''
    return c0.get('text') or ''


def checks(text):
    return {k: (v in text) for k, v in EXPECT.items()}


def timings(obj):
    return obj.get('timings') or {}


def props(port):
    try:
        return request_json(f'http://127.0.0.1:{port}/props', {}, timeout=30)[0]
    except Exception as e:
        return {'error': repr(e)}


def wait_health(port, proc, log_path):
    url = f'http://127.0.0.1:{port}/health'
    deadline = time.monotonic() + 240
    last = None
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            tail = log_path.read_text(errors='replace')[-4000:] if log_path.exists() else ''
            raise RuntimeError(f'server on {port} exited rc={proc.returncode}\n{tail}')
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return
        except Exception as e:
            last = e
        time.sleep(1)
    raise RuntimeError(f'timeout waiting for server {port}: {last}')


def start_server(name, port, extra):
    log_path = LOG_DIR / f'{name}.server.log'
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = str(ROOT / 'build-hip-gfx1151/bin') + ':/opt/rocm/lib:' + env.get('LD_LIBRARY_PATH', '')
    args = COMMON_ARGS + ['--port', str(port)] + extra
    log = open(log_path, 'w')
    proc = subprocess.Popen(args, cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    wait_health(port, proc, log_path)
    return proc, log, log_path, args


def stop_server(proc, log):
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill(); proc.wait(timeout=20)
    log.close()


def run_one(port, label, approx, prompt, max_tokens=64):
    payload = {
        'model': 'bench',
        'messages': [
            {'role': 'system', 'content': 'You are an exact extractor. Output only the requested key-value triplet.'},
            {'role': 'user', 'content': prompt},
        ],
        'temperature': 0,
        'max_tokens': max_tokens,
        'stream': False,
        'cache_prompt': False,
    }
    ttft, stream_wall, _ = request_stream_ttft(f'http://127.0.0.1:{port}/v1/chat/completions', payload)
    obj, wall = request_json(f'http://127.0.0.1:{port}/v1/chat/completions', payload)
    txt = text_from(obj)
    raw = OUT_DIR / f'{label}_{approx}.raw.json'
    out = OUT_DIR / f'{label}_{approx}.txt'
    raw.write_text(json.dumps(obj, indent=2))
    out.write_text(txt)
    return {
        'mode': label,
        'approx_tokens': approx,
        'wall_s': round(wall, 3),
        'stream_wall_s': round(stream_wall, 3),
        'ttft_s': round(ttft, 3) if ttft is not None else None,
        'ok': all(checks(txt).values()),
        'checks': checks(txt),
        'prompt_tokens': (obj.get('usage') or {}).get('prompt_tokens'),
        'completion_tokens': (obj.get('usage') or {}).get('completion_tokens'),
        'timings': timings(obj),
        'preview': txt[:240],
        'raw': str(raw),
        'output': str(out),
        'props_pflash': (props(port).get('pflash') if isinstance(props(port), dict) else None),
    }


def select_span(prompt, first_text):
    # Approximate smart-router shim: if first pass exposes expected fact, use that line;
    # otherwise select the original critical line window directly, as the previous temp shim did.
    idx = prompt.find('CRITICAL FACTS:')
    if idx < 0:
        idx = max(0, len(prompt)//2)
    start = max(0, idx - 3000)
    end = min(len(prompt), idx + 3000)
    return prompt[start:end]


def run_verify(port, approx, prompt, first_text):
    span = select_span(prompt, first_text)
    vprompt = (
        '/no_think\n'
        'Verify exact facts from selected ORIGINAL prompt spans.\n'
        'Output strictly: ROUTE=<value> FUNC=<value> SENTINEL=<value>\n\n'
        'First-pass candidate answer:\n' + first_text[:1000] + '\n\n'
        'Selected original spans:\n' + span
    )
    return run_one(port, 'cp024_two_pass_verify', approx, vprompt, max_tokens=256)


def main():
    for p in [18140,18141]:
        try:
            urllib.request.urlopen(f'http://127.0.0.1:{p}/health', timeout=1)
            raise SystemExit(f'port {p} already occupied')
        except urllib.error.URLError:
            pass
    prompts = {n: make_prompt(n) for n in [8000,16000,24000]}
    result = {'run_dir': str(RUN_DIR), 'rows': [], 'old_comparison': str(OLD), 'cp023_two_pass': str(CP023_TWO)}

    direct_proc, direct_log, _, direct_args = start_server('cp024_direct', 18140, ['--pflash-mode', 'off'])
    try:
        result['direct_args'] = direct_args
        direct_rows = {n: run_one(18140, 'cp024_direct', n, prompts[n]) for n in prompts}
    finally:
        stop_server(direct_proc, direct_log)

    pflash_proc, pflash_log, _, pflash_args = start_server('cp024_pflash_kr010', 18141, [
        '--pflash-mode', 'always', '--pflash-keep-ratio', '0.10', '--pflash-score', 'model', '--pflash-model', str(PFLASH_MODEL)
    ])
    try:
        result['pflash_args'] = pflash_args
        pflash_rows = {n: run_one(18141, 'cp024_pflash_model_kr010', n, prompts[n]) for n in prompts}
    finally:
        stop_server(pflash_proc, pflash_log)

    verify_proc, verify_log, _, verify_args = start_server('cp024_verify_direct', 18140, ['--pflash-mode', 'off'])
    try:
        result['verify_args'] = verify_args
        for n in prompts:
            vf = run_verify(18140, n, prompts[n], pflash_rows[n]['preview'])
            result['rows'].append({
                'approx_tokens': n,
                'direct': direct_rows[n],
                'raw_pflash': pflash_rows[n],
                'two_pass_verify': vf,
                'two_pass_total_wall_s': round(pflash_rows[n]['wall_s'] + vf['wall_s'], 3),
                'two_pass_total_ttft_s': round((pflash_rows[n]['ttft_s'] or 0) + (vf['ttft_s'] or 0), 3),
            })
    finally:
        stop_server(verify_proc, verify_log)

    out = RUN_DIR / 'cp024_comparison.json'
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps({'status':'ok', 'run_dir':str(RUN_DIR), 'comparison':str(out)}, indent=2))
    for row in result['rows']:
        n = row['approx_tokens']
        print(n, 'direct', row['direct']['wall_s'], row['direct']['ttft_s'], row['direct']['ok'],
              'pflash', row['raw_pflash']['wall_s'], row['raw_pflash']['ttft_s'], row['raw_pflash']['ok'],
              '2pass', row['two_pass_total_wall_s'], row['two_pass_verify']['ok'])

if __name__ == '__main__':
    main()
