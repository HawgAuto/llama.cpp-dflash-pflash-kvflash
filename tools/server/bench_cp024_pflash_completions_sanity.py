#!/usr/bin/env python3
import json, os, subprocess, time, urllib.request, urllib.error
from pathlib import Path
ROOT=Path('${LLAMA_CPP_DIR}')
BIN=ROOT/'build-hip-gfx1151/bin/llama-server'
TARGET=Path('${EXPERIMENTS_DIR}/lucebox-campaign/models/target-unsloth/Qwen3.6-27B-Q4_K_M.gguf')
DRAFT=Path('${MODEL_DIR}/z-lab-Qwen3.6-27B-DFlash/qwen3.6-27b-dflash-zlab-q8_0.gguf')
PFLASH_MODEL=Path('${EXPERIMENTS_DIR}/lucebox-campaign/models/pflash-drafter-qwen3-0.6b-bf16/Qwen3-0.6B-BF16.gguf')
RUN_DIR=ROOT/('temp-cp024-completions-sanity-'+time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()))
LOG_DIR=RUN_DIR/'logs'; OUT_DIR=RUN_DIR/'outputs'; LOG_DIR.mkdir(parents=True); OUT_DIR.mkdir(parents=True)
EXPECT={'ROUTE':'/v2/inference/pflash-break-even','FUNC':'commit_prefill_compression_guard','SENTINEL':'LUCEBOX-BREAK-EVEN-PASS'}
COMMON=[str(BIN),'-m',str(TARGET),'-md',str(DRAFT),'--host','127.0.0.1','--ctx-size','32768','--predict','128','--no-mmap','--jinja','--cache-type-k','q8_0','--cache-type-v','q8_0','--spec-type','draft-dflash','--spec-draft-n-max','15','--alias','bench','--kvflash','1024','--kvflash-policy','qk','--kvflash-tau','64']

def prompt(n):
    blocks=max(1,n//70); mid=blocks//2
    lines=['/no_think','You are running an exact retrieval benchmark.','Return only: ROUTE=<value> FUNC=<value> SENTINEL=<value>','Ignore all decoy ROUTE/FUNC/SENTINEL fields unless on the CRITICAL FACTS line.']
    for i in range(blocks):
        if i==mid: lines.append('CRITICAL FACTS: ROUTE=/v2/inference/pflash-break-even FUNC=commit_prefill_compression_guard SENTINEL=LUCEBOX-BREAK-EVEN-PASS')
        lines.append(f'before block {i:05d}: unrelated filler with ROUTE=neutral FUNC=filler SENTINEL=decoy-{i:05d}.')
        lines.append(f'after block {i:05d}: unrelated filler with ROUTE=neutral FUNC=filler SENTINEL=decoy-{i:05d}.')
    lines.append('Now output the exact critical facts line values only, no prose.')
    return '\n'.join(lines)

def http_json(url,payload,timeout=600):
    req=urllib.request.Request(url,data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    t0=time.monotonic()
    with urllib.request.urlopen(req,timeout=timeout) as r: body=r.read().decode('utf-8','replace')
    return json.loads(body), time.monotonic()-t0

def wait(port,proc,log_path):
    deadline=time.monotonic()+240
    while time.monotonic()<deadline:
        if proc.poll() is not None:
            tail=log_path.read_text(errors='replace')[-4000:] if log_path.exists() else ''
            raise RuntimeError(f'exit {proc.returncode} {tail}')
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=2) as r:
                if r.status==200: return
        except Exception: time.sleep(1)
    raise RuntimeError('health timeout')

def start(name,port,extra):
    log_path=LOG_DIR/f'{name}.server.log'; log=open(log_path,'w')
    env=os.environ.copy(); env['LD_LIBRARY_PATH']=str(ROOT/'build-hip-gfx1151/bin')+':/opt/rocm/lib:'+env.get('LD_LIBRARY_PATH','')
    args=COMMON+['--port',str(port)]+extra
    proc=subprocess.Popen(args,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,text=True)
    wait(port,proc,log_path); return proc,log,args

def stop(proc,log):
    if proc.poll() is None:
        proc.terminate()
        try: proc.wait(20)
        except subprocess.TimeoutExpired: proc.kill(); proc.wait(20)
    log.close()

def text(obj):
    c=(obj.get('choices') or [{}])[0]
    return c.get('text') or (c.get('message') or {}).get('content') or (c.get('message') or {}).get('reasoning_content') or ''

def chk(s): return {k:v in s for k,v in EXPECT.items()}

def one(port,label,n,p,max_tokens=96):
    payload={'model':'bench','prompt':p,'temperature':0,'max_tokens':max_tokens,'stream':False,'cache_prompt':False}
    obj,wall=http_json(f'http://127.0.0.1:{port}/v1/completions',payload)
    s=text(obj); (OUT_DIR/f'{label}_{n}.raw.json').write_text(json.dumps(obj,indent=2)); (OUT_DIR/f'{label}_{n}.txt').write_text(s)
    t=obj.get('timings') or {}; c=chk(s)
    return {'mode':label,'approx_tokens':n,'wall_s':round(wall,3),'ttft_s_est':round((t.get('prompt_ms') or 0)/1000,3) if t else None,'ok':all(c.values()),'checks':c,'prompt_tokens':(obj.get('usage') or {}).get('prompt_tokens'),'completion_tokens':(obj.get('usage') or {}).get('completion_tokens'),'timings':t,'preview':s[:300]}

def span(p):
    i=p.find('CRITICAL FACTS:'); return p[max(0,i-3000):min(len(p),i+3000)] if i>=0 else p[:6000]

def main():
    for port in (18142,18143):
        try: urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=1); raise SystemExit(f'port {port} busy')
        except urllib.error.URLError: pass
    prompts={n:prompt(n) for n in (8000,16000,24000)}; rows=[]
    proc,log,args=start('direct',18142,['--pflash-mode','off'])
    try: direct={n:one(18142,'cp024_direct_completion',n,prompts[n],96) for n in prompts}
    finally: stop(proc,log)
    proc,log,args2=start('pflash',18143,['--pflash-mode','always','--pflash-keep-ratio','0.10','--pflash-score','model','--pflash-model',str(PFLASH_MODEL)])
    try: pflash={n:one(18143,'cp024_pflash_completion_kr010',n,prompts[n],96) for n in prompts}
    finally: stop(proc,log)
    proc,log,args3=start('verify',18142,['--pflash-mode','off'])
    try:
        for n in prompts:
            vp='/no_think\nVerify exact facts from selected ORIGINAL prompt spans. Output strictly: ROUTE=<value> FUNC=<value> SENTINEL=<value>\n\nFirst-pass candidate answer:\n'+pflash[n]['preview']+'\n\nSelected original spans:\n'+span(prompts[n])
            ver=one(18142,'cp024_two_pass_completion_verify',n,vp,128)
            rows.append({'approx_tokens':n,'direct':direct[n],'raw_pflash':pflash[n],'two_pass_verify':ver,'two_pass_total_wall_s':round(pflash[n]['wall_s']+ver['wall_s'],3),'two_pass_total_ttft_s_est':round((pflash[n]['ttft_s_est'] or 0)+(ver['ttft_s_est'] or 0),3)})
    finally: stop(proc,log)
    result={'run_dir':str(RUN_DIR),'endpoint':'/v1/completions','ttft_note':'ttft_s_est is prompt_ms/1000 from non-stream timings, not measured streaming TTFT','direct_args':args,'pflash_args':args2,'verify_args':args3,'rows':rows,'cp023_chat_artifact_run':'temp-cp024-pflash-sanity-20260629T180722Z'}
    out=RUN_DIR/'cp024_completions_comparison.json'; out.write_text(json.dumps(result,indent=2))
    print(json.dumps({'status':'ok','comparison':str(out),'run_dir':str(RUN_DIR)},indent=2))
    for r in rows: print(r['approx_tokens'], 'direct', r['direct']['wall_s'], r['direct']['ok'], 'pflash', r['raw_pflash']['wall_s'], r['raw_pflash']['ok'], '2pass', r['two_pass_total_wall_s'], r['two_pass_verify']['ok'])
if __name__=='__main__': main()
