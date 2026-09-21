"""Official LCB/IFEval/IFBench adapters and frozen M2RL GPQA scorer."""
import argparse, base64, contextlib, copy, hashlib, importlib.util, io, json, os, pickle, random, re, signal, subprocess, sys, zlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from benchmark_generate import dump,read_rows,digest,now
SOURCES=ROOT/'analysis/2026-09-16-sft-benchmarks/sources'

def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def load_if(name):
    os.environ.setdefault('NLTK_DATA','/home/yangdejin/.mopd-benchmark-nltk')
    if name=='ifeval':
        sys.path.insert(0,str(SOURCES/'google-research'))
        from instruction_following_eval import evaluation_lib
        return evaluation_lib
    sys.path.insert(0,str(SOURCES/'IFBench'))
    return load_module('benchmark_ifbench_eval',SOURCES/'IFBench/evaluation_lib.py')

class StringUnpickler(pickle.Unpickler):
    def find_class(self,module,name):raise ValueError('Unexpected executable pickle content in LCB data')

def lcb_sample(row):
    m=row['metadata']; public=json.loads(m['public_test_cases']); private=m['private_test_cases']
    try: private=json.loads(private)
    except (ValueError,TypeError):
        unpacked=StringUnpickler(io.BytesIO(zlib.decompress(base64.b64decode(private)))).load()
        assert isinstance(unpacked,(str,bytes));private=json.loads(unpacked)
    tests=public+private;assert tests and all(t['testtype'] in ('stdin','functional') for t in tests)
    meta=m['metadata'];meta=json.loads(meta) if isinstance(meta,str) else meta
    fn=meta.get('func_name');assert all(t['testtype']==('functional' if fn else 'stdin') for t in tests)
    return {'input_output':json.dumps({'inputs':[t['input'] for t in tests],'outputs':[t['output'] for t in tests],'fn_name':fn})},len(tests)

def code_from_final(text):
    # Same generic extraction as official LCB: last pair of fence lines.
    lines=text.split('\n'); idx=[i for i,l in enumerate(lines) if '```' in l]
    return '\n'.join(lines[idx[-2]+1:idx[-1]]) if len(idx)>=2 else ''

def lcb_score(sample,n,code):
    if not code.strip():return {'correct':False,'reason':'missing_complete_code_block','tests_total':n,'tests':[]}
    envroot=Path(sys.executable).parent.parent
    bwrap=ROOT/'analysis/2026-09-16-sft-benchmarks/tools/bubblewrap/usr/bin/bwrap'
    cmd=[str(bwrap),'--unshare-all','--die-with-parent','--new-session',
         '--ro-bind','/usr','/usr','--ro-bind','/lib','/lib','--ro-bind','/lib64','/lib64',
         '--symlink','usr/bin','/bin','--proc','/proc','--dev','/dev','--tmpfs','/tmp',
         '--dir','/etc','--ro-bind','/etc/ld.so.cache','/etc/ld.so.cache',
         '--ro-bind',str(envroot),str(envroot),'--dir','/judge',
         '--ro-bind',str(SOURCES/'LiveCodeBench/lcb_runner/evaluation/testing_util.py'),'/judge/testing_util.py',
         '--ro-bind',str(ROOT/'scripts/experiments/benchmark_lcb_worker.py'),'/judge/worker.py',
         '--clearenv','--setenv','PATH',str(envroot/'bin')+':/usr/bin:/bin','--setenv','HOME','/tmp',
         '--setenv','PYTHONDONTWRITEBYTECODE','1','--setenv','PYTHONHASHSEED','0',
         '--setenv','OPENBLAS_NUM_THREADS','1','--setenv','OMP_NUM_THREADS','1',
         '--setenv','CUDA_VISIBLE_DEVICES','','--chdir','/tmp',str(envroot/'bin/python'),'/judge/worker.py']
    payload={'sample':sample,'code':code,'cpu_limit':7*n+30}
    try:cp=subprocess.run(cmd,input=json.dumps(payload),text=True,capture_output=True,timeout=7*n+60)
    except subprocess.TimeoutExpired:return {'correct':False,'reason':'outer_code_timeout','tests_total':n,'tests':[]}
    if cp.returncode:raise RuntimeError(f'LCB sandbox infrastructure/process failure {cp.returncode}: {cp.stderr[-1500:]}')
    try:res=json.loads(cp.stdout)
    except ValueError as e:raise RuntimeError(f'LCB sandbox invalid result: {cp.stdout[-500:]} {cp.stderr[-500:]}') from e
    tests=res['tests'];assert tests
    return dict(correct=bool(len(tests)==n and all(x is True for x in tests)),tests_total=n,tests=tests,metadata=res['metadata'])

def if_score(lib,row,text):
    random.seed(int(hashlib.sha256(row['id'].encode()).hexdigest()[:16],16))
    from langdetect import DetectorFactory
    DetectorFactory.seed=42
    m=row['metadata'];kwargs=[{k:v for k,v in (x or {}).items() if v is not None} for x in m['kwargs']]
    inp=lib.InputExample(key=0,instruction_id_list=m['instruction_id_list'],prompt=m['prompt_text'],kwargs=kwargs)
    strict=lib.test_instruction_following_strict(copy.deepcopy(inp),{inp.prompt:text})
    loose=lib.test_instruction_following_loose(copy.deepcopy(inp),{inp.prompt:text})
    return {'correct':bool(strict.follow_all_instructions),'prompt_strict':bool(strict.follow_all_instructions),
            'prompt_loose':bool(loose.follow_all_instructions),'instruction_strict':strict.follow_instruction_list,
            'instruction_loose':loose.follow_instruction_list}

def score_text(name,row,text,lib=None):
    if name in ('ifeval','ifbench'):return if_score(lib,row,text)
    if name=='gpqa_diamond':
        pred=lib._extract_letter_from_response(text,'ABCD')
        return {'correct':bool(lib.compute_gpqa_reward(text,row['answer'],row['metadata'])),'prediction':pred}
    from bandit_mopd.verifiers.text import math as score_math
    return score_math(row,text)

def alarm_handler(*_):raise TimeoutError('Text benchmark verifier exceeded 30 seconds')

def summarize(records,name):
    n=len(records);correct=sum(r['result']['correct'] for r in records)
    summary={'name':name,'n':n,'correct':correct,'accuracy':correct/n,'truncated':sum(r['truncated'] for r in records),
        'truncation_rate':sum(r['truncated'] for r in records)/n,'mean_tokens':sum(r['response_tokens'] for r in records)/n,
        'thinking_unclosed':sum(not r['thinking_closed'] for r in records),'infrastructure_errors':0}
    if name in ('ifeval','ifbench'):
        for mode in ('strict','loose'):
            summary['prompt_'+mode]=sum(r['result']['prompt_'+mode] for r in records)/n
            constraints=[x for r in records for x in r['result']['instruction_'+mode]]
            summary['instruction_'+mode]=sum(constraints)/len(constraints)
    return summary

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--dataset',required=True);a=ap.parse_args()
    cfg=json.loads(Path(a.config).read_text());ds=next(x for x in cfg['datasets'] if x['name']==a.dataset);out=ROOT/cfg['output']/a.dataset
    assert (out/'generation-complete.json').exists()
    predictions=read_rows(out/'results.jsonl');assert len(predictions)==ds['n'];preds={r['id']:r for r in predictions}
    old=read_rows(out/'scores.jsonl') if (out/'scores.jsonl').exists() else []
    assert [x['id'] for x in old]==[x['id'] for x in predictions[:len(old)]]
    existing={r['id'] for r in old};lib=load_if(a.dataset) if a.dataset in ('ifeval','ifbench') else None
    if a.dataset=='gpqa_diamond':lib=load_module('benchmark_gpqa',SOURCES/'M2RL/gpqa.py')
    assert digest(ROOT/ds['path'])==json.loads((out/'input-manifest.json').read_text())['source_sha256']
    signal.signal(signal.SIGALRM,alarm_handler)
    with (ROOT/ds['path']).open() as source, (out/'scores.jsonl').open('a') as dest:
        for line in source:
            row=json.loads(line);p=preds[row['id']]
            if row['id'] in existing:continue
            try:
                if a.dataset=='lcb_release_v5':
                    sample,n=lcb_sample(row);result=lcb_score(sample,n,code_from_final(p['final_answer']))
                else:
                    signal.alarm(30)
                    try:result=score_text(a.dataset,row,p['final_answer'],lib)
                    finally:signal.alarm(0)
            except Exception as e:
                dump(out/'evaluation-error.json',{'id':row['id'],'error':repr(e),'timestamp':now()});raise
            rec={k:p[k] for k in ('id','benchmark','domain','response_tokens','truncated','thinking_closed')};rec['result']=result
            dest.write(json.dumps(rec,ensure_ascii=False)+'\n');dest.flush();os.fsync(dest.fileno());old.append(rec)
            dump(out/'score-progress.json',{'completed':len(old),'total':ds['n'],'correct':sum(x['result']['correct'] for x in old),'updated':now()})
            print(json.dumps({'id':row['id'],'correct':result['correct'],'completed':len(old)}),flush=True)
    assert len(old)==ds['n'];summary=summarize(old,a.dataset);summary['source_results_sha256']=digest(out/'results.jsonl')
    dump(out/'summary.json',summary);dump(out/'complete.json',{'status':'complete','n':len(old),'timestamp':now()});print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
