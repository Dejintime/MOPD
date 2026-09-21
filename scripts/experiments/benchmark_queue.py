"""Sequential persistent benchmark queue, holding a single-owner flock."""
import argparse, fcntl, hashlib, json, os, subprocess, sys, time, traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from benchmark_generate import dump,now,runtime_settings

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--runtime-config');ap.add_argument('--external-evaluation',nargs='*',default=[]);a=ap.parse_args()
    cfgpath=Path(a.config).resolve();cfg=json.loads(cfgpath.read_text());out=ROOT/cfg['output'];out.mkdir(exist_ok=True)
    assert json.loads((out/'manifest.json').read_text())['config_sha256']==hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest(),'Config differs from prepared inputs'
    runtime_settings(cfg,a.runtime_config)
    runtimepath=Path(a.runtime_config).resolve() if a.runtime_config else None
    lock=(out/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state={'status':'running','pid':os.getpid(),'started':now(),'order':[d['name'] for d in cfg['datasets']],'completed':[]}
    dump(out/'queue-state.json',state)
    env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='8',VLLM_NO_USAGE_STATS='1',DO_NOT_TRACK='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',VLLM_WORKER_MULTIPROC_METHOD='spawn',NLTK_DATA='/home/yangdejin/.mopd-benchmark-nltk')
    try:
        for ds in cfg['datasets']:
            name=ds['name'];dest=out/name
            if (dest/'complete.json').exists():state['completed'].append(name);continue
            state.update(current=name,stage='waiting_for_verifier_readiness',updated=now());dump(out/'queue-state.json',state)
            while not (out/'readiness'/f'{name}.json').exists():time.sleep(10)
            for stage,script,marker in [('generation','benchmark_generate.py','generation-complete.json'),('evaluation','benchmark_evaluate.py','complete.json')]:
                if (dest/marker).exists():continue
                if stage=='evaluation' and name in a.external_evaluation:
                    state.update(stage='waiting_for_external_evaluation',child_pid=None,updated=now());dump(out/'queue-state.json',state)
                    while not (dest/marker).exists():
                        error=dest/'external-evaluation-error.json'
                        if error.exists():raise RuntimeError(error.read_text())
                        time.sleep(10)
                    continue
                state.update(stage=stage,updated=now());dump(out/'queue-state.json',state)
                with (dest/f'{stage}.log').open('a') as log:
                    command=[sys.executable,'-u',str(ROOT/'scripts/experiments'/script),'--config',str(cfgpath),'--dataset',name]
                    if runtimepath and stage=='generation':command+=['--runtime-config',str(runtimepath)]
                    child=subprocess.Popen(command,cwd=ROOT,env=env if stage=='generation' else dict(env,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT)
                    state.update(child_pid=child.pid);dump(out/'queue-state.json',state)
                    code=child.wait()
                if code:raise RuntimeError(f'{name} {stage} exited {code}; see {stage}.log')
                assert (dest/marker).exists(),(name,stage,'completion marker missing')
            state['completed'].append(name);state.update(updated=now());dump(out/'queue-state.json',state)
            summaries=[json.loads((out/n/'summary.json').read_text()) for n in state['completed']]
            dump(out/'summaries.json',summaries)
            print(json.dumps({'completed':name,'summary':summaries[-1]}),flush=True)
        state.update(status='complete',stage='complete',current=None,child_pid=None,updated=now());dump(out/'queue-state.json',state)
        print('ALL_BENCHMARKS_COMPLETE',flush=True)
    except BaseException as e:
        state.update(status='failed',error=repr(e),updated=now());dump(out/'queue-state.json',state);raise

if __name__=='__main__':main()
