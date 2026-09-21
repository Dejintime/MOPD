"""Local coordinator for a GPU host and the original isolated CPU LCB judge.

Uses existing SSH connections only; no passwords are saved. Completion markers
are delivered last so the GPU queue cannot advance before scores are durable.
"""
import argparse
import fcntl
import hashlib
import json
import shlex
import subprocess
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUTPUT='analysis/2026-09-16-sft-benchmarks'
REMOTE_ROOT='/root/autodl-tmp/MOPD'
JUDGE_ROOT='/home/yangdejin/MOPD'
PYTHON='/home/yangdejin/miniconda3/envs/mopd/bin/python'

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--gpu-host',required=True)
    ap.add_argument('--gpu-port',required=True)
    ap.add_argument('--gpu-socket',required=True)
    ap.add_argument('--judge-host')
    ap.add_argument('--judge-socket')
    ap.add_argument('--local-judge',action='store_true')
    a=ap.parse_args()
    out=ROOT/OUTPUT
    lock=(out/'bridge.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    gpu=['ssh','-p',a.gpu_port,'-S',a.gpu_socket,'-o','BatchMode=yes','-o','ConnectTimeout=20','-o','ServerAliveInterval=30','-o','ServerAliveCountMax=6']
    if a.local_judge:
        assert ROOT==Path(JUDGE_ROOT),'Run local judge coordinator from the original project'
        judge=[]
    else:
        if not a.judge_host or not a.judge_socket:ap.error('Provide judge SSH options or --local-judge')
        judge=['ssh','-S',a.judge_socket,'-o','BatchMode=yes','-o','ConnectTimeout=20','-o','ServerAliveInterval=30','-o','ServerAliveCountMax=6']

    def run(args,**kwargs):
        return subprocess.run(args,check=True,text=True,**kwargs)

    def rsync(ssh,source,dest,includes=None):
        cmd=['rsync','-az','--no-owner','--no-group']
        if includes:
            cmd+=['--include='+x for x in includes]+['--exclude=*']
        run(cmd+['-e',shlex.join(ssh),source,dest],timeout=600)

    def refresh():
        # Runtime records and all prediction/score outputs, without large sources.
        cmd=['rsync','-az','--no-owner','--no-group','--exclude=sources/','--exclude=tools/','--exclude=nltk-packages/','--exclude=*.lock',
             '--exclude=bridge.log','--exclude=bridge-state.json','-e',shlex.join(gpu),
             a.gpu_host+':'+REMOTE_ROOT+'/'+OUTPUT+'/',str(out)+'/']
        run(cmd,timeout=600)

    previous=None
    while True:
        if not a.local_judge:run(judge+[a.judge_host,'true'],timeout=30)
        cp=run(gpu+[a.gpu_host,'cat '+REMOTE_ROOT+'/'+OUTPUT+'/queue-state.json'],capture_output=True,timeout=30)
        state=json.loads(cp.stdout)
        key=(state['status'],state.get('current'),state.get('stage'))
        if key!=previous:
            print(json.dumps(state),flush=True)
            previous=key
        if state.get('stage')=='waiting_for_external_evaluation':
            name=state['current']
            if name!='lcb_release_v5':raise ValueError('Unexpected external benchmark '+name)
            refresh()
            local=out/name
            remote=JUDGE_ROOT+'/'+OUTPUT+'/'+name+'/'
            # Input identity is verified by the judge against its frozen dataset.
            if not a.local_judge:
                rsync(judge,str(local)+'/',a.judge_host+':'+remote,
                      ['results.jsonl','generation-complete.json','input-manifest.json'])
            command='cd '+shlex.quote(JUDGE_ROOT)+' && '+shlex.join([PYTHON,'-u','scripts/experiments/benchmark_evaluate.py',
                    '--config',OUTPUT+'/config.json','--dataset',name])+' >> '+shlex.quote(OUTPUT+'/'+name+'/evaluation.log')+' 2>&1'
            try:
                process=subprocess.Popen(['bash','-lc',command] if a.local_judge else judge+[a.judge_host,command])
                while process.poll() is None:
                    # Keep the authenticated GPU connection alive during a long
                    # CPU evaluation; the GPU queue is waiting for its marker.
                    run(gpu+[a.gpu_host,'true'],timeout=30)
                    time.sleep(30)
                if process.returncode:raise subprocess.CalledProcessError(process.returncode,command)
            except subprocess.CalledProcessError as exc:
                error={'status':'failed','error':str(exc),'benchmark':name}
                (local/'external-evaluation-error.json').write_text(json.dumps(error)+'\n')
                rsync(gpu,str(local/'external-evaluation-error.json'),a.gpu_host+':'+REMOTE_ROOT+'/'+OUTPUT+'/'+name+'/')
                raise
            files=['scores.jsonl','summary.json','score-progress.json','evaluation.log']
            if not a.local_judge:rsync(judge,a.judge_host+':'+remote,str(local)+'/',files)
            summary=json.loads((local/'summary.json').read_text())
            assert summary['n']==880 and summary['source_results_sha256']==hashlib.sha256((local/'results.jsonl').read_bytes()).hexdigest()
            rsync(gpu,str(local)+'/',a.gpu_host+':'+REMOTE_ROOT+'/'+OUTPUT+'/'+name+'/',files)
            if not a.local_judge:rsync(judge,a.judge_host+':'+remote+'complete.json',str(local)+'/')
            rsync(gpu,str(local/'complete.json'),a.gpu_host+':'+REMOTE_ROOT+'/'+OUTPUT+'/'+name+'/')
            print('LCB_EXTERNAL_EVALUATION_COMPLETE',flush=True)
        else:
            refresh()
        (out/'bridge-state.json').write_text(json.dumps({'updated_unix':time.time(),'queue':state},indent=2)+'\n')
        if state['status']=='complete':
            print('ALL_BENCHMARKS_COMPLETE_AND_SYNCED',flush=True)
            return
        if state['status']=='failed':raise RuntimeError(state.get('error','GPU queue failed'))
        time.sleep(30)

if __name__=='__main__':main()
