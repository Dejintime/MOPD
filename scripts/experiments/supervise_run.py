"""Launch one experiment and record process/resource status until it exits."""
import argparse
from datetime import datetime, timezone
import json
import os
import re
from pathlib import Path
import subprocess
import time


def recent_metrics(path, previous=None):
    """Failure events need not contain loss/reward or a completed update."""
    result = dict(previous or {'completed_steps': 0})
    try:
        with Path(path).open('rb') as f:
            f.seek(0, 2)
            end = f.tell()
            f.seek(max(0, end-262144))
            data = f.read()
    except FileNotFoundError:
        return result
    except OSError as error:
        result['metrics_error'] = str(error)
        return result
    for line in data.splitlines():
        try:
            event = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            continue
        if not isinstance(event, dict):
            continue
        for key in ('attempt','skipped_batches','consecutive_empty_batches'):
            if key in event: result[key] = event[key]
        result['last_event_status'] = event.get('status')
        result['last_event_step'] = event.get('optimizer_step', event.get('step', -1)+1)
        if event.get('status') != 'updated':
            continue
        step = event.get('optimizer_step', 0)
        if step < result.get('completed_steps', 0):
            continue
        result['completed_steps'] = step
        for source, target in [('reward','last_reward'),('loss','last_loss'),
                               ('step_seconds','last_step_seconds'),('eta_seconds','last_eta_seconds')]:
            result[target] = event.get(source)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--log-prefix',required=True)
    parser.add_argument('--gpu', help='One physical GPU index or UUID; student must use logical cuda:0')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[2]
    os.chdir(root)
    cfg=json.loads(Path(args.config).read_text())
    gpu_id=args.gpu if args.gpu is not None else os.environ.get('CUDA_VISIBLE_DEVICES','0')
    if not re.fullmatch(r'(\d+|GPU-[A-Za-z0-9-]+)',gpu_id):
        parser.error('Select exactly one physical GPU index or UUID with --gpu')
    if cfg.get('student_device','cuda:0') != 'cuda:0':
        parser.error('The selected physical GPU is exposed as logical student_device cuda:0')
    prefix=Path(args.log_prefix)
    prefix.parent.mkdir(parents=True,exist_ok=True)
    output=Path(cfg['output'])
    if output.exists():raise FileExistsError(output)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu_id,CUDA_DEVICE_ORDER='PCI_BUS_ID',
        OMP_NUM_THREADS=str(cfg['cpu_threads']),MKL_NUM_THREADS=str(cfg['cpu_threads']),
        OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1')
    if cfg.get('inference_backend') == 'vllm':
        # CUDA IPC weight transfer cannot export expandable-segment allocations
        # in this container (pidfd_getfd is denied).
        env.pop('PYTORCH_ALLOC_CONF', None)
        env.pop('PYTORCH_CUDA_ALLOC_CONF', None)
    else:
        env['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
    python=env.get('MOPD_PYTHON','/home/yangdejin/miniconda3/envs/mopd/bin/python')
    command=[python,'-u','-m','bandit_mopd.train','--config',args.config]
    # Exclusive creation prevents overwriting any previous run's log.
    with Path(str(prefix)+'.log').open('x',buffering=1) as log, Path(str(prefix)+'.resources.jsonl').open('x',buffering=1) as resources:
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env)
        started=time.time()
        base=dict(supervisor_pid=os.getpid(),training_pid=process.pid,command=command,
            config=str(Path(args.config).resolve()),output=str(output.resolve()),
            started_at=datetime.now(timezone.utc).isoformat(),steps=cfg['steps'],save_steps=cfg['save_steps'],
            physical_gpu=gpu_id,cuda_visible_devices=gpu_id,student_device='cuda:0')
        status_path=Path(str(prefix)+'.status.json')
        latest_metrics = None
        while True:
            code=process.poll()
            record=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.time()-started)
            try:
                lines=dict(x.split(':',1) for x in Path('/proc/meminfo').read_text().splitlines())
                record['host_available_bytes']=int(lines['MemAvailable'].split()[0])*1024
                p=Path(f'/proc/{process.pid}/status')
                if p.exists():
                    info=dict(x.split(':',1) for x in p.read_text().splitlines())
                    for key in ['VmRSS','VmSwap','VmHWM']:
                        record[key+'_bytes']=int(info.get(key,'0').split()[0])*1024
                gpu=subprocess.run(['nvidia-smi',f'--id={gpu_id}','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
                if gpu.returncode==0:
                    used,util=gpu.stdout.strip().split(',');record.update(gpu_used_mib=int(used),gpu_utilization_percent=int(util))
            except (OSError,ValueError,subprocess.TimeoutExpired) as error:
                record['monitor_error']=str(error)
            latest_metrics=recent_metrics(output/'metrics.jsonl',latest_metrics)
            record.update(latest_metrics)
            resources.write(json.dumps(record)+'\n')
            state='running' if code is None else ('completed' if code==0 else 'failed')
            status=dict(base,status=state,exit_code=code,**record)
            temp=status_path.with_suffix('.json.tmp');temp.write_text(json.dumps(status,indent=2));temp.replace(status_path)
            if code is not None:break
            try:process.wait(timeout=30)
            except subprocess.TimeoutExpired:pass
    return code


if __name__=='__main__':raise SystemExit(main())
