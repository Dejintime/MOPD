import json
import sys
from types import SimpleNamespace
from pathlib import Path
import pytest
from scripts.experiments import supervise_run


def write_events(path):
    path.write_text(json.dumps(dict(status='updated',optimizer_step=3,reward=-1.8,loss=.0036,step_seconds=1222))+'\n'+json.dumps(dict(status='failed_empty_team',step=3))+'\n'+ '{"status":')


def test_failure_event_preserves_last_completed_step_and_metrics(tmp_path):
    p=tmp_path/'metrics.jsonl';write_events(p)
    s=supervise_run.recent_metrics(p)
    assert s['completed_steps']==3 and s['last_reward']==-1.8
    assert s['last_event_status']=='failed_empty_team' and s['last_event_step']==4
    p.write_text(json.dumps(dict(status='failed_empty_team',step=3))+'\n')
    assert supervise_run.recent_metrics(p,s)==s


def test_failure_before_first_update_needs_no_reward(tmp_path):
    p=tmp_path/'metrics.jsonl';p.write_text(json.dumps(dict(status='failed_empty_team',step=0))+'\n')
    s=supervise_run.recent_metrics(p)
    assert s['completed_steps']==0 and s['last_event_step']==1


def test_supervisor_records_failed_exit_instead_of_crashing(monkeypatch,tmp_path):
    output=tmp_path/'run';prefix=tmp_path/'observed'
    cfg=tmp_path/'config.json';cfg.write_text(json.dumps(dict(output=str(output),steps=200,save_steps=100,cpu_threads=1)))
    def start(*args,**kwargs):
        output.mkdir();write_events(output/'metrics.jsonl')
        return SimpleNamespace(pid=99999999,poll=lambda:1)
    monkeypatch.chdir(Path(supervise_run.__file__).resolve().parents[2])
    monkeypatch.setattr(supervise_run.subprocess,'Popen',start)
    monkeypatch.setattr(supervise_run.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=1))
    monkeypatch.setattr(sys,'argv',['supervise_run','--config',str(cfg),'--log-prefix',str(prefix)])
    assert supervise_run.main()==1
    status=json.loads(Path(str(prefix)+'.status.json').read_text())
    assert status['status']=='failed' and status['exit_code']==1
    assert status['completed_steps']==3 and status['last_event_status']=='failed_empty_team'


@pytest.mark.parametrize('gpu_id', ['1', 'GPU-f3ed3c37-a52e-c126-79e9-e0a0783c584d'])
def test_selected_gpu_is_used_for_training_and_monitoring(monkeypatch,tmp_path,gpu_id):
    output=tmp_path/'run';prefix=tmp_path/'observed'
    cfg=tmp_path/'config.json'
    cfg.write_text(json.dumps(dict(output=str(output),steps=200,save_steps=50,cpu_threads=10,student_device='cuda:0')))
    def start(command,**kwargs):
        assert kwargs['env']['CUDA_VISIBLE_DEVICES']==gpu_id
        assert kwargs['env']['CUDA_DEVICE_ORDER']=='PCI_BUS_ID'
        return SimpleNamespace(pid=99999999,poll=lambda:0)
    def monitor(command,**kwargs):
        assert f'--id={gpu_id}' in command
        return SimpleNamespace(returncode=0,stdout='1234, 99\n')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    monkeypatch.setattr(supervise_run.subprocess,'Popen',start)
    monkeypatch.setattr(supervise_run.subprocess,'run',monitor)
    # Provide Linux host memory on any test host, including macOS.
    original_read=Path.read_text
    monkeypatch.setattr(Path,'read_text',lambda p,*a,**k: 'MemAvailable: 100000 kB\n' if str(p)=='/proc/meminfo' else original_read(p,*a,**k))
    monkeypatch.setattr(sys,'argv',['supervise_run','--config',str(cfg),'--log-prefix',str(prefix),'--gpu',gpu_id])
    assert supervise_run.main()==0
    status=json.loads(Path(str(prefix)+'.status.json').read_text())
    assert status['physical_gpu']==gpu_id and status['student_device']=='cuda:0'
    assert status['gpu_used_mib']==1234 and status['gpu_utilization_percent']==99
    assert status['save_steps']==50
