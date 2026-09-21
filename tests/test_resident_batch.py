import copy
import json
import sys

import numpy as np
import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from bandit_mopd.chunked import compact_target, backward_resident_batch
from bandit_mopd.core import mixture_log_probs, reverse_kl_loss
from bandit_mopd.train import log_probs
from bandit_mopd.data import read_training_prompts


@pytest.mark.parametrize('micro,accumulation,backward_size',[(1,2,1),(2,1,2),(2,2,2),(4,2,1)])
def test_resident_microbatch_and_accumulation_match_dense_gradient(micro,accumulation,backward_size):
    torch.manual_seed(61)
    cfg=Qwen3Config(vocab_size=19,hidden_size=16,intermediate_size=24,num_hidden_layers=2,
        num_attention_heads=2,num_key_value_heads=1,head_dim=8,tie_word_embeddings=True)
    reference=Qwen3ForCausalLM(cfg).eval()
    actual=copy.deepcopy(reference)
    actual.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    samples=[]; expected=[]
    for i in range(micro*accumulation):
        ids=torch.tensor([[1,2,3,4,5,6,7][:4+i]])
        prompt=2+i%2; count=ids.shape[1]-prompt
        teachers=torch.randn(2,count,19).log_softmax(-1)
        old=torch.randn(count,19).log_softmax(-1)
        union,valid,target,_=compact_target(teachers,old,[.3,.7],.05,3)
        sample=dict(ids=ids,prompt_length=prompt,response_ids=ids[0,prompt:],
                    union_ids=union,union_valid=valid,target=target)
        samples.append(sample)
        q=mixture_log_probs(teachers,[.3,.7],old,.05)
        expected.append(reverse_kl_loss(log_probs(reference,ids,prompt),q,teachers,3))
    torch.stack(expected).mean().backward()
    actual.train()
    got=backward_resident_batch(actual,samples,dict(micro_batch_size=micro,
        gradient_accumulation_steps=accumulation,student_backward_batch_size=backward_size,
        logit_chunk_tokens=2),pad_token_id=0)
    assert np.allclose(got,[x.item() for x in expected],atol=1e-6)
    for (name,p),(_,r) in zip(actual.named_parameters(),reference.named_parameters()):
        assert p.grad is not None,name
        assert torch.allclose(p.grad,r.grad,atol=2e-6,rtol=2e-5),name


def test_prompt_reader_drops_unused_metadata_and_rejects_agent(tmp_path):
    path=tmp_path/'train.jsonl'
    row=dict(id='x',domain='math',messages=[dict(role='user',content='problem')],metadata={'unit_tests':'large'},answer='secret')
    path.write_text(json.dumps(row)+'\n')
    assert read_training_prompts(path)==[{k:row[k] for k in ('id','domain','messages')}]
    row['domain']='agent';path.write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError,match='Agent'): read_training_prompts(path)


def test_agent_filter_preserves_retained_bytes_and_does_not_match_prompt_mentions(tmp_path):
    from scripts.experiments.remove_agent_data import purge
    raw=tmp_path/'raw';raw.mkdir()
    keep=b'{"domain":"math", "messages": [{"role":"user","content":"an agent solves math"}]}\n'
    remove=b'{"dataset":"nano_v3_sft_profiled_workbench"}\n'
    path=raw/'train.jsonl';path.write_bytes(keep+remove)
    report=purge(tmp_path,apply=True)
    assert path.read_bytes()==keep
    assert report['removed_rows']==1
    assert report['files'][0]['removed_bytes']==len(remove)


@pytest.mark.parametrize('empty_mode',['none','once','always','after_update','save_after_skip'])
def test_dense_only_resident_training_never_loads_or_evaluates_probes(monkeypatch,tmp_path,empty_mode):
    from bandit_mopd import train
    from test_probes import Tokenizer, row
    cfg_model=Qwen3Config(vocab_size=17,hidden_size=16,intermediate_size=24,num_hidden_layers=1,
        num_attention_heads=2,num_key_value_heads=1,head_dim=8)
    for name in ('student','teacher'):
        Qwen3ForCausalLM(cfg_model).save_pretrained(tmp_path/name)
    cfg=json.loads(__import__('pathlib').Path('configs/experiments/bandit_four_resident_16k.json').read_text())
    domains=['math','code','science','if']
    cfg.update(student=str(tmp_path/'student'),student_device='cpu',cpu_threads=1,
        teachers=[dict(name=d,domain=d,path=str(tmp_path/'teacher')) for d in domains],
        steps=2 if empty_mode in ('after_update','save_after_skip') else 1,k=2,micro_batch_size=2,gradient_accumulation_steps=1,
        max_consecutive_empty_batches=2,save_steps=1 if empty_mode=='save_after_skip' else 100,
        save_student=empty_mode in ('after_update','save_after_skip'),save_optimizer=False,output=str(tmp_path/'output'),
        train_data=str(tmp_path/'train.jsonl'),probe_data='/must-not-be-read/probe.jsonl',kl_gate=100)
    assert 'task' not in cfg['reward']
    (tmp_path/'train.jsonl').write_text(''.join(json.dumps(row(d,'train'+str(i)))+'\n' for d in domains for i in range(2)))
    config_path=tmp_path/'config.json';config_path.write_text(json.dumps(cfg))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys,'argv',['train','--config',str(config_path)])
    monkeypatch.setattr(train,'inspect',lambda cfg:{})
    # These tiny model fixtures do not need the production 4B checkpoint reserve.
    monkeypatch.setattr(train,'disk_reserve_bytes',lambda cfg:1024*1024)
    monkeypatch.setattr(train,'peak_gpu_memory',lambda cfg:{})
    monkeypatch.setattr(train.AutoTokenizer,'from_pretrained',lambda *a,**kw:Tokenizer())
    monkeypatch.setattr(train,'read_jsonl',lambda *a:pytest.fail('Probe data were read'))
    monkeypatch.setattr(train,'ProbeEvaluator',lambda *a:pytest.fail('Probe evaluator was initialized'))
    monkeypatch.setattr(train,'rollout',lambda *a,**kw:(torch.tensor([[1,2,3,4,5]]),3,'response',False))
    original_select=train.CombinatorialLinUCB.select
    calls=[]
    def select(*args):
        calls.append(1)
        empty=(empty_mode=='always' or empty_mode in ('once','save_after_skip') and len(calls)==1
               or empty_mode=='after_update' and len(calls)>1)
        return ([],np.zeros(4),-np.ones(4)) if empty else original_select(*args)
    monkeypatch.setattr(train.CombinatorialLinUCB,'select',select)
    if empty_mode in ('after_update','save_after_skip'):
        monkeypatch.setattr(Tokenizer,'save_pretrained',lambda self,path:None,raising=False)
    if empty_mode in ('always','after_update'):
        with pytest.raises(RuntimeError,match='No eligible team selected in 2 consecutive batches'):
            train.main()
        events=[json.loads(x) for x in (tmp_path/'output/metrics.jsonl').read_text().splitlines()]
        assert [e['status'] for e in events[-2:]]==['skipped_empty_team','failed_empty_team_limit']
        event=events[-1]
        assert event['completed_steps']==int(empty_mode=='after_update')
        assert event['reason']=='negative_initial_marginal_gains'
        assert event['eligible']==[True]*4 and all(g<0 for g in event['initial_marginal_gains'])
        from scripts.experiments.supervise_run import recent_metrics
        status=recent_metrics(tmp_path/'output/metrics.jsonl')
        assert status['completed_steps']==int(empty_mode=='after_update')
        assert status['last_event_status']=='failed_empty_team_limit'
        summary=json.loads((tmp_path/'output/summary.json').read_text())
        assert summary['status']=='failed' and summary['skipped_batches']==2
        if empty_mode=='after_update':
            assert (tmp_path/'output/checkpoint-1/student/model.safetensors').exists()
            state=json.loads((tmp_path/'output/checkpoint-1/trainer_state.json').read_text())
            assert state['save_reason']=='empty_team_limit' and state['optimizer_step']==1
        return
    train.main()
    events=[json.loads(x) for x in (tmp_path/'output/metrics.jsonl').read_text().splitlines()]
    if empty_mode=='once':
        assert [e['status'] for e in events]==['skipped_empty_team','updated']
        assert events[-1]['attempt']==2 and events[-1]['optimizer_step']==1
        assert events[-1]['train_samples_seen']==2
    if empty_mode=='save_after_skip':
        assert [e['status'] for e in events]==['skipped_empty_team','updated','updated']
        assert [e['optimizer_step'] for e in events[1:]]==[1,2]
        assert [e['attempt'] for e in events[1:]]==[2,3]
        assert (tmp_path/'output/checkpoint-1/student/model.safetensors').exists()
        assert (tmp_path/'output/checkpoint-2/student/model.safetensors').exists()
        assert not (tmp_path/'output/checkpoint-3').exists()
        summary=json.loads((tmp_path/'output/summary.json').read_text())
        assert summary['steps']==2 and summary['attempted_batches']==3 and summary['skipped_batches']==1
    event=events[-1]
    assert event['weighted_reward_terms']['cost']==0
    assert event['selection_cost_unit']=='relative_excess_prefill'
    assert all(0<=c<=1 for c in event['selection_costs'])
    assert np.allclose(np.array(event['contexts'])[:,6],event['selection_costs'])
    assert event['effective_batch_size']==2
    assert event['probe_seconds']==0 and event['task_gain'] is None
    assert all(x['resident'] and x['device']=='cpu' for x in event['teacher_costs'])
    assert len(event['contexts'])==4 and len(event['contexts'][0])==7
    assert json.loads((tmp_path/'output/summary.json').read_text())['reward_scope']=='dense_proxy_only'
