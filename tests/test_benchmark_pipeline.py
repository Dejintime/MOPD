"""Tests critical scoring boundaries, including real sandboxed code execution."""
import importlib.util, json, sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/experiments'))
from benchmark_generate import final_answer,runtime_settings
from benchmark_evaluate import lcb_score, lcb_sample, code_from_final, load_if, if_score, score_text

def test_runtime_override_preserves_frozen_experiment(tmp_path):
    cfg=json.loads((ROOT/'analysis/2026-09-16-sft-benchmarks/config.json').read_text())
    before=json.dumps(cfg,sort_keys=True)
    batch,engine=runtime_settings(cfg,ROOT/'analysis/2026-09-16-sft-benchmarks/runtime-48gb.json')
    assert batch==24 and engine['max_num_seqs']==24 and engine['gpu_memory_utilization']==0.94
    assert engine['dtype']==cfg['vllm']['dtype'] and engine['max_model_len']==cfg['vllm']['max_model_len']
    assert json.dumps(cfg,sort_keys=True)==before
    path=tmp_path/'invalid.json'
    for forbidden in ({'sampling':{'temperature':1}}, {'vllm':{'dtype':'float16'}}, {'vllm':{'max_model_len':8192}}, {'batch_size':0}):
        path.write_text(json.dumps(forbidden))
        with pytest.raises(ValueError):runtime_settings(cfg,path)

def test_open_prefilled_thinking_is_not_a_final_answer():
    assert final_answer('The answer might be 7. Answer: \\boxed{7}')==''
    assert final_answer('Reasoning</think>Answer: \\boxed{7}')=='Answer: \\boxed{7}'
    assert final_answer('Reasoning</think><think>Restarted')==''

def test_aime_gold_integer_and_wrong_answer():
    assert score_text('aime24',{'answer':'007'},'Answer: \\boxed{7}')['correct']
    assert not score_text('aime24',{'answer':'8'},'Answer: \\boxed{7}')['correct']

def test_lcb_missing_final_fence_does_not_repair_model_code():
    assert code_from_final('```python\nprint(5)')==''
    assert code_from_final('```python\nprint(5)\n```')=='print(5)'

def test_lcb_stdin_and_functional_official_checker():
    for fn,code,expected in [(None,'a,b=map(int,input().split());print(a+b)',True),
          ('add','class Solution:\n def add(self,a,b): return a+b',True),
          ('add','class Solution:\n def add(self,a,b): return a-b',False)]:
        sample={'input_output':json.dumps({'inputs':['2\n3' if fn else '2 3\n'],'outputs':['5'],'fn_name':fn})}
        assert lcb_score(sample,1,code)['correct'] is expected

@pytest.mark.parametrize('name',['ifeval','ifbench'])
def test_if_official_positive_negative_and_empty(name):
    lib=load_if(name)
    row={'id':'toy:if','metadata':{'instruction_id_list':['keywords:existence'],'kwargs':[{'keywords':['apple']}],'prompt_text':'Include apple.'}}
    assert if_score(lib,row,'apple')['correct']
    assert not if_score(lib,row,'orange')['correct']
    assert not if_score(lib,row,'')['correct']
