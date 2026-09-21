"""Validate frozen prompts, source data and persisted answers before resuming."""
import argparse
import hashlib
import json
from pathlib import Path
from benchmark_generate import ROOT,digest,read_rows,now,dump,final_answer
from bandit_mopd.prompting import render_rollout_prompt

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--report',required=True);a=ap.parse_args()
    cfg=json.loads(Path(a.config).read_text());out=ROOT/cfg['output']
    fingerprint=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
    assert cfg['enable_thinking'] is True and cfg['max_new_tokens']==8192
    assert cfg['sampling']==dict(temperature=0.6,top_p=0.95,top_k=20,min_p=0.0,repetition_penalty=1.0)
    assert cfg['rollout_system_prompt']==(out/'system-prompt.txt').read_text().strip()
    assert json.loads((out/'manifest.json').read_text())['config_sha256']==fingerprint
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(cfg['student'],local_files_only=True)
    report={'checked_utc':now(),'config_sha256':fingerprint,'hostname':__import__('os').uname().nodename,'datasets':[]}
    for ds in cfg['datasets']:
        dest=out/ds['name'];meta=json.loads((dest/'input-manifest.json').read_text())
        assert meta['config_sha256']==fingerprint and digest(ROOT/ds['path'])==meta['source_sha256']
        prompts=read_rows(dest/'prompts.jsonl');rows=read_rows(ROOT/ds['path'])
        assert len(prompts)==len(rows)==ds['n']
        for i,(p,row) in enumerate(zip(prompts,rows)):
            expected=render_rollout_prompt(tokenizer,row['messages'],cfg,tokenize=False,add_generation_prompt=True)
            if not expected.endswith('<think>\n'):expected+='<think>\n'
            assert p['index']==i and p['id']==row['id']==meta['ids'][i]
            assert p['prompt']==expected and p['prompt_token_ids']==tokenizer.encode(expected,add_special_tokens=False)
            assert len(p['prompt_token_ids'])+8192<=cfg['vllm']['max_model_len']
        saved=read_rows(dest/'results.jsonl') if (dest/'results.jsonl').exists() else []
        assert [r['id'] for r in saved]==[p['id'] for p in prompts[:len(saved)]]
        for r in saved:
            assert r['response_tokens']==len(r['response_ids'])<=8192
            assert r['final_answer']==final_answer(r['response'])
            assert r['truncated']==(r['finish_reason']=='length')
        report['datasets'].append({'name':ds['name'],'prompts_checked':len(prompts),'saved_answers_checked':len(saved),
            'prompts_sha256':digest(dest/'prompts.jsonl'),'source_sha256':meta['source_sha256'],
            'saved_results_sha256':digest(dest/'results.jsonl') if saved else None})
    report['status']='passed';dump(a.report,report);print(json.dumps(report),flush=True)

if __name__=='__main__':main()
