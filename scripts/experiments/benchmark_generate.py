"""Full benchmark inference; restart from persisted answers without reselection."""
import argparse, hashlib, json, os, sys, time, importlib.metadata
from pathlib import Path
from datetime import datetime, timezone
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from bandit_mopd.prompting import render_rollout_prompt

def now(): return datetime.now(timezone.utc).isoformat()
def dump(path,obj):
    path=Path(path); tmp=path.with_suffix(path.suffix+'.part');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)
def read_rows(path):
    with Path(path).open() as f: return [json.loads(l) for l in f if l.strip()]
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def final_answer(response):
    # Opening <think> is prefilled by the tokenizer, outside generated text.
    if '</think>' not in response:return ''
    answer=response.rsplit('</think>',1)[1].strip()
    return '' if '<think>' in answer else answer

def runtime_settings(cfg,path=None):
    """Only scheduling/memory overrides; frozen scientific config stays intact."""
    runtime=json.loads(Path(path).read_text()) if path else {}
    if set(runtime)-{'batch_size','vllm'}:raise ValueError('Runtime overrides may only change batch_size/vllm scheduling')
    overrides=runtime.get('vllm',{})
    if set(overrides)-{'gpu_memory_utilization','max_num_seqs','max_num_batched_tokens'}:
        raise ValueError('Runtime overrides cannot change model, precision, context or sampling')
    batch=runtime.get('batch_size',cfg['batch_size'])
    engine=dict(cfg['vllm'],**overrides)
    if not isinstance(batch,int) or isinstance(batch,bool) or batch<1:raise ValueError('Invalid batch_size')
    for key in ('max_num_seqs','max_num_batched_tokens'):
        if not isinstance(engine[key],int) or isinstance(engine[key],bool) or engine[key]<1:raise ValueError('Invalid '+key)
    if not 0<engine['gpu_memory_utilization']<1:raise ValueError('Invalid GPU memory utilization')
    return batch,engine

def prepare(cfg,tokenizer):
    out=ROOT/cfg['output']; out.mkdir(parents=True,exist_ok=True)
    config_hash=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
    manifest={'created_utc':now(),'config_sha256':config_hash,'enable_thinking':True,'max_new_tokens':cfg['max_new_tokens'],'datasets':[],
              'versions':{n:importlib.metadata.version(n) for n in ('torch','transformers','vllm')}}
    for ds in cfg['datasets']:
        path=ROOT/ds['path']; target=out/ds['name'];target.mkdir(exist_ok=True)
        prompts=[];lengths=[];seen=set();h=hashlib.sha256()
        with path.open('rb') as f:
            for index,line in enumerate(f):
                h.update(line);row=json.loads(line)
                assert row['usage']=='final_evaluation_only' and row['id'] not in seen
                seen.add(row['id'])
                prompt=render_rollout_prompt(tokenizer,row['messages'],cfg,tokenize=False,add_generation_prompt=True)
                if not prompt.endswith('<think>\n'):
                    assert prompt.endswith('<|im_start|>assistant\n'),repr(prompt[-60:])
                    prompt += '<think>\n'
                ids=tokenizer.encode(prompt,add_special_tokens=False)
                assert len(ids)+cfg['max_new_tokens']<=cfg['vllm']['max_model_len'],(ds['name'],row['id'],len(ids))
                lengths.append(len(ids))
                prompts.append({'id':row['id'],'index':index,'domain':row['domain'],'prompt':prompt,'prompt_token_ids':ids})
        assert len(prompts)==ds['n'],(ds,len(prompts))
        old=target/'input-manifest.json'
        item={'name':ds['name'],'n':len(prompts),'source_sha256':h.hexdigest(),'config_sha256':config_hash,'prompt_min':min(lengths),'prompt_max':max(lengths),'ids':[p['id'] for p in prompts]}
        if old.exists():assert json.loads(old.read_text())==item,'Config/data changed; choose a new output directory'
        else:
            dump(old,item)
            with (target/'prompts.jsonl').open('w') as f:
                for p in prompts:f.write(json.dumps(p,ensure_ascii=False)+'\n')
        manifest['datasets'].append(item)
        print(json.dumps({'prepared':ds['name'],'n':len(prompts),'prompt_max':max(lengths)}),flush=True)
    dump(out/'manifest.json',manifest)
    return manifest

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--dataset');ap.add_argument('--prepare-only',action='store_true');ap.add_argument('--runtime-config');a=ap.parse_args()
    cfg=json.loads(Path(a.config).read_text());assert cfg['enable_thinking'] is True and isinstance(cfg['max_new_tokens'],int) and cfg['max_new_tokens']>0
    out=ROOT/cfg['output']
    if a.prepare_only:
        from transformers import AutoTokenizer
        prepare(cfg,AutoTokenizer.from_pretrained(cfg['student'],local_files_only=True));return
    ds=next(d for d in cfg['datasets'] if d['name']==a.dataset);dest=out/a.dataset
    meta=json.loads((dest/'input-manifest.json').read_text())
    assert meta['config_sha256']==hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
    assert digest(ROOT/ds['path'])==meta['source_sha256'],'Evaluation data changed'
    batch_size,engine=runtime_settings(cfg,a.runtime_config)
    prompts=read_rows(dest/'prompts.jsonl'); existing=read_rows(dest/'results.jsonl') if (dest/'results.jsonl').exists() else []
    assert [r['id'] for r in existing]==[r['id'] for r in prompts[:len(existing)]]
    if len(existing)==ds['n']:
        if not (dest/'generation-complete.json').exists():dump(dest/'generation-complete.json',{'n':len(existing),'timestamp':now(),'recovered_from_complete_results':True})
        return
    from vllm import LLM,SamplingParams
    session={'started':now(),'resume_offset':len(existing),'config_sha256':meta['config_sha256'],
             'prompts_sha256':digest(dest/'prompts.jsonl'),'batch_size':batch_size,'vllm':engine,
             'hostname':os.uname().nodename,'versions':{n:importlib.metadata.version(n) for n in ('torch','transformers','vllm')}}
    runtime_id=hashlib.sha256(json.dumps(session,sort_keys=True).encode()).hexdigest()
    sessions=dest/'runtime-sessions';sessions.mkdir(exist_ok=True);dump(sessions/(runtime_id+'.json'),session)
    llm=LLM(model=cfg['student'],seed=cfg['seed'],**engine)
    started=time.monotonic()
    for offset in range(len(existing),len(prompts),batch_size):
        batch=prompts[offset:offset+batch_size]
        params=[SamplingParams(max_tokens=cfg['max_new_tokens'],seed=cfg['seed']+p['index'],ignore_eos=False,**cfg['sampling']) for p in batch]
        dump(dest/'progress.json',{'status':'generating','completed':offset,'total':len(prompts),'updated':now()})
        outputs=llm.generate([{'prompt_token_ids':p['prompt_token_ids']} for p in batch],params,use_tqdm=True)
        assert len(outputs)==len(batch)
        with (dest/'results.jsonl').open('a') as f:
            for p,o in zip(batch,outputs):
                assert list(o.prompt_token_ids)==p['prompt_token_ids']
                ans=o.outputs[0]; assert ans.finish_reason in ('stop','length') and len(ans.token_ids)<=cfg['max_new_tokens']
                result={'id':p['id'],'index':p['index'],'benchmark':a.dataset,'domain':p['domain'],'response':ans.text,
                    'final_answer':final_answer(ans.text),'response_ids':list(ans.token_ids),'response_tokens':len(ans.token_ids),
                    'prompt_tokens':len(p['prompt_token_ids']),'finish_reason':ans.finish_reason,'stop_reason':ans.stop_reason,
                    'truncated':ans.finish_reason=='length','thinking_closed':'</think>' in ans.text,'timestamp':now(),'runtime_id':runtime_id}
                f.write(json.dumps(result,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno());existing.append(result)
        progress={'status':'generating','completed':len(existing),'total':len(prompts),'truncated':sum(r['truncated'] for r in existing),'mean_tokens':sum(r['response_tokens'] for r in existing)/len(existing),'updated':now()}
        dump(dest/'progress.json',progress);print(json.dumps(progress),flush=True)
    dump(dest/'generation-complete.json',{'n':len(existing),'timestamp':now(),'generation_seconds':time.monotonic()-started})
    llm.llm_engine.engine_core.shutdown()

if __name__=='__main__':main()
