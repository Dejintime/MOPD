"""Offline student probe with configurable thinking mode and vLLM diagnostics.

This command performs inference only. It never supplies answers or verifier
metadata to the model, and never silently shortens an over-budget prompt.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bandit_mopd.prompting import render_rollout_prompt, rollout_messages

def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

def installed_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None

def choose_samples(source, tokenizer, cfg):
    pools = {d: [] for d in cfg['domains']}
    counts, overlong = Counter(), Counter()
    seen = set()
    digest = hashlib.sha256()
    with source.open('rb') as f:
        for line in f:  # JSON strings can contain Unicode line separators.
            digest.update(line)
            row = json.loads(line)
            if row['domain'] not in pools:
                continue
            if row.get('usage') == 'final_evaluation_only':
                raise ValueError('Final benchmarks cannot be used as probes')
            if row['id'] in seen:
                raise ValueError('Duplicate probe ID')
            seen.add(row['id'])
            d = row['domain']; counts[d] += 1
            prompt = render_rollout_prompt(tokenizer, row['messages'], cfg, tokenize=False,
                add_generation_prompt=True)
            if cfg['enable_thinking'] is True and not prompt.endswith('<think>\n'):
                # Match the benchmark generators for tokenizer templates that do
                # not inject Qwen's thinking prefix themselves.
                prompt += '<think>\n'
            ids = tokenizer.encode(prompt, add_special_tokens=False)
            if len(ids) > cfg['max_prompt_tokens']:
                overlong[d] += 1
                continue
            # Verify this Qwen3-derived student actually implements the selected mode.
            if cfg['enable_thinking'] is False and not prompt.endswith('<think>\n\n</think>\n\n'):
                raise ValueError('Tokenizer did not render the expected Qwen3 no-thinking prefix')
            if cfg['enable_thinking'] is True and not prompt.endswith('<think>\n'):
                raise ValueError('Tokenizer did not render the expected Qwen3 thinking prefix')
            rank = hashlib.sha256(f"{cfg['seed']}:probe:{row['id']}".encode()).hexdigest()
            pools[d].append((rank, row, prompt, ids))
    chosen = []
    for d, pool in pools.items():
        pool.sort(key=lambda x: x[0])
        if len(pool) < cfg['samples_per_domain']:
            raise ValueError(f'Not enough eligible probes in {d}')
        chosen.extend(pool[:cfg['samples_per_domain']])
    return chosen, {'source_sha256': digest.hexdigest(), 'domain_population': dict(counts),
        'filtered_overlong_prompts': dict(overlong),
        'selection': 'First N per domain by SHA256(seed:probe:id), matching existing probe ordering; no output-based selection.'}

def summarize(records, domains):
    summary = {}
    for mode in sorted({r['mode'] for r in records}):
        summary[mode] = {}
        for d in [*domains, 'overall']:
            rs = [r for r in records if r['mode']==mode and (d=='overall' or r['domain']==d)]
            if not rs:
                continue
            summary[mode][d] = {'samples':len(rs), 'truncated':sum(r['truncated'] for r in rs),
                'finish_reasons':dict(Counter(r['finish_reason'] for r in rs)),
                'mean_tokens':sum(r['response_tokens'] for r in rs)/len(rs),
                'max_tokens':max(r['response_tokens'] for r in rs),
                'generated_think_tags':sum(r['think_open_count']+r['think_close_count'] for r in rs),
                'empty_responses':sum(not r['response'].strip() for r in rs)}
    return summary

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args=parser.parse_args()
    cfg=json.loads(Path(args.config).read_text())
    if cfg['inference_backend']!='vllm' or not isinstance(cfg['enable_thinking'], bool):
        raise ValueError('This experiment requires vLLM and an explicit boolean enable_thinking setting')
    if cfg['vllm']['max_model_len'] < cfg['max_prompt_tokens']+cfg['max_new_tokens']:
        raise ValueError('Context length must accommodate full prompt plus output budget')
    out=ROOT/cfg['output'];out.mkdir(parents=True,exist_ok=True)
    if (out/'results.jsonl').exists():
        raise FileExistsError('Choose a new output directory; existing generations are immutable')
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(cfg['student'],local_files_only=True)
    source = ROOT/cfg.get('fixed_samples_data', cfg['probe_data'])
    chosen, manifest=choose_samples(source,tokenizer,cfg)
    if cfg.get('fixed_samples_data'):
        with source.open() as f: fixed_rows=[json.loads(line) for line in f if line.strip()]
        if [r['id'] for _,r,_,_ in chosen] != [r['id'] for r in fixed_rows]:
            raise ValueError('Fixed probes or their ordering changed')
        manifest['selection']='Exact prior probe IDs and order; no reselection.'
    manifest['sampling_source']=str(source)
    dump(out/'config.json',cfg)
    (out/'samples.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for _,r,_,_ in chosen))
    manifest.update(model=cfg['student'],inference_backend='vllm',enable_thinking=cfg['enable_thinking'],
        max_new_tokens=cfg['max_new_tokens'],max_model_len=cfg['vllm']['max_model_len'],
        prompt_policy='Original messages plus the explicitly configured system instruction. No answers, tests, or prompt truncation.',
        rollout_system_prompt=cfg.get('rollout_system_prompt',''),
        scope='Small probe completion audit, not a benchmark or training run.',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        model_config_sha256=hashlib.sha256((Path(cfg['student'])/'config.json').read_bytes()).hexdigest(),
        tokenizer_config_sha256=hashlib.sha256((Path(cfg['student'])/'tokenizer_config.json').read_bytes()).hexdigest(),
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        versions={name:installed_version(name) for name in ('vllm','torch','transformers')},
        selected=[{'id':r['id'],'domain':r['domain'],'prompt_tokens':len(ids),'no_thinking_prefix_verified':True}
                  for _,r,prompt,ids in chosen])
    dump(out/'manifest.json',manifest)
    dump(out/'rendered-prompts.json',[{'id':r['id'],'messages':rollout_messages(r['messages'],cfg),
        'prompt':prompt,'prompt_token_ids':ids} for _,r,prompt,ids in chosen])
    print(json.dumps({'prepared':manifest},ensure_ascii=False),flush=True)
    if args.prepare_only:
        return
    from vllm import LLM, SamplingParams
    started=time.perf_counter()
    llm=LLM(model=cfg['student'],seed=cfg['seed'],**cfg['vllm'])
    load_seconds=time.perf_counter()-started
    records=[];mode_seconds={}
    for mode in cfg['decode_modes']:
        settings=dict(cfg['sampling'])
        if mode=='greedy': settings['temperature']=0.0
        elif mode!='sampled': raise ValueError('Unsupported decode mode')
        params=[SamplingParams(max_tokens=cfg['max_new_tokens'], seed=cfg['seed']+i,
            ignore_eos=False,**settings) for i in range(len(chosen))]
        print(json.dumps({'starting_mode':mode,'samples':len(chosen),'max_tokens':cfg['max_new_tokens']}),flush=True)
        start=time.perf_counter()
        outputs=llm.generate([{'prompt_token_ids':ids} for _,r,p,ids in chosen],params,use_tqdm=True)
        mode_seconds[mode]=time.perf_counter()-start
        if len(outputs)!=len(chosen): raise RuntimeError('Wrong output count')
        for (_,r,prompt,ids),result in zip(chosen,outputs):
            assert list(result.prompt_token_ids)==ids
            answer=result.outputs[0]
            record={'id':r['id'],'domain':r['domain'],'mode':mode,'prompt_tokens':len(ids),
                'response_tokens':len(answer.token_ids),'response':answer.text,'response_ids':list(answer.token_ids),
                'finish_reason':answer.finish_reason,'stop_reason':answer.stop_reason,
                'last_token_id':answer.token_ids[-1] if answer.token_ids else None,
                'eos_token_id':tokenizer.eos_token_id,
                'truncated':answer.finish_reason=='length',
                'think_open_count':list(answer.token_ids).count(tokenizer.convert_tokens_to_ids('<think>')),
                'think_close_count':list(answer.token_ids).count(tokenizer.convert_tokens_to_ids('</think>')),
                'sampling_params':str(params[len(records)%len(chosen)]),
                'timestamp_utc':datetime.now(timezone.utc).isoformat()}
            if answer.finish_reason not in ('stop','length'): raise RuntimeError(f'Unexpected finish: {answer.finish_reason}')
            records.append(record)
            with (out/'results.jsonl').open('a') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
            print(json.dumps({k:record[k] for k in ('id','domain','mode','response_tokens','finish_reason','truncated')},ensure_ascii=False),flush=True)
        dump(out/'summary.json',summarize(records,cfg['domains']))
    dump(out/'timing.json',{'engine_load_seconds':load_seconds,'mode_seconds':mode_seconds})
    dump(out/'complete.json',{'status':'complete','records':len(records),'timestamp_utc':datetime.now(timezone.utc).isoformat()})
    print(json.dumps({'complete':summarize(records,cfg['domains'])},ensure_ascii=False),flush=True)

if __name__=='__main__': main()
