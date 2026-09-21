"""Fixed-weight inference ablations; gold metadata never enters generation."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bandit_mopd.prompting import render_rollout_prompt

DOMAINS = ['code', 'if', 'math', 'science']
BALANCED = ('Solve the task accurately and finish within 8192 output tokens. '
            'Use a concise, single solution; do not repeatedly restart or reconsider the same approach. '
            'Spend only the necessary space on reasoning and reserve enough space for the complete final answer. '
            'Preserve every requested format, language, and length constraint. '
            'For programming tasks, give a complete Python program in one closed python code block; '
            'check input parsing, variable definitions, edge cases and algorithmic complexity before finishing. '
            'For math, finish with the final answer in \\boxed{}. '
            'For multiple-choice questions, finish in the exact answer format requested. '
            'For instruction-following tasks, output only the requested deliverable with no extra commentary. '
            'Stop once the answer is complete.')
FINALIZE = ('The remaining generation budget is 4096 tokens. Finish the original task now. '
            'Use the useful work above, but do not repeat the analysis. '
            'Return a self-contained final answer in the originally requested format. '
            'For code, output one complete runnable Python program in a closed python code block, '
            'with all variables defined. For math, include the final answer in \\boxed{}. '
            'Do not discuss this budget instruction.')


def read_rows(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def policies():
    common = dict(temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, repetition_penalty=1.0)
    return {
        'baseline': {'sampling': dict(temperature=1., top_p=1., top_k=-1, repetition_penalty=1.)},
        'sampling': {'sampling': common},
        'balanced': {'sampling': common, 'rollout_system_prompt': BALANCED},
        'presence': {'sampling': dict(common, presence_penalty=1.5), 'rollout_system_prompt': BALANCED},
        'reserve': {'sampling': common, 'rollout_system_prompt': BALANCED, 'first_budget': 4096},
    }


def effective_policy(cfg, policy, domain):
    if 'per_domain' in policy:
        return cfg['policies'][policy['per_domain'][domain]]
    return policy


def summary(records):
    result = {}
    for name in sorted({r['policy'] for r in records}):
        result[name] = {}
        for domain in [*DOMAINS, 'overall']:
            rs = [r for r in records if r['policy'] == name and (domain == 'overall' or r['domain'] == domain)]
            if rs:
                result[name][domain] = {'n': len(rs), 'truncated': sum(r['truncated'] for r in rs),
                    'mean_generated_tokens': sum(r['total_generated_tokens'] for r in rs)/len(rs),
                    'recovered_requests': sum(r['recovery_used'] for r in rs),
                    'finish_reasons': dict(Counter(r['finish_reason'] for r in rs))}
    return result


def prepare(cfg, tokenizer, out):
    # Replay frozen historical holdout IDs verbatim; never reselect after scoring.
    dev = read_rows(ROOT / cfg['development_samples'])
    heldout = read_rows(ROOT / cfg['fixed_holdout_samples'])
    assert len(heldout) == 24 and Counter(r['domain'] for r in heldout) == dict.fromkeys(DOMAINS, 6)
    assert not ({r['id'] for r in dev} & {r['id'] for r in heldout})
    assert len({r['id'] for r in heldout}) == len(heldout)
    selected = []
    for row in heldout:
        assert row.get('usage') != 'final_evaluation_only'
        lengths = {}
        for name, policy in cfg['policies'].items():
            prompt = render_rollout_prompt(tokenizer, row['messages'], dict(policy, enable_thinking=False),
                                           tokenize=False, add_generation_prompt=True)
            lengths[name] = len(tokenizer.encode(prompt, add_special_tokens=False))
            assert lengths[name] <= cfg['max_prompt_tokens'], (row['id'], lengths)
            assert prompt.endswith('<think>\n\n</think>\n\n')
        selected.append({'id': row['id'], 'domain': row['domain'], 'prompt_tokens': lengths})
    for name, rows in [('development', dev), ('holdout', heldout)]:
        path = out / f'{name}-samples.jsonl'
        data = ''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows)
        if path.exists():
            assert path.read_text() == data
        else:
            path.write_text(data)
    dump(out/'manifest.json', {
        'source_sha256': hashlib.sha256((ROOT/cfg['probe_data']).read_bytes()).hexdigest(),
        'frozen_samples_sha256': hashlib.sha256((ROOT/cfg['fixed_holdout_samples']).read_bytes()).hexdigest(),
        'selection': 'Exact 24 previously frozen held-out IDs, 6/domain; no reselection or input truncation.',
        'selected': selected, 'seeds': cfg.get('seeds', [42]), 'enable_thinking': False, 'max_new_tokens': 8192,
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'model_config_sha256': hashlib.sha256((Path(cfg['student'])/'config.json').read_bytes()).hexdigest(),
        'versions': {k: importlib.metadata.version(k) for k in ['vllm', 'torch', 'transformers']}})
    return {'development': dev, 'holdout': heldout}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--split', choices=['development', 'holdout'], default='development')
    parser.add_argument('--policies', nargs='+', required=True)
    parser.add_argument('--seeds', nargs='+', type=int, default=[42])
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    cfg['seeds'] = args.seeds
    out = ROOT / cfg['output']; out.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg['student'], local_files_only=True)
    split_rows = prepare(cfg, tokenizer, out)
    dump(out/'config.json', cfg)
    if args.prepare_only:
        return
    rows = split_rows[args.split]
    jobs = [(name, seed, out/f'{args.split}-{name}-seed{seed}')
            for seed in args.seeds for name in args.policies]
    for _, _, path in jobs:
        if path.exists():
            raise FileExistsError(f'Immutable output already exists: {path}')
    from vllm import LLM, SamplingParams
    llm = LLM(model=cfg['student'], seed=42, **cfg['vllm'])
    for name, seed, dest in jobs:
        dest.mkdir()
        policy = cfg['policies'][name]
        row_policies = [effective_policy(cfg, policy, r['domain']) for r in rows]
        prompts = [render_rollout_prompt(tokenizer, r['messages'], dict(p, enable_thinking=False),
                    tokenize=False, add_generation_prompt=True) for r, p in zip(rows, row_policies)]
        ids = [tokenizer.encode(p, add_special_tokens=False) for p in prompts]
        assert max(map(len, ids)) <= cfg['max_prompt_tokens']
        assert all(p.endswith('<think>\n\n</think>\n\n') for p in prompts)
        dump(dest/'prompts.json', [{'id': r['id'], 'prompt': p} for r, p in zip(rows, prompts)])
        first_budgets = [p.get('first_budget', 8192) for p in row_policies]
        params = [SamplingParams(max_tokens=first_budgets[i], seed=seed+i, ignore_eos=False,
                                **p['sampling']) for i, p in enumerate(row_policies)]
        dump(dest/'run-config.json', {'policy': policy, 'seed': seed, 'split': args.split,
             'total_generation_budget': 8192, 'first_budgets': first_budgets,
             'sampling_params': [str(p) for p in params]})
        print(json.dumps({'starting': str(dest), 'n': len(rows)}), flush=True)
        start = time.perf_counter()
        outputs = llm.generate([{'prompt_token_ids': x} for x in ids], params, use_tqdm=True)
        # Save the first stage before any longer-context recovery call can fail.
        dump(dest/'first-stage.json', [{'id': row['id'], 'response': output.outputs[0].text,
             'response_ids': list(output.outputs[0].token_ids),
             'finish_reason': output.outputs[0].finish_reason,
             'stop_reason': output.outputs[0].stop_reason}
             for row, output in zip(rows, outputs)])
        recover = [i for i, r in enumerate(outputs) if first_budgets[i] < 8192 and r.outputs[0].finish_reason == 'length']
        final_outputs = {}
        if recover:
            recovery_ids, recovery_params, recovery_prompts = [], [], []
            for i in recover:
                # Preserve the exact initial prefix and generated draft token IDs.
                # A new user turn requests a complete final answer; no gold or test feedback.
                suffix = '<|im_end|>\n<|im_start|>user\n'+FINALIZE+'<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
                draft = outputs[i].outputs[0]
                prefix = ids[i] + list(draft.token_ids) + tokenizer.encode(suffix, add_special_tokens=False)
                budget = 8192 - len(draft.token_ids)
                assert budget == 4096 and len(prefix)+budget <= cfg['vllm']['max_model_len']
                recovery_ids.append({'prompt_token_ids': prefix})
                recovery_params.append(SamplingParams(max_tokens=budget, seed=seed+i+100000,
                    ignore_eos=False, **row_policies[i]['sampling']))
                recovery_prompts.append({'id': rows[i]['id'], 'prompt_token_ids': prefix,
                                         'max_tokens': budget, 'sampling_params': str(recovery_params[-1])})
            dump(dest/'recovery-prompts.json', recovery_prompts)
            rs = llm.generate(recovery_ids, recovery_params, use_tqdm=True)
            final_outputs = dict(zip(recover, rs))
        records = []
        for i, (row, output) in enumerate(zip(rows, outputs)):
            draft = output.outputs[0]
            answer = final_outputs.get(i, output).outputs[0]
            recovered = i in final_outputs
            total = len(draft.token_ids) + (len(answer.token_ids) if recovered else 0)
            assert total <= 8192
            record = {'id': row['id'], 'domain': row['domain'], 'policy': name,
                'component_policy': policy.get('per_domain', {}).get(row['domain'], name),
                'seed': seed, 'split': args.split, 'prompt_tokens': len(ids[i]),
                'response': answer.text, 'response_ids': list(answer.token_ids),
                'response_tokens': len(answer.token_ids), 'total_generated_tokens': total,
                'finish_reason': answer.finish_reason, 'stop_reason': answer.stop_reason,
                'truncated': answer.finish_reason == 'length', 'recovery_used': recovered,
                'first_finish_reason': draft.finish_reason,
                'draft_response': draft.text if recovered else None,
                'draft_ids': list(draft.token_ids) if recovered else None}
            assert answer.finish_reason in ['stop', 'length']
            records.append(record)
        (dest/'results.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in records))
        dump(dest/'summary.json', summary(records))
        dump(dest/'complete.json', {'status': 'complete', 'n': len(records),
             'seconds': time.perf_counter()-start, 'timestamp': datetime.now(timezone.utc).isoformat()})
        print(json.dumps({'complete': str(dest), 'summary': summary(records)}), flush=True)


if __name__ == '__main__':
    main()
