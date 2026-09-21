"""Normalize restored Nemotron JSONL and make immutable, disjoint student splits."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

DOMAINS = {
    'nano_v3_sft_profiled_dapo17k': 'math',
    'nano_v3_sft_profiled_skywork_no_omni': 'math',
    'nano_v3_sft_profiled_stem_mcqa': 'science',
    'nano_v3_sft_profiled_instruction_following': 'if',
    'nano_v3_sft_profiled_comp_coding_50tests': 'code',
    'nano_v3_sft_profiled_workbench': 'agent',
}


def read_jsonl(path):
    # str.splitlines() also splits U+2028/U+2029 inside valid JSON strings.
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def read_training_prompts(path):
    """Keep only fields consumed by distillation; do not retain large unit tests."""
    rows = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            row = json.loads(line)
            if row['domain'] == 'agent':
                raise ValueError('Agent data are excluded from the four-domain training scope')
            if row.get('usage') == 'final_evaluation_only':
                raise ValueError('Final evaluation data cannot supply training prompts')
            rows.append({key:row[key] for key in ('id','domain','messages')})
    return rows


def identity(messages):
    # Ignore whitespace/case when deduplicating; keep roles and every message.
    text = '\n'.join(m['role'] + ':' + re.sub(r'\s+', ' ', m['content']).strip().casefold()
                     for m in messages)
    return hashlib.sha256(text.encode()).hexdigest()


def normalize(row, source_index):
    domain = DOMAINS.get(row.get('dataset'))
    if domain is None:
        return None, 'unsupported_category'
    if row.get('_hf_placeholder'):
        return None, 'unrestored_math_placeholder'
    params = row.get('responses_create_params') or {}
    messages = params.get('input')
    if not isinstance(messages, list) or not messages:
        return None, 'missing_messages'
    if not all(isinstance(m.get('content'), str) and m.get('role') in ('system', 'user')
               for m in messages):
        return None, 'requires_multiturn_adapter'
    if domain == 'agent':
        return None, 'requires_workplace_environment'
    messages = [dict(m) for m in messages]
    # Match M2RL's math wrapper while preserving complete multi-message input.
    if domain == 'math':
        idx = next(i for i, m in enumerate(messages) if m['role'] == 'user')
        messages[idx]['content'] = ('Solve the following math problem step by step. '
            'The last line of your response should be of the form Answer: $Answer '
            '(without quotes) where $Answer is the answer to the problem.\n\n' + messages[idx]['content'])
    return {
        'id': identity(messages), 'domain': domain, 'messages': messages,
        'source': row['dataset'], 'source_index': source_index,
        'answer': row.get('expected_answer'),
        'metadata': {key: row.get(key) for key in ('template_metadata', 'instruction_id_list',
                     'kwargs', 'verifier_metadata', 'uuid', 'hash_id')},
    }, None


def prepare(input_path, output_dir, seed=42, smoke=False):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    groups, seen, rejected = defaultdict(list), set(), Counter()
    digest = hashlib.sha256()
    with open(input_path, 'rb') as f:
        for i, line in enumerate(f):
            digest.update(line)
            if not line.strip():
                continue
            record, reason = normalize(json.loads(line), i)
            if reason:
                rejected[reason] += 1
                continue
            if record['id'] in seen:
                rejected['duplicate_prompt'] += 1
                continue
            seen.add(record['id'])
            groups[record['domain']].append(record)
    if rejected['unrestored_math_placeholder'] and not smoke:
        raise ValueError('Restore mathematical placeholders with the pinned NVIDIA helper first.')
    if not groups:
        raise ValueError('No usable examples')
    splits = {key: [] for key in ('train', 'probe', 'dev')}
    for domain, rows in groups.items():
        rows.sort(key=lambda r: hashlib.sha256(f"{seed}:{r['id']}".encode()).hexdigest())
        if len(rows) < 3:
            raise ValueError(f'{domain}: need at least 3 distinct prompts')
        count = max(1, len(rows)//10)
        splits['probe'].extend(rows[:count])
        splits['dev'].extend(rows[count:2*count])
        splits['train'].extend(rows[2*count:])
    hashes = {}
    for split, rows in splits.items():
        p = out / f'{split}.jsonl'
        p.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
        hashes[split] = hashlib.sha256(p.read_bytes()).hexdigest()
    report = {'input_sha256': digest.hexdigest(), 'seed': seed, 'smoke_prefix_only': smoke,
              'counts': {s: dict(Counter(r['domain'] for r in rows)) for s, rows in splits.items()},
              'rejected': dict(rejected), 'split_sha256': hashes,
              'teacher_exposure': 'May have been used in teacher training; not teacher-unseen evaluation.',
              'test': 'External frozen benchmarks required; dev is not a final benchmark.'}
    (out/'manifest.json').write_text(json.dumps(report, indent=2))
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--smoke-prefix', action='store_true')
    a = p.parse_args()
    print(json.dumps(prepare(a.input, a.output, a.seed, a.smoke_prefix), indent=2))


if __name__ == '__main__':
    main()
