"""Summarize fixed probe inference, preserving paired question-level uncertainty."""
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[2]


def aggregate(records):
    result = {}
    for domain in ['code', 'if', 'math', 'science', 'overall']:
        rs = [r for r in records if domain == 'overall' or r['domain'] == domain]
        if not rs:
            continue
        n = len(rs)
        result[domain] = {'n': n, 'unique_questions': len({r['id'] for r in rs}),
            'correct': sum(r['result']['correct'] for r in rs),
            'acc': sum(r['result']['correct'] for r in rs)/n,
            'truncated': sum(r['truncated'] for r in rs),
            'truncation_rate': sum(r['truncated'] for r in rs)/n,
            'mean_total_tokens': sum(r['total_generated_tokens'] for r in rs)/n,
            'recovery_used': sum(r['recovery_used'] for r in rs)}
    return result


def bootstrap_pairs(before, after):
    a = {(r['id'], r['seed']): r for r in before}
    b = {(r['id'], r['seed']): r for r in after}
    assert set(a) == set(b), 'Paired question/seed mismatch'
    groups = defaultdict(lambda: defaultdict(list))
    won = lost = 0
    for key in a:
        x, y = a[key], b[key]
        dx = int(y['result']['correct'])-int(x['result']['correct'])
        won += dx == 1; lost += dx == -1
        groups[x['domain']][x['id']].append((dx, int(y['truncated'])-int(x['truncated']),
            y['total_generated_tokens']-x['total_generated_tokens']))
    # Average seeds first; resample questions within each of the four domains.
    blocks = [[tuple(sum(v[j] for v in values)/len(values) for j in range(3))
               for values in by_id.values()] for by_id in groups.values()]
    rng = random.Random(20260914)
    draws = [[], [], []]
    for _ in range(10000):
        sample = [rng.choice(domain) for domain in blocks for _ in domain]
        for j in range(3):
            draws[j].append(sum(x[j] for x in sample)/len(sample))
    intervals = {}
    for name, values in zip(['acc_delta', 'truncation_delta', 'mean_total_tokens_delta'], draws):
        values.sort()
        intervals[name] = [values[249], values[9749]]
    return {'paired_responses': len(a), 'unique_questions': sum(len(g) for g in groups.values()),
        'correctness_wins': won, 'correctness_losses': lost,
        'bootstrap_95pct_intervals': intervals, 'bootstrap_seed': 20260914,
        'bootstrap_replicates': 10000,
        'method': 'Question-cluster bootstrap stratified by domain; seeds averaged within question.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='analysis/2026-09-14-decoding-research')
    args = parser.parse_args()
    out = ROOT / args.root
    groups = defaultdict(list)
    for path in sorted(out.glob('*-seed*/verification.json')):
        if not (path.parent/'verification-complete.json').exists():
            continue
        rs = json.loads(path.read_text())
        for r in rs:
            groups[(r['split'], r['policy'])].append(r)
    historical = ROOT/'analysis/2026-09-14-system-prompt-probe'
    old = json.loads((historical/'all-verification.json').read_text())
    for condition, policy, run in [('before', 'baseline', ROOT/'analysis/2026-09-14-vllm-8k-probe/run'),
                                   ('after', 'historical_strict', historical/'run')]:
        with (run/'results.jsonl').open() as f:
            responses = {r['id']: r for r in map(json.loads, f) if r['mode'] == 'sampled'}
        for record in old:
            if record['condition'] != condition:
                continue
            p = responses[record['id']]
            groups[('development', policy)].append(dict(record, policy=policy, split='development',
                seed=42, total_generated_tokens=p['response_tokens'], response_tokens=p['response_tokens'], recovery_used=False))
    result = {'groups': {}, 'comparisons': {}}
    flat = []
    for (split, policy), rs in groups.items():
        result['groups'][f'{split}/{policy}'] = aggregate(rs)
        for r in rs:
            flat.append({k: r[k] for k in ['split', 'policy', 'seed', 'id', 'domain', 'truncated',
                         'total_generated_tokens', 'response_tokens', 'recovery_used']} | {
                         'correct': r['result']['correct'], 'reason': r['result'].get('reason', ''),
                         'tests_passed': r['result'].get('tests_passed', ''),
                         'tests_total': r['result'].get('tests_total', '')})
    for (split, policy), rs in groups.items():
        if split == 'holdout' and policy != 'baseline' and ('holdout', 'baseline') in groups:
            before = groups[('holdout', 'baseline')]
            if {(r['id'], r['seed']) for r in before} == {(r['id'], r['seed']) for r in rs}:
                result['comparisons'][policy] = bootstrap_pairs(before, rs)
    (out/'analysis.json').write_text(json.dumps(result, indent=2)+'\n')
    if flat:
        with (out/'per-response.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)
    print(json.dumps({k: v['overall'] for k, v in result['groups'].items()}, indent=2))
    print(json.dumps(result['comparisons'], indent=2))


if __name__ == '__main__':
    main()
