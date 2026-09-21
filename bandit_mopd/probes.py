"""Paired, domain-balanced evaluations for manuscript Eq.22."""
from collections import Counter
import hashlib
import math
import time
from .verifiers import VerifierError, VerifierSuite, validate_row
from .prompting import render_rollout_prompt


class ProbeEvaluator:
    def __init__(self, rows, tokenizer, cfg, suite=None):
        self.cfg = cfg
        self.settings = cfg['probe']
        self.domains = sorted(cfg['domain_filter'])
        self.size = self.settings['samples_per_domain']
        if not isinstance(self.size, int) or self.size < 1:
            raise ValueError('probe.samples_per_domain must be a positive integer')
        if self.settings.get('scope', 'all_domains') not in ('all_domains', 'current_domain'):
            raise ValueError('Unknown probe scope')
        scale = self.settings.get('gain_scale', 1.0)
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError('probe.gain_scale must be positive and finite')
        self.suite = suite or VerifierSuite(self.settings['verifiers'])
        self.pools = {d: [] for d in self.domains}
        overlong = Counter()
        seen = set()
        for row in rows:
            if row.get('usage') == 'final_evaluation_only':
                raise VerifierError('Final benchmarks cannot supply task-gain feedback')
            d = row['domain']
            if d not in self.pools:
                continue
            if row['id'] in seen:
                raise VerifierError('Duplicate probe ID')
            seen.add(row['id'])
            validate_row(row)
            length = len(render_rollout_prompt(tokenizer, row['messages'], cfg, tokenize=True,
                         add_generation_prompt=True, return_dict=False))
            if length > self.settings.get('max_prompt_tokens', cfg['max_prompt_tokens']):
                overlong[d] += 1
                continue
            self.pools[d].append(row)
        for d, pool in self.pools.items():
            pool.sort(key=lambda r: hashlib.sha256(f"{cfg['seed']}:probe:{r['id']}".encode()).hexdigest())
            if len(pool) < self.size:
                raise VerifierError(f'{d}: need {self.size} usable probes, found {len(pool)}')
        self.suite.check_ready(self.domains)
        self.manifest = {'scope': self.settings.get('scope', 'all_domains'),
                         'samples_per_domain': self.size, 'filtered_overlong': dict(overlong),
                         'eligible_ids': {d: [r['id'] for r in pool] for d, pool in self.pools.items()},
                         'aggregation': 'equal_weight_domain_mean',
                         'normalization': 'clip((post-pre)/gain_scale, -1, 1)',
                         'gain_scale': scale}

    def batch(self, step, domain):
        domains = self.domains if self.settings.get('scope', 'all_domains') == 'all_domains' else [domain]
        return [self.pools[d][(step*self.size+j) % len(self.pools[d])]
                for d in domains for j in range(self.size)]

    def evaluate(self, model, tokenizer, rows, rollout):
        start = time.perf_counter()
        rollout_cfg = dict(self.cfg)
        for key in ('max_prompt_tokens', 'max_new_tokens'):
            rollout_cfg[key] = self.settings.get(key, self.cfg[key])
        records = []
        # Greedy decoding and identical rows/settings pair the two evaluations.
        # Do not send answers, constraints metadata, or unit tests to the model.
        for row in rows:
            model_row = {'id': row['id'], 'domain': row['domain'], 'messages': row['messages']}
            _, _, response, truncated = rollout(model, tokenizer, model_row, rollout_cfg, greedy=True)
            result = self.suite.score(row, response)
            records.append({'id': row['id'], 'domain': row['domain'], 'response': response,
                            'truncated': truncated, **result})
        domains = sorted({r['domain'] for r in records})
        scores = {d: sum(r['score'] for r in records if r['domain'] == d) /
                     sum(r['domain'] == d for r in records) for d in domains}
        return {'score': sum(scores.values())/len(scores), 'domain_scores': scores,
                'records': records, 'seconds': time.perf_counter()-start}

    def gain(self, before, after):
        keys = lambda x: [(r['domain'], r['id']) for r in x['records']]
        if keys(before) != keys(after):
            raise VerifierError('Task gain requires identical pre/post probe IDs in the same order')
        delta = after['score'] - before['score']
        scale = self.settings.get('gain_scale', 1.0)
        return {'raw_gain': delta, 'normalized_gain': max(-1., min(1., delta/scale)),
                'before': before['score'], 'after': after['score'],
                'domain_gains': {d: after['domain_scores'][d]-s for d, s in before['domain_scores'].items()},
                'probe_ids': [r['id'] for r in before['records']]}


def main():
    """Audit the probe pool and verifier runtime without loading a student/GPU."""
    import argparse
    import importlib.metadata
    import json
    import os
    from pathlib import Path
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    from transformers import AutoTokenizer
    from .data import read_jsonl, identity
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--audit-output', required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    rows = read_jsonl(cfg['probe_data'])
    train_ids = {identity(r['messages']) for r in read_jsonl(cfg['train_data'])}
    if train_ids & {identity(r['messages']) for r in rows}:
        raise VerifierError('Train/probe prompt leakage')
    tok = AutoTokenizer.from_pretrained(cfg['student'], local_files_only=True)
    evaluator = ProbeEvaluator(rows, tok, cfg)
    report = {'status': 'passed', 'eligible_counts': {d: len(p) for d, p in evaluator.pools.items()},
              'manifest': evaluator.manifest, 'model_loaded': False,
              'versions': {name: importlib.metadata.version(name) for name in
                           ('wasmtime', 'math-verify', 'nltk', 'langdetect', 'immutabledict', 'absl-py')}}
    output = Path(args.audit_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != 'manifest'}, indent=2))


if __name__ == '__main__':
    main()
