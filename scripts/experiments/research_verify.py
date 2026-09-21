"""Judge completed research generations without altering or regenerating them."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bandit_mopd.verifiers import VerifierSuite


def rows(path):
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='analysis/2026-09-14-decoding-research')
    args = parser.parse_args()
    out = ROOT/args.root
    cfg = {'text_timeout_seconds': 20, 'code': {
        'backend': 'wasmtime', 'runtime_dir': str(ROOT/'data/verifier_runtime/python-3.13.15-wasi'),
        'timeout_seconds': 5, 'memory_mb': 512, 'max_output_bytes': 1048576,
        'float_tolerance': '0.000001'}}
    suite = VerifierSuite(cfg)
    pending = [p for p in sorted(out.glob('*-seed*/results.jsonl'))
               if (p.parent/'complete.json').exists() and not (p.parent/'verification-complete.json').exists()]
    if not pending:
        print('No unjudged completed runs.', flush=True)
        return
    suite.check_ready(['code', 'if', 'math', 'science'])
    for source in pending:
        predictions = rows(source)
        samples = {r['id']: r for r in rows(out/f'{predictions[0]["split"]}-samples.jsonl')}
        dest = source.parent
        records = []
        for prediction in predictions:
            record = {k: prediction[k] for k in ['id', 'domain', 'policy', 'seed', 'split',
                'truncated', 'response_tokens', 'total_generated_tokens', 'recovery_used']}
            result = suite.score(samples[prediction['id']], prediction['response'])
            record['result'] = result
            records.append(record)
            (dest/'verification.json').write_text(json.dumps(records, indent=2)+'\n')
            print(json.dumps({**{k: v for k, v in record.items() if k != 'result'},
                 'correct': result['correct'], 'reason': result.get('reason'),
                 'tests_passed': result.get('tests_passed'), 'tests_total': result.get('tests_total'),
                 'test_statuses': dict(Counter(t['status'] for t in result.get('test_results', [])))}), flush=True)
        summary = {}
        for domain in ['code', 'if', 'math', 'science', 'overall']:
            rs = [r for r in records if domain == 'overall' or r['domain'] == domain]
            summary[domain] = {'n': len(rs), 'correct': sum(r['result']['correct'] for r in rs),
                'truncated': sum(r['truncated'] for r in rs),
                'mean_tokens': sum(r['total_generated_tokens'] for r in rs)/len(rs)}
        (dest/'verification-complete.json').write_text(json.dumps({
            'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'verifier_config': cfg, 'summary': summary, 'infrastructure_errors': 0}, indent=2)+'\n')
        print(json.dumps({'finished': str(dest), 'summary': summary}), flush=True)


if __name__ == '__main__':
    main()
