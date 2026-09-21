import json
import math
import os
from pathlib import Path
import subprocess
import sys


class VerifierError(RuntimeError):
    """Invalid metadata or broken infrastructure: never silently reward as zero."""


def code_tests(row):
    value = (row.get('metadata', {}).get('verifier_metadata') or {}).get('unit_tests')
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict) or value.get('fn_name'):
        raise VerifierError('Code probes require stdin/stdout unit_tests')
    inputs, outputs = value.get('inputs'), value.get('outputs')
    if (not isinstance(inputs, list) or not inputs or not isinstance(outputs, list)
            or len(inputs) != len(outputs) or not all(isinstance(s, str) for s in inputs + outputs)):
        raise VerifierError('Missing/misaligned code inputs and outputs')
    return inputs, outputs


def validate_row(row):
    domain = row['domain']
    if domain in ('math', 'science'):
        if row.get('answer') is None or not str(row['answer']).strip():
            raise VerifierError(f'{domain}: missing gold answer')
    elif domain == 'if':
        from third_party.ifevalg.instructions_registry import INSTRUCTION_DICT
        ids = row.get('metadata', {}).get('instruction_id_list')
        kwargs = row.get('metadata', {}).get('kwargs')
        if not isinstance(ids, list) or not ids or not isinstance(kwargs, list) or len(ids) != len(kwargs):
            raise VerifierError('IF instruction IDs and kwargs must align exactly')
        if any(name not in INSTRUCTION_DICT for name in ids):
            raise VerifierError(f'Unsupported IF instruction: {set(ids) - set(INSTRUCTION_DICT)}')
        if any(k is not None and not isinstance(k, dict) for k in kwargs):
            raise VerifierError('IF kwargs must be dictionaries or null')
    elif domain == 'code':
        code_tests(row)
    else:
        raise VerifierError(f'No verifier for domain {domain}')


class VerifierSuite:
    def __init__(self, config):
        self.config = config

    def check_ready(self, domains):
        if 'math' in domains:
            import math_verify  # noqa: F401
        if 'if' in domains:
            import nltk
            from third_party.ifevalg import instructions_registry  # noqa: F401
            nltk.data.find('tokenizers/punkt_tab/english')
        if 'code' in domains:
            from .wasm import WasmSandbox
            WasmSandbox(self.config['code']).check_ready()

    def score(self, row, response):
        validate_row(row)
        if row['domain'] == 'code':
            from .code import score
            result = score(row, response, self.config['code'])
        else:
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1')
            timeout = self.config.get('text_timeout_seconds', 20)
            try:
                completed = subprocess.run([sys.executable, '-m', 'bandit_mopd.verifiers.worker'],
                    input=json.dumps({'row': row, 'response': response}), text=True,
                    capture_output=True, timeout=timeout, env=env,
                    cwd=Path(__file__).resolve().parents[2])
            except subprocess.TimeoutExpired as exc:
                raise VerifierError(f'{row["domain"]} verifier timed out; task gain aborted') from exc
            try:
                payload = json.loads(completed.stdout)
                result = payload['result']
            except (ValueError, KeyError) as exc:
                raise VerifierError(f'{row["domain"]} verifier failed: {completed.stdout[-1000:]} {completed.stderr[-1000:]}') from exc
            if completed.returncode:
                raise VerifierError(f'Verifier exited with {completed.returncode}')
        value = result.get('score')
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise VerifierError(f'Invalid verifier score: {value}')
        return result
