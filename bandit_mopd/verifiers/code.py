from decimal import Decimal, InvalidOperation
import re
from .api import code_tests
from .text import final_response
from .wasm import WasmSandbox


def output_matches(actual, expected, tolerance='0.000001'):
    a, b = actual.split(), expected.split()
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if x == y:
            continue
        # Integers require exact equality, including very large integers.
        if re.fullmatch(r'[+-]?\d+', x) and re.fullmatch(r'[+-]?\d+', y):
            if int(x) != int(y):
                return False
            continue
        try:
            p, q = Decimal(x), Decimal(y)
            if not p.is_finite() or not q.is_finite() or abs(p-q) > Decimal(tolerance)*max(Decimal(1), abs(q)):
                return False
        except InvalidOperation:
            return False
    return True


def score(row, response, config):
    inputs, outputs = code_tests(row)
    text = final_response(response)
    blocks = re.findall(r'```(?:python|py)\s*\n(.*?)```', text, re.S | re.I)
    if not blocks:
        return {'score': 0.0, 'correct': False, 'reason': 'missing_python_code_block',
                'tests_total': len(inputs), 'tests_passed': 0}
    results = WasmSandbox(config).run(blocks[-1], inputs)
    checks = [r['status'] == 'ok' and output_matches(r['stdout'], gold,
              str(config.get('float_tolerance', '0.000001'))) for r, gold in zip(results, outputs)]
    # Gold outputs stay in the trusted judge; never enter the WASM guest.
    return {'score': float(all(checks)), 'correct': all(checks), 'backend': 'wasmtime',
            'tests_total': len(inputs), 'tests_passed': sum(checks),
            'test_results': [{'passed': passed, 'status': r['status'], 'seconds': r['seconds'],
                              'stderr': r['stderr'][:1000]} for passed, r in zip(checks, results)]}
