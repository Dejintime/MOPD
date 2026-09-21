import json
import os
from pathlib import Path
import pytest
from bandit_mopd.verifiers.wasm import WasmSandbox
from bandit_mopd.verifiers.code import score


@pytest.fixture(scope='module')
def sandbox_config():
    root = Path(os.environ.get('MOPD_WASI_RUNTIME', 'data/verifier_runtime/python-3.13.15-wasi'))
    if not root.is_dir():
        pytest.skip('Install the WASI runtime with setup_verifiers.py to run sandbox integration tests')
    return {'backend': 'wasmtime', 'runtime_dir': str(root.resolve()), 'timeout_seconds': 2,
            'memory_mb': 128, 'max_output_bytes': 8192}


def test_wasm_correct_wrong_runtime_error_and_new_instance(sandbox_config):
    s = WasmSandbox(sandbox_config); s.check_ready()
    r = {'metadata': {'verifier_metadata': {'unit_tests': {'inputs': ['2 3\n', '9 1\n'], 'outputs': ['5\n', '10\n']}}}}
    assert score(r, '```python\na,b=map(int,input().split()); print(a+b)\n```', sandbox_config)['score'] == 1
    assert score(r, '```python\nprint(5)\n```', sandbox_config)['score'] == 0
    assert s.run('raise ValueError("wrong")', [''])[0]['status'] == 'runtime_error'
    assert [r['stdout'].strip() for r in s.run('x=globals().get("x",0)+1;print(x)', ['', ''])] == ['1', '1']


def test_wasm_cannot_access_host_or_write_stdlib(sandbox_config, tmp_path):
    secret = tmp_path/'private.txt'; secret.write_text('HOST_SENTINEL')
    code = f'''import os
for path in [{str(secret)!r}, '/etc/passwd', '/proc/self/environ']:
    try:
        open(path).read()
        print('UNEXPECTED_READ')
    except OSError:
        print('blocked')
try:
    open('/lib/python3.13/mopd_sandbox_write_test', 'w').write('x')
    print('UNEXPECTED_WRITE')
except OSError:
    print('readonly')
print(os.environ.get('MOPD_TEST_SECRET', 'absent'))
try:
    import socket
    socket.socket()
    print('UNEXPECTED_SOCKET')
except (ImportError, OSError, AttributeError):
    print('no_socket')
'''
    os.environ['MOPD_TEST_SECRET'] = 'DO_NOT_INHERIT'
    try:
        result = WasmSandbox(sandbox_config).run(code, [''])[0]
    finally:
        os.environ.pop('MOPD_TEST_SECRET')
    assert result['status'] == 'ok', result
    assert result['stdout'].split() == ['blocked', 'blocked', 'blocked', 'readonly', 'absent', 'no_socket']


def test_wasm_limits(sandbox_config):
    s = WasmSandbox(sandbox_config)
    assert s.run('while True: pass', [''])[0]['status'] == 'timeout'
    assert s.run('while True: print("x"*1000)', [''])[0]['status'] == 'output_limit'
    assert s.run('x=bytearray(512*1024*1024)', [''])[0]['status'] == 'runtime_error'
