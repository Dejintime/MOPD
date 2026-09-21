"""Unprivileged CPython/WASI sandbox: no host code execution fallback."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from .api import VerifierError


class WasmSandbox:
    def __init__(self, config):
        self.config = config
        if config.get('backend') != 'wasmtime':
            raise VerifierError('Code verifier requires the wasmtime backend')

    def check_ready(self):
        import wasmtime  # noqa: F401
        root = Path(self.config['runtime_dir'])
        manifest = json.loads((root/'manifest.json').read_text())
        for name, expected in manifest['files_sha256'].items():
            if hashlib.sha256((root/name).read_bytes()).hexdigest() != expected:
                raise VerifierError(f'WASI runtime integrity mismatch: {name}')
        result = self.run("import sys, math; print(sys.platform); print(math.isqrt(81))", [''])
        if len(result) != 1 or result[0]['status'] != 'ok' or result[0]['stdout'].split() != ['wasi', '9']:
            raise VerifierError(f'WASI self-test failed: {result}')

    def run(self, code, inputs):
        if not inputs:
            raise VerifierError('No code test inputs')
        config = dict(self.config, runtime_dir=str(Path(self.config['runtime_dir']).resolve()))
        timeout = 60 + len(inputs)*(config['timeout_seconds']+1)
        try:
            result = subprocess.run([sys.executable, '-m', 'bandit_mopd.verifiers.wasm_worker'],
                input=json.dumps({'config': config, 'code': code, 'inputs': inputs}),
                text=True, capture_output=True, timeout=timeout,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', RAYON_NUM_THREADS='1'),
                cwd=Path(__file__).resolve().parents[2])
        except subprocess.TimeoutExpired as exc:
            raise VerifierError('Sandbox worker exceeded its outer deadline') from exc
        try:
            output = json.loads(result.stdout)
        except ValueError as exc:
            raise VerifierError(f'Sandbox worker failed: {result.stderr[-1000:]}') from exc
        if result.returncode or 'error' in output:
            raise VerifierError(f'Sandbox infrastructure failure: {output}')
        if len(output['results']) != len(inputs):
            raise VerifierError('Sandbox returned wrong test count')
        return output['results']
