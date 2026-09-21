"""Only WASM guest instructions execute submitted code. No exec/eval on host."""
import json
from pathlib import Path
import resource
import sys
import tempfile
import threading
import time
import wasmtime


def run(payload):
    cfg = payload['config']
    runtime = Path(cfg['runtime_dir'])
    resource.setrlimit(resource.RLIMIT_CPU, (int(60 + len(payload['inputs'])*(cfg['timeout_seconds']+1)),)*2)
    config = wasmtime.Config()
    config.epoch_interruption = True
    config.cache = True
    engine = wasmtime.Engine(config)
    module = wasmtime.Module.from_file(engine, str(runtime/'python.wasm'))
    linker = wasmtime.Linker(engine)
    linker.define_wasi()
    results = []
    for input_text in payload['inputs']:
        with tempfile.TemporaryDirectory(prefix='mopd-wasi-') as directory:
            stdin = Path(directory)/'stdin'
            stdin.write_text(input_text)
            stdout, stderr = bytearray(), bytearray()
            overflow = threading.Event()
            expired = threading.Event()

            def capture(buffer):
                def write(data):
                    remaining = max(0, cfg['max_output_bytes']-len(buffer))
                    buffer.extend(data[:remaining])
                    if len(data) > remaining:
                        overflow.set()
                        engine.increment_epoch()
                    return len(data)
                return write

            wasi = wasmtime.WasiConfig()
            wasi.argv = ['python', '-B', '-s', '-P', '-c', payload['code']]
            # No inherited host env, stdio, sockets, project paths or credentials.
            wasi.env = [('PYTHONHOME', '/'), ('PYTHONHASHSEED', '0')]
            wasi.preopen_dir(str(runtime/'lib'), '/lib', fs_mutable=False)
            wasi.stdin_file = str(stdin)
            wasi.stdout_custom = capture(stdout)
            wasi.stderr_custom = capture(stderr)
            store = wasmtime.Store(engine)
            store.set_limits(memory_size=cfg['memory_mb']*1024**2, memories=1, instances=1)
            store.set_wasi(wasi)
            store.set_epoch_deadline(1)

            def interrupt():
                expired.set()
                engine.increment_epoch()

            timer = threading.Timer(cfg['timeout_seconds'], interrupt)
            timer.daemon = True
            timer.start()
            start = time.perf_counter()
            status, exit_code = 'ok', 0
            try:
                instance = linker.instantiate(store, module)
                instance.exports(store)['_start'](store)
            except wasmtime.ExitTrap as exc:
                exit_code = exc.code
                if exit_code:
                    status = 'runtime_error'
            except wasmtime.Trap:
                status = 'runtime_error'
            finally:
                timer.cancel()
                timer.join()
                store.close()
            if expired.is_set():
                status = 'timeout'
            if overflow.is_set():
                status = 'output_limit'
            results.append({'status': status, 'exit_code': exit_code,
                            'stdout': stdout.decode('utf-8', errors='replace'),
                            'stderr': stderr.decode('utf-8', errors='replace'),
                            'seconds': time.perf_counter()-start})
    return results


if __name__ == '__main__':
    try:
        print(json.dumps({'results': run(json.load(sys.stdin))}))
    except Exception as exc:
        print(json.dumps({'error': f'{type(exc).__name__}: {exc}'}))
        sys.exit(2)
