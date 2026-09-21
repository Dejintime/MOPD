#!/root/miniconda3/bin/python
"""Shut down this AutoDL instance after 10 continuous minutes of zero GPU memory."""

import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import time


IDLE_SECONDS = 600
POLL_SECONDS = 10
LOCK_PATH = Path('/root/autodl-tmp/gpu-idle-shutdown.lock')
GPU_QUERY = [
    '/usr/bin/nvidia-smi', '--query-gpu=memory.used',
    '--format=csv,noheader,nounits',
]


def gpu_memory_mib():
    result = subprocess.run(GPU_QUERY, capture_output=True, text=True,
                            check=True, timeout=15)
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise ValueError('nvidia-smi returned no GPUs')
    values = [int(line) for line in lines]
    if any(value < 0 for value in values):
        raise ValueError(f'Invalid GPU memory values: {values}')
    return values


class IdleTimer:
    def __init__(self, threshold=IDLE_SECONDS):
        self.threshold = threshold
        self.since = None

    def observe(self, memory_mib, now):
        if not memory_mib or any(value != 0 for value in memory_mib):
            self.since = None
            return False
        if self.since is None:
            self.since = now
        return now - self.since >= self.threshold


def log(message):
    print(f'{time.strftime("%Y-%m-%d %H:%M:%S")} {message}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-once', action='store_true',
                        help='Print current GPU memory and exit without shutting down')
    args = parser.parse_args()
    if args.check_once:
        print(gpu_memory_mib())
        return

    with LOCK_PATH.open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('GPU idle shutdown monitor is already running')
        lock.write(str(os.getpid()))
        lock.flush()
        timer = IdleTimer()
        log(f'Monitor started: all GPUs must use 0 MiB for {IDLE_SECONDS} seconds')
        while True:
            try:
                memory = gpu_memory_mib()
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                timer.observe(None, time.monotonic())
                log(f'GPU query failed; idle timer reset: {error}')
                time.sleep(POLL_SECONDS)
                continue

            was_idle = timer.since is not None
            expired = timer.observe(memory, time.monotonic())
            if timer.since is not None and not was_idle:
                log('GPU memory is 0 MiB; idle timer started')
            elif timer.since is None and was_idle:
                log(f'GPU memory increased to {memory} MiB; idle timer reset')

            if expired:
                try:
                    confirmation = gpu_memory_mib()
                except (OSError, ValueError, subprocess.SubprocessError) as error:
                    timer.observe(None, time.monotonic())
                    log(f'Final GPU query failed; idle timer reset: {error}')
                else:
                    if all(value == 0 for value in confirmation):
                        log('All GPUs remained at 0 MiB for 10 minutes; running /usr/bin/shutdown')
                        subprocess.run(['/usr/bin/shutdown'], check=True)
                        return
                    timer.observe(confirmation, time.monotonic())
                    log(f'Final GPU query showed {confirmation} MiB; idle timer reset')
            time.sleep(POLL_SECONDS)


if __name__ == '__main__':
    main()
