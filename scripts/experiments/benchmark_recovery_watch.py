"""Finalize the queue only after the independent post-LCB runner succeeds."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "experiments"))
from benchmark_generate import dump, now


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg_path = Path(args.config).resolve()
    cfg = json.loads(cfg_path.read_text())
    out = ROOT / cfg["output"]
    state_path = out / "recovery-watch-state.json"
    dump(state_path, {"status": "waiting_for_parallel_runner", "pid": os.getpid(), "started": now()})
    while True:
        parallel = json.loads((out / "parallel-remaining-state.json").read_text())
        if parallel["status"] == "failed":
            dump(state_path, {"status": "failed", "error": parallel.get("error"), "updated": now()})
            raise RuntimeError("parallel benchmark runner failed")
        if parallel["status"] == "complete":
            break
        time.sleep(20)
    command = [sys.executable, "-u", str(ROOT / "scripts" / "experiments" / "benchmark_finalize_queue.py"), "--config", str(cfg_path)]
    with (out / "recovery-watch.log").open("a") as log:
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        dump(state_path, {"status": "failed", "error": "finalizer exited nonzero", "updated": now()})
        raise SystemExit(result.returncode)
    dump(state_path, {"status": "complete", "updated": now()})


if __name__ == "__main__":
    main()
