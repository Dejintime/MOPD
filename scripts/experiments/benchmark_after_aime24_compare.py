"""After a current AIME24 job finishes, stop its queue and run one comparison."""
import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_generate import dump, now


def matching_pids(fragment):
    rows = subprocess.check_output(["ps", "-eo", "pid=,args="], text=True).splitlines()
    return [int(row.strip().split(None, 1)[0]) for row in rows if fragment in row]


def stop_current_queue(fragment):
    for _ in range(3):
        pids = matching_pids(fragment)
        for pid in pids:
            if pid != os.getpid():
                os.kill(pid, signal.SIGTERM)
        time.sleep(2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--current-output", required=True)
    parser.add_argument("--comparison-config", required=True)
    args = parser.parse_args()
    current = ROOT / args.current_output
    config_path = Path(args.comparison_config).resolve()
    cfg = json.loads(config_path.read_text())
    out = ROOT / cfg["output"]
    out.mkdir(parents=True, exist_ok=True)
    with (out / "queue.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = out / "queue-state.json"
        state = {"status": "waiting_for_current_aime24", "started": now(), "current_output": args.current_output}
        dump(state_path, state)
        while not (current / "aime24" / "complete.json").exists():
            state["updated"] = now()
            dump(state_path, state)
            time.sleep(30)
        stop_current_queue("benchmark_aime_avgk_queue.py --config " + args.current_output)
        stop_current_queue("benchmark_aime_avgk.py")
        stop_current_queue("VLLM::EngineCore")
        state.update({"status": "generating", "current_aime24_completed": now()})
        dump(state_path, state)
        generator = Path(__file__).with_name("benchmark_aime_avgk.py")
        with (out / "run.log").open("a") as log:
            subprocess.run([sys.executable, "-u", str(generator), "--config", str(config_path), "--dataset", "aime24"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        state.update({"status": "complete", "completed": now()})
        dump(state_path, state)


if __name__ == "__main__":
    main()
