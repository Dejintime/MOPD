"""Start an LCB pass@1 run only after the specified AIME queue completes."""
import argparse
import fcntl
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_generate import dump, now


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aime-output", required=True)
    parser.add_argument("--lcb-config", required=True)
    args = parser.parse_args()
    aime_root = ROOT / args.aime_output
    config_path = Path(args.lcb_config).resolve()
    cfg = json.loads(config_path.read_text())
    lcb_root = ROOT / cfg["output"]
    lcb_root.mkdir(parents=True, exist_ok=True)
    with (lcb_root / "queue.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = lcb_root / "queue-state.json"
        state = {"status": "waiting_for_aime", "started": now(), "aime_output": args.aime_output}
        dump(state_path, state)
        while True:
            aime_state = json.loads((aime_root / "queue-state.json").read_text())
            if aime_state["status"] == "complete":
                break
            if aime_state["status"] == "failed":
                raise RuntimeError("AIME queue failed; LiveCodeBench was not started")
            state["updated"] = now()
            dump(state_path, state)
            time.sleep(60)
        state.update({"status": "generating", "aime_completed": now()})
        dump(state_path, state)
        generator = Path(__file__).with_name("benchmark_generate.py")
        evaluator = Path(__file__).with_name("benchmark_evaluate.py")
        dataset = "lcb_release_v5"
        with (lcb_root / "generation.log").open("a") as log:
            subprocess.run([sys.executable, "-u", str(generator), "--config", str(config_path), "--dataset", dataset], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        state.update({"status": "evaluating", "generation_completed": now()})
        dump(state_path, state)
        with (lcb_root / "evaluation.log").open("a") as log:
            subprocess.run([sys.executable, "-u", str(evaluator), "--config", str(config_path), "--dataset", dataset], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        state.update({"status": "complete", "completed": now()})
        dump(state_path, state)


if __name__ == "__main__":
    main()
