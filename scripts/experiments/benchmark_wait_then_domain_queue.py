"""Run the remaining benchmark suite after a designated generation completes."""
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
    parser.add_argument("--wait-output", required=True)
    parser.add_argument("--wait-dataset", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    wait_complete = ROOT / args.wait_output / args.wait_dataset / "complete.json"
    config_path = Path(args.config).resolve()
    cfg = json.loads(config_path.read_text())
    out = ROOT / cfg["output"]
    out.mkdir(parents=True, exist_ok=True)
    with (out / "queue.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = out / "queue-state.json"
        state = {"status": "waiting", "started": now(), "wait_for": str(wait_complete), "datasets": {}}
        dump(state_path, state)
        while not wait_complete.exists():
            state["updated"] = now()
            dump(state_path, state)
            time.sleep(30)
        generator = Path(__file__).with_name("benchmark_generate.py")
        evaluator = Path(__file__).with_name("benchmark_evaluate.py")
        for dataset in ("lcb_release_v5", "gpqa_diamond", "ifeval", "ifbench"):
            if (out / dataset / "complete.json").exists():
                state["datasets"][dataset] = "complete"
                dump(state_path, state)
                continue
            state.update({"status": "generating", "active_dataset": dataset})
            state["datasets"][dataset] = "generating"
            dump(state_path, state)
            with (out / dataset / "generation.log").open("a") as log:
                subprocess.run([sys.executable, "-u", str(generator), "--config", str(config_path), "--dataset", dataset], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            if dataset == "lcb_release_v5":
                state["datasets"][dataset] = "remote_evaluation_pending"
                state["status"] = "remote_lcb_evaluation_pending"
                dump(state_path, state)
                continue
            state["status"] = "evaluating"
            dump(state_path, state)
            with (out / dataset / "evaluation.log").open("a") as log:
                subprocess.run([sys.executable, "-u", str(evaluator), "--config", str(config_path), "--dataset", dataset], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            state["datasets"][dataset] = "complete"
            state["updated"] = now()
            dump(state_path, state)
        state.update({"status": "complete" if state["datasets"].get("lcb_release_v5") == "complete" else "local_complete_remote_lcb_evaluation_pending", "completed": now()})
        state.pop("active_dataset", None)
        dump(state_path, state)


if __name__ == "__main__":
    main()
