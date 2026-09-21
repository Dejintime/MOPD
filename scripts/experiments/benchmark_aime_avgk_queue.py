"""Run the two isolated AIME avg@k jobs sequentially and retain queue state."""
import argparse
import fcntl
import json
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_generate import dump, now


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    cfg = json.loads(config_path.read_text())
    output = ROOT / cfg["output"]
    output.mkdir(parents=True, exist_ok=True)
    with (output / "queue.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = output / "queue-state.json"
        state = {"started": now(), "status": "running", "datasets": {}}
        dump(state_path, state)
        for dataset in ("aime24", "aime25"):
            complete = output / dataset / "complete.json"
            if complete.exists():
                state["datasets"][dataset] = "complete"
                dump(state_path, state)
                continue
            state["active_dataset"] = dataset
            state["datasets"][dataset] = "running"
            dump(state_path, state)
            log_path = output / dataset / "run.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with log_path.open("a") as log:
                    subprocess.run(
                        [sys.executable, "-u", str(Path(__file__).with_name("benchmark_aime_avgk.py")),
                         "--config", str(config_path), "--dataset", dataset],
                        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True,
                    )
            except Exception:
                state["status"] = "failed"
                state["datasets"][dataset] = "failed"
                state["failed_dataset"] = dataset
                state["error"] = traceback.format_exc()
                state["updated"] = now()
                dump(state_path, state)
                raise
            state["datasets"][dataset] = "complete"
            state["updated"] = now()
            dump(state_path, state)
        state["status"] = "complete"
        state.pop("active_dataset", None)
        state["completed"] = now()
        dump(state_path, state)


if __name__ == "__main__":
    main()
