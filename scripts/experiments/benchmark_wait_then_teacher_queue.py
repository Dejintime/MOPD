"""Run teacher benchmarks only after the current student domain queue is done."""
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


def invoke(args, log):
    with log.open("a") as stream:
        subprocess.run(args, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-output", required=True)
    args = parser.parse_args()
    student = ROOT / args.student_output
    out = ROOT / "analysis/2026-09-16-teacher-benchmark-queue"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "queue.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = out / "queue-state.json"
        state = {"status": "waiting_for_student_domains", "started": now(), "datasets": {}}
        dump(state_path, state)
        required = [student / "lcb_release_v5" / "generation-complete.json", student / "gpqa_diamond" / "complete.json", student / "ifeval" / "complete.json", student / "ifbench" / "complete.json"]
        while not all(path.exists() for path in required):
            state["updated"] = now()
            dump(state_path, state)
            time.sleep(30)
        py = sys.executable
        exp = Path(__file__).resolve().parent
        configs = [
            ROOT / "analysis/2026-09-16-teacher-code-lcb-pass1-concise/config.json",
            ROOT / "analysis/2026-09-16-teacher-science-gpqa-concise/config.json",
            ROOT / "analysis/2026-09-16-teacher-if-concise/config.json",
        ]
        for config in configs:
            invoke([py, "-u", str(exp / "benchmark_generate.py"), "--config", str(config), "--prepare-only"], out / "preparation.log")
        jobs = [
            ("math", [py, "-u", str(exp / "benchmark_aime_avgk_queue.py"), "--config", str(ROOT / "analysis/2026-09-16-teacher-math-avg8-concise/config.json")]),
            ("code_generation", [py, "-u", str(exp / "benchmark_generate.py"), "--config", str(ROOT / "analysis/2026-09-16-teacher-code-lcb-pass1-concise/config.json"), "--dataset", "lcb_release_v5"]),
            ("science", [py, "-u", str(exp / "benchmark_generate.py"), "--config", str(ROOT / "analysis/2026-09-16-teacher-science-gpqa-concise/config.json"), "--dataset", "gpqa_diamond"]),
            ("if_generation", [py, "-u", str(exp / "benchmark_generate.py"), "--config", str(ROOT / "analysis/2026-09-16-teacher-if-concise/config.json"), "--dataset", "ifeval"]),
            ("ifbench_generation", [py, "-u", str(exp / "benchmark_generate.py"), "--config", str(ROOT / "analysis/2026-09-16-teacher-if-concise/config.json"), "--dataset", "ifbench"]),
        ]
        for name, command in jobs:
            state.update({"status": "running", "active": name}); state["datasets"][name] = "running"; dump(state_path, state)
            invoke(command, out / f"{name}.log")
            if name == "code_generation":
                state["datasets"][name] = "remote_evaluation_pending"; dump(state_path, state); continue
            if name == "science":
                invoke([py, "-u", str(exp / "benchmark_evaluate.py"), "--config", str(ROOT / "analysis/2026-09-16-teacher-science-gpqa-concise/config.json"), "--dataset", "gpqa_diamond"], out / "science-evaluation.log")
            if name in ("if_generation", "ifbench_generation"):
                dataset = "ifeval" if name == "if_generation" else "ifbench"
                invoke([py, "-u", str(exp / "benchmark_evaluate.py"), "--config", str(ROOT / "analysis/2026-09-16-teacher-if-concise/config.json"), "--dataset", dataset], out / f"{dataset}-evaluation.log")
            state["datasets"][name] = "complete"; dump(state_path, state)
        state.update({"status": "local_complete_remote_code_evaluation_pending", "completed": now()}); state.pop("active", None); dump(state_path, state)


if __name__ == "__main__":
    main()
