"""Start the SFT probe only after the active teacher benchmark queue completes."""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEACHER_STATE = ROOT / "analysis/2026-09-16-teacher-benchmark-queue/queue-state.json"
CONFIG = ROOT / "analysis/2026-09-17-sft-probe-eval-8k/config.json"
OUTPUT = ROOT / "analysis/2026-09-17-sft-probe-eval-8k"


def write_state(status, **extra):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "queue-state.json").write_text(
        json.dumps({"status": status, **extra}, ensure_ascii=False, indent=2) + "\n"
    )


def main():
    while True:
        state = json.loads(TEACHER_STATE.read_text())
        status = state.get("status")
        if status == "local_complete_remote_code_evaluation_pending":
            break
        if status in {"failed", "cancelled"}:
            raise RuntimeError(f"Teacher queue ended unexpectedly: {status}")
        write_state("waiting_for_teacher_queue", teacher_status=status, active=state.get("active"))
        time.sleep(30)
    write_state("running")
    log = OUTPUT / "run.log"
    with log.open("a") as stream:
        subprocess.run(
            [sys.executable, "-u", str(ROOT / "scripts/experiments/probe_vllm.py"), "--config", str(CONFIG)],
            cwd=ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )
    write_state("complete")


if __name__ == "__main__":
    main()
