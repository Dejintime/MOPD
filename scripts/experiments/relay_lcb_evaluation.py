"""Relay completed LCB generations from the GPU host to the dedicated judge host."""
import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def stamp():
    return datetime.now(timezone.utc).isoformat()


def run(command):
    subprocess.run(command, check=True)


def write_state(path, state):
    state["updated"] = stamp()
    temp = path.with_suffix(".json.part")
    temp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu-host", default="root@connect.westc.seetacloud.com")
    parser.add_argument("--gpu-port", default="14471")
    parser.add_argument("--gpu-root", default="/root/autodl-tmp/MOPD")
    parser.add_argument("--judge-host", default="yangdejin@222.19.197.47")
    parser.add_argument("--judge-root", default="/home/yangdejin/MOPD")
    parser.add_argument("--judge-key", required=True)
    args = parser.parse_args()
    out = ROOT / args.output
    state_path = out / "lcb-relay-state.json"
    state = {"status": "waiting_for_generation", "started": stamp(), "output": args.output}
    write_state(state_path, state)
    complete = f"{args.gpu_root}/{args.output}/lcb_release_v5/generation-complete.json"
    ssh_gpu = ["ssh", "-o", "ControlPath=/tmp/mopd-newserver-%C", "-p", args.gpu_port, args.gpu_host]
    while subprocess.run(ssh_gpu + [f"test -f {complete}"], check=False).returncode:
        time.sleep(30)
    state["status"] = "syncing_to_judge"
    write_state(state_path, state)
    run(["rsync", "-az", "-e", f"ssh -o ControlPath=/tmp/mopd-newserver-%C -p {args.gpu_port}", f"{args.gpu_host}:{args.gpu_root}/{args.output}/", str(out) + "/"])
    run(["rsync", "-az", "-e", f"ssh -i {args.judge_key}", str(out) + "/", f"{args.judge_host}:{args.judge_root}/{args.output}/"])
    state["status"] = "judge_running"
    write_state(state_path, state)
    command = (
        f"cd {args.judge_root} && nohup /home/yangdejin/miniconda3/envs/mopd/bin/python -u "
        f"scripts/experiments/benchmark_evaluate.py --config {args.output}/config.json --dataset lcb_release_v5 "
        f"> {args.output}/lcb_release_v5/evaluation.log 2>&1 < /dev/null &"
    )
    subprocess.run(["ssh", "-f", "-i", args.judge_key, args.judge_host, command], check=True)
    state.update({"status": "judge_started", "judge_started": stamp()})
    write_state(state_path, state)


if __name__ == "__main__":
    main()
