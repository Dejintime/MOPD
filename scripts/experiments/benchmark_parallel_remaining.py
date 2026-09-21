"""Run independent post-LCB benchmarks while an external LCB judge is pending.

The primary queue owns its own lock and waits for the LiveCodeBench completion
marker.  This helper deliberately touches only named benchmark directories, so
the primary queue will later observe their completed markers and skip them.
"""
import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "experiments"))
from benchmark_generate import dump, now, runtime_settings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--runtime-config", required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    runtime_path = Path(args.runtime_config).resolve()
    cfg = json.loads(cfg_path.read_text())
    out = ROOT / cfg["output"]
    config_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    assert json.loads((out / "manifest.json").read_text())["config_sha256"] == config_hash
    runtime_settings(cfg, runtime_path)
    known = {dataset["name"] for dataset in cfg["datasets"]}
    assert set(args.datasets).issubset(known)
    assert "lcb_release_v5" not in args.datasets

    lock = (out / "parallel-remaining.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    env = dict(
        os.environ,
        CUDA_DEVICE_ORDER="PCI_BUS_ID",
        CUDA_VISIBLE_DEVICES="0",
        OMP_NUM_THREADS="8",
        VLLM_NO_USAGE_STATS="1",
        DO_NOT_TRACK="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
        NLTK_DATA="/home/yangdejin/.mopd-benchmark-nltk",
    )
    state = {
        "status": "running",
        "pid": os.getpid(),
        "started": now(),
        "datasets": args.datasets,
        "config_sha256": config_hash,
    }
    dump(out / "parallel-remaining-state.json", state)
    try:
        for name in args.datasets:
            dest = out / name
            if (dest / "complete.json").exists():
                continue
            readiness = out / "readiness" / f"{name}.json"
            assert readiness.exists(), f"Verifier readiness missing: {readiness}"
            for stage, script, marker, stage_env in (
                ("generation", "benchmark_generate.py", "generation-complete.json", env),
                ("evaluation", "benchmark_evaluate.py", "complete.json", dict(env, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1")),
            ):
                if (dest / marker).exists():
                    continue
                state.update(current=name, stage=stage, updated=now())
                dump(out / "parallel-remaining-state.json", state)
                command = [sys.executable, "-u", str(ROOT / "scripts" / "experiments" / script), "--config", str(cfg_path), "--dataset", name]
                if stage == "generation":
                    command += ["--runtime-config", str(runtime_path)]
                with (dest / f"{stage}.log").open("a") as log:
                    child = subprocess.Popen(command, cwd=ROOT, env=stage_env, stdout=log, stderr=subprocess.STDOUT)
                    state.update(child_pid=child.pid, updated=now())
                    dump(out / "parallel-remaining-state.json", state)
                    if child.wait() != 0:
                        raise RuntimeError(f"{name} {stage} failed; see {dest / (stage + '.log')}")
                assert (dest / marker).exists(), f"{name} {stage} marker missing"
        state.update(status="complete", current=None, child_pid=None, updated=now())
        dump(out / "parallel-remaining-state.json", state)
    except BaseException as exc:
        state.update(status="failed", error=repr(exc), updated=now())
        dump(out / "parallel-remaining-state.json", state)
        raise


if __name__ == "__main__":
    main()
