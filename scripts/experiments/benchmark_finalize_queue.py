"""Mark the benchmark queue complete after independently resumed stages finish."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT / "scripts" / "experiments"))
from benchmark_generate import dump, now


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    out = ROOT / cfg["output"]
    summaries = []
    for dataset in cfg["datasets"]:
        path = out / dataset["name"]
        assert (path / "complete.json").exists(), f"Incomplete benchmark: {dataset['name']}"
        summaries.append(json.loads((path / "summary.json").read_text()))
    dump(out / "summaries.json", summaries)
    dump(out / "queue-state.json", {
        "status": "complete",
        "stage": "complete",
        "current": None,
        "child_pid": None,
        "completed": [dataset["name"] for dataset in cfg["datasets"]],
        "updated": now(),
        "recovered_after_parallel_resume": True,
    })
    print("ALL_BENCHMARKS_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
