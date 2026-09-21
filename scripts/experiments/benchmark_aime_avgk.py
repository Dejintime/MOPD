"""Persistent AIME avg@k generation and scoring with fixed independent samples."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bandit_mopd.prompting import render_rollout_prompt
from benchmark_generate import digest, dump, final_answer, now, read_rows


def config_hash(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


def load_dataset(path):
    rows = read_rows(path)
    assert all(row["usage"] == "final_evaluation_only" for row in rows)
    assert len({row["id"] for row in rows}) == len(rows)
    return rows


def prepare(cfg):
    from transformers import AutoTokenizer

    out = ROOT / cfg["output"]
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg["student"], local_files_only=True)
    fingerprint = config_hash(cfg)
    manifest = {"created_utc": now(), "config_sha256": fingerprint, "num_samples": cfg["num_samples"], "datasets": []}
    for ds in cfg["datasets"]:
        target = out / ds["name"]
        target.mkdir(exist_ok=True)
        source = ROOT / ds["path"]
        rows = load_dataset(source)
        assert len(rows) == ds["n"]
        prompts = []
        for index, row in enumerate(rows):
            prompt = render_rollout_prompt(tokenizer, row["messages"], cfg, tokenize=False, add_generation_prompt=True)
            if not prompt.endswith("<think>\n"):
                assert prompt.endswith("<|im_start|>assistant\n"), repr(prompt[-60:])
                prompt += "<think>\n"
            ids = tokenizer.encode(prompt, add_special_tokens=False)
            assert len(ids) + cfg["max_new_tokens"] <= cfg["vllm"]["max_model_len"]
            prompts.append({"id": row["id"], "index": index, "domain": row["domain"], "prompt_token_ids": ids})
        item = {
            "name": ds["name"], "n": ds["n"], "num_samples": cfg["num_samples"],
            "source_sha256": digest(source), "config_sha256": fingerprint,
            "ids": [prompt["id"] for prompt in prompts],
        }
        old = target / "input-manifest.json"
        if old.exists():
            assert json.loads(old.read_text()) == item, "Config or evaluation data changed; choose a new output directory"
        else:
            dump(old, item)
            with (target / "prompts.jsonl").open("w") as file:
                for prompt in prompts:
                    file.write(json.dumps(prompt, ensure_ascii=False) + "\n")
        manifest["datasets"].append(item)
    manifest["versions"] = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "vllm")}
    dump(out / "manifest.json", manifest)


def expected_keys(prompts, samples):
    return [(prompt["id"], sample) for prompt in prompts for sample in range(samples)]


def generate(cfg, ds):
    from vllm import LLM, SamplingParams

    out = ROOT / cfg["output"]
    dest = out / ds["name"]
    prompts = read_rows(dest / "prompts.jsonl")
    meta = json.loads((dest / "input-manifest.json").read_text())
    assert meta["config_sha256"] == config_hash(cfg)
    assert meta["source_sha256"] == digest(ROOT / ds["path"])
    samples = cfg["num_samples"]
    existing = read_rows(dest / "results.jsonl") if (dest / "results.jsonl").exists() else []
    keys = expected_keys(prompts, samples)
    assert [(row["id"], row["sample"]) for row in existing] == keys[:len(existing)]
    if len(existing) == len(keys):
        dump(dest / "generation-complete.json", {"n_questions": len(prompts), "n_samples": len(existing), "timestamp": now()})
        return
    assert len(existing) % samples == 0, "Interrupted batch must contain complete question samples"
    session = {
        "started": now(), "resume_questions": len(existing) // samples,
        "config_sha256": meta["config_sha256"], "num_samples": samples,
        "vllm": cfg["vllm"], "hostname": os.uname().nodename,
    }
    session_id = hashlib.sha256(json.dumps(session, sort_keys=True).encode()).hexdigest()
    sessions = dest / "runtime-sessions"
    sessions.mkdir(exist_ok=True)
    dump(sessions / f"{session_id}.json", session)
    llm = LLM(model=cfg["student"], seed=cfg["base_seed"], **cfg["vllm"])
    for question in prompts[len(existing) // samples:]:
        params = [SamplingParams(max_tokens=cfg["max_new_tokens"], seed=cfg["base_seed"] + question["index"] * samples + sample,
                                 ignore_eos=False, **cfg["sampling"]) for sample in range(samples)]
        outputs = []
        batch_size = cfg["vllm"]["max_num_seqs"]
        for start in range(0, samples, batch_size):
            batch_params = params[start:start + batch_size]
            outputs.extend(llm.generate(
                [{"prompt_token_ids": question["prompt_token_ids"]} for _ in batch_params],
                batch_params, use_tqdm=True,
            ))
        assert len(outputs) == samples
        with (dest / "results.jsonl").open("a") as file:
            for sample, output in enumerate(outputs):
                answer = output.outputs[0]
                assert list(output.prompt_token_ids) == question["prompt_token_ids"]
                assert answer.finish_reason in ("stop", "length") and len(answer.token_ids) <= cfg["max_new_tokens"]
                record = {
                    "id": question["id"], "index": question["index"], "sample": sample,
                    "benchmark": ds["name"], "domain": question["domain"], "response": answer.text,
                    "final_answer": final_answer(answer.text), "response_ids": list(answer.token_ids),
                    "response_tokens": len(answer.token_ids), "prompt_tokens": len(question["prompt_token_ids"]),
                    "finish_reason": answer.finish_reason, "stop_reason": answer.stop_reason,
                    "truncated": answer.finish_reason == "length", "thinking_closed": "</think>" in answer.text,
                    "timestamp": now(), "runtime_id": session_id,
                }
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
                file.flush()
                os.fsync(file.fileno())
                existing.append(record)
        dump(dest / "progress.json", {
            "status": "generating", "questions_completed": len(existing) // samples, "questions_total": len(prompts),
            "samples_completed": len(existing), "samples_total": len(keys), "updated": now(),
        })
    dump(dest / "generation-complete.json", {"n_questions": len(prompts), "n_samples": len(existing), "timestamp": now()})
    llm.llm_engine.engine_core.shutdown()


def summarize(records, samples):
    assert records and len(records) % samples == 0
    grouped = defaultdict(list)
    for record in records:
        grouped[record["id"]].append(record)
    assert all(len(group) == samples for group in grouped.values())
    question_scores = [sum(row["correct"] for row in group) / samples for group in grouped.values()]
    return {
        "questions": len(grouped), "samples_per_question": samples, "candidate_correct": sum(row["correct"] for row in records),
        "candidate_accuracy": sum(row["correct"] for row in records) / len(records),
        "avg_at_k": sum(question_scores) / len(question_scores), "avg_at_k_name": f"avg@{samples}",
        "question_correct_counts": {question_id: sum(row["correct"] for row in group) for question_id, group in grouped.items()},
        "truncated": sum(row["truncated"] for row in records),
        "truncation_rate": sum(row["truncated"] for row in records) / len(records),
        "mean_tokens": sum(row["response_tokens"] for row in records) / len(records),
        "thinking_unclosed": sum(not row["thinking_closed"] for row in records),
        "infrastructure_errors": 0,
    }


def evaluate(cfg, ds):
    from bandit_mopd.verifiers.text import math as score_math

    out = ROOT / cfg["output"]
    dest = out / ds["name"]
    assert (dest / "generation-complete.json").exists()
    predictions = read_rows(dest / "results.jsonl")
    rows = load_dataset(ROOT / ds["path"])
    samples = cfg["num_samples"]
    assert len(predictions) == len(rows) * samples
    expected = [(row["id"], sample) for row in rows for sample in range(samples)]
    assert [(row["id"], row["sample"]) for row in predictions] == expected
    scored = []
    with (dest / "scores.jsonl").open("w") as file:
        source = {row["id"]: row for row in rows}
        for prediction in predictions:
            result = score_math(source[prediction["id"]], prediction["final_answer"])
            record = {key: prediction[key] for key in ("id", "index", "sample", "response_tokens", "truncated", "thinking_closed")}
            record["correct"] = bool(result["correct"])
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
            scored.append(record)
    summary = summarize(scored, samples)
    summary.update({"name": ds["name"], "source_results_sha256": digest(dest / "results.jsonl")})
    dump(dest / "summary.json", summary)
    dump(dest / "complete.json", {"status": "complete", "questions": len(rows), "samples": len(scored), "timestamp": now()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--dataset", choices=("aime24", "aime25"))
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    assert cfg["enable_thinking"] is True and cfg["max_new_tokens"] == 16384 and isinstance(cfg["num_samples"], int) and cfg["num_samples"] > 0
    assert cfg["sampling"] == {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "repetition_penalty": 1.0}
    prepare(cfg)
    if args.prepare_only:
        return
    ds = next(dataset for dataset in cfg["datasets"] if dataset["name"] == args.dataset)
    generate(cfg, ds)
    evaluate(cfg, ds)


if __name__ == "__main__":
    main()
