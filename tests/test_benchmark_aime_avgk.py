import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "experiments"))
from benchmark_aime_avgk import summarize


def test_avg_at_k_averages_each_question_over_all_samples():
    records = []
    for question, corrects in (("a", (True, False, True, False)), ("b", (False, False, False, False))):
        for sample, correct in enumerate(corrects):
            records.append({"id": question, "sample": sample, "correct": correct, "truncated": False,
                            "thinking_closed": True, "response_tokens": 10})
    result = summarize(records, 4)
    assert result["avg_at_k_name"] == "avg@4"
    assert result["avg_at_k"] == 0.25
    assert result["candidate_accuracy"] == 0.25
    assert result["question_correct_counts"] == {"a": 2, "b": 0}
