import json
import sys
import argparse
from pathlib import Path
from src.pipeline import ask
def load_golden(path):
    return [
        json.loads(line)
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]
def source_hit(result, expected_sources):
    if not expected_sources:
        return True
    actual_files = {
        s["metadata"].get("file")
        for s in result["sources"]
    }
    return any(src in actual_files for src in expected_sources)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", default="src/eval/golden.jsonl")
    parser.add_argument("--fail-below", default="")
    args = parser.parse_args()
    golden = load_golden(args.golden)
    rows = []
    for ex in golden:
        result = ask(ex["question"])
        validation = result["validation"]
        expected_type = ex.get("expected_answer_type", "ANSWER")
        expects_refusal = expected_type == "REFUSE"
        refused = validation["refused"]
        row = {
            "id": ex["id"],
            "question": ex["question"],
            "citation_ok": validation["ok"],
            "source_hit": source_hit(result, ex.get("expected_sources", [])),
            "refusal_ok": refused if expects_refusal else not refused,
            "answer": result["answer"]
        }
        rows.append(row)
        print(json.dumps(row, indent=2))
    total = len(rows)
    scores = {
        "citation_ok_rate": sum(r["citation_ok"] for r in rows) / total,
        "source_hit_rate": sum(r["source_hit"] for r in rows) / total,
        "refusal_ok_rate": sum(r["refusal_ok"] for r in rows) / total
    }
    print("=== SCORES ===")
    print(json.dumps(scores, indent=2))
    if args.fail_below:
        thresholds = dict(p.split("=") for p in args.fail_below.split(","))
        for metric, threshold in thresholds.items():
            if scores.get(metric, 0) < float(threshold):
                print(f"FAIL: {metric}={scores.get(metric, 0):.3f} < {threshold}")
                sys.exit(1)
    print("PASS")
if __name__ == "__main__":
    main()
