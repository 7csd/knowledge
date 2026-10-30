"""Audit one actual output/trace pair and optionally score an explicitly reviewed annotation."""

import argparse
import json
import sys
from pathlib import Path

from knowledge_agent.answer_evaluation import audit_run, load_trace, review_template, score_review

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
parser = argparse.ArgumentParser()
parser.add_argument("--artifact", type=Path, required=True)
parser.add_argument("--trace", type=Path, required=True)
parser.add_argument("--review", type=Path)
parser.add_argument("--case-id", help="额外按验收用例核查状态、来源及补查动作")
parser.add_argument("--cases", type=Path, default=Path("data/evaluation_cases.json"))
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
artifact = json.loads(args.artifact.read_text(encoding="utf-8-sig"))
actual = artifact.get("actual", artifact)
events = load_trace(args.trace)
review = json.loads(args.review.read_text(encoding="utf-8-sig")) if args.review else review_template(actual)
expectations = artifact.get("input", {})
if not isinstance(expectations, dict):
    expectations = {}
if args.case_id:
    expectations = next(c for c in json.loads(args.cases.read_text(encoding="utf-8-sig"))
                        if c["case_id"] == args.case_id)
result = {"mechanical_audit": audit_run(actual, events, expectations),
          "semantic_metrics": score_review(actual, events, review), "review": review,
          "live_acceptance": "pending" if actual["model_mode"] != "live" else "requires_semantic_review"}
args.output.parent.mkdir(parents=True, exist_ok=True)
with args.output.open("x", encoding="utf-8") as stream:
    stream.write(json.dumps(result, ensure_ascii=False, indent=2))
print(json.dumps({"mechanical_pass": result["mechanical_audit"]["mechanical_pass"],
                  "semantic_metrics": result["semantic_metrics"], "live_acceptance": result["live_acceptance"]},
                 ensure_ascii=False, indent=2))
