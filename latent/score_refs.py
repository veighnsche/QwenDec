"""Score, before training: what does the adapter need from a Score request?

No training. The accepted fork-mode adapter (fork-v1) answers the 199 development
Score cases with the status options, in three forms of the request:

  full          question and rubric as words, the adapter's own format
  no rubric     question as words, the rubric left out
  no question   empty question

A case is right when its status is right and, if it is assessed, the picked level is
within 0.05 of the target. These figures do not enter the pass rule.
"""

import json
import sys
from collections import Counter
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "gpu_loop"))
sys.path.insert(0, str(ROOT / "baseline"))
sys.path.insert(0, str(HERE))
from measure_decision4b import STATUS_OPTIONS, build  # noqa: E402
from measure_fork import Gate, probabilities  # noqa: E402
from model import load_text_model, require_gpu  # noqa: E402
from score_common import right_levels  # noqa: E402

FORMS = ("full", "no rubric", "no question")


def task_for(case, form):
    task, values = build(case, "status")
    if form == "no rubric":
        q = case["question"]
        task["question"] = f"What score between 0 and 1 applies? {q['proposition']}. Criterion: {q['criterion']}."
    if form == "no question":
        task["question"] = ""
    return task, values


@torch.no_grad()
def main():
    from peft import PeftModel
    from transformers import AutoTokenizer

    device = require_gpu()
    tok = AutoTokenizer.from_pretrained(ROOT / "Qwen3.5-4B")
    model = load_text_model(str(ROOT / "Qwen3.5-4B"), device)
    peft = PeftModel.from_pretrained(model, str(ROOT / "runs/fork-v1/adapter")).eval()
    gate = Gate(peft)
    cases = [c for c in map(json.loads, open(ROOT / "data/qwenonetwo/development.jsonl")) if c["operation"] == "Score"]

    report, letter_ids = {}, {}
    for form in FORMS:
        right, total = Counter(), Counter()
        for case in cases:
            task, values = task_for(case, form)
            probs, _, _ = probabilities(peft, tok, gate, task, "fork", device, letter_ids)
            picked = values[max(range(len(probs)), key=probs.__getitem__)]
            status = picked if picked in STATUS_OPTIONS else "Assessed"
            if case["status"] != "Assessed":
                kind, ok = "a status", status == case["status"]
            else:
                lo, hi = case["value"]["score_interval"]
                kind = "in between" if 0 < lo and hi < 1 else "an end of the scale"
                ok = status == "Assessed" and picked in right_levels([lo, hi])
            for key in ("all", f"target: {kind}"):
                total[key] += 1
                right[key] += ok
        report[form] = {k: f"{right[k]}/{total[k]}" for k in sorted(total)}
    (HERE / "results/score-refs.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
