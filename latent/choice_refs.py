"""Choice, before any design: what does the adapter need from a Choice request?

No training. The accepted fork-mode adapter (fork-v1) answers the development Choice
cases with the status options, in four forms of the request:

  full                 question text; each alternative with its name and its properties
  names only           question text; each alternative with its name only
  no question          empty question; each alternative with its name and properties
  no question, names   empty question; each alternative with its name only

Each form is also run with the alternatives listed in reverse order.
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
from measure_decision4b import STATUS_OPTIONS, build  # noqa: E402
from measure_fork import Gate, probabilities  # noqa: E402
from model import load_text_model, require_gpu  # noqa: E402

FORMS = {"full": (True, True), "names only": (True, False), "no question": (False, True), "no question, names": (False, False)}


def task_for(case, question, properties, reverse):
    task, values = build(case, "status")
    n = len(case["alternatives"])
    options, alt_values = task["options"][:n], values[:n]
    if reverse:
        options, alt_values = options[::-1], alt_values[::-1]
    if not properties:
        options = [{"label": o["label"], "key": o["key"]} for o in options]
    options = options + task["options"][n:]
    for i, o in enumerate(options):
        o["label"] = "ABCDEFGHIJKLMNOP"[i]
    task["options"] = options
    if not question:
        task["question"] = ""
    return task, alt_values + values[n:]


@torch.no_grad()
def main():
    from peft import PeftModel
    from transformers import AutoTokenizer

    device = require_gpu()
    tok = AutoTokenizer.from_pretrained(ROOT / "Qwen3.5-4B")
    model = load_text_model(str(ROOT / "Qwen3.5-4B"), device)
    peft = PeftModel.from_pretrained(model, str(ROOT / "runs/fork-v1/adapter")).eval()
    gate = Gate(peft)
    cases = [c for c in map(json.loads, open(ROOT / "data/qwenonetwo/development.jsonl")) if c["operation"] == "Choice"]

    report, letter_ids = {}, {}
    for name, (question, properties) in FORMS.items():
        for reverse in (False, True):
            right, total = Counter(), Counter()
            for case in cases:
                task, values = task_for(case, question, properties, reverse)
                probs, _, _ = probabilities(peft, tok, gate, task, "fork", device, letter_ids)
                picked = values[max(range(len(probs)), key=probs.__getitem__)]
                status = picked if picked in STATUS_OPTIONS else "Assessed"
                ok = status == case["status"] and (status != "Assessed" or picked in case["value"]["acceptable_ids"])
                for key in ("all", f"status {case['status']}"):
                    total[key] += 1
                    right[key] += ok
            report[name + (", reversed order" if reverse else "")] = {k: f"{right[k]}/{total[k]}" for k in sorted(total)}
    (HERE / "results/choice-refs.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
