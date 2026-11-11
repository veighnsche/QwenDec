"""Measure decision-4b when the state is read with the adapter off.

Same cases, prompts and scoring as measure_decision4b.py. The only change is where
the adapter is active: off for every position up to the end of the state, on from
the question onward. In one causal forward this is exactly a fork of a natively
encoded prefix: the prefix positions, their attention keys and values, and the
recurrent state they leave behind are all computed without the adapter.

The switch is a multiplier on each LoRA branch, per position. Running it with the
multiplier at 1 everywhere must reproduce the earlier adapter-on results; that is
checked before the fork numbers are trusted. Inference only, no training.
"""

import json
import sys
from collections import Counter
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "gpu_loop"))
sys.path.insert(0, str(HERE))
from measure_decision4b import STATUS_OPTIONS, SYSTEM, build, interval_distance, summarise  # noqa: E402
from model import load_text_model, require_gpu  # noqa: E402

STATE_END = '", "question": "'  # unescaped, so it cannot occur inside the JSON-escaped state


class Gate:
    """Per-position on/off multiplier applied to every LoRA branch."""

    def __init__(self, peft):
        self.mask = None
        self.branches = 0
        for module in peft.modules():
            if hasattr(module, "lora_B"):
                for linear in module.lora_B.values():
                    linear.register_forward_hook(self.apply)
                    self.branches += 1

    def apply(self, module, inputs, output):
        assert output.shape[1] == self.mask.shape[1], "LoRA branch is not [batch, position, features]"
        return output * self.mask.to(output.dtype)


@torch.no_grad()
def probabilities(model, tok, gate, task, mode, device, letter_ids):
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": json.dumps(task, ensure_ascii=False)}]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False, return_offsets_mapping=True)
    ids = enc["input_ids"].to(device)
    length = ids.shape[1]

    if mode == "full":
        on = torch.ones(length, dtype=torch.bool)
    else:
        boundary = prompt.index(STATE_END) + 1  # first character after the state's closing quote
        on = enc["offset_mapping"][0, :, 1] > boundary
    gate.mask = on.to(device).view(1, length, 1)

    logits = model(input_ids=ids).logits[0, -1]
    letters = [o["label"] for o in task["options"]]
    for letter in letters:
        if letter not in letter_ids:
            letter_ids[letter] = tok.encode(prompt + letter, add_special_tokens=False)[-1]
    picked = torch.tensor([letter_ids[x] for x in letters], device=device)
    return torch.softmax(logits[picked].float(), dim=0).tolist(), int(on.sum()), length


def is_correct(row):
    picked = row["values"][row["argmax"]]
    if row["view"] == "status":
        status = picked if isinstance(picked, str) and picked in STATUS_OPTIONS else "Assessed"
        if status != row["status"]:
            return False
        if status != "Assessed":
            return True
    if row["operation"] == "Noul":
        return picked == bool(row["gold"]["p_true"])
    if row["operation"] == "Choice":
        return picked in row["gold"]["acceptable_ids"]
    if row["view"] == "status":
        return interval_distance(picked, row["gold"]["score_interval"]) <= 0.05
    expect = sum(p * v for p, v in zip(row["probs"], row["values"]))
    return interval_distance(expect, row["gold"]["score_interval"]) <= 0.05


def main():
    import argparse

    from peft import PeftModel
    from transformers import AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=str(ROOT / "decision-4b"), help="adapter folder to measure")
    ap.add_argument("--tag", default="decision-4b", help="name used in the output files")
    args = ap.parse_args()
    stock = args.tag == "decision-4b"

    device = require_gpu()
    tok = AutoTokenizer.from_pretrained(ROOT / "Qwen3.5-4B")
    model = load_text_model(str(ROOT / "Qwen3.5-4B"), device)
    peft = PeftModel.from_pretrained(model, args.adapter).eval()
    gate = Gate(peft)
    print(f"gated LoRA branches: {gate.branches}")

    cases = [json.loads(x) for x in open(ROOT / "data/qwenonetwo/development.jsonl")]
    train = [json.loads(x) for x in open(ROOT / "data/qwenonetwo/train.jsonl")]
    earlier = [json.loads(x) for x in open(HERE / "cases-decision-4b.jsonl")]

    runs, letter_ids = {}, {}
    for mode in ("full", "fork"):
        rows, on_tokens, all_tokens = [], 0, 0
        for case in cases:
            views = ["status"] + (["values"] if case["status"] == "Assessed" else [])
            if case["operation"] == "Choice" and case["status"] == "Assessed":
                views.append("reversed")
            for view in views:
                task, values = build(case, view)
                probs, on, length = probabilities(peft, tok, gate, task, mode, device, letter_ids)
                on_tokens += on
                all_tokens += length
                rows.append({"id": case["id"], "operation": case["operation"], "status": case["status"],
                             "view": view, "gold": case["value"], "values": values, "probs": probs,
                             "argmax": max(range(len(probs)), key=probs.__getitem__)})
        runs[mode] = rows
        print(f"{mode}: {len(rows)} forward passes, adapter on for {100 * on_tokens / all_tokens:.1f}% of positions")

        if mode == "full" and stock:  # the switch at 1 everywhere must reproduce the earlier run
            assert [(r["id"], r["view"]) for r in rows] == [(r["id"], r["view"]) for r in earlier]
            gap = max(abs(a - b) for r, e in zip(rows, earlier) for a, b in zip(r["probs"], e["probs"]))
            flips = sum(r["argmax"] != e["argmax"] for r, e in zip(rows, earlier))
            print(f"switch check against the earlier adapter-on run: max probability gap {gap:.2e}, changed picks {flips}")
            assert gap < 1e-3 and flips == 0, "the gated forward does not reproduce the ungated one"

    with open(HERE / f"cases-{args.tag}-fork.jsonl", "w") as f:
        for r in runs["fork"]:
            f.write(json.dumps(r) + "\n")

    paired = {}
    for full, fork in zip(runs["full"], runs["fork"]):
        key = f"{full['operation']} / {full['view']}"
        a, b = is_correct(full), is_correct(fork)
        paired.setdefault(key, Counter())[("both right", "lost in fork", "gained in fork", "both wrong")[(not a) * 2 + (not b)]] += 1
    paired = {k: dict(v) for k, v in sorted(paired.items())}

    report = {"full": summarise(runs["full"], train), "fork": summarise(runs["fork"], train), "paired": paired}
    (HERE / ("results-fork.json" if stock else f"results-fork-{args.tag}.json")).write_text(json.dumps(report, indent=1) + "\n")

    def show(x):
        return f"{x['correct']}/{x['of']} ({100 * x['rate']:.0f}%)"

    for mode in ("full", "fork"):
        m = report[mode]
        print(f"\n== {mode}")
        print(f" Noul values   {show(m['noul_values'])}  brier skill {m['noul_values']['brier_skill']}")
        print(f" Choice values {show(m['choice_values'])}  reversed: correct {show(m['choice_reversed_order']['correct_when_reversed'])}")
        s = m["score_values"]
        print(f" Score         within 0.05 {show(s['within_0.05'])}  interior {show(s['within_0.05_interior'])}  "
              f"mean distance {s['mean_distance']} interior {s['mean_distance_interior']}")
        for op in ("noul", "choice", "score"):
            w = m[f"{op}_with_status"]
            print(f" {op:6s} with status: status {show(w['status_right'])}  status+value {show(w['status_and_value_right'])}")
    print("\npaired, same case under full vs fork:")
    for k, v in paired.items():
        print(f" {k:18s} {v}")

    confusion = Counter()
    for r in runs["fork"]:
        if r["view"] == "status":
            picked = r["values"][r["argmax"]]
            confusion[(r["status"], picked if isinstance(picked, str) and picked in STATUS_OPTIONS else "Assessed")] += 1
    print("\nfork status confusion (true -> predicted):")
    for true in ("Assessed", "Insufficient", "Conflicting", "Unresolved", "NoFeasibleChoice"):
        print(f" {true:17s}", {p: n for (t, p), n in sorted(confusion.items()) if t == true})


if __name__ == "__main__":
    main()
