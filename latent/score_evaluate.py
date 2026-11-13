"""Measure one arm of the Score request pilot on the development variants.

One forward per question; the pick is the option letter with the highest probability
at the last position. A question is right when its status is right and, if it is
assessed, the picked level is within 0.05 of the target (of one minus the target for
a complement question).

Reported for the pass rule:
  all questions
  status or end of the scale   the questions the conclusiveness condition counts
  steering pairs               assessed pairs with no level right for both questions, both right
  in between                   questions whose target lies strictly between 0 and 1
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "gpu_loop"))
sys.path.insert(0, str(ROOT / "baseline"))
sys.path.insert(0, str(HERE))
from measure_decision4b import interval_distance  # noqa: E402
from measure_fork import Gate  # noqa: E402
from model import load_text_model, require_gpu  # noqa: E402
from score_common import REQUESTS, VALUES, is_right, request_text, tokenise  # noqa: E402


def rate(hits):
    return f"{sum(hits)}/{len(hits)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--mode", choices=("text", "latent"), required=True)
    ap.add_argument("--cases", default=str(ROOT / "data/latent/score-development.jsonl"))
    ap.add_argument("--vectors", default=str(ROOT / "data/latent/question-vectors-k16-score.pt"))
    ap.add_argument("--projection", help="trained projection file (latent mode)")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    from peft import PeftModel
    from transformers import AutoTokenizer

    device = require_gpu()
    tok = AutoTokenizer.from_pretrained(ROOT / "Qwen3.5-4B")
    model = load_text_model(str(ROOT / "Qwen3.5-4B"), device)
    peft = PeftModel.from_pretrained(model, args.adapter).eval()
    gate = Gate(peft)
    cases = [json.loads(x) for x in open(args.cases)]
    letters = torch.arange(32, 32 + len(VALUES), device=device)  # token ids of A, B, C, ...
    assert [tok.decode([int(i)]) for i in letters[:7]] == list("ABCDEFG")

    k = 0
    if args.mode == "latent":
        store = torch.load(args.vectors)
        assert [t for t, _ in sorted(store["index"].items(), key=lambda x: x[1])] == REQUESTS, "request vectors are in another order"
        vectors, index, k = store["vectors"].to(device), store["index"], store["vectors"].shape[1]
        projection = torch.load(args.projection)["projection"].to(device) if args.projection else None

    rows = []
    with torch.no_grad():
        for case in cases:
            ids, gate_start, q_start = tokenise(tok, case, args.mode, k)
            ids = torch.tensor([ids], device=device)
            n = ids.shape[1]
            gate.mask = (torch.arange(n, device=device) >= gate_start).view(1, n, 1)
            if args.mode == "latent":
                q = vectors[index[request_text(case)]].float()
                if projection is not None:
                    q = q @ projection.T
                embeds = model.model.embed_tokens(ids)
                embeds[0, q_start:q_start + k] = q.to(embeds.dtype)
                logits = peft(inputs_embeds=embeds).logits[0, -1]
            else:
                logits = peft(input_ids=ids).logits[0, -1]
            probs = torch.softmax(logits[letters].float(), dim=0).tolist()
            pick = max(range(len(probs)), key=probs.__getitem__)
            rows.append({"id": case["id"], "pick": pick, "picked": VALUES[pick], "right": is_right(case, pick), "probs": probs})

    right = {r["id"]: r["right"] for r in rows}
    picked = {r["id"]: r["picked"] for r in rows}
    groups, distance = defaultdict(list), defaultdict(list)
    for c in cases:
        ok = right[c["id"]]
        keys = ["all questions", f"kind {c['kind']}", f"target: {c['target_type']}",
                "status or end of the scale" if c["target_type"] != "in between" else "in between",
                f"status of the question: {c['status']}",
                "rule wording " + ("as in training" if c["rule_wording_in_training"] else "unlike training")]
        if c["rubric_adds_midpoint"]:
            keys.append("rubric adds a midpoint")
        if c["status"] == "Assessed":
            keys.append(f"kind {c['kind']}, {c['target_type']}")
            if not isinstance(picked[c["id"]], str):  # a level was picked: how far from the target
                distance[c["target_type"]].append(interval_distance(picked[c["id"]], c["interval"]))
        for key in keys:
            groups[key].append(ok)
    pairs = defaultdict(list)
    for c in cases:
        if c["kind"] == "score" and c["pair_has_no_shared_level"]:
            both = right[c["id"]] and right[c["twin"]]
            pairs["all"].append(both)
            pairs[c["target_type"]].append(both)
    report = {
        "tag": args.tag, "mode": args.mode, "adapter": args.adapter,
        "all_questions": rate(groups["all questions"]),
        "status_or_end_of_the_scale": rate(groups["status or end of the scale"]),
        "in_between": rate(groups["in between"]),
        "steering_pairs_both_right": rate(pairs["all"]),
        "steering_pairs_by_target": {key: rate(v) for key, v in sorted(pairs.items()) if key != "all"},
        "accuracy": {key: rate(v) for key, v in sorted(groups.items())},
        "mean_distance_of_picked_level": {key: round(sum(v) / len(v), 4) for key, v in sorted(distance.items())},
        "assessed_questions_answered_with_a_status": sum(c["status"] == "Assessed" and isinstance(picked[c["id"]], str) for c in cases),
        "picks": dict(Counter(str(r["picked"]) for r in rows)),
    }
    (HERE / "results" / f"{args.tag}.json").write_text(json.dumps(report, indent=1) + "\n")
    with open(HERE / "results" / f"{args.tag}.cases.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
