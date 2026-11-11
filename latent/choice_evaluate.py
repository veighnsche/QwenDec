"""Measure one arm of the Choice request pilot on the development variants.

One forward per question; the pick is the option letter with the highest probability
at the last position. A question is right when its status is right and, if it is
assessed, the picked alternative is an acceptable one.

Reported for the pass rule:
  all questions
  kind pairs        a select and a ruled-out question on the same state and order, both right
  order-consistent  assessed states whose select question is right in every order asked
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
from choice_common import is_right, question_text, tokenise, values  # noqa: E402
from measure_fork import Gate  # noqa: E402
from model import load_text_model, require_gpu  # noqa: E402


def rate(hits):
    return f"{sum(hits)}/{len(hits)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--mode", choices=("text", "latent"), required=True)
    ap.add_argument("--cases", default=str(ROOT / "data/latent/choice-development.jsonl"))
    ap.add_argument("--vectors", default=str(ROOT / "data/latent/question-vectors-k16-choice.pt"))
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
    all_letters = torch.arange(32, 32 + 16, device=device)  # token ids of A, B, C, ...
    assert [tok.decode([int(i)]) for i in all_letters[:7]] == list("ABCDEFG")

    k = 0
    if args.mode == "latent":
        store = torch.load(args.vectors)
        vectors, index, k = store["vectors"].to(device), store["index"], store["vectors"].shape[1]
        projection = torch.load(args.projection)["projection"].to(device) if args.projection else None

    rows = []
    with torch.no_grad():
        for case in cases:
            ids, gate_start, q_start, _ = tokenise(tok, case, args.mode, k)
            ids = torch.tensor([ids], device=device)
            n = ids.shape[1]
            gate.mask = (torch.arange(n, device=device) >= gate_start).view(1, n, 1)
            if args.mode == "latent":
                q = vectors[index[question_text(case)]].float()
                if projection is not None:
                    q = q @ projection.T
                embeds = model.model.embed_tokens(ids)
                embeds[0, q_start:q_start + k] = q.to(embeds.dtype)
                logits = peft(inputs_embeds=embeds).logits[0, -1]
            else:
                logits = peft(input_ids=ids).logits[0, -1]
            probs = torch.softmax(logits[all_letters[: len(values(case))]].float(), dim=0).tolist()
            pick = max(range(len(probs)), key=probs.__getitem__)
            rows.append({"id": case["id"], "pick": pick, "picked": values(case)[pick], "right": is_right(case, pick),
                         "probs": probs})

    right = {r["id"]: r["right"] for r in rows}
    groups = defaultdict(list)
    for c in cases:
        ok = right[c["id"]]
        for key in ("all questions", f"kind {c['kind']}", f"status of the question: {c['status']}",
                    "alternatives named by single letters" if c["letter_names"] else "alternatives with other names",
                    f"kind {c['kind']}, {'listed' if c['listed_order'] else 'changed'} order"):
            groups[key].append(ok)
    by_key = {(c["base_id"], c["kind"], c["order"]): right[c["id"]] for c in cases}
    pairs = [ok and by_key[(b, "ruledout", o)] for (b, kind, o), ok in by_key.items()
             if kind == "select" and (b, "ruledout", o) in by_key]
    orders = defaultdict(list)
    for c in cases:
        if c["kind"] == "select" and c["status"] == "Assessed":
            orders[c["base_id"]].append(right[c["id"]])
    listed = [right[c["id"]] for c in cases if c["kind"] == "select" and c["listed_order"]]
    report = {
        "tag": args.tag, "mode": args.mode, "adapter": args.adapter,
        "all_questions": rate(groups["all questions"]),
        "kind_pairs_both_right": rate(pairs),
        "assessed_states_right_in_every_order": rate([all(v) for v in orders.values()]),
        "select_questions_in_listed_order": rate(listed),
        "accuracy": {key: rate(v) for key, v in sorted(groups.items())},
        "picks": dict(Counter("an alternative" if r["picked"] in [a["identity"] for a in c["alternatives"]] else r["picked"]
                              for r, c in zip(rows, cases))),
    }
    (HERE / "results" / f"{args.tag}.json").write_text(json.dumps(report, indent=1) + "\n")
    with open(HERE / "results" / f"{args.tag}.cases.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
