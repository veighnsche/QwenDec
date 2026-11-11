"""Measure an adapter on the Noul question variants, in one question mode.

Inference only on the RTX 3090, state read with the adapter off. Reports accuracy
on status and value together, and the share of plain/negated twin pairs where both
are right. A twin pair shares its state and differs only in the question, so a
model that ignores the question cannot get both of an assessed pair right.

Latent mode replaces K question positions with vectors from the fixed encoder,
passed through the trained projection. `--ablate zero` feeds zero vectors;
`--ablate swap` feeds the vectors of the twin question on the same state.
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
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(HERE))
from common import VALUES, gold_index, question_text, tokenise  # noqa: E402
from model import load_text_model, require_gpu  # noqa: E402
from step import Gate  # noqa: E402


def rate(hits):
    return f"{sum(hits)}/{len(hits)} ({100 * sum(hits) / max(1, len(hits)):.0f}%)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--mode", choices=("text", "none", "latent"), required=True)
    ap.add_argument("--cases", default=str(ROOT / "data/latent/noul-development.jsonl"))
    ap.add_argument("--vectors", help="question vectors file (latent mode)")
    ap.add_argument("--projection", help="trained projection file (latent mode)")
    ap.add_argument("--ablate", choices=("zero", "swap"))
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
    by_id = {c["id"]: c for c in cases}
    letters = torch.arange(32, 32 + len(VALUES), device=device)  # token ids of A, B, C, ...
    assert [tok.decode([int(i)]) for i in letters] == list("ABCDE")

    if args.mode == "latent":
        store = torch.load(args.vectors)
        vectors, index, k = store["vectors"].to(device), store["index"], store["vectors"].shape[1]
        projection = torch.load(args.projection)["projection"].to(device) if args.projection else None

    rows = []
    with torch.no_grad():
        for case in cases:
            ids, gate_start, q_start = tokenise(tok, case, args.mode, k if args.mode == "latent" else 0)
            ids = torch.tensor([ids], device=device)
            n = ids.shape[1]
            gate.mask = (torch.arange(n, device=device) >= gate_start).view(1, n, 1)
            if args.mode == "latent":
                source = by_id[case["twin"]] if args.ablate == "swap" else case
                q = vectors[index[question_text(source)]].float()
                if projection is not None:
                    q = q @ projection.T
                if args.ablate == "zero":
                    q = torch.zeros_like(q)
                embeds = model.model.embed_tokens(ids)
                embeds[0, q_start:q_start + k] = q.to(embeds.dtype)
                logits = peft(inputs_embeds=embeds).logits[0, -1]
            else:
                logits = peft(input_ids=ids).logits[0, -1]
            probs = torch.softmax(logits[letters].float(), dim=0).tolist()
            pick = max(range(len(probs)), key=probs.__getitem__)
            rows.append({"id": case["id"], "pick": pick, "gold": gold_index(case), "probs": probs})

    right = {r["id"]: r["pick"] == r["gold"] for r in rows}
    groups = defaultdict(list)
    for c in cases:
        groups["all questions"].append(right[c["id"]])
        groups[f"status {c['status']}"].append(right[c["id"]])
        if c["parsed"]:
            groups[f"kind {c['kind']}{' negated' if c['negated'] else ''}"].append(right[c["id"]])
    pairs = defaultdict(list)
    for c in cases:
        if not c["negated"]:
            twin = by_id[c["twin"]]
            both = right[c["id"]] and right[twin["id"]]
            pairs["all twin pairs"].append(both)
            if c["status"] == "Assessed":
                pairs["assessed twin pairs"].append(both)
                if c["parsed"]:
                    pairs[f"assessed pairs, kind {c['kind']}"].append(both)
    # A state is fully right only if every one of its questions is right.
    state = defaultdict(list)
    for c in cases:
        if c["parsed"]:
            state[c["base_id"]].append(right[c["id"]])
    report = {
        "tag": args.tag, "mode": args.mode, "ablate": args.ablate, "adapter": args.adapter,
        "accuracy": {k: rate(v) for k, v in sorted(groups.items())},
        "both_twins_right": {k: rate(v) for k, v in sorted(pairs.items())},
        "states_with_all_eight_right": rate([all(v) for v in state.values()]),
        # Only meaningful with --ablate swap: does the answer follow the question that was swapped in?
        "answer_matches_twin_question": rate([r["pick"] == gold_index(by_id[by_id[r["id"]]["twin"]]) for r in rows
                                              if by_id[r["id"]]["status"] == "Assessed"]),
        "picks": dict(Counter(str(VALUES[r["pick"]]) for r in rows)),
    }
    out = ROOT / "latent" / "results"
    out.mkdir(exist_ok=True)
    (out / f"{args.tag}.json").write_text(json.dumps(report, indent=1) + "\n")
    with open(out / f"{args.tag}.cases.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
