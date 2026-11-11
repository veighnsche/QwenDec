"""Compare candidate fixed encoders before any training. No labels, no fitting.

Pilot 1 failed because its encoder gave almost the same vectors to two questions
that differ in one word ("first" against "second"). This measures, for each
candidate way of turning a question into K vectors, how far apart such minimal
pairs are, relative to the typical distance between two questions.

A candidate is a layer of the frozen base model and a pooling:
  layer    0 is the input embeddings, "final" is the normalised last layer
  spans-K  mean over K equal spans of the question
  last-K   the last K positions
Each vector is rescaled to the typical input-embedding length, as in pilot 1.

separation = median distance within a kind of pair / median distance over all
pairs of questions (after removing the mean question).
"""

import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "gpu_loop"))
sys.path.insert(0, str(HERE))
from common import question_text  # noqa: E402
from model import load_text_model, require_gpu  # noqa: E402

LAYERS = (0, 4, 8, 16, 24, "final")
POOLINGS = (("spans", 4), ("spans", 16), ("last", 4), ("last", 16))


def pool(states, n, how, k):
    if how == "last":
        return states[n - k:n]
    edges = [round(j * n / k) for j in range(k + 1)]
    return torch.stack([states[edges[j]:edges[j + 1]].mean(0) for j in range(k)])


@torch.no_grad()
def main():
    from transformers import AutoTokenizer

    device = require_gpu()
    tok = AutoTokenizer.from_pretrained(ROOT / "Qwen3.5-4B")
    model = load_text_model(str(ROOT / "Qwen3.5-4B"), device)
    typical = model.model.embed_tokens.weight.float().norm(dim=-1).mean()

    cases = [json.loads(x) for x in open(ROOT / "data/latent/noul-development.jsonl")]
    states = {}
    for c in cases:
        if c["parsed"]:
            states.setdefault(c["base_id"], {})[(c["kind"], c["negated"])] = question_text(c)
    states = [s for s in states.values() if len(s) == 8]
    texts = sorted({t for s in states for t in s.values()})
    row = {t: i for i, t in enumerate(texts)}
    tokens = [tok(t, add_special_tokens=False)["input_ids"] for t in texts]

    def pairs(a, b):
        return sorted({(row[s[a]], row[s[b]]) for s in states})

    groups = {
        "first vs second prerequisite": pairs(("pa", False), ("pb", False)) + pairs(("pa", True), ("pb", True)),
        "question vs its negation": [p for kind in ("and", "or", "pa", "pb") for p in pairs((kind, False), (kind, True))],
        "rule met vs at least one": pairs(("and", False), ("or", False)),
    }
    unrelated = [(a, b) for a in range(len(texts)) for b in range(a + 1, len(texts))]  # every pair of questions

    # One forward per question; keep every layer's states.
    per_question = []
    for ids in tokens:
        out = model.model(input_ids=torch.tensor([ids], device=device), use_cache=False, output_hidden_states=True)
        layers = {i: out.hidden_states[i][0].float() for i in LAYERS if i != "final"}
        layers["final"] = out.last_hidden_state[0].float()
        per_question.append(layers)

    report = []
    for layer in LAYERS:
        for how, k in POOLINGS:
            v = torch.stack([pool(q[layer], len(ids), how, k) for q, ids in zip(per_question, tokens)])
            v = v / v.norm(dim=-1, keepdim=True) * typical
            v = (v - v.mean(0, keepdim=True)).flatten(1)

            def distance(pairs_):
                a = torch.tensor([p[0] for p in pairs_], device=device)
                b = torch.tensor([p[1] for p in pairs_], device=device)
                return float((v[a] - v[b]).norm(dim=1).median())

            scale = distance(unrelated)
            report.append({"layer": layer, "pooling": f"{how}-{k}",
                           **{name: round(distance(p) / scale, 3) for name, p in groups.items()}})

    out = {"questions": len(texts), "states": len(states), "unrelated_pairs": len(unrelated),
           "pairs": {name: len(p) for name, p in groups.items()}, "separation": report}
    (HERE / "results/encoder-candidates.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"{'layer':>6} {'pooling':>9} " + " ".join(f"{name[:24]:>26}" for name in groups))
    for r in report:
        print(f"{str(r['layer']):>6} {r['pooling']:>9} " + " ".join(f"{r[name]:>26.3f}" for name in groups))


if __name__ == "__main__":
    main()
