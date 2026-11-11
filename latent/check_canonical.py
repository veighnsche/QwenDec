"""Pilot 3, before any training: do the fixed wordings say enough?

The text arm of pilot 1 was trained on each question's own wording. Here it reads the
canonical wording instead (evaluate.py --mode text --canonical, tag check-canonical-text),
with no further training. This breaks both results down by type of question.
Single-prerequisite questions keep their own wording in this check: their canonical
form needs a pointer, which text mode does not have.
"""

import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TEMPLATES = ("The eligibility rule is met for", "At least one prerequisite holds for")


def group(c):
    if c["kind"] in ("pa", "pb"):
        return "single prerequisite (own wording in both)"
    if c["kind"] == "or":
        return "at least one prerequisite"
    if not c["proposition"].removeprefix("Negation of: ").startswith(TEMPLATES):
        return "whole rule, free-form state"
    return "whole rule, templated state" + (', prompt asks "is it FALSE that"' if "Is it FALSE" in c["prompt"] else "")


def main():
    cases = {c["id"]: c for c in map(json.loads, open(ROOT / "data/latent/noul-development.jsonl"))}
    out = {}
    for name, tag in (("own wording, trained (pilot 1 text arm)", "pilot-text"),
                      ("canonical wording, same adapter, not trained on it", "check-canonical-text")):
        right, total = Counter(), Counter()
        for r in map(json.loads, open(HERE / f"results/{tag}.cases.jsonl")):
            for key in ("all questions", group(cases[r["id"]])):
                total[key] += 1
                right[key] += r["pick"] == r["gold"]
        out[name] = {k: f"{right[k]}/{total[k]}" for k in sorted(total)}
    (HERE / "results/check-canonical-text-breakdown.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


def vectors():
    """Every question of both splits maps to one of the six requests, and the six vector sets differ."""
    import sys

    import torch

    sys.path.insert(0, str(ROOT / "gpu_loop"))
    sys.path.insert(0, str(HERE))
    from common import pointed, question_text
    from model import require_gpu

    device = require_gpu()
    store = torch.load(ROOT / "data/latent/question-vectors-k16-canonical.pt")
    index, v = store["index"], store["vectors"].to(device).float()
    used = {split: dict() for split in ("train", "development")}
    for split in used:
        for c in map(json.loads, open(ROOT / f"data/latent/noul-{split}.jsonl")):
            text = question_text(c, True, True)
            assert text in index, text
            assert (pointed(c) is not None) == text.startswith(("Is this proposition true? The indicated",
                                                                "Is this proposition true? Negation of: The indicated"))
            used[split][text] = used[split].get(text, 0) + 1
    flat = torch.nn.functional.normalize((v - v.mean(0, keepdim=True)).flatten(1), dim=1)
    cosine = flat @ flat.T
    off = cosine[~torch.eye(len(v), dtype=torch.bool, device=device)]
    smallest_gap = min(float((v[a] - v[b]).abs().max()) for a in range(len(v)) for b in range(a + 1, len(v)))
    out = {"requests": len(index), "questions_per_request": used,
           "largest_cosine_between_two_requests_after_removing_the_mean": round(float(off.max()), 3),
           "smallest_largest_difference_between_two_requests": round(smallest_gap, 4)}
    assert len(index) == 6 and smallest_gap > 0
    (HERE / "results/check-canonical-vectors.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    import sys as _sys

    vectors() if _sys.argv[1:] == ["vectors"] else main()
