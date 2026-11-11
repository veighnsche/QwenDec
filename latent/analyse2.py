"""Pilot 2: where each pointer arm stands, beyond the pass rule's five numbers.

1. Development accuracy split by whether the state parses (only parsed states have
   single-prerequisite questions and pointers) and by gold status.
2. What moving the pointer to the other prerequisite changed: how many answers
   changed at all, and by how much.
3. How alike the one-vector pointers of a state's two prerequisites are (mean
   input embedding of the name's tokens), in development and in training.

No training. The embedding arithmetic of part 3 runs on the RTX 3090.
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
from common import MARK, pointed, pointer_tokens, render  # noqa: E402
from model import load_text_model, require_gpu  # noqa: E402

ARMS = {"text": "pilot-text", "16 vectors (pilot 1)": "pilot-latent-k16",
        "pointer as one vector": "pilot2-pointer-vector", "pointer as replayed span": "pilot2-pointer-span"}


def rows(tag):
    return {r["id"]: r for r in map(json.loads, open(HERE / f"results/{tag}.cases.jsonl"))}


@torch.no_grad()
def main():
    from transformers import AutoTokenizer

    device = require_gpu()
    cases = {c["id"]: c for c in map(json.loads, open(ROOT / "data/latent/noul-development.jsonl"))}
    report = {"by_state_kind": {}, "pointer_moved": {}, "pointer_vectors": {}}

    for name, tag in ARMS.items():
        right, total = Counter(), Counter()
        for i, r in rows(tag).items():
            c = cases[i]
            kind = "states that parse" if c["parsed"] else "states that do not parse"
            for key in (kind, f"{kind}, {c['status']}"):
                total[key] += 1
                right[key] += r["pick"] == r["gold"]
        report["by_state_kind"][name] = {k: f"{right[k]}/{total[k]}" for k in sorted(total)}

    with_pointer = [i for i, c in cases.items() if pointed(c) is not None]
    for arm in ("vector", "span"):
        a, b = rows(f"pilot2-pointer-{arm}"), rows(f"pilot2-pointer-{arm}-swap-pointer")
        report["pointer_moved"][arm] = {
            "questions_with_pointer": len(with_pointer),
            "answers_changed": sum(a[i]["pick"] != b[i]["pick"] for i in with_pointer),
            "largest_probability_change": round(max(abs(x - y) for i in with_pointer
                                                    for x, y in zip(a[i]["probs"], b[i]["probs"])), 4),
            "other_questions_changed": sum(a[i]["probs"] != b[i]["probs"] for i in a if i not in set(with_pointer)),
        }

    tok = AutoTokenizer.from_pretrained(ROOT / "Qwen3.5-4B")
    table = load_text_model(str(ROOT / "Qwen3.5-4B"), device).model.embed_tokens.weight.float()
    for split in ("train", "development"):
        cosines, seen = [], set()
        for c in map(json.loads, open(ROOT / f"data/latent/noul-{split}.jsonl")):
            if pointed(c) is None or c["base_id"] in seen:
                continue
            seen.add(c["base_id"])
            before, _ = render(tok, c).split(MARK)
            head = tok(before, add_special_tokens=False, return_offsets_mapping=True)
            first, second = (table[torch.tensor(pointer_tokens(before, head, c, other), device=device)].mean(0)
                             for other in (False, True))
            cosines.append(float(torch.nn.functional.cosine_similarity(first, second, dim=0)))
        cosines.sort()
        report["pointer_vectors"][split] = {
            "states": len(cosines), "cosine_between_a_states_two_pointers": {
                "min": round(cosines[0], 3), "median": round(cosines[len(cosines) // 2], 3), "max": round(cosines[-1], 3)}}

    (HERE / "results/analysis-pilot2.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
