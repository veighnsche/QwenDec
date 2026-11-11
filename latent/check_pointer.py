"""Checks of the pilot 2 request before any training. Tokenizer only, no model.

1. Every pointer range, taken from the tokenised state, decodes to exactly the
   prerequisite's name (apart from the leading space), for both prerequisites of
   every parsed state in both splits.
2. The two single-prerequisite questions of a state have the same name-free text,
   so their question vectors are identical and only the pointer tells them apart.
3. The two pointers of a state are different token ranges.
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from common import MARK, pointed, pointer_tokens, question_text, render  # noqa: E402


def main():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(ROOT / "Qwen3.5-4B")
    report = {}
    for split in ("train", "development"):
        cases = [json.loads(x) for x in open(ROOT / f"data/latent/noul-{split}.jsonl")]
        by_id = {c["id"]: c for c in cases}
        checked, lengths, names = 0, [], set()
        for c in cases:
            name = pointed(c)
            if name is None:
                assert question_text(c, True) == question_text(c), "a question without a pointer changed"
                continue
            before, _ = render(tok, c).split(MARK)
            head = tok(before, add_special_tokens=False, return_offsets_mapping=True)
            for other in (False, True):
                ids = pointer_tokens(before, head, c, other)
                want = json.dumps(pointed(c, other), ensure_ascii=False)[1:-1]
                assert tok.decode(ids).strip() == want, (c["id"], tok.decode(ids), want)
            mine, theirs = pointer_tokens(before, head, c), pointer_tokens(before, head, c, True)
            assert mine != theirs, f"{c['id']}: both pointers are the same tokens"
            sibling = by_id[f"{c['base_id']}-{'pb' if c['kind'] == 'pa' else 'pa'}{'-neg' if c['negated'] else ''}"]
            assert question_text(c, True) == question_text(sibling, True), "sibling questions differ in text"
            assert name not in question_text(c, True), "the name-free question contains the name"
            checked += 1
            lengths.append(len(mine))
            names.add(name)
        lengths.sort()
        report[split] = {"questions": len(cases), "with_pointer": checked, "distinct_names": len(names),
                         "pointer_tokens_min_median_max": [lengths[0], lengths[len(lengths) // 2], lengths[-1]]}
    print(json.dumps(report, indent=1))
    (HERE / "results/check-pointer.json").write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
