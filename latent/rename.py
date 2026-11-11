"""Development's single-prerequisite questions with the two prerequisites renamed.

Development states all call their prerequisites "the first documented prerequisite
holds" and "the second documented prerequisite holds". Those names are unseen in
training, are clauses rather than noun phrases, and differ in one word. This writes
two edited copies of the 464 single-prerequisite questions, for measuring arms that
are already trained. Labels do not change: only the names do.

  nouns    distinct noun phrases, none of them a training name
  clauses  distinct clauses ending in "holds", none of them a training name

Twelve name pairs per version, assigned to states in turn. Tokenizer only, no model.
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from cases import RULE  # noqa: E402
from common import MARK, pointed, pointer_tokens, render  # noqa: E402

OLD = ("the first documented prerequisite holds", "the second documented prerequisite holds")
NAMES = {
    "nouns": [
        ("signed locker agreement", "returned visitor badge"), ("valid parking permit", "completed site induction"),
        ("paid deposit receipt", "approved leave form"), ("current first-aid certificate", "logged equipment check"),
        ("confirmed delivery window", "stamped customs form"), ("registered bicycle tag", "cleared library fines"),
        ("filed expense report", "renewed software licence"), ("passed hearing test", "submitted reference letter"),
        ("booked loading dock", "verified shipping address"), ("charged spare radio", "sealed sample container"),
        ("updated emergency contact", "returned borrowed projector"), ("posted notice period", "countersigned handover sheet"),
    ],
    "clauses": [
        ("the residency requirement holds", "the paid deposit condition holds"),
        ("the age limit condition holds", "the signed waiver requirement holds"),
        ("the insurance cover condition holds", "the notice period requirement holds"),
        ("the membership status condition holds", "the safety briefing requirement holds"),
        ("the stock level condition holds", "the manager approval requirement holds"),
        ("the opening hours condition holds", "the valid ticket requirement holds"),
        ("the weight limit condition holds", "the booking reference requirement holds"),
        ("the weather window condition holds", "the crew size requirement holds"),
        ("the budget ceiling condition holds", "the supplier rating requirement holds"),
        ("the room capacity condition holds", "the fire warden requirement holds"),
        ("the data backup condition holds", "the access badge requirement holds"),
        ("the warranty period condition holds", "the proof of purchase requirement holds"),
    ],
}


def swap(value, pair):
    if isinstance(value, list):
        return [swap(v, pair) for v in value]
    return value.replace(OLD[0], pair[0]).replace(OLD[1], pair[1])


def main():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(ROOT / "Qwen3.5-4B")
    seen = set()
    for c in map(json.loads, open(ROOT / "data/latent/noul-train.jsonl")):
        if c["parsed"]:
            seen |= {n.lower() for n in RULE.match(c["rules"][0]).groups()[1:]}
    cases = [c for c in map(json.loads, open(ROOT / "data/latent/noul-development.jsonl")) if pointed(c) is not None]
    states = sorted({c["base_id"] for c in cases})
    report = {}
    for version, pairs in NAMES.items():
        assert not {n.lower() for p in pairs for n in p} & seen, "a new name occurs in training"
        out = []
        for c in cases:
            pair = pairs[states.index(c["base_id"]) % len(pairs)]
            new = {k: swap(c[k], pair) if k in ("prompt", "facts", "rules", "proposition", "criterion") else c[k] for k in c}
            assert OLD[0] not in json.dumps(new) and OLD[1] not in json.dumps(new)
            before, _ = render(tok, new).split(MARK)
            head = tok(before, add_special_tokens=False, return_offsets_mapping=True)
            for other in (False, True):  # the pointer still decodes to the name
                assert tok.decode(pointer_tokens(before, head, new, other)).strip() == pointed(new, other)
            out.append(new)
        path = ROOT / f"data/latent/noul-development-renamed-{version}.jsonl"
        path.write_text("".join(json.dumps(c) + "\n" for c in out))
        report[version] = {"questions": len(out), "states": len(states), "name_pairs": len(pairs)}
    print(json.dumps(report))


if __name__ == "__main__":
    main()
