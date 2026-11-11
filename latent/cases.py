"""Build Noul question variants so that the question, not the state, decides the answer.

In the imported cases every state has exactly one question, and the state implies
it, so a model could answer without ever reading the question. Here each state
keeps its text unchanged and gets up to eight questions whose answers differ:

  and   the eligibility rule is met (the original question)
  or    at least one prerequisite holds
  pa    the first prerequisite holds
  pb    the second prerequisite holds
  each also as its negation

Labels come from the stated facts by exact rule, with no model involved. A fact
that is not stated is unknown: a question that depends on it is "insufficient
evidence", and negating it leaves it insufficient. States whose facts do not parse
(conflicting records, ambiguous targets) keep only the original question and its
negation, with the status unchanged.
"""

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RULE = re.compile(r"^For (.+), eligibility holds if and only if both prerequisites hold: (.+); (.+)\.$")
FACT = re.compile(r"^(.+): (yes|no)\.$")


def parse(case):
    """(target, A, B, {name: bool}) when the state is two named prerequisites with yes/no facts."""
    if len(case["rules"]) != 1 or not (m := RULE.match(case["rules"][0])):
        return None
    target, a, b = m.groups()
    facts = {}
    for line in case["facts"]:
        if not (f := FACT.match(line)):
            return None
        facts[f.group(1)] = f.group(2) == "yes"
    return (target, a, b, facts) if set(facts) <= {a, b} else None


def truth(kind, a, b):
    """Three-valued: True, False, or None when the stated facts cannot settle it."""
    if kind == "pa":
        return a
    if kind == "pb":
        return b
    if kind == "and":
        return False if (a is False or b is False) else (True if (a and b) else None)
    return True if (a or b) else (False if (a is False and b is False) else None)


def wording(kind, target, a, b):
    if kind == "and":
        return f"The eligibility rule is met for {target}", f"Both {a} and {b}"
    if kind == "or":
        return f"At least one prerequisite holds for {target}", f"Either {a} or {b}"
    name = a if kind == "pa" else b
    return f"This prerequisite holds for {target}: {name}", f"{name} alone"


def variants(case):
    base = {k: case[k] for k in ("split", "family", "domain", "prompt", "facts", "rules")}
    out = []
    parsed = parse(case) if case["status"] in ("Assessed", "Insufficient") else None
    if parsed:
        target, a, b, facts = parsed
        for kind in ("and", "or", "pa", "pb"):
            proposition, criterion = wording(kind, target, a, b)
            value = truth(kind, facts.get(a), facts.get(b))
            for negated in (False, True):
                out.append({
                    **base, "id": f"{case['id']}-{kind}{'-neg' if negated else ''}", "base_id": case["id"],
                    "kind": kind, "negated": negated,
                    "proposition": f"Negation of: {proposition}" if negated else proposition, "criterion": criterion,
                    "status": "Insufficient" if value is None else "Assessed",
                    "p_true": None if value is None else int(value != negated),
                })
    else:
        q = case["question"]
        for negated in (False, True):
            out.append({
                **base, "id": f"{case['id']}-and{'-neg' if negated else ''}", "base_id": case["id"],
                "kind": "and", "negated": negated,
                "proposition": f"Negation of: {q['proposition']}" if negated else q["proposition"],
                "criterion": q["criterion"], "status": case["status"],
                "p_true": None if case["status"] != "Assessed" else int(bool(case["value"]["p_true"]) != negated),
            })
    for v in out:
        v["twin"] = v["id"][:-4] if v["negated"] else v["id"] + "-neg"
        v["parsed"] = parsed is not None
    return out


def main():
    target = ROOT / "data" / "latent"
    target.mkdir(parents=True, exist_ok=True)
    summary = {}
    for split in ("train", "development"):
        cases = [json.loads(x) for x in open(ROOT / f"data/qwenonetwo/{split}.jsonl")]
        noul = [c for c in cases if c["operation"] == "Noul"]
        rows = [v for c in noul for v in variants(c)]
        disagree = []  # where the rule and the source label differ on the original question
        for c in noul:
            if (p := parse(c)) and c["status"] in ("Assessed", "Insufficient") and not c["question"]["proposition"].startswith("Negation"):
                value = truth("and", p[3].get(p[1]), p[3].get(p[2]))
                if (value is None) != (c["status"] == "Insufficient") or (value is not None and int(value) != c["value"]["p_true"]):
                    disagree.append(c["id"])
        with open(target / f"noul-{split}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        answers = Counter((r["kind"], r["negated"], "true" if r["p_true"] == 1 else "false" if r["p_true"] == 0 else r["status"]) for r in rows)
        per_state = Counter(len({(r["status"], r["p_true"]) for r in rows if r["base_id"] == c["id"]}) for c in noul)
        summary[split] = {
            "noul_states": len(noul), "states_with_eight_questions": sum(parse(c) is not None and c["status"] in ("Assessed", "Insufficient") for c in noul),
            "source_labels_contradicting_the_stated_facts": disagree,
            "questions": len(rows), "by_status": dict(Counter(r["status"] for r in rows)),
            "distinct_answers_per_state": dict(sorted(per_state.items())),
            "answers": {f"{k}{' neg' if n else ''} -> {a}": v for (k, n, a), v in sorted(answers.items())},
        }
    (target / "noul-summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
