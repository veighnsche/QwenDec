"""Judge the produced requests against the rule in REQUEST.md. Bookkeeping only, no model.

  judge.py panels   parts 1 and 2 from what the head produced, and the two development
                    panels rewritten to carry the produced pointers in place of the handed ones
  judge.py final    part 3, after the accepted adapter has answered those panels

A produced range counts as an item if it reads exactly as the item's name, at any of its
mentions. A state's items are right when every item is found once and nothing else is.
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from reading import characters_of, native_pass, reads_as, state_text  # noqa: E402

NEEDED = {"1 operation": 639, "2 items": 312, "3 Noul end to end": 1082, "3 Choice end to end": 910, "3 Score end to end": 282}
HANDED = {"Noul": "words-noul", "Choice": "words-choice", "Score": "words-score"}


def load(path):
    return [json.loads(x) for x in open(path)]


def match(record, produced):
    """Which item each produced text reads as; None if the item set is not exactly right."""
    names = [i["name"] for i in record["items"]]
    found = []
    for text in produced["texts"]:
        hits = [n for n in names if reads_as(text, n)]
        if len(hits) != 1:
            return None
        found.append(hits[0])
    return found if sorted(found) == sorted(names) else None


def status():
    """Per state: is the operation right, and are the items right."""
    produced = {p["id"]: p for p in json.loads((HERE / "results/produced.json").read_text())}
    out = {}
    for split in ("train", "development"):
        for r in load(ROOT / f"data/native/states-{split}.jsonl"):
            if r["id"] not in produced:
                continue
            p = produced[r["id"]]
            found = match(r, p)
            out[r["id"]] = {"split": split, "operation": r["operation"], "produced_operation": p["operation"],
                            "operation_right": p["operation"] == r["operation"], "has_items": bool(r["items"]),
                            "items_right": found is not None, "found": found, "ranges": p["ranges"], "record": r}
    return out


def rate(hits):
    return f"{sum(hits)}/{len(hits)}"


def panels():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(ROOT / "Qwen3.5-4B")
    st = status()
    # Where each produced Choice pointer lies in the state, as characters; the panel is rewritten with these below.
    where, moved = {}, Counter()
    choice_states = {r["id"]: r for r in load(ROOT / "data/native/states-development.jsonl") if r["operation"] == "Choice"}
    for i, r in choice_states.items():
        s = st[i]
        if not s["items_right"]:
            continue
        state = state_text(r)
        _, offsets, _, before = native_pass(tok, state)
        spans = {name: characters_of(offsets, before, state, lo, hi) for name, (lo, hi) in zip(s["found"], s["ranges"])}
        if any(v is None for v in spans.values()):
            s["items_right"] = False  # a range that covers no whole character of the state
            continue
        where[i] = {item["identity"]: spans[item["name"]] for item in r["items"]}
        produced_range = dict(zip(s["found"], s["ranges"]))
        moved["Choice pointers at another place than the handed one"] += sum(list(produced_range[item["name"]]) != item["token_range"] for item in r["items"])
    dev = [s for s in st.values() if s["split"] == "development"]
    held = [s for s in st.values() if s["split"] == "train"]
    hand = json.loads((ROOT / "latent/choice_names.json").read_text())["names"]
    groups = defaultdict(list)
    for s in dev:
        r = s["record"]
        groups[f"operation, {s['operation']} states"].append(s["operation_right"])
        if s["has_items"]:
            groups[f"items, {s['operation']} states"].append(s["items_right"])
            if s["operation"] == "Choice":
                letters = all(len(i["identity"]) == 1 for i in r["items"])
                by_hand = any(i["identity"] in hand.get(r["family"], {}) for i in r["items"])
                groups["items, Choice, " + ("single letters" if letters else "other names, mapped by hand" if by_hand else "other names, found by rule")].append(s["items_right"])
        else:
            groups[f"no items produced where there are none, {s['operation']} states"].append(not s["ranges"])
    report = {
        "part 1 operation": {"measured": sum(s["operation_right"] for s in dev), "of": len(dev), "needed": NEEDED["1 operation"]},
        "part 2 items": {"measured": sum(s["items_right"] for s in dev if s["has_items"]), "of": sum(s["has_items"] for s in dev), "needed": NEEDED["2 items"]},
        "development": {k: rate(v) for k, v in sorted(groups.items())},
        "operations produced, development": dict(Counter(f"{s['operation']} read as {s['produced_operation']}" for s in dev)),
        "held-out training states": {"operation": rate([s["operation_right"] for s in held]),
                                     "items": rate([s["items_right"] for s in held if s["has_items"]])},
    }
    for part in ("part 1 operation", "part 2 items"):
        report[part]["met"] = report[part]["measured"] >= report[part]["needed"]

    # The panels again, with the produced pointers where the state's items are right.
    # Where they are not, the handed pointers stay; those questions are counted wrong in part 3 whatever the answer.
    noul = load(ROOT / "data/latent/noul-development-panel4.jsonl")
    for c in noul:
        s = st[c["base_id"]]
        if s["has_items"] and s["items_right"]:
            ids, _, _, _ = native_pass(tok, state_text(c))
            a_name, b_name = [i["name"] for i in s["record"]["items"]]  # in the order of the rule sentence
            spans = {name: ids[lo:hi + 1] for name, (lo, hi) in zip(s["found"], s["ranges"])}
            c["produced"] = {"a": spans[a_name], "b": spans[b_name]}
            handed = {i["name"]: ids[i["token_range"][0]:i["token_range"][1] + 1] for i in s["record"]["items"]}
            if c["kind"] == "and" and not c["negated"]:  # once per state
                moved["Noul pointers at another place than the handed one"] += sum(spans[n] != handed[n] for n in handed)
    (ROOT / "data/native/noul-development-panel4-produced.jsonl").write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in noul))
    choice = load(ROOT / "data/latent/choice-development.jsonl")
    for c in choice:
        if c["base_id"] in where:
            for a in c["alternatives"]:
                a["span"] = where[c["base_id"]][a["identity"]]
    (ROOT / "data/native/choice-development-produced.jsonl").write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in choice))
    report["pointers moved"] = dict(moved)
    (HERE / "results/request-status.json").write_text(json.dumps(
        {i: {k: v for k, v in s.items() if k != "record"} for i, s in st.items()}, ensure_ascii=False, indent=1) + "\n")
    (HERE / "results/parts-1-2.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))


def final():
    st = json.loads((HERE / "results/request-status.json").read_text())
    report = json.loads((HERE / "results/parts-1-2.json").read_text())
    sources = {"Noul": ("native-noul", "data/latent/noul-development-panel4.jsonl", lambda c, r: r["pick"] == r["gold"],
                        lambda c: c["kind"] in ("pa", "pb")),
               "Choice": ("native-choice", "data/latent/choice-development.jsonl", lambda c, r: r["right"], lambda c: True),
               # a Score request has no pointers, so the adapter's answers are those it gave with handed requests
               "Score": ("words-score", "data/latent/score-development.jsonl", lambda c, r: r["right"], lambda c: False)}
    for op, (tag, file, decided, needs_pointers) in sources.items():
        cases = {c["id"]: c for c in load(ROOT / file)}
        rows = {r["id"]: r for r in load(ROOT / f"latent/results/{tag}.cases.jsonl")}
        handed = {r["id"]: r for r in load(ROOT / f"latent/results/{HANDED[op]}.cases.jsonl")}
        assert rows.keys() == cases.keys()
        right, lost = [], Counter()
        for i, c in cases.items():
            s = st[c["base_id"]]
            request = s["operation_right"] and (s["items_right"] or not needs_pointers(c))
            answer = decided(c, rows[i])
            right.append(request and answer)
            if not s["operation_right"]:
                lost["operation wrong"] += 1
            elif not request:
                lost["items wrong"] += 1
            elif not answer and decided(c, handed[i]):
                lost["answer wrong with the produced pointer, right with the handed one"] += 1
        key = f"3 {op} end to end"
        report[f"part {key}"] = {"measured": sum(right), "of": len(right), "needed": NEEDED[key],
                                 "with handed requests": sum(decided(cases[i], handed[i]) for i in cases),
                                 "met": sum(right) >= NEEDED[key], "questions lost to": dict(lost)}
    report["passed"] = all(v["met"] for k, v in report.items() if k.startswith("part "))
    (HERE / "results/judged.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    {"panels": panels, "final": final}[sys.argv[1]]()
