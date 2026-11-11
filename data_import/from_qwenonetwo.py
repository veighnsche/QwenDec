"""Import the annotation records from QwenOneTwo into a neutral format.

Takes the content of each case: what the user asked, the supplied facts and rules,
the question being decided, and the labelled outcome. Leaves behind everything that
described how QwenOneTwo fed or trained its model (roles, packets, token mappings,
exposure and loss settings). The locked evaluation file is not read.
"""

import hashlib
import json
from collections import Counter
from pathlib import Path

SOURCE = Path("/home/vince/QwenOneTwo/data/training-collections-20261105")
TARGET = Path(__file__).resolve().parent.parent / "data" / "qwenonetwo"
FILES = {"train": "train-examples.jsonl", "development": "development-examples.jsonl"}


def neutral(record: dict) -> dict:
    labels, semantics = record["labels"], record["semantics"]
    operation, payload = record["operation"], labels["payload"]
    if payload is None:
        value = None
    elif operation == "Noul":
        value = {"p_true": payload["p_true"]}
    elif operation == "Choice":
        value = {"acceptable_ids": payload["acceptable_selection_ids"]}
    else:
        value = {"score_interval": payload["score_interval"]}
    assert (value is not None) == (labels["status"] == "Assessed"), record["id"]

    out = {
        "split": record["split"],
        "family": record["family"],
        "domain": record["domain"],
        "operation": operation,
        "prompt": record["human_input"]["prompt"],
        "facts": record["supplied_facts"],
        "rules": record["supplied_rules"],
        "question": semantics["question"],
        "target": semantics["target"]["referent"],
        "status": labels["status"],
        "value": value,
        "reason": record["reason"]["support"] if record["reason"] else None,
        "answer": record["guidance"]["acceptable_continuations"][0],
        "label_reviewed": not labels.get("unreviewed", False),
    }
    if operation == "Choice":
        out["alternatives"] = semantics["alternatives"]
        out["feasibility"] = labels["feasibility"]
    if operation == "Score":
        out["rubric"] = semantics["rubric"]
    return out


def main():
    TARGET.mkdir(parents=True, exist_ok=True)
    manifest = {"source": str(SOURCE), "locked_file_read": False, "files": {}}
    for split, name in FILES.items():
        raw = (SOURCE / name).read_bytes()
        seen, rows = {}, []
        for line in raw.splitlines():
            record = json.loads(line)
            row = neutral(record)
            key = hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]
            if key in seen:
                seen[key]["source_ids"].append(record["id"])
                continue
            row = {"id": f"{split}-{key}", **row, "source_ids": [record["id"]]}
            seen[key] = row
            rows.append(row)
        with open(TARGET / f"{split}.jsonl", "w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        cells = Counter(f"{r['operation']}/{r['status']}" for r in rows)
        manifest["files"][split] = {
            "source_file": name,
            "source_sha256": hashlib.sha256(raw).hexdigest(),
            "source_records": len(raw.splitlines()),
            "distinct_cases": len(rows),
            "families": len({r["family"] for r in rows}),
            "unreviewed_labels": sum(not r["label_reviewed"] for r in rows),
            "cells": dict(sorted(cells.items())),
        }
    (TARGET / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
