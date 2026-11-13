"""The two Score request vector sets differ from each other and from the eight existing ones.

Arithmetic on stored vectors only. No model is run.
"""

import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent


def main():
    score = torch.load(ROOT / "data/latent/question-vectors-k16-score.pt")
    joint = torch.load(ROOT / "data/latent/question-vectors-k16-joint.pt")
    unit = lambda v: torch.nn.functional.normalize(v.float().flatten(1), dim=1)
    a, b = unit(score["vectors"]), unit(joint["vectors"])
    report = {"score against complement": round(float(a[0] @ a[1]), 3),
              "highest against a Noul or Choice request": round(float((a @ b.T).max()), 3),
              "lowest against a Noul or Choice request": round(float((a @ b.T).min()), 3)}
    assert report["score against complement"] < 0.999 and report["highest against a Noul or Choice request"] < 0.999
    (ROOT / "latent/results/score-vectors.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
