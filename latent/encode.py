"""Turn each question into K vectors with a fixed encoder. No training, no labels.

The frozen base model (adapter off) reads the question text on its own. Its final
hidden states are averaged over K equal spans of the question, and each average is
rescaled to the typical length of an input embedding. Nothing here ever sees a
label, the state, or a decision, and nothing updates it.
"""

import argparse
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


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, required=True)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    device = require_gpu()
    tok = AutoTokenizer.from_pretrained(ROOT / "Qwen3.5-4B")
    model = load_text_model(str(ROOT / "Qwen3.5-4B"), device)
    texts = sorted({question_text(json.loads(x)) for split in ("train", "development")
                    for x in open(ROOT / f"data/latent/noul-{split}.jsonl")})
    tokens = [tok(t, add_special_tokens=False)["input_ids"] for t in texts]
    assert min(len(t) for t in tokens) >= args.k
    order = sorted(range(len(texts)), key=lambda i: len(tokens[i]))
    typical = model.model.embed_tokens.weight.float().norm(dim=-1).mean()

    vectors = torch.zeros(len(texts), args.k, model.config.hidden_size, dtype=torch.bfloat16)
    for start in range(0, len(order), 64):
        batch = order[start:start + 64]
        width = max(len(tokens[i]) for i in batch)
        ids = torch.zeros(len(batch), width, dtype=torch.long, device=device)
        for row, i in enumerate(batch):
            ids[row, : len(tokens[i])] = torch.tensor(tokens[i])
        hidden = model.model(input_ids=ids, use_cache=False).last_hidden_state.float()  # right padding is causal-safe
        for row, i in enumerate(batch):
            n = len(tokens[i])
            edges = [round(j * n / args.k) for j in range(args.k + 1)]
            spans = torch.stack([hidden[row, edges[j]:edges[j + 1]].mean(0) for j in range(args.k)])
            vectors[i] = (spans / spans.norm(dim=-1, keepdim=True) * typical).to(torch.bfloat16).cpu()

    out = ROOT / f"data/latent/question-vectors-k{args.k}.pt"
    torch.save({"vectors": vectors, "index": {t: i for i, t in enumerate(texts)}, "k": args.k}, out)
    lengths = [len(t) for t in tokens]
    print(json.dumps({"questions": len(texts), "k": args.k, "question_tokens_mean": round(sum(lengths) / len(lengths), 1),
                      "question_tokens_max": max(lengths), "embedding_norm": round(float(typical), 4), "file": str(out)}))


if __name__ == "__main__":
    main()
