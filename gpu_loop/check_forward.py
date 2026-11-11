"""Check the lean forward against the stock Transformers forward on the GPU. No training."""

import sys

import torch

from model import lean_hidden, load_text_model, require_gpu, rope_tables


@torch.no_grad()
def main(path: str, length: int = 512):
    device = require_gpu()
    model = load_text_model(path, device)
    text = model.model
    print("class", type(model).__name__, "| text model", type(text).__name__, "| layers", len(text.layers))
    print("weights on", {p.device.type for p in model.parameters()}, "| dtype", next(model.parameters()).dtype)

    gen = torch.Generator(device=device).manual_seed(0)
    ids = torch.randint(0, 200_000, (2, length), device=device, generator=gen)
    cos, sin = rope_tables(text, length, device)

    ref = model(input_ids=ids, use_cache=False).logits.float()
    got = model.lm_head(lean_hidden(text, ids, cos, sin)).float()
    diff = (got - ref).abs()
    agree = (got.argmax(-1) == ref.argmax(-1)).float().mean().item()
    print(
        f"bit-identical {torch.equal(got, ref)}  max|dlogit| {diff.max().item():.6f}  "
        f"ref logit scale {ref.abs().mean().item():.3f}  top-1 agreement {agree:.4f}"
    )
    print(f"peak memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 512)
