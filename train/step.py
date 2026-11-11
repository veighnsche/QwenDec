"""Fork-mode training step and the on-device loop that runs it.

A step reads its own row of the schedule on the GPU, so the host never chooses a
batch. For each row the state is read with the adapter off and the question and
options with it on; the loss is taken on the option-letter logits at the answer
position. Loss, backward, clipping, AdamW, the non-finite guard, logging and the
step counter are all tensor work.

`Trainer.record()` wraps the step in a CUDA graph while-node. Inside it, one
if-node per length bucket runs when the schedule names that bucket, and one more
applies the update only when the loss and gradient norm are finite. The host
writes `stop`, launches once, and sleeps.
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "gpu_loop"))
from model import lean_hidden, rope_tables  # noqa: E402


class Gate:
    """Per-position on/off multiplier on every LoRA branch. Off positions are native."""

    def __init__(self, peft):
        self.mask = None
        self.branches = 0
        for module in peft.modules():
            if hasattr(module, "lora_B"):
                for linear in module.lora_B.values():
                    linear.register_forward_hook(self.apply)
                    self.branches += 1

    def apply(self, module, inputs, output):
        return output * self.mask.to(output.dtype)


class Trainer:
    def __init__(self, model, peft, data, device, *, lr=5e-5, warmup=50, weight_decay=0.01,
                 clip=1.0, recompute=True, autocast=False, bad_limit=5):
        self.text, self.head = model.model, model.lm_head
        self.gate = Gate(peft)
        self.params = [p for p in peft.parameters() if p.requires_grad]
        # Latent question interface: K positions take question vectors through a trained projection.
        self.vectors = data["vectors"].to(device) if "vectors" in data else None
        if self.vectors is not None:
            hidden = self.vectors.shape[-1]
            self.k_range = torch.arange(self.vectors.shape[1], device=device)
            self.projection = torch.eye(hidden, device=device).requires_grad_(True)
            self.params.append(self.projection)
        self.recompute, self.clip, self.weight_decay = recompute, clip, weight_decay
        # autocast=False keeps the stock numerics: BF16 backbone, FP32 LoRA branch and delta rule.
        self.autocast = autocast
        self.meta = data["meta"]
        self.steps = self.meta["steps"]

        # Data and schedule, resident on the GPU.
        self.buckets = []
        for b in data["buckets"]:
            b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
            b["positions"] = torch.arange(b["length"], device=device)
            self.buckets.append(b)
        self.bucket_of_step = data["bucket_of_step"].to(device)
        self.rows = data["rows"].to(device)
        self.weight = data["weight"].to(device)
        self.letters = data["letters"].to(device)
        longest = max(b["length"] for b in self.buckets)
        self.cos, self.sin = rope_tables(self.text, longest, device)
        ramp = torch.arange(1, self.steps + 1, device=device, dtype=torch.float32) / max(1, warmup)
        self.lr_table = lr * ramp.clamp(max=1.0)

        # Run state, also on the GPU.
        self.t = torch.zeros(1, dtype=torch.long, device=device)
        self.stop = torch.zeros(1, dtype=torch.long, device=device)
        self.bad = torch.zeros(1, dtype=torch.long, device=device)
        self.bad_limit = bad_limit
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.loss_log = torch.full((self.steps,), float("nan"), device=device)
        self.gnorm_log = torch.full((self.steps,), float("nan"), device=device)

    # ---- state ----------------------------------------------------------------
    def state(self):
        return {"t": self.t, "bad": self.bad, "loss_log": self.loss_log, "gnorm_log": self.gnorm_log,
                **{f"p{i}": p for i, p in enumerate(self.params)},
                **{f"m{i}": x for i, x in enumerate(self.m)},
                **{f"v{i}": x for i, x in enumerate(self.v)}}

    def snapshot(self):
        return {k: x.detach().clone() for k, x in self.state().items()}

    @torch.no_grad()
    def restore(self, snap):
        for k, x in self.state().items():
            x.copy_(snap[k])

    def save(self, path):
        torch.save({"data_sha256": self.meta.get("file_sha256"), "state": self.snapshot()}, path)

    def load(self, path):
        saved = torch.load(path, map_location=self.t.device)
        assert saved["data_sha256"] == self.meta.get("file_sha256"), "checkpoint belongs to different prepared data"
        self.restore(saved["state"])

    # ---- one step -------------------------------------------------------------
    def loss_for(self, b, rows, weight):
        d = self.buckets[b]
        ids = d["tokens"].index_select(0, rows).long()
        start = d["gate_start"].index_select(0, rows)
        self.gate.mask = (d["positions"][None, :] >= start[:, None]).unsqueeze(-1)

        embeds = self.text.embed_tokens(ids)
        if self.vectors is not None:
            q = self.vectors.index_select(0, d["q_index"].index_select(0, rows)).float() @ self.projection.T
            where = d["q_start"].index_select(0, rows)[:, None] + self.k_range[None, :]
            embeds = embeds.scatter(1, where[..., None].expand(-1, -1, embeds.shape[-1]), q.to(embeds.dtype))
            if "ptr_ids" in d:
                # Pointer as one vector: the mean input embedding of the pointed tokens, in the position
                # after the K vectors. Not projected. A question with no pointer gets a zero vector.
                mask = d["ptr_mask"].index_select(0, rows)
                tokens = self.text.embed_tokens(d["ptr_ids"].index_select(0, rows).long()).float()
                mean = (tokens * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp(min=1.0)
                at = (d["q_start"].index_select(0, rows) + self.k_range.numel())[:, None, None]
                embeds = embeds.scatter(1, at.expand(-1, 1, embeds.shape[-1]), mean[:, None].to(embeds.dtype))
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.autocast):
            hidden = lean_hidden(self.text, ids, self.cos[:, : d["length"]], self.sin[:, : d["length"]],
                                 self.recompute, inputs_embeds=embeds)
        pos = d["answer_pos"].index_select(0, rows)
        h = hidden.gather(1, pos.view(-1, 1, 1).expand(-1, 1, hidden.shape[-1])).squeeze(1).float()
        logits = h @ self.head.weight.index_select(0, self.letters).float().T  # [batch, options]

        valid = d["valid"].index_select(0, rows)
        target = d["target"].index_select(0, rows)
        logp = logits.masked_fill(~valid, float("-inf")).log_softmax(-1)
        soft = -(target * logp.masked_fill(~valid, 0.0)).sum(-1)
        as_set = -logp.masked_fill(target <= 0, float("-inf")).logsumexp(-1)
        per_row = torch.where(d["credit_set"].index_select(0, rows), as_set, soft)
        return (per_row * weight).sum() / weight.sum()

    def update(self, grads, gnorm):
        scale = (self.clip / (gnorm + 1e-6)).clamp(max=1.0)
        tf = (self.t + 1).float()
        lr = self.lr_table.index_select(0, self.t)
        c1, c2 = 1 - 0.9**tf, 1 - 0.999**tf
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            g = g * scale
            m.mul_(0.9).add_(g, alpha=0.1)
            v.mul_(0.999).addcmul_(g, g, value=0.001)
            p.mul_(1 - lr * self.weight_decay)
            p.sub_(lr / c1 * m / ((v / c2).sqrt() + 1e-8))

    def step_bucket(self, b, graph):
        batch = self.buckets[b]["batch"]
        rows = self.rows.index_select(0, self.t)[0, :batch]
        weight = self.weight.index_select(0, self.t)[0, :batch]
        loss = self.loss_for(b, rows, weight)
        grads = torch.autograd.grad(loss, self.params)
        with torch.no_grad():
            gnorm = torch.stack([g.pow(2).sum() for g in grads]).sum().sqrt()
            ok = (torch.isfinite(loss) & torch.isfinite(gnorm)).reshape(())
            self.loss_log.index_copy_(0, self.t, loss.detach().view(1))
            self.gnorm_log.index_copy_(0, self.t, gnorm.view(1))
            self.bad.add_((~ok).long())
            self.branch(graph, ok, lambda: self.update(grads, gnorm))  # skip the update on a non-finite step

    # ---- control flow that lives on the GPU when recording -------------------------
    @staticmethod
    def branch(graph, pred, body):
        if graph is None:  # eager reference path only
            if bool(pred):
                body()
            return
        graph.begin_capture_to_if_node(pred)
        try:
            body()
        finally:
            graph.end_capture_to_conditional_node()

    def more(self):
        return ((self.t < self.stop) & (self.bad < self.bad_limit)).reshape(())

    def iteration(self, graph=None):
        bucket = self.bucket_of_step.index_select(0, self.t)
        for b in range(len(self.buckets)):
            self.branch(graph, (bucket == b).reshape(()), lambda b=b: self.step_bucket(b, graph))
        with torch.no_grad():
            self.t.add_(1)

    def warm_up(self):
        """Run every bucket once eagerly so nothing initialises lazily during recording."""
        snap = self.snapshot()
        for b in range(len(self.buckets)):
            self.t.fill_(int((self.bucket_of_step == b).nonzero()[0]))
            self.step_bucket(b, None)
        self.restore(snap)  # also puts the step counter back
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    def record(self):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph.begin_capture_to_while_node(self.more())
            self.iteration(graph)
            graph.set_conditional_handle_for_current_node(self.more())
            graph.end_capture_to_conditional_node()
        return graph

    def run_eager(self, until):
        """Reference path: the same steps driven from Python."""
        while int(self.t) < until and int(self.bad) < self.bad_limit:
            self.iteration(None)
