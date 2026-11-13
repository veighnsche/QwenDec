"""The reader on the native model's states, and the on-device loop that trains it.

The native model is frozen and is not part of this file: its final-layer states are
computed once and held on the GPU as one flat tensor. The head reads them.

  operation  one linear layer on the state's last position
  items      SLOTS slots, in the order the items appear. For each slot, attention from
             the last position over the state's tokens gives where the item starts and
             where it ends; one more option per slot says the slot is empty.

`HeadTrainer.record()` wraps the step in a CUDA graph while-node, as train/step.py does
for the adapter: a step reads its own row of the schedule on the GPU, and loss,
backward, clipping, AdamW and the step counter are tensor work. The host writes `stop`,
launches once and sleeps.
"""

import torch

SLOTS, DIM, OPERATIONS = 3, 256, 3


class Head(torch.nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.norm = torch.nn.LayerNorm(hidden)
        self.operation = torch.nn.Linear(hidden, OPERATIONS)
        self.keys = torch.nn.Linear(hidden, DIM)
        self.queries = torch.nn.Linear(hidden, 2 * SLOTS * DIM)  # a start and an end query per slot
        self.empty = torch.nn.Linear(hidden, 2 * SLOTS)

    def forward(self, states, last, pointable):
        """states [batch, length, hidden]; last [batch] index of the state's last position;
        pointable [batch, length] true on the state's own tokens.

        Returns operation logits [batch, 3] and place logits [batch, 2*SLOTS, length+1];
        the extra last column is "this slot is empty".
        """
        h = self.norm(states)
        end = h.gather(1, last.view(-1, 1, 1).expand(-1, 1, h.shape[-1])).squeeze(1)
        q = self.queries(end).view(-1, 2 * SLOTS, DIM)
        scores = torch.bmm(q, self.keys(h).transpose(1, 2)) / DIM ** 0.5
        scores = scores.masked_fill(~pointable.unsqueeze(1), float("-inf"))
        return self.operation(end), torch.cat([scores, self.empty(end).unsqueeze(-1)], dim=-1)


def read_out(place_logits, length):
    """Token ranges from place logits of one state: a list of (first, last), slots in order, empty slots dropped."""
    out = []
    for slot in range(SLOTS):
        start = int(place_logits[2 * slot].argmax())
        if start == length:  # the slot is empty
            continue
        ends = place_logits[2 * slot + 1, :length].clone()
        ends[:start] = float("-inf")  # an item cannot end before it starts
        out.append((start, int(ends.argmax())))
    return out


class HeadTrainer:
    def __init__(self, head, data, device, *, lr=1e-3, weight_decay=0.01, clip=1.0, bad_limit=5):
        self.head, self.clip, self.weight_decay, self.lr = head, clip, weight_decay, lr
        self.params = [p for p in head.parameters()]
        self.features = data["features"]  # [all tokens, hidden], on the GPU
        self.offset, self.tokens, self.first = data["offset"], data["tokens"], data["first"]
        self.operation, self.places = data["operation"], data["places"]
        self.rows, self.weight = data["rows"], data["weight"]  # [steps, batch]
        self.steps, self.length = self.rows.shape[0], data["length"]
        self.positions = torch.arange(self.length, device=device)
        self.limit = self.features.shape[0] - 1

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

    # ---- one step -------------------------------------------------------------
    def batch(self, rows):
        """The states of some rows, padded to the common length, with what the head needs to read them."""
        n = self.tokens.index_select(0, rows)
        where = self.offset.index_select(0, rows)[:, None] + self.positions[None, :]
        states = self.features.index_select(0, where.clamp(max=self.limit).reshape(-1)).view(rows.shape[0], self.length, -1).float()
        inside = self.positions[None, :] < n[:, None]
        pointable = inside & (self.positions[None, :] >= self.first.index_select(0, rows)[:, None])
        return states, n - 1, pointable

    def loss_for(self, rows, weight):
        states, last, pointable = self.batch(rows)
        operation, places = self.head(states, last, pointable)
        per_row = torch.nn.functional.cross_entropy(operation, self.operation.index_select(0, rows), reduction="none")
        target = self.places.index_select(0, rows)  # [batch, 2*SLOTS], the common length meaning "empty"
        place_loss = torch.nn.functional.cross_entropy(places.reshape(-1, places.shape[-1]), target.reshape(-1), reduction="none")
        per_row = per_row + place_loss.view(rows.shape[0], -1).mean(1)
        return (per_row * weight).sum() / weight.sum()

    def update(self, grads, gnorm):
        scale = (self.clip / (gnorm + 1e-6)).clamp(max=1.0)
        tf = (self.t + 1).float()
        c1, c2 = 1 - 0.9**tf, 1 - 0.999**tf
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            g = g * scale
            m.mul_(0.9).add_(g, alpha=0.1)
            v.mul_(0.999).addcmul_(g, g, value=0.001)
            p.mul_(1 - self.lr * self.weight_decay)
            p.sub_(self.lr / c1 * m / ((v / c2).sqrt() + 1e-8))

    def step(self, graph):
        rows = self.rows.index_select(0, self.t)[0]
        weight = self.weight.index_select(0, self.t)[0]
        loss = self.loss_for(rows, weight)
        grads = torch.autograd.grad(loss, self.params)
        with torch.no_grad():
            gnorm = torch.stack([g.pow(2).sum() for g in grads]).sum().sqrt()
            ok = (torch.isfinite(loss) & torch.isfinite(gnorm)).reshape(())
            self.loss_log.index_copy_(0, self.t, loss.detach().view(1))
            self.gnorm_log.index_copy_(0, self.t, gnorm.view(1))
            self.bad.add_((~ok).long())
            self.branch(graph, ok, lambda: self.update(grads, gnorm))  # skip the update on a non-finite step
            self.t.add_(1)

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

    def warm_up(self):
        """Run one step eagerly so nothing initialises lazily during recording."""
        snap = self.snapshot()
        self.step(None)
        self.restore(snap)
        torch.cuda.synchronize()

    def record(self):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph.begin_capture_to_while_node(self.more())
            self.step(graph)
            graph.set_conditional_handle_for_current_node(self.more())
            graph.end_capture_to_conditional_node()
        return graph

    def run_eager(self, until):
        """Reference path: the same steps driven from Python."""
        while int(self.t) < until and int(self.bad) < self.bad_limit:
            self.step(None)
