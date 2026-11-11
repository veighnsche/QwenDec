"""Mechanical proof: complete training steps with no host work in the loop.

Everything a step needs lives on the GPU: the dataset, the sampling order, the
learning-rate table, the step counter, the optimizer state and the metric logs.
One step = sample batch -> forward -> combined loss -> backward -> clip -> AdamW ->
log -> advance counter.

Three ways to run the same steps are measured:
  eager  ordinary PyTorch; Python and the autograd engine launch every kernel
  graph  one step recorded as a CUDA graph; the host replays it once per step
  loop   the step recorded inside an on-device while loop; the host launches once
         and the GPU runs every step by itself until the counter reaches `stop`

This measures throughput and host CPU only. The data are random tokens and the
targets are random; nothing here is evidence of learning, and no weights are saved.
"""

import argparse
import json
import subprocess
import time

import torch
import torch.nn.functional as F

from model import RTX3090_UUID, lean_hidden, load_text_model, require_gpu, rope_tables


class GpuRun:
    def __init__(self, model, params, *, batch, length, steps, readouts, checkpoint, device):
        self.text, self.head, self.params = model.model, model.lm_head, params
        self.checkpoint = checkpoint
        gen = torch.Generator(device=device).manual_seed(1234)
        records = 2048

        # Dataset, order and schedule: resident on the GPU for the whole run.
        self.tokens = torch.randint(0, 200_000, (records, length), device=device, generator=gen)
        self.teacher = torch.rand(records, 4, device=device, generator=gen).softmax(-1)
        self.order = torch.randint(0, records, (steps, batch), device=device, generator=gen)
        warm = max(1, steps // 10)
        ramp = torch.arange(1, steps + 1, device=device, dtype=torch.float32) / warm
        self.lr_table = 1e-4 * ramp.clamp(max=1.0)
        self.readout_pos = torch.linspace(0, length - 2, readouts, device=device).long()
        self.option_rows = torch.arange(1000, 5000, 1000, device=device)
        self.cos, self.sin = rope_tables(self.text, length, device)

        # Run state: also on the GPU, so a step never asks the host for anything.
        self.t = torch.zeros(1, dtype=torch.long, device=device)
        self.stop = torch.zeros(1, dtype=torch.long, device=device)
        self.m = [torch.zeros_like(p) for p in params]
        self.v = [torch.zeros_like(p) for p in params]
        self.loss_log = torch.zeros(steps, device=device)
        self.gnorm_log = torch.zeros(steps, device=device)

    def state(self):
        return [self.t, self.loss_log, self.gnorm_log, *self.params, *self.m, *self.v]

    def snapshot(self):
        return [x.detach().clone() for x in self.state()]

    @torch.no_grad()
    def restore(self, snap):
        for x, s in zip(self.state(), snap):
            x.copy_(s)

    def more(self):
        """On-device loop condition: a boolean scalar that never leaves the GPU."""
        return (self.t < self.stop).reshape(())

    def step(self):
        idx = self.order.index_select(0, self.t).view(-1)
        ids = self.tokens.index_select(0, idx)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = lean_hidden(self.text, ids, self.cos, self.sin, self.checkpoint)
        # Objective 1: next-token cross-entropy at fixed readout positions.
        logits = self.head(hidden.index_select(1, self.readout_pos)).float()
        targets = ids.index_select(1, self.readout_pos + 1)
        ce = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        # Objective 2: distillation to cached teacher probabilities over four option rows.
        rows = self.head.weight.index_select(0, self.option_rows).float()
        option_logits = hidden[:, -1].float() @ rows.T
        kd = -(self.teacher.index_select(0, idx) * option_logits.log_softmax(-1)).sum(-1).mean()
        loss = ce + kd

        grads = torch.autograd.grad(loss, self.params)  # one backward for the combined loss
        with torch.no_grad():
            gnorm = torch.stack([g.pow(2).sum() for g in grads]).sum().sqrt()
            clip = (1.0 / (gnorm + 1e-6)).clamp(max=1.0)
            tf = (self.t + 1).float()
            lr = self.lr_table.index_select(0, self.t)
            c1, c2 = 1 - 0.9**tf, 1 - 0.999**tf
            for p, g, m, v in zip(self.params, grads, self.m, self.v):
                g = g * clip
                m.mul_(0.9).add_(g, alpha=0.1)
                v.mul_(0.999).addcmul_(g, g, value=0.001)
                p.mul_(1 - lr * 0.01)
                p.sub_(lr / c1 * m / ((v / c2).sqrt() + 1e-8))
            self.loss_log.index_copy_(0, self.t, loss.detach().view(1))
            self.gnorm_log.index_copy_(0, self.t, gnorm.view(1))
            self.t.add_(1)


class Waiter:
    """Wait for the GPU without burning a CPU core."""

    def __init__(self, blocking: bool, poll_s: float):
        self.blocking, self.poll_s = blocking, poll_s

    def __call__(self):
        if self.blocking:
            torch.cuda.synchronize()  # sleeps in the kernel when blocking sync is enabled
            return
        event = torch.cuda.Event()
        event.record()
        while not event.query():
            time.sleep(self.poll_s)


def smi(field: str) -> str:
    return subprocess.run(
        ["nvidia-smi", "-i", RTX3090_UUID, f"--query-gpu={field}", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


class Meter:
    """Wall time, CPU time of this process (all threads) and sampled GPU utilisation."""

    def __enter__(self):
        self.smi = subprocess.Popen(
            ["nvidia-smi", "-i", RTX3090_UUID, "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits", "-lms", "200"],
            stdout=subprocess.PIPE, text=True,
        )
        time.sleep(0.6)
        self.cpu, self.wall = time.process_time(), time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.wall = time.perf_counter() - self.wall
        self.cpu = time.process_time() - self.cpu
        self.smi.terminate()
        samples = [int(x) for x in self.smi.communicate()[0].split()[3:]]
        self.gpu_util = sum(samples) / max(1, len(samples))

    def report(self, name, steps, positions):
        row = {
            "phase": name,
            "steps": steps,
            "wall_ms_per_step": round(1000 * self.wall / steps, 1),
            "cpu_ms_per_step": round(1000 * self.cpu / steps, 2),
            "cpu_percent_of_one_core": round(100 * self.cpu / self.wall, 2),
            "gpu_util_percent": round(self.gpu_util, 1),
            "positions_per_s": round(positions * steps / self.wall, 1),
            "gpu_memory_used_mib": int(smi("memory.used")),
        }
        print(json.dumps(row), flush=True)
        return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--length", type=int, default=512)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--readouts", type=int, default=64)
    ap.add_argument("--blocking-sync", action="store_true")
    ap.add_argument("--poll-ms", type=float, default=10.0)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--checkpoint", action="store_true", help="recompute layer activations in backward to save memory")
    ap.add_argument("--skip", default="", help="comma list of phases to skip: eager,graph,loop")
    args = ap.parse_args()
    skip = set(filter(None, args.skip.split(",")))

    from peft import PeftModel

    device = require_gpu(blocking_sync=args.blocking_sync)
    if args.deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
    wait = Waiter(args.blocking_sync, args.poll_ms / 1000)

    model = load_text_model(args.model, device)
    peft = PeftModel.from_pretrained(model, args.adapter, is_trainable=True)
    params = [p for p in peft.parameters() if p.requires_grad]
    assert all(p.is_cuda for p in peft.parameters()), "a model tensor is on the CPU"
    print(f"trainable tensors {len(params)}  trainable params {sum(p.numel() for p in params):,}  dtype {params[0].dtype}")

    n = args.steps
    run = GpuRun(model, params, batch=args.batch, length=args.length, steps=n,
                 readouts=args.readouts, checkpoint=args.checkpoint, device=device)
    positions = args.batch * args.length
    init = run.snapshot()
    rows, losses, finals = [], {}, {}

    def finish(name, meter):
        rows.append(meter.report(name, n, positions))
        losses[name] = run.loss_log.clone()
        finals[name] = torch.cat([p.detach().flatten() for p in params]).clone()
        assert int(run.t) == n, f"{name}: ran {int(run.t)} steps, expected {n}"
        run.restore(init)
        wait()

    # Warm-up (cuBLAS/cuDNN handles, allocator); needed before any capture.
    run.step(); wait(); run.restore(init); wait()

    if "eager" not in skip:
        with Meter() as meter:
            for _ in range(n):
                run.step()
            wait()
        finish("eager", meter)
    torch.cuda.empty_cache()

    if "graph" not in skip:
        graph = torch.cuda.CUDAGraph()
        t0 = time.perf_counter()
        with torch.cuda.graph(graph):
            run.step()
        print(f"graph capture took {time.perf_counter() - t0:.2f} s")
        run.restore(init); wait()
        launch = []
        with Meter() as meter:
            for _ in range(n):
                t0 = time.perf_counter()
                graph.replay()
                launch.append(time.perf_counter() - t0)
                wait()
        print(f"host time inside replay(): median {1000 * sorted(launch)[n // 2]:.2f} ms")
        finish("graph", meter)
        del graph
        torch.cuda.empty_cache()

    if "loop" not in skip:
        loop = torch.cuda.CUDAGraph()
        t0 = time.perf_counter()
        with torch.cuda.graph(loop):
            loop.begin_capture_to_while_node(run.more())
            run.step()
            loop.set_conditional_handle_for_current_node(run.more())
            loop.end_capture_to_conditional_node()
        print(f"loop capture took {time.perf_counter() - t0:.2f} s")
        run.restore(init); wait()
        with Meter() as meter:
            run.stop.fill_(n)  # the host's whole contribution: "run until step n"
            t0 = time.perf_counter()
            loop.replay()
            launch_ms = 1000 * (time.perf_counter() - t0)
            wait()
        print(f"host time inside the single launch: {launch_ms:.2f} ms for {n} steps")
        finish("loop", meter)

    # The only host reads of training state happen here, after the runs.
    names = list(losses)
    for name in names:
        print(f"{name:6s} loss {[round(x, 4) for x in losses[name].tolist()[:5]]} ... finite {bool(torch.isfinite(losses[name]).all())}")
    for a, b in zip(names, names[1:]):
        rel = ((losses[a] - losses[b]).abs() / losses[a].abs()).max().item()
        print(f"{a} vs {b}: losses bit-identical {torch.equal(losses[a], losses[b])}, max relative gap {rel:.2e}; "
              f"final weights bit-identical {torch.equal(finals[a], finals[b])}")
    print(json.dumps({"config": vars(args), "rows": rows}))


if __name__ == "__main__":
    main()
