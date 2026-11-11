"""Lean, capture-clean forward for Qwen3.5 text models.

The stock Transformers forward builds masks, caches and position ids on every call
and has host-side branches. This forward does only tensor work on the GPU, with
static shapes, so a whole training step can be recorded once as a CUDA graph and
replayed with no Python, no autograd engine and no per-kernel host launch.
"""

import os

import torch
import torch.nn.functional as F

RTX3090_UUID = "GPU-225509c3-d702-6c27-de03-76b5e99b33b1"
CHUNK = 64


def enable_blocking_sync() -> None:
    """Make the host thread sleep, not spin, while it waits for the GPU.

    CUDA's default is to busy-wait in synchronize(), which pins one CPU core at 100%
    for as long as the GPU works. The flag must be set before the CUDA context exists.
    """
    import ctypes

    with open("/proc/self/maps") as maps:
        paths = {line.split()[-1] for line in maps if "libcudart.so" in line}
    if len(paths) != 1:
        raise RuntimeError(f"expected one loaded libcudart, found {sorted(paths)}")
    code = ctypes.CDLL(paths.pop()).cudaSetDeviceFlags(4)  # cudaDeviceScheduleBlockingSync
    if code != 0:
        raise RuntimeError(f"cudaSetDeviceFlags failed with code {code}")


def require_gpu(blocking_sync: bool = False) -> torch.device:
    """Select the RTX 3090 by UUID. Never fall back to CPU or another card."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != RTX3090_UUID:
        raise RuntimeError(f"run with CUDA_VISIBLE_DEVICES={RTX3090_UUID}")
    if blocking_sync:
        enable_blocking_sync()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("the selected GPU is not available")
    name = torch.cuda.get_device_name(0)
    if "3090" not in name:
        raise RuntimeError(f"expected the RTX 3090, got {name}")
    return torch.device("cuda:0")


def load_text_model(path: str, device: torch.device, dtype=torch.bfloat16):
    from transformers import Qwen3_5ForCausalLM

    model = Qwen3_5ForCausalLM.from_pretrained(path, dtype=dtype, device_map={"": device})
    model.requires_grad_(False)
    return model


def chunk_gated_delta_rule(query, key, value, g, beta):
    """Gated delta rule over a full sequence, chunked. Inputs are [B, L, H, D]; g and beta are [B, L, H].

    Always FP32 with autocast off, matching the reference. L must be a multiple of CHUNK
    so every shape is static.
    """
    with torch.autocast("cuda", enabled=False):
        out_dtype = query.dtype
        query, key, value, beta, g = (x.transpose(1, 2).float() for x in (query, key, value, beta, g))
        batch, heads, length, _ = key.shape
        if length % CHUNK:
            raise ValueError(f"sequence length {length} is not a multiple of {CHUNK}")
        chunks = length // CHUNK

        query = query * torch.rsqrt((query * query).sum(-1, keepdim=True) + 1e-6)
        key = key * torch.rsqrt((key * key).sum(-1, keepdim=True) + 1e-6)
        query = query * query.shape[-1] ** -0.5

        v_beta = value * beta.unsqueeze(-1)
        k_beta = key * beta.unsqueeze(-1)
        query, key, k_beta, v_beta = (
            x.reshape(batch, heads, chunks, CHUNK, x.shape[-1]) for x in (query, key, k_beta, v_beta)
        )
        cum = g.reshape(batch, heads, chunks, CHUNK).cumsum(-1)

        upper = torch.ones(CHUNK, CHUNK, dtype=torch.bool, device=query.device).triu(1)
        pair = (cum.unsqueeze(-1) - cum.unsqueeze(-2)).masked_fill(upper, float("-inf")).exp()

        key_t = key.transpose(-1, -2)
        system = (k_beta @ key_t) * pair
        intra = (query @ key_t) * pair
        decayed_k_beta = k_beta * cum.exp().unsqueeze(-1)

        # Forward substitution. An explicit inverse by repeated squaring was tried and
        # rejected: it overflowed to NaN on the 4B model.
        new_values = torch.linalg.solve_triangular(system, v_beta, upper=False, unitriangular=True)
        k_cumdecay = torch.linalg.solve_triangular(system, decayed_k_beta, upper=False, unitriangular=True)

        query = query * cum.exp().unsqueeze(-1)
        key_t = (key * (cum[..., -1:] - cum).exp().unsqueeze(-1)).transpose(-1, -2)
        chunk_decay = cum[..., -1].exp()[..., None, None]

        state = torch.zeros(batch, heads, key.shape[-1], value.shape[-1], dtype=torch.float32, device=query.device)
        outs = []
        for i in range(chunks):  # unrolled once at capture time; free on replay
            v_new = new_values[:, :, i] - k_cumdecay[:, :, i] @ state
            outs.append(query[:, :, i] @ state + intra[:, :, i] @ v_new)
            state = state * chunk_decay[:, :, i] + key_t[:, :, i] @ v_new
        out = torch.stack(outs, dim=2).reshape(batch, heads, length, -1)
        return out.transpose(1, 2).to(out_dtype)


def delta_layer(m, x):
    batch, length, _ = x.shape
    qkv = m.in_proj_qkv(x).transpose(1, 2)
    z = m.in_proj_z(x)
    b = m.in_proj_b(x)
    a = m.in_proj_a(x)

    w = m.conv1d.weight
    qkv = F.conv1d(qkv.to(w.dtype), w, m.conv1d.bias, padding=w.shape[-1] - 1, groups=w.shape[0])[:, :, :length]
    qkv = F.silu(qkv).to(x.dtype).transpose(1, 2)
    query, key, value = torch.split(qkv, [m.key_dim, m.key_dim, m.value_dim], dim=-1)
    query = query.reshape(batch, length, -1, m.head_k_dim)
    key = key.reshape(batch, length, -1, m.head_k_dim)
    value = value.reshape(batch, length, -1, m.head_v_dim)

    beta = b.sigmoid()
    g = -m.A_log.float().exp() * F.softplus(a.float() + m.dt_bias)
    rep = m.num_v_heads // m.num_k_heads
    if rep > 1:
        query = query.repeat_interleave(rep, dim=2)
        key = key.repeat_interleave(rep, dim=2)

    out = chunk_gated_delta_rule(query, key, value, g, beta)
    out = m.norm(out.reshape(-1, m.head_v_dim), z.reshape(-1, m.head_v_dim))
    return m.out_proj(out.reshape(batch, length, -1))


def attention_layer(m, x, cos, sin):
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb, repeat_kv

    batch, length, _ = x.shape
    hd = m.head_dim
    query, gate = m.q_proj(x).view(batch, length, -1, 2 * hd).chunk(2, dim=-1)
    gate = gate.reshape(batch, length, -1)
    query = m.q_norm(query).transpose(1, 2)
    key = m.k_norm(m.k_proj(x).view(batch, length, -1, hd)).transpose(1, 2)
    value = m.v_proj(x).view(batch, length, -1, hd).transpose(1, 2)
    query, key = apply_rotary_pos_emb(query, key, cos, sin)
    key = repeat_kv(key, m.num_key_value_groups)
    value = repeat_kv(value, m.num_key_value_groups)
    out = F.scaled_dot_product_attention(query, key, value, is_causal=True)
    out = out.transpose(1, 2).reshape(batch, length, -1) * torch.sigmoid(gate)
    return m.o_proj(out)


@torch.no_grad()
def rope_tables(text_model, length: int, device):
    """cos/sin for positions 0..length-1, computed once and kept on the GPU."""
    pos = torch.arange(length, device=device).view(1, 1, -1).expand(3, 1, -1)
    probe = torch.zeros(1, dtype=text_model.embed_tokens.weight.dtype, device=device)
    cos, sin = text_model.rotary_emb(probe, pos)
    return cos.contiguous(), sin.contiguous()


def block(layer, h, cos, sin):
    x = layer.input_layernorm(h)
    if layer.block_type == "linear_attention":
        x = delta_layer(layer.linear_attn, x)
    else:
        x = attention_layer(layer.self_attn, x, cos, sin)
    h = h + x
    return h + layer.mlp(layer.post_attention_layernorm(h))


def lean_hidden(text_model, input_ids, cos, sin, checkpoint: bool = False, inputs_embeds=None):
    """Final hidden states [B, L, hidden] for full sequences with no padding and no cache.

    With checkpoint=True each layer keeps only its input and recomputes its activations
    during the backward pass. That is extra GPU work in exchange for memory; it adds no
    host work, because the recomputation is recorded into the same graph.
    """
    from torch.utils.checkpoint import checkpoint as recompute

    h = text_model.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
    for layer in text_model.layers:
        if checkpoint:
            # preserve_rng_state touches the generator, which an on-device loop cannot record.
            h = recompute(block, layer, h, cos, sin, use_reentrant=False, preserve_rng_state=False)
        else:
            h = block(layer, h, cos, sin)
    return text_model.norm(h)
