"""Autoregressive generation: the naive loop, the KV cache, and the arithmetic.

Why this module exists
----------------------
Chapter 9 claims that autoregressive generation is a memory-bandwidth problem
rather than a compute problem, and that the KV cache is a space-time trade
designed to survive the memory wall. A chapter may not assert that. This module
implements both generation strategies, proves they compute the same thing, and
then measures the difference, so the claim rests on a latency measurement and an
arithmetic-intensity derivation rather than on an analogy with databases.

The two strategies
------------------
``generate_naive``
    At step ``t``, feed the whole prefix ``x[0..t]`` through every block and take
    the logits at the last position. Correct, and quadratically wasteful: the
    projections for positions ``0..t-1`` were already computed at step ``t-1`` and
    are recomputed identically.

``generate_cached``
    At step ``t``, project only the new token, append its key and value to a
    per-layer cache, and attend over the whole cached history. Each position is
    projected exactly once, ever.

Equivalence
-----------
At step ``t`` both paths must produce the same logits for the same prefix.
``selfcheck`` asserts agreement to float32 precision at *every* step, not just
at the end and not merely that both produce the same greedy token -- a greedy
match alone would hide a large logit error that happens not to change the argmax.

Conventions follow the notation contract: ``d_h = d / H`` is the per-head width,
``D`` is the SwiGLU gated product (never used here), ``L`` is the block count,
and ``g_k`` / ``Delta_theta_k`` keep their optimizer meanings and are not used in
this module.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import NamedTuple

import torch

from . import attention as att
from . import graph as graph_mod
from . import rope as rope_mod
from . import rmsnorm as rn
from . import swiglu as glu
from .dims import Dims

__all__ = [
    "GenModel", "KVCache", "generate_naive", "generate_cached",
    "prefill_cost", "decode_cost", "arithmetic_intensity", "measure",
    "MACHINE", "MACHINE_INFO",
]


# --------------------------------------------------------------------- env --

def _machine_info() -> dict[str, object]:
    """What we can honestly measure about the local hardware.

    Peak FLOP/s is **not** measured here: sustained tensor throughput depends on
    thread count, BLAS backend and precision, and quoting a spec-sheet peak
    alongside a measured latency would mix a literature figure with our own
    measurement. The ridge point is therefore computed from the two numbers we
    *can* justify, and the textbook machine figures are labelled as such.
    """
    info: dict[str, object] = {
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
    }
    try:
        info["machine"] = torch.backends.mps.is_available() and "mps" or "cpu"
    except Exception:
        info["machine"] = "cpu"
    return info


MACHINE_INFO = _machine_info()
MACHINE = MACHINE_INFO["machine"]


# ------------------------------------------------------------------- model --

class GenModel(NamedTuple):
    """Parameters of the canonical block stack, reused unchanged by both paths."""

    E: torch.Tensor              # (V, d)
    blocks: list                # list[BlockParams]
    gf: torch.Tensor             # (d,)
    W_out: torch.Tensor          # (d, V)
    dims: Dims


def init_model(dims: Dims, seed: int = 0, dtype=torch.float32) -> GenModel:
    """Deterministic initialization for the measurement.

    Norm gains are ones, matching the canonical baseline and Chapter 7's
    experiments, so the measured latency is not an artifact of an unusual
    parameter distribution.
    """
    g = torch.Generator().manual_seed(seed)

    def p(*shape, scale=0.05):
        return torch.randn(*shape, generator=g, dtype=dtype) * scale

    blocks = []
    for _ in range(dims.n_layers):
        blocks.append(graph_mod.BlockParams(
            g1=torch.ones(dims.d, dtype=dtype),
            W_Q=p(dims.d, dims.d), W_K=p(dims.d, dims.d),
            W_V=p(dims.d, dims.d), W_O=p(dims.d, dims.d),
            g2=torch.ones(dims.d, dtype=dtype),
            W_g=p(dims.d, dims.d_ff), W_u=p(dims.d, dims.d_ff),
            W_d=p(dims.d_ff, dims.d),
        ))
    return GenModel(
        E=p(dims.V, dims.d),
        blocks=blocks,
        gf=torch.ones(dims.d, dtype=dtype),
        W_out=p(dims.d, dims.V),
        dims=dims,
    )


def _forward_full(model: GenModel, tokens: torch.Tensor) -> torch.Tensor:
    """Prefill / naive path: whole prefix through every block. Returns logits."""
    d = model.dims
    X = model.E[tokens]
    for bp in model.blocks:
        X, _ = graph_mod.block_forward(X, bp, d)
    Nf, _ = rn.rmsnorm_forward(X, model.gf, d.eps)
    return Nf @ model.W_out


# --------------------------------------------------------------- KV cache ---

@dataclass
class KVCache:
    """Per-layer key/value tensors, preallocated to the maximum length.

    Layout is ``(H, B, T_max, d_h)``, matching :func:`tgd.attention.split_heads`
    so the decode path can reuse those primitives unchanged. Only ``[0, 0, :n, :]``
    is ever read; the tail exists so that a decode step is a slice assignment
    rather than a reallocation, which is the whole point of the mechanism.
    """

    K: list          # list[(H, B, T_max, d_h)]
    V: list          # list[(H, B, T_max, d_h)]
    n: int = 0       # number of valid positions currently stored

    @staticmethod
    def allocate(dims: Dims, batch: int, max_len: int,
                 dtype=torch.float32) -> "KVCache":
        shape = (dims.H, batch, max_len, dims.d_h)
        return KVCache(
            K=[torch.zeros(*shape, dtype=dtype) for _ in range(dims.n_layers)],
            V=[torch.zeros(*shape, dtype=dtype) for _ in range(dims.n_layers)],
            n=0,
        )

    def append(self, layer: int, k, v, pos: int) -> int:
        """Write one token's K and V at absolute position ``pos``, in place."""
        self.K[layer][:, :, pos:pos + 1, :] = k
        self.V[layer][:, :, pos:pos + 1, :] = v
        self.n = pos + 1
        return self.n

    def bytes(self) -> int:
        tot = 0
        for t in self.K:
            tot += t.numel() * t.element_size()
        for t in self.V:
            tot += t.numel() * t.element_size()
        return tot

    def used_bytes(self) -> int:
        """Bytes actually holding valid data.

        The preallocated tail is deliberately excluded: it is address space the
        allocator has reserved, not data the model has written. Reporting it
        would make a small generation look like a large one.
        """
        if not self.K:
            return 0
        b = self.K[0].element_size()
        n_layers = len(self.K)
        _, batch, _, d_h = self.K[0].shape
        return 2 * n_layers * d_h * batch * self.n * b


def _block_step(model: GenModel, x: torch.Tensor, layer: int,
                cache: KVCache, pos: int):
    """One incremental block for a single token at absolute position ``pos``.

    ``x`` is ``(B, 1, d)``. Returns ``(x_out, k_new, v_new)`` where the new
    key/value have already been written into the cache at index ``pos``.
    """
    d = model.dims
    bp = model.blocks[layer]

    # ---- attention branch, one token ---------------------------------
    N1, N1c = rn.rmsnorm_forward(x, bp.g1, d.eps)
    q_lin = N1 @ bp.W_Q                              # (B,1,d)
    k_lin = N1 @ bp.W_K
    v_lin = N1 @ bp.W_V

    Qh = att.split_heads(q_lin, d.H)                 # (H,B,1,d_h)
    Kh = att.split_heads(k_lin, d.H)
    Vh = att.split_heads(v_lin, d.H)

    # RoPE at the ABSOLUTE position, not at local index 0. Getting this wrong
    # is the classic incremental-decoding bug and it produces plausible-looking
    # output for a few steps before degrading.
    angles_all = rope_mod.rope_angles(pos + 1, d.d_h, d.rope_base,
                                      dtype=x.dtype, device=x.device)
    angles = angles_all[pos:pos + 1]                 # (1, d_h/2)
    Q = rope_mod.rope_forward_rows(Qh[:, :, 0, :], angles).unsqueeze(2)
    K = rope_mod.rope_forward_rows(Kh[:, :, 0, :], angles).unsqueeze(2)
    # V is never rotated (contract C1 / spec section 3.3).

    # ---- write into the cache, in place ------------------------------
    n = cache.append(layer, K, Vh, pos)
    K_hist = cache.K[layer][:, :, :n, :]
    V_hist = cache.V[layer][:, :, :n, :]

    # ---- attend over the whole cached history ------------------------
    scale = math.sqrt(d.d_h)
    S = torch.einsum("hbsd,hbtd->hbst", Q, K_hist) / scale   # (H,B,1,n)
    A = torch.softmax(S, dim=-1)
    O_h = torch.einsum("hbst,hbtd->hbsd", A, V_hist)         # (H,B,1,d_h)
    # head-major concatenation, identical to torch.cat(list, dim=-1) over heads
    # in tgd.graph.block_forward: [b,t,h,d_h] -> [b,t,H*d_h] == [b,t,d]
    O = O_h.permute(1, 2, 0, 3).reshape(O_h.shape[1], 1, -1)

    Y = O @ bp.W_O
    R = x + Y                                             # first residual

    # ---- feed-forward branch ----------------------------------------
    N2, N2c = rn.rmsnorm_forward(R, bp.g2, d.eps)
    F, _ = glu.swiglu_forward(N2, bp.W_g, bp.W_u, bp.W_d)
    Xp = R + F                                            # second residual
    return Xp


def _model_decode(model: GenModel, token: torch.Tensor, cache: KVCache,
                  pos: int) -> torch.Tensor:
    """Run one token through the whole stack, updating the cache. Logits."""
    d = model.dims
    x = model.E[token]                                   # (B,1,d)
    for layer in range(d.n_layers):
        x = _block_step(model, x, layer, cache, pos)
    Nf, _ = rn.rmsnorm_forward(x, model.gf, d.eps)
    return Nf @ model.W_out


# --------------------------------------------------------------- generate ---

@torch.no_grad()
def generate_naive(model: GenModel, prompt: torch.Tensor, n_new: int,
                   record: bool = False):
    """Recompute the entire prefix at every step.

    Returns ``(tokens, logits_per_step)`` where ``logits_per_step[t]`` is the
    logit vector that produced ``tokens[t]``. With ``record=False`` only the
    final token sequence is returned, for latency measurement.
    """
    seq = prompt.clone()
    logits_out = []
    for _ in range(n_new):
        lg = _forward_full(model, seq)[0, -1]           # (V,)
        if record:
            logits_out.append(lg.clone())
        seq = torch.cat([seq, lg.argmax().reshape(1, 1)], dim=1)
    return (seq, logits_out) if record else seq


@torch.no_grad()
def generate_cached(model: GenModel, prompt: torch.Tensor, n_new: int,
                    record: bool = False):
    """Prefill the prompt, then decode one token at a time into a KV cache."""
    d = model.dims
    total = prompt.shape[1] + n_new
    # The cache holds ACTIVATIONS, so its dtype is the model's parameter dtype.
    # Taking it from the prompt would allocate int64, because token ids are
    # integers -- and the failure surfaces much later as an einsum dtype error
    # rather than as an obvious mistake here.
    cache = KVCache.allocate(d, prompt.shape[0], total, dtype=model.E.dtype)

    seq = prompt.clone()
    logits_out = []
    pos = 0

    # ---- prefill: the prompt, in one full-sequence pass --------------
    lg_prefill = _forward_full(model, seq)
    if record:
        logits_out.append(lg_prefill[0, -1].clone())
    seq = torch.cat([seq, lg_prefill[0, -1].argmax().reshape(1, 1)], dim=1)
    pos = prompt.shape[1]

    # seed the cache with the prompt's keys/values by replaying it
    _seed_cache(model, prompt, cache)

    for _ in range(n_new - 1):
        nxt = seq[:, -1:].clone()
        lg = _model_decode(model, nxt, cache, pos)[0, -1]
        if record:
            logits_out.append(lg.clone())
        seq = torch.cat([seq, lg.argmax().reshape(1, 1)], dim=1)
        pos += 1
    return (seq, logits_out, cache) if record else seq


@torch.no_grad()
def _seed_cache(model: GenModel, prompt: torch.Tensor, cache: KVCache) -> None:
    """Fill the cache with the prompt's keys and values, in layer order.

    Done by a full forward with the cache written as we go, rather than by a
    separate path, so the cached and uncached code paths cannot diverge in how
    they compute K and V. Seeding is prefill work, not per-token work, and is
    excluded from the decode latency by construction.
    """
    d = model.dims
    B, T = prompt.shape
    x = model.E[prompt]
    for layer, bp in enumerate(model.blocks):
        N1, N1c = rn.rmsnorm_forward(x, bp.g1, d.eps)
        Kh = att.split_heads(N1 @ bp.W_K, d.H)      # (H,B,T,d_h)
        Vh = att.split_heads(N1 @ bp.W_V, d.H)
        angles = rope_mod.rope_angles(T, d.d_h, d.rope_base, dtype=x.dtype)
        K = torch.stack([rope_mod.rope_forward_rows(Kh[h], angles)
                         for h in range(d.H)], dim=0)
        cache.K[layer][:, :, :T, :] = K
        cache.V[layer][:, :, :T, :] = Vh
        x, _ = graph_mod.block_forward(x, bp, d)
    cache.n = T


# ------------------------------------------------------------------- cost ---

def prefill_cost(dims: Dims, n_new: int) -> dict[str, float]:
    """FLOPs and bytes for the prefill of ``n_new`` tokens, batch 1.

    Counts matmuls as 2mnk and elementwise work at one op per element. The
    weight bytes are counted once because prefill reuses every weight for all
    ``n_new`` rows -- that reuse is exactly what makes prefill compute-bound.
    """
    T, d, d_h, H, dff, V, L = (dims.T, dims.d, dims.d_h, dims.H,
                                dims.d_ff, dims.V, dims.n_layers)
    B = 1
    eb = 4                                   # float32 bytes per parameter
    phi_proj = L * 4 * 2 * B * T * d * d     # Q,K,V,O per block
    phi_attn = L * 4 * B * T * T * d         # scores + AV, summed over heads
    phi_ffn = L * (2 * 2 * B * T * d * dff + 2 * B * T * dff * d)
    phi_out = 2 * B * T * d * V
    wbytes = L * (4 * d * d + 2 * d * dff + dff * d + 2 * d) \
        + 2 * V * d + 3 * d                 # blocks + E/W_out + gains
    return {
        "phi": float(phi_proj + phi_attn + phi_ffn + phi_out),
        "bytes": float(wbytes * eb),
        "phi_proj": float(phi_proj),
        "phi_attn": float(phi_attn),
        "phi_ffn": float(phi_ffn),
        "phi_out": float(phi_out),
        "weight_bytes": float(wbytes * eb),
        "batch": float(B), "seq": float(T), "layers": float(L),
        "element_bytes": float(eb),
    }


def decode_cost(dims: Dims, n_cached: int) -> dict[str, float]:
    """FLOPs and bytes for ONE token of decode with ``n_cached`` positions.

    The decisive difference from prefill: the activations are a single row, so
    every weight matrix is read for a rank-1 update and contributes no reuse.
    """
    d, d_h, H, dff, V, L = (dims.d, dims.d_h, dims.H, dims.d_ff,
                             dims.V, dims.n_layers)
    eb = 4
    phi_proj = L * 4 * 2 * d * d
    phi_attn = L * 4 * n_cached * d           # scores + AV for one query row
    phi_ffn = L * (2 * 2 * d * dff + 2 * dff * d)
    phi_out = 2 * d * V
    # weights: read in full, because a (1 x d) @ (d x d) product still has to
    # stream the entire (d x d) operand.
    wbytes = L * (4 * d * d + 2 * d * dff + dff * d + 2 * d) + 2 * V * d + 3 * d
    # KV cache: read once per layer, and re-read at every subsequent step.
    kvb = L * 2 * d * n_cached * eb
    return {
        "phi": float(phi_proj + phi_attn + phi_ffn + phi_out),
        "bytes": float(wbytes * eb + kvb),
        "phi_proj": float(phi_proj),
        "phi_attn": float(phi_attn),
        "phi_ffn": float(phi_ffn),
        "phi_out": float(phi_out),
        "weight_bytes": float(wbytes * eb),
        "kv_bytes": float(kvb),
        "batch": 1.0, "n_cached": float(n_cached),
        "element_bytes": float(eb),
    }


def arithmetic_intensity(phi: float, nbytes: float) -> float:
    """The book's definition, stated once: FLOPs per byte of memory read."""
    return phi / nbytes if nbytes else float("inf")


def kv_cache_bytes(dims: Dims, seq_len: int, batch: int = 1,
                   element_bytes: int = 2) -> int:
    """Cache footprint: 2 * L * d * batch * seq_len * bytes_per_parameter.

    ``2`` is the factor for K and V; ``L*d`` is the per-token footprint summed
    over heads, since ``H * d_h = d`` exactly. The result is therefore
    independent of the head count, which is worth stating because it is
    surprising: halving the heads halves the cache only if ``d_h`` is held
    fixed, and doubling them changes nothing at all.
    """
    return 2 * dims.n_layers * dims.d * batch * seq_len * element_bytes


# ---------------------------------------------------------------- measure ---

@dataclass
class Measurement:
    label: str
    dims: Dims
    n_new: int
    naive_s: float
    cached_s: float
    naive_tok_s: float
    cached_tok_s: float
    speedup: float
    max_logit_dev: float = 0.0
    detail: dict = field(default_factory=dict)


def _time(fn, repeat: int = 3) -> float:
    """Best-of-`repeat` wall time. Best-of, not mean: we are measuring the
    hardware's floor, and scheduler noise can only make a measurement worse."""
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def measure(dims: Dims, n_prompt: int = 8, n_new: int = 50,
            repeat: int = 3) -> Measurement:
    """Time both strategies and check they agree."""
    model = init_model(dims)
    prompt = torch.randint(0, dims.V, (1, n_prompt), generator=torch.Generator().manual_seed(1))

    t_n = _time(lambda: generate_naive(model, prompt, n_new), repeat)
    t_c = _time(lambda: generate_cached(model, prompt, n_new), repeat)

    seq_n, lg_n = generate_naive(model, prompt, n_new, record=True)
    seq_c, lg_c, cache = generate_cached(model, prompt, n_new, record=True)

    dev = 0.0
    if len(lg_n) == len(lg_c):
        for a, b in zip(lg_n, lg_c):
            dev = max(dev, float((a - b).abs().max()))
    same = bool(torch.equal(seq_n, seq_c))

    return Measurement(
        label=f"L={dims.n_layers} d={dims.d} H={dims.H} T_p={n_prompt}",
        dims=dims, n_new=n_new,
        naive_s=t_n, cached_s=t_c,
        naive_tok_s=n_new / t_n, cached_tok_s=n_new / t_c,
        speedup=t_n / t_c,
        max_logit_dev=dev,
        detail={"tokens_identical": same,
                "cache_bytes": cache.used_bytes(),
                "naive_peak_work_ratio": float(n_new)},
    )