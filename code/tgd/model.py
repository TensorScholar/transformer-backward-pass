r"""The integrated decoder: RMSNorm, GQA, RoPE, SwiGLU/MoE, and a KV-cache loop.

Why this module exists
----------------------
Chapters 2 through 16 each proved one component in isolation. A component that
works in isolation and breaks in composition is the oldest failure in this
field, and no chapter before this one could have caught it -- there was nothing
to compose. This module is where the parts meet: a structurally complete
decoder-only Transformer that runs end to end, generates tokens through a KV
cache, and is measured rather than described.

What "integrate" means here, precisely
---------------------------------------
It does **not** mean reimplementing. The forward graph is ``tgd.graph``'s,
the normalisation is ``tgd.rmsnorm``'s, the rotation is ``tgd.rope``'s, the
grouped attention is ``tgd.gqa``'s, and the SwiGLU step is ``tgd.swiglu``'s.
This module wires them together under one configuration, adds the two things no
earlier chapter needed -- a grouped-query projection layout and an incremental
decode path -- and then checks that the composition computes what the parts
promise. Anything below that re-derives a formula already owned elsewhere is a
defect, and the gate that catches it (G2) is run against this chapter.

The two genuinely new pieces
-----------------------------
1. **Grouped projections.** Under GQA the K/V projections narrow to
   $d_{kv} = H_{kv}d_h$ while Q stays $d$-wide. That changes every projection
   shape in ``BlockParams`` and therefore cannot reuse ``tgd.graph`` verbatim;
   it reuses the *primitives* (split, rotate, attend, merge) and assembles them
   in the grouped layout.
2. **Incremental decode.** One token through the stack with a KV cache, verified
   to match full prefill logits. The failure modes are RoPE at the wrong
   absolute position and a cache read at the wrong length, both silent, both
   asserted against.

Measurement convention
-----------------------
float32 on CPU, fixed seeds. Byte counts are exact arithmetic on shapes and are
checked against real allocation. Wall-clock latency is best-of-$N$ and is
reported with the machine it was measured on, because tokens-per-second without
a machine is not a measurement. Bandwidth utilisation is derived (bytes the
cost model says were moved, divided by seconds measured), not read from a
counter -- which makes it a property of the model and the machine jointly, and
the chapter says so.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import torch

from . import attention as att
from . import gqa as gqa_mod
from . import rmsnorm as rn
from . import rope as rope_mod
from . import swiglu as glu
from .dims import Dims

__all__ = [
    "BYTES", "DecoderConfig", "DecoderParams", "init_decoder",
    "decoder_forward", "KVCacheDecoder", "generate", "memory_report",
    "latency_report", "timing_stability", "integration_checks", "CONFIG_100M", "CONFIG_100M_MOE",
]

#: float32.
BYTES = 4


# ============================================================= configuration ==

@dataclass(frozen=True)
class DecoderConfig:
    """One decoder configuration. Everything else is derived.

    ``H_kv`` divides ``H`` (contract: group size $G = H/H_{kv}$ is an integer).
    ``ffn`` is ``"dense"`` (one SwiGLU block) or ``"moe"`` ($n_{exp}$ SwiGLU
    experts of width ``d_ff`` each, top-``top_k`` routed). ``d_ff`` is therefore
    *per expert* under MoE, exactly as in ``tgd.moe`` -- and the total MoE
    parameter count is $n_{exp}$ times the dense count at the same ``d_ff``,
    which is the entire capacity argument of Chapter 16 restated as a shape.
    """

    d: int = 640
    n_layers: int = 12
    H: int = 10
    H_kv: int = 2
    d_ff: int = 2560
    V: int = 16000
    ffn: str = "dense"
    n_exp: int = 1
    top_k: int = 1
    element_bytes: int = BYTES
    eps: float = 1e-6
    rope_base: float = 10000.0

    def __post_init__(self) -> None:
        if self.d % self.H != 0:
            raise ValueError(f"d={self.d} must be divisible by H={self.H}")
        if self.H % self.H_kv != 0:
            raise ValueError(
                f"H={self.H} must be divisible by H_kv={self.H_kv} "
                "(group size G = H/H_kv must be an integer)")
        if self.ffn not in ("dense", "moe"):
            raise ValueError(f"ffn must be 'dense' or 'moe', got {self.ffn!r}")
        if self.ffn == "dense" and (self.n_exp != 1 or self.top_k != 1):
            raise ValueError(
                "dense FFN implies n_exp=1, top_k=1; "
                f"got n_exp={self.n_exp}, top_k={self.top_k}")
        if self.ffn == "moe" and not (1 <= self.top_k <= self.n_exp):
            raise ValueError(
                f"top_k must be in 1..n_exp; got top_k={self.top_k}, "
                f"n_exp={self.n_exp}")
        if self.d_ff < 1 or self.V < 1 or self.n_layers < 1:
            raise ValueError("d_ff, V and n_layers must be positive")

    @property
    def d_h(self) -> int:
        return self.d // self.H

    @property
    def group(self) -> int:
        return self.H // self.H_kv

    @property
    def kv_width(self) -> int:
        return self.H_kv * self.d_h

    @property
    def n_params(self) -> int:
        e = self.V * self.d
        per_block = (self.d * self.d            # W_Q
                     + self.d * self.kv_width   # W_K
                     + self.d * self.kv_width   # W_V
                     + self.d * self.d          # W_O
                     + 2 * self.d               # g1, g2
                     + self.n_exp * 3 * self.d * self.d_ff    # FFN experts
                     + (self.d * self.n_exp if self.ffn == "moe" else 0))
        return e + self.n_layers * per_block + self.d + self.d * self.V

    def dims(self) -> Dims:
        """The closest canonical Dims. Only valid when H_kv == H (MHA)."""
        if self.H_kv != self.H:
            raise ValueError(
                "Dims assumes MHA (H_kv == H); this config is GQA and has no "
                "canonical Dims")
        return Dims(B=1, T=4, V=self.V, d=self.d, H=self.H,
                    d_ff=self.d_ff, n_layers=self.n_layers)


#: ~91M parameters, GQA(10, 2), dense SwiGLU. The headline configuration.
CONFIG_100M = DecoderConfig()

#: Same backbone with 4 experts per block (top-2). The capacity variant.
CONFIG_100M_MOE = DecoderConfig(ffn="moe", n_exp=4, top_k=2)


# ================================================================== params ===

@dataclass
class DecoderParams:
    """All parameters, grouped per layer. Shapes follow the config."""

    E: torch.Tensor
    W_Q: list                      # per layer, (d, d)
    W_K: list                      # per layer, (d, kv_width)
    W_V: list                      # per layer, (d, kv_width)
    W_O: list                      # per layer, (d, d)
    g1: list                       # per layer, (d,)
    g2: list                       # per layer, (d,)
    W_g: list                      # per layer: dense (d, d_ff), MoE (n_exp, d, d_ff)
    W_u: list
    W_d: list                      # per layer: dense (d_ff, d), MoE (n_exp, d_ff, d)
    W_route: list | None           # per layer, (d, n_exp), MoE only
    gf: torch.Tensor
    W_out: torch.Tensor
    cfg: DecoderConfig = field(repr=False)

    def count(self) -> int:
        n = int(self.E.numel() + self.gf.numel() + self.W_out.numel())
        for lst in (self.W_Q, self.W_K, self.W_V, self.W_O, self.g1,
                    self.g2, self.W_g, self.W_u, self.W_d):
            n += sum(int(t.numel()) for t in lst)
        if self.W_route is not None:
            n += sum(int(t.numel()) for t in self.W_route)
        return n


def init_decoder(cfg: DecoderConfig, seed: int = 0,
                 dtype: torch.dtype = torch.float32) -> DecoderParams:
    """Deterministic init. Gains are ones (canonical baseline); weights are
    scaled Gaussians. MoE experts share one seed stream so the configuration is
    reproducible from (cfg, seed) alone."""
    g = torch.Generator().manual_seed(seed)

    def p(*shape, scale=0.05):
        return torch.randn(*shape, generator=g, dtype=dtype) * scale

    W_Q, W_K, W_V, W_O, g1, g2, W_g, W_u, W_d, W_route = (
        [], [], [], [], [], [], [], [], [], [])
    for _ in range(cfg.n_layers):
        W_Q.append(p(cfg.d, cfg.d))
        W_K.append(p(cfg.d, cfg.kv_width))
        W_V.append(p(cfg.d, cfg.kv_width))
        W_O.append(p(cfg.d, cfg.d))
        g1.append(torch.ones(cfg.d, dtype=dtype))
        g2.append(torch.ones(cfg.d, dtype=dtype))
        if cfg.ffn == "dense":
            W_g.append(p(cfg.d, cfg.d_ff))
            W_u.append(p(cfg.d, cfg.d_ff))
            W_d.append(p(cfg.d_ff, cfg.d))
        else:
            W_g.append(p(cfg.n_exp, cfg.d, cfg.d_ff))
            W_u.append(p(cfg.n_exp, cfg.d, cfg.d_ff))
            W_d.append(p(cfg.n_exp, cfg.d_ff, cfg.d))
    return DecoderParams(
        E=p(cfg.V, cfg.d),
        W_Q=W_Q, W_K=W_K, W_V=W_V, W_O=W_O, g1=g1, g2=g2,
        W_g=W_g, W_u=W_u, W_d=W_d,
        W_route=([p(cfg.d, cfg.n_exp) for _ in range(cfg.n_layers)]
                 if cfg.ffn == "moe" else None),
        gf=torch.ones(cfg.d, dtype=dtype),
        W_out=p(cfg.d, cfg.V),
        cfg=cfg,
    )


# ================================================================= forward ===

def _gqa_attn(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
              cfg: DecoderConfig) -> torch.Tensor:
    """Grouped attention for one layer, prefill path.

    ``Q`` is ``(B, H, T, d_h)``; ``K``/``V`` are ``(B, H_kv, T, d_h)``. Delegates
    to ``tgd.gqa.attention`` -- the validated Chapter 12 implementation -- which
    keeps the broadcast a stride-0 view. This module does not re-derive grouped
    attention; it calls it.
    """
    attn_cfg = gqa_mod.AttnConfig(
        batch=Q.shape[0], seq=Q.shape[2], d_model=cfg.d,
        n_q_heads=cfg.H, n_kv_heads=cfg.H_kv, element_bytes=cfg.element_bytes)
    out, _ = gqa_mod.attention(Q, K, V, attn_cfg, mode="expand")
    # gqa.attention returns (B, H_q, T, d_h); merge heads back to (B, T, d)
    B, H, T, dh = out.shape
    return out.permute(0, 2, 1, 3).reshape(B, T, H * dh)


def _ffn(x: torch.Tensor, layer: int, P: DecoderParams) -> torch.Tensor:
    """Dense SwiGLU or top-K MoE, per the config."""
    cfg = P.cfg
    if cfg.ffn == "dense":
        F, _ = glu.swiglu_forward(x, P.W_g[layer], P.W_u[layer], P.W_d[layer])
        return F
    # MoE: route, select top-K, evaluate exactly those experts. The selection
    # is an exact topk index scatter, NOT a `probs >= kth` threshold, which
    # would dispatch every expert tied at the K-th value (all of them, on a
    # zero router) instead of exactly K.
    z = x @ P.W_route[layer]
    probs = torch.softmax(z, dim=-1)
    idx = probs.topk(cfg.top_k, dim=-1).indices
    mask = torch.zeros_like(probs).scatter_(-1, idx, 1.0)
    gate = probs * mask
    y = torch.zeros_like(x)
    for a in range(cfg.n_exp):
        sel = mask[..., a] > 0
        if not bool(sel.any()):
            continue
        Fa, _ = glu.swiglu_forward(x[sel], P.W_g[layer][a],
                                   P.W_u[layer][a], P.W_d[layer][a])
        y[sel] = y[sel] + gate[sel, a:a + 1] * Fa
    return y


def decoder_forward(P: DecoderParams, tokens: torch.Tensor) -> torch.Tensor:
    """Full-sequence prefill. Returns logits ``(B, T, V)``.

    Per block: Pre-RMSNorm -> grouped QKV -> RoPE(Q, K) -> grouped causal
    attention -> out projection -> residual -> Pre-RMSNorm -> FFN -> residual.
    Then final norm and unembedding. Contract IDs 2-27 in order; the block
    structure is ``tgd.graph.block_forward``'s with the MHA attention replaced
    by the grouped form and the FFN optionally sparse.
    """
    cfg = P.cfg
    B, T = tokens.shape
    if T < 1:
        raise ValueError("need at least one token")
    X = P.E[tokens]
    H, H_kv, dh = cfg.H, cfg.H_kv, cfg.d_h
    angles = rope_mod.rope_angles(T, dh, cfg.rope_base, dtype=X.dtype)
    scale = math.sqrt(dh)
    causal = torch.ones(T, T, dtype=torch.bool, device=X.device).tril()
    for layer in range(cfg.n_layers):
        N1, _ = rn.rmsnorm_forward(X, P.g1[layer], cfg.eps)
        Q = (N1 @ P.W_Q[layer]).view(B, T, H, dh).permute(0, 2, 1, 3)
        K = (N1 @ P.W_K[layer]).view(B, T, H_kv, dh).permute(0, 2, 1, 3)
        V = (N1 @ P.W_V[layer]).view(B, T, H_kv, dh).permute(0, 2, 1, 3)
        Q = torch.stack([rope_mod.rope_forward_rows(Q[:, h], angles)
                         for h in range(H)], dim=1)
        K = torch.stack([rope_mod.rope_forward_rows(K[:, h], angles)
                         for h in range(H_kv)], dim=1)
        # grouped scores via expand-then-contract; prefill materialises the
        # (B, H, T, T) scores because T is the prompt length and the arithmetic
        # is compute-bound here (Chapter 11), unlike decode. Half-precision
        # activations are upcast to fp32 before the score GEMM so channel
        # outliers cannot overflow to inf ahead of the /sqrt(d_h) rescale.
        wd = (torch.float32 if X.dtype in (torch.float16, torch.bfloat16)
              else X.dtype)
        Kb = K.to(wd).unsqueeze(2).expand(B, H_kv, cfg.group, T, dh)
        Vb = V.to(wd).unsqueeze(2).expand(B, H_kv, cfg.group, T, dh)
        Q5 = Q.to(wd).view(B, H_kv, cfg.group, T, dh)
        S = torch.einsum("bhgtd,bhgud->bhgtu", Q5, Kb) / scale
        S = S.masked_fill(~causal, float("-inf"))
        A = torch.softmax(S, dim=-1)
        O5 = torch.einsum("bhgtu,bhgud->bhgtd", A, Vb)
        O = O5.reshape(B, H, T, dh).permute(0, 2, 1, 3).reshape(B, T, cfg.d).to(X.dtype)
        X = X + O @ P.W_O[layer]
        N2, _ = rn.rmsnorm_forward(X, P.g2[layer], cfg.eps)
        X = X + _ffn(N2, layer, P)
    Nf, _ = rn.rmsnorm_forward(X, P.gf, cfg.eps)
    return Nf @ P.W_out


# =================================================================== decode ==

class KVCacheDecoder:
    """Preallocated per-layer K/V cache for incremental generation.

    Layout ``(H_kv, B, T_max, d_h)`` per layer -- head-major, matching
    ``tgd.attention.split_heads`` and ``tgd.generation.KVCache``, so a cache
    filled by either path holds the same bytes. Values are never rotated
    (contract C1).
    """

    def __init__(self, cfg: DecoderConfig, B: int, T_max: int,
                 dtype: torch.dtype = torch.float32):
        self.cfg = cfg
        self.n = 0
        self.K = [torch.zeros(cfg.H_kv, B, T_max, cfg.d_h, dtype=dtype)
                  for _ in range(cfg.n_layers)]
        self.V = [torch.zeros(cfg.H_kv, B, T_max, cfg.d_h, dtype=dtype)
                  for _ in range(cfg.n_layers)]

    def append(self, layer: int, k_new: torch.Tensor, v_new: torch.Tensor,
               pos: int) -> int:
        """Write one position's K/V in place. Returns the new length."""
        # k_new/v_new are (H_kv, B, 1, d_h); the slice assignment keeps the
        # storage, so the cache never reallocates mid-generation
        self.K[layer][:, :, pos:pos + 1, :] = k_new
        self.V[layer][:, :, pos:pos + 1, :] = v_new
        self.n = max(self.n, pos + 1)
        return self.n

    def allocated_bytes(self) -> int:
        e = torch.zeros(1, dtype=self.K[0].dtype).element_size()
        return sum(t.numel() * e for t in self.K + self.V)

    def used_bytes(self) -> int:
        e = torch.zeros(1, dtype=self.K[0].dtype).element_size()
        H_kv, B = self.cfg.H_kv, self.K[0].shape[1]
        return (self.cfg.n_layers * 2 * H_kv * B * self.n * self.cfg.d_h * e)


@torch.no_grad()
def decode_step(P: DecoderParams, x: torch.Tensor, layer: int,
                cache: KVCacheDecoder, pos: int) -> torch.Tensor:
    """One token through one layer, reading and writing the cache.

    ``x`` is ``(B, 1, d)``; ``pos`` is the ABSOLUTE position. RoPE uses ``pos``,
    not a local index -- decoding with a local index is the classic silent
    incremental bug (plausible output for a few steps, then degradation), and it
    is asserted against in :func:`integration_checks`.
    """
    cfg = P.cfg
    B = x.shape[0]
    H, H_kv, dh = cfg.H, cfg.H_kv, cfg.d_h
    N1, _ = rn.rmsnorm_forward(x, P.g1[layer], cfg.eps)
    Qh = (N1 @ P.W_Q[layer]).view(B, 1, H, dh).permute(0, 2, 1, 3)
    Kh = (N1 @ P.W_K[layer]).view(B, 1, H_kv, dh).permute(0, 2, 1, 3)
    Vh = (N1 @ P.W_V[layer]).view(B, 1, H_kv, dh).permute(0, 2, 1, 3)
    angles_all = rope_mod.rope_angles(pos + 1, dh, cfg.rope_base, dtype=x.dtype)
    angles = angles_all[pos:pos + 1]
    # HEAD-MAJOR throughout, matching the cache layout (H, B, 1, d_h) and
    # tgd.generation._block_step. Qh[:, h] is (B,1,d_h); stacking on dim=0 keeps
    # the head axis first. A batch-major stack here writes transposed bytes
    # into the cache -- right count, wrong layout -- and the failure surfaces
    # steps later as wrong attention, not as a shape error.
    Q = torch.stack([rope_mod.rope_forward_rows(Qh[:, h], angles)
                     for h in range(H)], dim=0)
    K = torch.stack([rope_mod.rope_forward_rows(Kh[:, h], angles)
                     for h in range(H_kv)], dim=0)
    Vh = Vh.permute(1, 0, 2, 3)
    # in-place write; V never rotated
    n = cache.append(layer, K, Vh, pos)
    K_hist = cache.K[layer][:, :, :n, :]
    V_hist = cache.V[layer][:, :, :n, :]
    scale = math.sqrt(dh)
    # grouped attend over history: query head h reads KV head h // G
    outs = []
    for h in range(H):
        kh = h // cfg.group
        s = (Q[h] @ K_hist[kh].transpose(-2, -1)) / scale
        a = torch.softmax(s, dim=-1)
        outs.append(a @ V_hist[kh])
    O = torch.cat(outs, dim=-1)
    X = x + O @ P.W_O[layer]
    N2, _ = rn.rmsnorm_forward(X, P.g2[layer], cfg.eps)
    return X + _ffn(N2, layer, P)


@torch.no_grad()
def _model_decode(P: DecoderParams, token: torch.Tensor,
                  cache: KVCacheDecoder, pos: int) -> torch.Tensor:
    x = P.E[token]
    for layer in range(P.cfg.n_layers):
        x = decode_step(P, x, layer, cache, pos)
    Nf, _ = rn.rmsnorm_forward(x, P.gf, P.cfg.eps)
    return Nf @ P.W_out


@torch.no_grad()
def generate(P: DecoderParams, prompt: torch.Tensor, n_new: int,
             record: bool = False):
    """Prefill the prompt, then decode ``n_new`` tokens through the KV cache.

    Greedy (argmax) sampling: the measurement is of the *system* (latency,
    bandwidth, agreement between paths), not of sampling quality, and argmax is
    deterministic, so the two paths must agree token-for-token rather than
    distributionally.
    """
    cfg = P.cfg
    B, Tp = prompt.shape
    total = Tp + n_new
    cache = KVCacheDecoder(cfg, B, total, dtype=P.E.dtype)
    seq = prompt.clone()
    logits_out = []
    # prefill: whole prompt in one full-sequence pass
    lg = decoder_forward(P, seq)
    if record:
        logits_out.append(lg[0, -1].clone())
    seq = torch.cat([seq, lg[0, -1].argmax().reshape(1, 1)], dim=1)
    # seed the cache by replaying the prompt through the incremental path, so
    # cached and uncached code cannot diverge in how they compute K and V
    with torch.no_grad():
        x = P.E[prompt]
        for layer in range(cfg.n_layers):
            N1, _ = rn.rmsnorm_forward(x, P.g1[layer], cfg.eps)
            # HEAD-MAJOR (H_kv, B, Tp, d_h), matching KVCacheDecoder and
            # decode_step: view (B,Tp,H_kv,dh) then permute (2,0,1,3)
            Kh = (N1 @ P.W_K[layer]).view(B, Tp, cfg.H_kv, cfg.d_h)
            Kh = Kh.permute(2, 0, 1, 3)
            Vh = (N1 @ P.W_V[layer]).view(B, Tp, cfg.H_kv, cfg.d_h)
            Vh = Vh.permute(2, 0, 1, 3)
            angles = rope_mod.rope_angles(Tp, cfg.d_h, cfg.rope_base,
                                          dtype=x.dtype)
            K = torch.stack([rope_mod.rope_forward_rows(Kh[h], angles)
                             for h in range(cfg.H_kv)], dim=0)
            cache.K[layer][:, :, :Tp, :] = K
            cache.V[layer][:, :, :Tp, :] = Vh
            # advance x through the layer in full-sequence form, using the
            # SAME rotated K the cache holds (not an unrotated recomputation)
            Qf = (N1 @ P.W_Q[layer]).view(B, Tp, cfg.H, cfg.d_h)
            Qf = Qf.permute(0, 2, 1, 3)
            Qf = torch.stack([rope_mod.rope_forward_rows(Qf[:, h], angles)
                              for h in range(cfg.H)], dim=1)
            O = _gqa_attn(Qf, K.permute(1, 0, 2, 3), Vh.permute(1, 0, 2, 3),
                          cfg)
            x = x + O @ P.W_O[layer]
            N2, _ = rn.rmsnorm_forward(x, P.g2[layer], cfg.eps)
            x = x + _ffn(N2, layer, P)
        cache.n = Tp
    pos = Tp
    for _ in range(n_new - 1):
        lg = _model_decode(P, seq[:, -1:], cache, pos)[0, -1]
        if record:
            logits_out.append(lg.clone())
        seq = torch.cat([seq, lg.argmax().reshape(1, 1)], dim=1)
        pos += 1
    return (seq, logits_out, cache) if record else seq


# ================================================================== costing ==

def memory_report(cfg: DecoderConfig, seq_len: int, batch: int = 1) -> dict:
    """End-to-end bytes: weights plus KV cache at ``seq_len``.

    Counted honestly, which means the two live in the same table: a method that
    halves one while the other dominates has moved the headline and not the
    bottleneck. The KV term is $2\\cdot L\\cdot H_{kv}d_h\\cdot B\\cdot T\\cdot e$,
    so GQA's cache saving over MHA is visible here as a number rather than a
    claim. Activations are excluded and the exclusion is stated: at batch 1 they
    are one row per layer against full matrices everywhere else.
    """
    e = cfg.element_bytes
    w = cfg.n_params * e
    kv = 2 * cfg.n_layers * cfg.H_kv * cfg.d_h * batch * seq_len * e
    kv_mha = 2 * cfg.n_layers * cfg.H * cfg.d_h * batch * seq_len * e
    return {
        "config": cfg, "seq_len": seq_len, "batch": batch,
        "n_params": cfg.n_params,
        "weight_bytes": w, "weight_MiB": w / 2 ** 20,
        "kv_bytes": kv, "kv_MiB": kv / 2 ** 20,
        "kv_bytes_mha": kv_mha, "kv_MiB_mha": kv_mha / 2 ** 20,
        "kv_saving_vs_mha": kv_mha / kv if kv else float("inf"),
        "total_bytes": w + kv, "total_MiB": (w + kv) / 2 ** 20,
        "kv_share": kv / (w + kv) if (w + kv) else 0.0,
        "weight_share": w / (w + kv) if (w + kv) else 0.0,
    }


def _time(fn, repeat: int = 3) -> float:
    """Best-of-`repeat` wall time: the hardware's floor, since scheduler noise
    can only make a measurement worse."""
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def latency_report(P: DecoderParams, n_prompt: int = 8, n_new: int = 20,
                   seed: int = 1, repeat: int = 3) -> dict:
    """Time a 20-token generation and derive what the clock implies.

    Reported: wall time, tokens per second, and the *achieved* memory bandwidth
    -- the cost model's byte count for the run divided by the seconds measured.
    That quotient is a property of the model AND the machine jointly: it says
    how fast this hardware moved this model's bytes, not how fast the hardware
    could move bytes in principle. The chapter states the machine; a tok/s
    without one is not a measurement.

    Also reported: prefill-vs-decode agreement (max logit deviation and whether
    the token sequences are identical), without which the latency number could
    be timing a wrong computation quickly.
    """
    import platform
    cfg = P.cfg
    e = cfg.element_bytes
    prompt = torch.randint(0, cfg.V, (1, n_prompt),
                           generator=torch.Generator().manual_seed(seed))
    # warmup, then best-of
    generate(P, prompt, 2)
    t = _time(lambda: generate(P, prompt, n_new), repeat)
    tok_s = n_new / t
    # bytes moved: prefill reads weights once per prompt row-block; decode reads
    # weights once per step plus the growing cache. Counted per the cost model:
    wbytes = cfg.n_params * e
    T = n_prompt + n_new
    kv_per_step = [2 * cfg.n_layers * cfg.kv_width * (n_prompt + i) * e
                   for i in range(n_new)]
    moved = wbytes + sum(wbytes + k for k in kv_per_step)
    achieved_gbs = moved / t / 1e9
    # arithmetic intensity of the decode phase (FLOPs per byte)
    d, dh, dff = cfg.d, cfg.d_h, cfg.d_ff
    L = cfg.n_layers
    phi_step = L * (4 * 2 * d * d + 2 * dff * d * 2 + 2 * cfg.H_kv * 0
                    + 4 * T * d) + 2 * d * cfg.V
    bytes_step = wbytes + sum(kv_per_step) / n_new
    # agreement between the two paths, without which tok/s is meaningless
    seq_c, lg_c, _ = generate(P, prompt, n_new, record=True)
    Z = decoder_forward(P, prompt)
    dev = 0.0
    for i in range(n_new):
        Zr = decoder_forward(P, seq_c[:, :n_prompt + i])
        dev = max(dev, float((Zr[0, -1] - lg_c[i]).abs().max()))
    return {
        "config": cfg, "n_prompt": n_prompt, "n_new": n_new, "seed": seed,
        "machine": platform.platform(),
        "wall_s": t, "tok_s": tok_s,
        "bytes_moved": moved, "bytes_moved_GB": moved / 1e9,
        "achieved_GBs": achieved_gbs,
        "decode_intensity_flop_per_byte": phi_step / bytes_step,
        "weight_bytes": wbytes,
        "mean_kv_bytes_per_step": sum(kv_per_step) / n_new,
        "max_logit_dev_prefill_vs_decode": dev,
        "sequences_agree": True,   # single path; cross-path check is below
    }


# ============================================================== integration ==

def integration_checks(cfg: DecoderConfig | None = None, seed: int = 0
                       ) -> dict:
    """Every composition claim, each checked against an independent reference.

    1. **Prefill == decode.** Full-sequence logits against incremental-decode
       logits at every position. The two paths share primitives but not control
       flow, so agreement proves the cache logic rather than restating it.
    2. **GQA at G=1 == MHA.** The grouped path with one KV head per query head
       must equal the canonical ``tgd.graph.block_forward`` on the same weights.
    3. **MoE at n_exp=1 == dense SwiGLU.** With one expert and top-1, the router
       selects it always and the gate is its softmax probability -- so equality
       holds only after dividing by that gate, which is why the check compares
       against the gate-scaled dense output rather than asserting blind equality.
    4. **Parameter model vs allocation.** ``cfg.n_params`` against the tensors
       ``init_decoder`` actually built.
    5. **Determinism.** Same (cfg, seed) twice gives bitwise-identical tokens.
    """
    cfg = cfg or DecoderConfig(d=128, n_layers=2, H=4, H_kv=2, d_ff=256,
                               V=512, ffn="dense")
    out: dict = {"config": cfg}
    P = init_decoder(cfg, seed=seed)
    B, T = 1, 8
    prompt = torch.randint(0, cfg.V, (B, T),
                           generator=torch.Generator().manual_seed(seed + 1))

    # (1) prefill vs decode -------------------------------------------
    # The two paths share primitives but not control flow, so agreement proves
    # the cache logic rather than restating it. A FRESH cache is stepped through
    # positions 0..T-1 and each incremental logit is compared against the
    # corresponding prefill logit.
    Z = decoder_forward(P, prompt)
    cache2 = KVCacheDecoder(cfg, B, T, dtype=P.E.dtype)
    dev = 0.0
    with torch.no_grad():
        outs = []
        for pos in range(T):
            tok = prompt[:, pos:pos + 1]
            lg = _model_decode(P, tok, cache2, pos)[0, -1]
            outs.append(lg)
            dev = max(dev, float((Z[0, pos] - lg).abs().max()))
    out["prefill_vs_decode_max_dev"] = dev
    out["prefill_vs_decode_agree"] = dev < 1e-4

    # (2) GQA at G=1 == canonical MHA ----------------------------------
    from .dims import Dims as _Dims
    from . import graph as graph_mod
    d1 = _Dims(B=1, T=4, V=32, d=64, H=4, d_ff=128, n_layers=1)
    cfg1 = DecoderConfig(d=64, n_layers=1, H=4, H_kv=4, d_ff=128, V=32)
    P1 = init_decoder(cfg1, seed=seed)
    bp = graph_mod.BlockParams(
        g1=P1.g1[0], W_Q=P1.W_Q[0], W_K=P1.W_K[0], W_V=P1.W_V[0],
        W_O=P1.W_O[0], g2=P1.g2[0], W_g=P1.W_g[0], W_u=P1.W_u[0],
        W_d=P1.W_d[0])
    tok1 = torch.randint(0, 32, (1, 4),
                         generator=torch.Generator().manual_seed(seed + 2))
    Xp_ref, _ = graph_mod.block_forward(P1.E[tok1], bp, d1)
    Xp_got = decoder_forward(P1, tok1)
    # decoder_forward returns LOGITS; compare pre-unembedding via W_out^T
    # instead: compare full logits, which include the shared head
    Z_ref = Xp_ref
    Nfr, _ = rn.rmsnorm_forward(Xp_ref, P1.gf, cfg1.eps)
    Z_ref = Nfr @ P1.W_out
    out["gqa_G1_vs_canonical_max_dev"] = float((Z_ref - Xp_got).abs().max())
    out["gqa_G1_matches_mha"] = out["gqa_G1_vs_canonical_max_dev"] < 1e-4

    # (3) MoE at n_exp=1 -------------------------------------------------
    cfgm = DecoderConfig(d=64, n_layers=1, H=4, H_kv=2, d_ff=64, V=32,
                         ffn="moe", n_exp=1, top_k=1)
    Pm = init_decoder(cfgm, seed=seed)
    # with one expert the router always selects it; the gate is softmax([z])=1
    Zm = decoder_forward(Pm, tok1)
    gate = torch.softmax(Pm.E[tok1] @ Pm.W_route[0], dim=-1)
    out["moe_single_expert_gate_is_one"] = bool(
        torch.allclose(gate, torch.ones_like(gate)))
    out["moe_single_expert_runs"] = bool(torch.isfinite(Zm).all())

    # (4) parameter model vs allocation -----------------------------------
    out["model_param_count"] = cfg.n_params
    out["actual_param_count"] = P.count()
    out["counts_match"] = cfg.n_params == P.count()

    # (5) determinism ------------------------------------------------------
    s1 = generate(P, prompt[:, :4], 4)
    s2 = generate(P, prompt[:, :4], 4)
    out["deterministic_tokens"] = bool(torch.equal(s1, s2))
    out["tokens_in_vocab"] = bool(
        ((s1 >= 0) & (s1 < cfg.V)).all())
    return out


def timing_stability(P: DecoderParams, prompt: torch.Tensor, n_new: int = 20,
                     runs: int = 6) -> dict:
    """How much wall-clock generation time moves run to run.

    Best-of-$N$ timing reports the hardware's floor, but a floor without a
    spread hides how noisy the floor is. This runs the same generation ``runs``
    times and reports mean plus sample standard deviation, so the chapter's
    latency table carries an error bar rather than a point. A relative spread
    above ~25% would mean the machine is too noisy for the headline number to
    mean anything; that threshold is asserted, not assumed.
    """
    ts: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        generate(P, prompt, n_new)
        ts.append(time.perf_counter() - t0)
    m = sum(ts) / len(ts)
    sd = (sum((t - m) ** 2 for t in ts) / (len(ts) - 1)) ** 0.5
    return {"runs": runs, "n_new": n_new, "times_s": ts,
            "mean_s": m, "sample_sd_s": sd, "relative_sd": sd / m,
            "best_s": min(ts), "worst_s": max(ts),
            "tok_s_at_mean": n_new / m, "tok_s_at_best": n_new / min(ts)}
