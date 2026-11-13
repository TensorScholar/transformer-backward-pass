"""Grouped-query attention: views versus copies, and the gradient merge.

Why this module exists
----------------------
Chapter 9 measured the KV cache at ``2 * L * d * batch * seq * bytes`` and
Chapter 11 measured that memory traffic is the currency. Grouped-query attention
is the mechanical answer to both: give fewer key/value heads than query heads, and
the cache shrinks in proportion.

But the saving is only real if the broadcast that lets 8 KV heads serve 32 query
heads does **not copy**. This module implements attention three ways -- MHA, GQA,
MQA -- and the broadcast two ways, and measures:

* the cache footprint of each, in bytes;
* the VRAM allocation of ``expand`` (a stride-0 view) against
  ``repeat_interleave`` (a real copy);
* that the gradient arriving at one KV head is exactly the sum over the query
  heads assigned to it.

The gradient fact is the mathematical content of GQA and it is a direct
application of the fan-in merge from Chapter 3: several consumers target one
tensor, and their adjoints are summed.

Conventions follow the notation contract: ``d_h = d / H_q`` is the query head
width and is held FIXED across all three variants, so the KV width is
``H_kv * d_h`` and narrows as ``H_kv`` falls. That is the convention under which
GQA actually saves memory; the alternative is discussed in the chapter.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch

__all__ = [
    "AttnConfig", "broadcast_kv", "attention", "attention_manual_backward",
    "cache_bytes", "kv_cache_report", "broadcast_allocation",
    "gradient_merge_check", "MHA_MQA_GQA",
]

Broadcast = Literal["expand", "repeat_interleave"]


@dataclass(frozen=True)
class AttnConfig:
    """Attention shape parameters. ``d_h`` is derived, never set directly."""

    batch: int
    seq: int
    d_model: int
    n_q_heads: int
    n_kv_heads: int
    element_bytes: int = 2

    @property
    def d_h(self) -> int:
        """Query head width, FIXED across MHA/GQA/MQA. This is the convention
        that makes the cache comparison meaningful: narrowing H_kv narrows the
        total KV width, which is where the saving comes from."""
        if self.d_model % self.n_q_heads:
            raise ValueError(
                f"d_model={self.d_model} not divisible by n_q_heads="
                f"{self.n_q_heads}")
        return self.d_model // self.n_q_heads

    @property
    def group(self) -> int:
        """Query heads per KV head. MHA 1, GQA intermediate, MQA n_q_heads."""
        if self.n_q_heads % self.n_kv_heads:
            raise ValueError(
                f"n_q_heads={self.n_q_heads} not divisible by n_kv_heads="
                f"{self.n_kv_heads}")
        return self.n_q_heads // self.n_kv_heads

    @property
    def kv_width(self) -> int:
        """Total KV channels per layer = H_kv * d_h."""
        return self.n_kv_heads * self.d_h


MHA_MQA_GQA = ("MHA", "GQA", "MQA")


# ------------------------------------------------------------ broadcast ----

def broadcast_kv(kv: torch.Tensor, cfg: AttnConfig, mode: Broadcast
                 ) -> torch.Tensor:
    """Give every query head its KV head, two ways.

    Returns a tensor whose **storage** is either the source (expand) or ``G``
    times larger (repeat_interleave). Both are numerically identical; only one
    is free.

    ``expand`` produces a 5-D tensor ``(B, H_kv, G, T, d_h)`` with **stride 0**
    on the group axis. It must be *used* in that shape -- any ``.reshape`` back to
    ``(B, H_q, T, d_h)`` silently materialises the copy and defeats the entire
    point. This is the single most common way GQA fails to deliver its saving.
    """
    B, H_kv, T, d_h = kv.shape
    if mode == "expand":
        return kv.unsqueeze(2).expand(B, H_kv, cfg.group, T, d_h)
    if mode == "repeat_interleave":
        rep = kv.repeat_interleave(cfg.group, dim=1)          # real copy
        return rep.view(B, H_kv, cfg.group, T, d_h)
    raise ValueError(f"unknown broadcast mode {mode!r}")


def storage_nbytes(t: torch.Tensor) -> int:
    """Bytes actually allocated, as opposed to bytes the tensor appears to span.

    ``numel() * element_size`` is the LOGICAL size and is identical for a view
    and a copy. The storage size is what the memory bus pays for, and it is the
    only number that distinguishes them.
    """
    return int(t.untyped_storage().nbytes())


# ------------------------------------------------------------- attention ---

def attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
              cfg: AttnConfig, mode: Broadcast = "expand"):
    """Scaled dot-product attention with GQA, returning ``(output, probs)``.

    ``K`` and ``V`` are ``(B, H_kv, T, d_h)``; ``Q`` is ``(B, H_q, T, d_h)``.
    Internally everything is 5-D ``(B, H_kv, G, T, d_h)`` so the broadcast
    stays a stride-0 view all the way through. Query head ``h`` reads KV head
    ``h // G``, which is the layout ``repeat_interleave`` produces and therefore
    the one autograd and the hand-written backward must agree on.

    No ``reshape`` appears after the broadcast. That is deliberate: reshaping a
    stride-0 view copies it and undoes the saving, silently.

    Half-precision operands are upcast to ``float32`` before the score GEMM:
    with 100x channel outliers a raw ``Q @ K^T`` element reaches ``d_h``
    products of magnitude ~1e4 each, overflowing fp16 (max 65504) to ``inf``
    before the ``/ sqrt(d_h)`` rescale and max-subtraction can make it finite
    again. The result is cast back to the input dtype at return.
    """
    B, H_q, T, d_h = Q.shape
    H_kv = cfg.n_kv_heads
    wd = torch.float32 if Q.dtype in (torch.float16, torch.bfloat16) else Q.dtype
    Q5 = Q.to(wd).view(B, H_kv, cfg.group, T, d_h)
    K5 = broadcast_kv(K.to(wd), cfg, mode)
    V5 = broadcast_kv(V.to(wd), cfg, mode)

    causal = torch.ones(T, T, dtype=torch.bool, device=Q.device).tril()
    scores = torch.einsum("bhgtd,bhgud->bhgtu", Q5, K5) / math.sqrt(d_h)
    scores = scores.masked_fill(~causal, float("-inf"))
    p = torch.softmax(scores, dim=-1)                       # (B,H_kv,G,T,T)
    ctx = torch.einsum("bhgtu,bhgud->bhgtd", p, V5)      # (B,H_kv,G,T,d_h)
    return ctx.reshape(B, H_q, T, d_h).to(Q.dtype), p.to(Q.dtype)


def attention_manual_backward(Q, K, V, dO, cfg: AttnConfig):
    """Hand-written backward through GQA, returning ``(dQ, dK_kv, dV_kv)``.

    This is the merge the chapter is about. The adjoint arriving at KV head ``h``
    is the **sum** over the ``G`` query heads that read it:

        dK_kv[h] = sum_{g=0..G-1}  (dK from query head h*G + g)
        dV_kv[h] = sum_{g=0..G-1}  (dV from query head h*G + g)

    Each query head has its own dQ and receives no merge -- the asymmetry
    between the query and key paths is the whole content of GQA.

    Half-precision inputs are upcast to ``float32`` internally, exactly as in
    :func:`attention`, so the score GEMM cannot overflow; the returned grads
    are cast back to the input dtype.
    """
    B, H_kv, T, d_h = K.shape
    G = cfg.group
    scale = 1.0 / math.sqrt(d_h)
    wd = torch.float32 if Q.dtype in (torch.float16, torch.bfloat16) else Q.dtype

    Q5 = Q.to(wd).view(B, H_kv, G, T, d_h)
    dO5 = dO.to(wd).view(B, H_kv, G, T, d_h)
    Kb = K.to(wd).unsqueeze(2).expand(B, H_kv, G, T, d_h)
    Vb = V.to(wd).unsqueeze(2).expand(B, H_kv, G, T, d_h)

    causal = torch.ones(T, T, dtype=torch.bool, device=Q.device).tril()
    scores = torch.einsum("bhgtd,bhgud->bhgtu", Q5, Kb) * scale
    scores = scores.masked_fill(~causal, float("-inf"))
    p = torch.softmax(scores, dim=-1)

    # --- dV_kv: SUM over the G query heads ------------------------------
    dV = torch.einsum("bhgtu,bhgtd->bhgud", p, dO5).sum(dim=2)
    # --- dP -> dscores ---------------------------------------------------
    dP = torch.einsum("bhgtd,bhgud->bhgtu", dO5, Vb)
    dScores = p * (dP - (p * dP).sum(dim=-1, keepdim=True)) * scale
    # --- dQ: per query head, no merge ------------------------------------
    dQ = torch.einsum("bhgtu,bhgud->bhgtd", dScores, Kb)
    # --- dK_kv: THE MERGE, summed over the G query heads -----------------
    dK = torch.einsum("bhgtu,bhgtd->bhgud", dScores, Q5).sum(dim=2)
    return (dQ.reshape(B, cfg.n_q_heads, T, d_h).to(Q.dtype),
            dK.to(K.dtype), dV.to(V.dtype))


# ------------------------------------------------------------ KV cache -----

def cache_bytes(cfg: AttnConfig, n_layers: int, seq_len: int | None = None
                ) -> int:
    """KV cache footprint: ``2 * n_layers * d_kv * batch * seq * element_bytes``.

    ``d_kv = H_kv * d_h``, so the cache is **proportional to H_kv** with the
    query head width held fixed. That proportionality is the entire memory
    argument for GQA and MQA.
    """
    T = cfg.seq if seq_len is None else seq_len
    return 2 * n_layers * cfg.kv_width * cfg.batch * T * cfg.element_bytes


def kv_cache_report(batch: int = 8, seq: int = 32768, d_model: int = 128,
                    n_q: int = 32, n_layers: int = 1,
                    element_bytes: int = 2) -> list[dict]:
    """MHA / GQA / MQA footprints at the stated shape."""
    rows = []
    for name, h_kv in (("MHA", n_q), ("GQA", n_q // 4), ("MQA", 1)):
        cfg = AttnConfig(batch=batch, seq=seq, d_model=d_model, n_q_heads=n_q,
                         n_kv_heads=h_kv, element_bytes=element_bytes)
        b = cache_bytes(cfg, n_layers, seq)
        rows.append({
            "name": name, "n_q": n_q, "n_kv": h_kv, "d_h": cfg.d_h,
            "group": cfg.group, "kv_width": cfg.kv_width,
            "bytes": b, "MiB": b / 2**20, "MB": b / 1e6,
        })
    base = rows[0]["bytes"]
    for r in rows:
        r["saving_vs_mha"] = base / r["bytes"]
    return rows


def broadcast_allocation(B: int = 8, T: int = 512, d_model: int = 128,
                        n_q: int = 32, n_kv: int = 8) -> dict:
    """Storage bytes for the KV broadcast, both mechanisms.

    The comparison is against ``untyped_storage().nbytes()``, not
    ``numel()``: a view and a copy have identical logical size and different
    allocations, and only the storage size distinguishes them.
    """
    cfg = AttnConfig(batch=B, seq=T, d_model=d_model, n_q_heads=n_q,
                     n_kv_heads=n_kv)
    K = torch.randn(B, n_kv, T, cfg.d_h)
    e = cfg.element_bytes
    view = broadcast_kv(K, cfg, "expand")
    copy = broadcast_kv(K, cfg, "repeat_interleave")
    src = storage_nbytes(K)
    nv, nc = storage_nbytes(view), storage_nbytes(copy)
    # Both outputs are already (B, H_kv, G, T, d_h), so this is a direct
    # comparison. An earlier version reshaped the view to compare layouts, which
    # is both unnecessary and wrong -- and it reported a mismatch that was
    # really a defect in the check, not in the two mechanisms.
    check_equal = torch.equal(view, copy)
    return {
        "config": cfg, "source_bytes": src,
        "expand_bytes": nv, "repeat_bytes": nc,
        "extra_from_expand": nv - src,
        "extra_from_repeat": nc - src,
        "ratio": nc / nv,
        "identical_values": bool(check_equal),
        "expand_is_view": bool(view._base is not None or view.storage().data_ptr()
                               == K.storage().data_ptr()),
    }


# ---------------------------------------------- the gradient merge proof ---

def gradient_merge_check(B: int = 2, T: int = 16, d_model: int = 32,
                         n_q: int = 8, n_kv: int = 2) -> dict:
    """Prove the GQA gradient merge against autograd.

    For GQA with group size G, the adjoint at KV head ``k`` must equal the sum of
    the adjoints contributed by the G query heads that read it. Autograd computes
    this implicitly through the ``expand`` view; this function computes it
    explicitly and compares, so the claim is checked rather than asserted.

    The comparison is exact because the merge is a plain sum, and ``expand``'s
    stride-0 backward is that same sum -- no approximation is involved.
    """
    cfg = AttnConfig(batch=B, seq=T, d_model=d_model, n_q_heads=n_q,
                     n_kv_heads=n_kv, element_bytes=4)
    G = cfg.group
    torch.manual_seed(0)
    Q = torch.randn(B, n_q, T, cfg.d_h, requires_grad=True)
    K = torch.randn(B, n_kv, T, cfg.d_h, requires_grad=True)
    V = torch.randn(B, n_kv, T, cfg.d_h, requires_grad=True)
    dO = torch.randn(B, n_q, T, cfg.d_h)

    O, _ = attention(Q, K, V, cfg, "expand")
    (O * dO).sum().backward()

    dQ_m, dK_m, dV_m = attention_manual_backward(Q.detach(), K.detach(),
                                                V.detach(), dO, cfg)
    eK = float((dK_m - K.grad).abs().max())
    eV = float((dV_m - V.grad).abs().max())
    eQ = float((dQ_m - Q.grad).abs().max())

    # and the explicit statement: head k of dK is the sum over its G query heads
    per_head = []
    for k in range(n_kv):
        q_heads = list(range(k * G, (k + 1) * G))
        # adjoint of KV head k computed from each query head separately
        contribs = []
        for h in q_heads:
            dO_h = torch.zeros_like(dO)
            dO_h[:, h] = dO[:, h]
            Qh = Q.detach().clone().requires_grad_(True)
            Kh = K.detach().clone().requires_grad_(True)
            Vh = V.detach().clone().requires_grad_(True)
            Oh, _ = attention(Qh, Kh, Vh, cfg, "expand")
            (Oh * dO_h).sum().backward()
            contribs.append(Kh.grad[:, k])
        summed = torch.stack(contribs, 0).sum(0)
        per_head.append({
            "kv_head": k, "q_heads": q_heads,
            "max_dev_from_sum": float((summed - K.grad[:, k]).abs().max()),
            "max_dev_manual": float((summed - dK_m[:, k]).abs().max()),
        })
    return {
        "config": cfg, "group": G,
        "err_dK_manual_vs_autograd": eK,
        "err_dV_manual_vs_autograd": eV,
        "err_dQ_manual_vs_autograd": eQ,
        "per_head": per_head,
        "worst_sum_deviation": max(p["max_dev_from_sum"] for p in per_head),
    }