"""The canonical forward graph, shape-asserted.

This module is the single implementation referred to throughout the book. It
builds one block of the canonical decoder from ``tgd`` primitives and records
every intermediate under its contract ID, so that:

* the manuscript's shape ledger is generated rather than typed;
* a shape change breaks a script instead of quietly invalidating a figure;
* chapters never restate the graph.

Chapters 6, 7, and 8 extend this with the reverse pass. The forward is frozen
now; ``BACKWARD_STATUS`` records, per ID, whether a reverse is implemented, and
the ledger checker reports any ID that is forward-only, so "zero missing
adjoints" is a machine-checked property rather than an aspiration.
"""
from __future__ import annotations

from typing import NamedTuple

import torch

from . import attention as att
from . import rope as rope_mod
from . import rmsnorm as rn
from . import swiglu as glu
from .dims import Dims

#: Contract §2, every forward ID and whether its reverse is implemented.
#: "pending" IDs are forward-only today; the ledger checker reports them.
BACKWARD_STATUS: dict[int, str] = {
    1: "implemented",   # E           -> tgd.loss.embedding_scatter_add
    2: "implemented",   # X_0         -> tgd.reverse (block caller)
    3: "implemented",   # N^(1)       -> tgd.rmsnorm.rmsnorm_backward
    4: "implemented",   # W_Q,K,V     -> tgd.reverse.block_backward
    5: "implemented",   # Q~,K~,V     -> tgd.reverse.block_backward
    6: "implemented",   # head slice  -> tgd.attention.split_heads
    7: "implemented",   # RoPE        -> tgd.rope
    8: "implemented",   # S^(h)       -> tgd.reverse.block_backward
    9: "implemented",   # A^(h)       -> tgd.attention.softmax_backward_rows
    10: "implemented",  # O^(h)       -> tgd.attention.attention_backward_full
    11: "implemented",  # concat      -> tgd.attention.merge_heads
    12: "implemented",  # W_O         -> tgd.reverse.block_backward
    13: "implemented",  # Y           -> tgd.reverse.block_backward
    14: "implemented",  # R           -> tgd.residual.residual_first_backward
    15: "implemented",  # N^(2)       -> tgd.rmsnorm.rmsnorm_backward
    16: "implemented",  # W_g,W_u     -> tgd.swiglu.swiglu_backward
    17: "implemented",  # G^pre       -> tgd.swiglu.swiglu_backward
    18: "implemented",  # U           -> tgd.swiglu.swiglu_backward
    19: "implemented",  # G           -> tgd.swiglu.swiglu_backward
    20: "implemented",  # D           -> tgd.swiglu.swiglu_backward
    21: "implemented",  # W_d         -> tgd.swiglu.swiglu_backward
    22: "implemented",  # F           -> tgd.reverse.block_backward
    23: "implemented",  # X'          -> tgd.residual.residual_second_backward
    24: "implemented",  # N_f         -> tgd.rmsnorm.rmsnorm_backward
    25: "implemented",  # W_out       -> tgd.reverse.output_backward
    26: "implemented",  # Z           -> tgd.loss.ce_backward
    27: "implemented",  # p           -> tgd.loss.ce_backward
    28: "n/a",          # m           -- a mask, not a differentiated tensor
    29: "implemented",  # loss        -> tgd.loss.masked_mean_ce
}

#: Human-readable name per contract ID, used to generate the ledger table.
ID_NAMES: dict[int, str] = {
    1: "E", 2: "X_0", 3: "N^(1)", 4: "W_Q, W_K, W_V", 5: "Q~, K~, V",
    6: "Q^(h), K^(h), V^(h)", 7: "Q, K (RoPE)", 8: "S^(h)", 9: "A^(h)",
    10: "O^(h)", 11: "O", 12: "W_O", 13: "Y", 14: "R", 15: "N^(2)",
    16: "W_g, W_u", 17: "G^pre", 18: "U", 19: "G", 20: "D", 21: "W_d",
    22: "F", 23: "X'", 24: "N_f", 25: "W_out", 26: "Z", 27: "p",
    28: "m", 29: "loss",
}


class BlockParams(NamedTuple):
    g1: torch.Tensor       # (d,)
    W_Q: torch.Tensor      # (d, d)
    W_K: torch.Tensor      # (d, d)
    W_V: torch.Tensor      # (d, d)
    W_O: torch.Tensor      # (d, d)
    g2: torch.Tensor       # (d,)
    W_g: torch.Tensor      # (d, d_ff)
    W_u: torch.Tensor      # (d, d_ff)
    W_d: torch.Tensor      # (d_ff, d)


class BlockCache(NamedTuple):
    """Every intermediate of one canonical block, keyed by contract ID."""

    X: torch.Tensor
    N1: rn.RMSNormCache
    Q_lin: torch.Tensor        # ID 5, pre-split dense projection, (B,T,d)
    K_lin: torch.Tensor
    V_lin: torch.Tensor
    Q_t: torch.Tensor          # ID 6, (H,B,T,d_h) pre-RoPE
    K_t: torch.Tensor
    V: torch.Tensor
    angles: torch.Tensor
    Q: torch.Tensor            # rotated
    K: torch.Tensor
    A: att.AttentionCache
    S: torch.Tensor            # ID 8, per-head masked scores (H,B,T,T)
    O: torch.Tensor            # concatenated, (B,T,d)
    Y: torch.Tensor
    R: torch.Tensor
    N2: rn.RMSNormCache
    glu: glu.SwiGLUCache
    F: torch.Tensor
    Xp: torch.Tensor


def expected_shapes(dims: Dims) -> dict[int, tuple]:
    """The shape each contract ID must have. Used for ledger generation."""
    B, T, d, d_h, d_ff, H, V = (
        dims.B, dims.T, dims.d, dims.d_h, dims.d_ff, dims.H, dims.V
    )
    return {
        2: (B, T, d),
        3: (B, T, d),
        5: (B, T, d),
        6: (H, B, T, d_h),
        7: (H, B, T, d_h),
        8: (H, B, T, T),
        9: (H, B, T, T),
        10: (H, B, T, d_h),
        11: (B, T, d),
        13: (B, T, d),
        14: (B, T, d),
        15: (B, T, d),
        17: (B, T, d_ff),
        18: (B, T, d_ff),
        19: (B, T, d_ff),
        20: (B, T, d_ff),
        22: (B, T, d),
        23: (B, T, d),
        24: (B, T, d),
        26: (B, T, V),
        27: (B, T, V),
    }


def block_forward(X: torch.Tensor, p: BlockParams, dims: Dims,
                  *, minimal_cut: bool = False
                  ) -> tuple[torch.Tensor, BlockCache]:
    """One canonical Pre-RMSNorm decoder block. Contract §2 IDs 2, 3, 5-23.

    ``X`` is the incoming residual stream, ``(B, T, d)``. Returns ``(X', cache)``
    where ``X'`` is the outgoing residual stream.

    ``minimal_cut=False`` (default) keeps every forward intermediate in the
    cache, which is what the shape ledger and the chapter traces read.
    ``minimal_cut=True`` drops the storages the reverse pass never re-read --
    the pre-split dense projections (``Q_lin``, ``K_lin``, ``V_lin``), the
    pre-RoPE head views (``Q_t``, ``K_t``), the raw residual (``R``), the
    post-attention projection (``Y``), the FFN output (``F``) and the block
    output itself (``Xp``); each is recomputable from its inputs and the
    reverse's own adjoints, and together they were measured at ~12.5% of the
    block's saved bytes (11.2 of 88.0 MiB at the B=2, T=64, V=1024, d=256
    benchmark shape). The fields are set to ``None`` rather than removed so
    existing attribute readers fail loudly instead of silently reading a
    stale tensor. Shape-ledger readers (``ch04_graph_ledger.py``) call the
    default mode and are unaffected.
    """
    B, T, d = X.shape
    if d != dims.d:
        raise ValueError(f"X width {d} does not match dims.d={dims.d}")
    H, d_h = dims.H, dims.d_h

    # --- ID 3: attention-side Pre-RMSNorm -------------------------------
    N1_y, N1 = rn.rmsnorm_forward(X, p.g1, dims.eps)

    # --- ID 4, 5: dense projections (row convention: X @ W) -------------
    Q_lin = N1_y @ p.W_Q
    K_lin = N1_y @ p.W_K
    V_lin = N1_y @ p.W_V

    # --- ID 6: head split ------------------------------------------------
    Q_t = att.split_heads(Q_lin, H)
    K_t = att.split_heads(K_lin, H)
    V_h = att.split_heads(V_lin, H)

    # --- ID 7: per-head RoPE on Q and K only -----------------------------
    angles = rope_mod.rope_angles(T, d_h, dims.rope_base, dtype=X.dtype)
    Q = torch.stack([rope_mod.rope_forward_rows(Q_t[h], angles) for h in range(H)])
    K = torch.stack([rope_mod.rope_forward_rows(K_t[h], angles) for h in range(H)])

    # --- ID 8, 9, 10: per-head causal attention --------------------------
    O_h, head_caches, S_h = [], [], []
    for h in range(H):
        O_head, head_cache = att.attention_forward(Q[h], K[h], V_h[h], d_h)
        O_h.append(O_head)
        head_caches.append(head_cache)
        S_h.append(head_cache.S)
    O = torch.cat(O_h, dim=-1)                     # ID 11

    # Stack the per-head softmax outputs so invariant I1 can be checked per head
    # rather than after summing over heads, which would hide a per-head failure.
    A_cache = att.AttentionCache(
        A=torch.stack([c.A for c in head_caches], dim=0),   # (H, B, T, T)
        S=torch.stack(S_h, dim=0),                          # (H, B, T, T)
        V=torch.stack([c.V for c in head_caches], dim=0),   # (H, B, T, d_h)
        d_h=d_h,
    )

    # --- ID 12, 13: output projection and first residual ----------------
    Y = O @ p.W_O
    R = X + Y

    # --- ID 15: feed-forward-side Pre-RMSNorm ---------------------------
    N2_y, N2 = rn.rmsnorm_forward(R, p.g2, dims.eps)

    # --- ID 16-22: SwiGLU and second residual ----------------------------
    F, glu_cache = glu.swiglu_forward(N2_y, p.W_g, p.W_u, p.W_d)
    Xp = R + F

    if minimal_cut:
        # Dead-for-the-reverse storages. The reverse pass recomputes R_bar's
        # identity term from Xp_bar, Y_bar from R_bar, and never touches the
        # pre-split / pre-RoPE / output-side tensors, so releasing them trims
        # the pinned set without changing any gradient.
        cache = BlockCache(
            X=X, N1=N1, Q_lin=None, K_lin=None, V_lin=None,      # type: ignore[arg-type]
            Q_t=None, K_t=None, V=V_h, angles=angles, Q=Q, K=K,  # type: ignore[arg-type]
            A=A_cache, S=A_cache.S, O=O, Y=None, R=None,         # type: ignore[arg-type]
            N2=N2, glu=glu_cache, F=None, Xp=None,               # type: ignore[arg-type]
        )
        return Xp, cache

    cache = BlockCache(
        X=X, N1=N1, Q_lin=Q_lin, K_lin=K_lin, V_lin=V_lin,
        Q_t=Q_t, K_t=K_t, V=V_h, angles=angles, Q=Q, K=K,
        A=A_cache, S=A_cache.S, O=O, Y=Y, R=R, N2=N2, glu=glu_cache, F=F, Xp=Xp,
    )
    return Xp, cache


def model_forward(E: torch.Tensor, x: torch.Tensor, blocks: list[BlockParams],
                  gf: torch.Tensor, W_out: torch.Tensor, dims: Dims,
                  y: torch.Tensor | None = None,
                  mask: torch.Tensor | None = None):
    """Full canonical forward: embedding -> L blocks -> final norm -> logits.

    Returns ``(loss_or_Z, record)`` where ``record`` maps contract IDs to
    tensors so the ledger can check shapes. When ``y`` is given the loss is
    returned; otherwise the logits ``Z`` are.
    """
    from . import loss as loss_mod

    X0 = E[x]                                                   # ID 2
    rec: dict[int, torch.Tensor] = {2: X0}
    X = X0
    for p in blocks:
        X, c = block_forward(X, p, dims)
        rec[23] = X
    Nf_y, Nf = rn.rmsnorm_forward(X, gf, dims.eps)               # ID 24
    Z = Nf_y @ W_out                                              # ID 26
    rec[24] = Nf_y
    rec[26] = Z
    if y is None:
        return Z, rec
    loss = loss_mod.masked_mean_ce(Z, y, mask)                   # ID 29
    return loss, rec
