"""Attention: head split, causal masking, row softmax, and its reverse.

Contract §2 IDs 6, 8, 9, 10, 11, and invariants I1, I2 and C2.

Shapes
------
    Q~, K~, V            (B, T, d)
    after split          (H, B, T, d_h)      with d_h = d / H
    S^(h)                (B, T, T)           per head
    A^(h)                (B, T, T)           per head
    O^(h)                (B, T, d_h)         per head
    O                    (B, T, d)           after concat

The scale factor is 1/sqrt(**d_h**), never 1/sqrt(d) (contract C2). These agree
exactly when H = 1, which is why the mistake is invisible in a one-head test and
appears only once the model is widened.

Invariant I2, which the reverse must respect: A masked column of ``A^(h)`` is
exactly zero, but the corresponding entry of ``A_bar^(h)`` need not be.
``A_bar`` is the adjoint of the softmax *output*; the projection that annihilates
masked columns happens one step later, inside the softmax reverse. A nonzero
masked ``A_bar`` with a zero masked ``S_bar`` is the expected behaviour.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import torch


def split_heads(z: torch.Tensor, H: int) -> torch.Tensor:
    """``(B, T, d) -> (H, B, T, d_h)`` — contract §2 ID 6.

    ``view(B, T, H, d_h).permute(2, 0, 1, 3)`` puts the head axis first so that
    every downstream per-head operation is a plain batched matmul over
    ``(B, T, d_h)`` and heads are checked independently. With ``H == 1`` this is
    an identity reshuffle, which is precisely why the one-head derivation cannot
    exhibit the multi-head reverse (Chapter 8).
    """
    B, T, d = z.shape
    if d % H != 0:
        raise ValueError(f"cannot split d={d} into H={H} heads")
    return z.view(B, T, H, d // H).permute(2, 0, 1, 3).contiguous()


def merge_heads(z: torch.Tensor) -> torch.Tensor:
    """Reverse of :func:`split_heads`: ``(H, B, T, d_h) -> (B, T, d)``.

    This is a **reshape**, i.e. the reverse of a split is a split of the adjoint.
    Summing across heads here instead of reshaping is the classic multi-head
    backprop bug; ``ch08_two_head_verify.py`` asserts it is caught.
    """
    H, B, T, d_h = z.shape
    return z.permute(1, 2, 0, 3).reshape(B, T, H * d_h)


def causal_mask(T: int, *, dtype: torch.dtype = torch.float32,
                device=None) -> torch.Tensor:
    """``M_ij = -inf`` for ``j > i``, else ``0``. Shape ``(T, T)``."""
    m = torch.full((T, T), float("-inf"), dtype=dtype, device=device)
    return torch.triu(m, diagonal=1)


def scores(q: torch.Tensor, k: torch.Tensor, d_h: int) -> torch.Tensor:
    """``S = Q K^T / sqrt(d_h)`` for one head. ``q``, ``k`` are ``(B, T, d_h)``."""
    return (q @ k.transpose(-2, -1)) / math.sqrt(d_h)


class AttentionCache(NamedTuple):
    A: torch.Tensor          # (H, B, T, T)  row-softmax probabilities
    S: torch.Tensor          # (H, B, T, T)  masked scores, -inf where illegal
    V: torch.Tensor          # (H, B, T, d_h)
    d_h: int


def attention_forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                      d_h: int) -> tuple[torch.Tensor, AttentionCache]:
    """One head: masked row softmax then ``O = A V``.

    ``q``, ``k``, ``v`` are ``(B, T, d_h)`` and the result is ``(B, T, d_h)``.
    """
    T = q.shape[-2]
    S = scores(q, k, d_h) + causal_mask(T, dtype=q.dtype, device=q.device)
    A = torch.softmax(S, dim=-1)
    O = A @ v
    return O, AttentionCache(A=A, S=S, V=v, d_h=d_h)


def softmax_backward_rows(A: torch.Tensor, A_bar: torch.Tensor
                          ) -> torch.Tensor:
    """Row-softmax reverse: ``s_bar = a * (a_bar - (a . a_bar) 1)``.

    ``A`` and ``A_bar`` are ``(..., T, T)`` with rows summing to 1. Masked columns
    carry ``A == 0``, hence ``S_bar == 0`` there automatically, which is how
    invariant I1 is satisfied without a special case.
    """
    if A.shape != A_bar.shape:
        raise ValueError(f"shape mismatch: {tuple(A.shape)} vs {tuple(A_bar.shape)}")
    row_dot = (A * A_bar).sum(dim=-1, keepdim=True)
    return A * (A_bar - row_dot)


def attention_backward_full(q: torch.Tensor, k: torch.Tensor,
                            O_bar: torch.Tensor, cache: AttentionCache
                            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reverse of :func:`attention_forward` for one head.

    Returns ``(q_bar, k_bar, v_bar)``.

        A_bar = O_bar V^T
        V_bar = A^T O_bar
        S_bar = softmax_backward_rows(A, A_bar)
        Q_bar = S_bar K / sqrt(d_h)
        K_bar = S_bar^T Q / sqrt(d_h)

    ``Q_bar`` needs ``K`` and ``K_bar`` needs ``Q``, so the rotated queries and
    keys are arguments rather than being stashed in the cache; stashing them
    would keep a rotated copy of every position alive for the whole block, which
    is memory the cache exists to avoid.
    """
    A, V, d_h = cache.A, cache.V, cache.d_h
    A_bar = O_bar @ V.transpose(-2, -1)          # (B, T, T)
    V_bar = A.transpose(-2, -1) @ O_bar          # (B, T, d_h)
    S_bar = softmax_backward_rows(A, A_bar)      # (B, T, T)
    scale = math.sqrt(d_h)
    Q_bar = S_bar @ k / scale
    K_bar = S_bar.transpose(-2, -1) @ q / scale
    return Q_bar, K_bar, V_bar
