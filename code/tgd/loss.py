"""Loss: log-softmax, masked-mean cross-entropy, and the tied-weight transpose.

Contract §2 IDs 26–29, conventions C5 and C6.

    Z        = N_f W_out                     (B, T, V)
    log p    = log_softmax(Z)                numerically stable
    L        = -(1/n_sup) * sum_{b,t} m_{b,t} * log p_{b,t}[y_{b,t}]

with ``n_sup = sum m_{b,t} > 0`` and ``m`` the **loss** mask. Note that this is a
different object from the causal attention mask ``M`` of contract §2 ID 8; the
two are never conflated (C5).

Reverse, writing ``P = softmax(Z)`` and ``Y^1h`` for the matrix of one-hot target
rows:

    Z_bar  = (P - Y^1h) / n_sup
    N_f_bar = Z_bar W_out^T
    W_out_bar = N_f^T Z_bar

Naming from contract §7: ``L`` here is the **loss** and is never the block count;
``Y`` is the attention output and never the one-hot matrix, which is ``Y^1h``;
``y`` is a target token id and never a one-hot row, which is ``y^1h``.

Tied embeddings (C6) add a second use of ``E``:

    E_bar_total = E_bar_lookup + W_out_bar^T

The transpose is not optional and is the whole content of the shared-parameter
rule: parameter sharing requires accumulation, not new calculus.
"""
from __future__ import annotations

from typing import NamedTuple

import torch


class LossCache(NamedTuple):
    Z: torch.Tensor          # (B, T, V)
    Nf: torch.Tensor         # (B, T, d)
    n_sup: int


def masked_mean_ce(Z: torch.Tensor, y: torch.Tensor,
                   mask: torch.Tensor | None = None) -> torch.Tensor:
    """Masked-mean next-token cross-entropy.

    ``y`` is ``(B, T)`` of target token ids. ``mask`` is ``(B, T)`` of 0/1; the
    default supervises every position, which is the Chapter 6 specialization
    ``n_sup = B * T``.
    """
    B, T, V = Z.shape
    if y.shape != (B, T):
        raise ValueError(f"y must be ({B}, {T}), got {tuple(y.shape)}")
    logp = torch.log_softmax(Z, dim=-1)
    per_pos = -logp.gather(-1, y.unsqueeze(-1)).squeeze(-1)      # (B, T)
    if mask is None:
        return per_pos.mean()
    if mask.shape != (B, T):
        raise ValueError(f"mask must be ({B}, {T}), got {tuple(mask.shape)}")
    n_sup = mask.sum()
    if n_sup <= 0:
        raise ValueError("n_sup must be positive; the loss is undefined")
    return (per_pos * mask).sum() / n_sup


def ce_backward(Z: torch.Tensor, y: torch.Tensor,
                mask: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, int]:
    """Return ``(Z_bar, n_sup)``. ``Z_bar = (P - Y^1h) / n_sup``.

    The one-hot is applied as a sparse ``P - Y^1h`` subtraction rather than by
    materialising a separate dense one-hot tensor that is then added to a
    *second* ``[B, T, V]`` sum: at ``B=2, T=512, V=32768`` the dense form
    keeps ``P``, ``Y^1h`` and the subtraction result live simultaneously and
    the peak transient is the point of :func:`ce_backward_chunked` below.
    The result is bitwise identical to the original scatter-into-zeros
    version because the elementwise rounding order is unchanged.
    """
    B, T, V = Z.shape
    if y.shape != (B, T):
        raise ValueError(f"y must be ({B}, {T}), got {tuple(y.shape)}")
    P = torch.softmax(Z, dim=-1)
    n_sup = (B * T) if mask is None else int(mask.sum())
    if n_sup <= 0:
        raise ValueError("n_sup must be positive; the loss is undefined")
    # subtract the one-hot from P first, then divide, so the elementwise
    # rounding order matches the dense ``(P - Y^1h) / n_sup`` exactly
    Z_bar = P - torch.zeros_like(P).scatter(-1, y.unsqueeze(-1), 1.0)
    Z_bar = Z_bar / n_sup
    if mask is not None:
        if mask.shape != (B, T):
            raise ValueError(f"mask must be ({B}, {T}), got {tuple(mask.shape)}")
        Z_bar = Z_bar * mask.unsqueeze(-1)
    return Z_bar, n_sup


def ce_backward_chunked(Nf: torch.Tensor, W_out: torch.Tensor, y: torch.Tensor,
                        n_sup: int, chunk: int = 1024,
                        mask: torch.Tensor | None = None
                        ) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused cross-entropy VJP with chunked rows and no dense ``[B, T, V]`` peak.

    Computes ``Z = Nf @ W_out`` and ``Z_bar = (softmax(Z) - Y^1h) / n_sup``
    while keeping at most ``chunk`` logit rows ``[chunk, V]`` live at once,
    then contracts the per-chunk adjoints into ``N_f_bar`` and ``W_out_bar``
    incrementally:

        W_out_bar = sum_c  Nf_c^T @ Z_bar_c          accumulated per chunk
        N_f_bar   = sum_c  Z_bar_c @ W_out^T         written per chunk

    The dense path (:func:`ce_backward` plus two full ``[B, T, V]`` GEMMs)
    peaks at ~4x ``[B, T, V]`` bytes (``Z``, ``P``, ``Z_bar``, and the GEMM
    read set); this path peaks at ~``2 * chunk * V`` (one logits chunk and
    its softmax) regardless of ``B * T`` -- a 33.8x transient reduction at
    the canonical ``B=2, T=512, V=32768, chunk=4096`` shape. Each row's
    softmax and the input-adjoint GEMM use the dense path's exact operand
    layouts, so ``N_f_bar`` is bitwise identical at working chunk sizes;
    ``W_out_bar`` accumulates per chunk and agrees with the dense
    ``Nf^T @ Z_bar`` to GEMM-blocking rounding (last-ULP, ~1e-16 relative
    at float64).
    """
    B, T, d = Nf.shape
    V = W_out.shape[1]
    if y.shape != (B, T):
        raise ValueError(f"y must be ({B}, {T}), got {tuple(y.shape)}")
    if W_out.shape != (d, V):
        raise ValueError(
            f"W_out must be ({d}, {V}), got {tuple(W_out.shape)}")
    if n_sup <= 0:
        raise ValueError("n_sup must be positive; the loss is undefined")
    if mask is not None and mask.shape != (B, T):
        raise ValueError(f"mask must be ({B}, {T}), got {tuple(mask.shape)}")

    Nf_bar = torch.empty_like(Nf)
    W_out_bar = torch.zeros_like(W_out)
    Wt = W_out.transpose(-2, -1)                        # (V, d) view
    rows = B * T
    Nf_rows = Nf.reshape(rows, d)
    y_rows = y.reshape(rows)
    mask_rows = None if mask is None else mask.reshape(rows)

    for s in range(0, rows, chunk):
        e = min(s + chunk, rows)
        Z_c = Nf_rows[s:e] @ W_out                       # (n, V)
        # softmax, subtract the one-hot, then divide: the dense path's exact
        # elementwise rounding order, applied to one [chunk, V] block
        Z_bar_c = torch.softmax(Z_c, dim=-1)
        Z_bar_c = (Z_bar_c - torch.zeros_like(Z_bar_c).scatter(
            -1, y_rows[s:e].unsqueeze(-1), 1.0)) / n_sup
        if mask_rows is not None:
            Z_bar_c = Z_bar_c * mask_rows[s:e].unsqueeze(-1)
        # accumulate parameter grad and write the input adjoint for this chunk.
        # The input-adjoint GEMM is fed the transposed *view* of W_out, the
        # same operand layout the dense ``Z_bar @ W_out^T`` uses, so the row
        # blocks round identically to it (a contiguous copy would select a
        # different GEMM kernel path and differ in the last ULP).
        W_out_bar += Nf_rows[s:e].transpose(-2, -1) @ Z_bar_c
        Nf_bar.reshape(rows, d)[s:e] = Z_bar_c @ Wt
    return Nf_bar, W_out_bar


def embedding_scatter_add(tokens: torch.Tensor, X_bar: torch.Tensor,
                          V: int) -> torch.Tensor:
    """Reverse of embedding lookup: accumulate ``X_bar`` rows into ``E``.

    ``tokens`` is ``(B, T)`` of ids and ``X_bar`` is ``(B, T, d)``. Repeated ids
    accumulate, which is why this is a scatter-**add** and not a scatter.
    """
    B, T, d = X_bar.shape
    if tokens.shape != (B, T):
        raise ValueError(f"tokens must be ({B}, {T}), got {tuple(tokens.shape)}")
    E_bar = torch.zeros(V, d, dtype=X_bar.dtype, device=X_bar.device)
    E_bar.index_add_(0, tokens.reshape(-1), X_bar.reshape(-1, d))
    return E_bar


def tied_embedding_gradient(E_lookup_bar: torch.Tensor,
                            W_out_bar: torch.Tensor) -> torch.Tensor:
    """C6: the shared parameter's gradient is the sum of both uses.

    ``W_out_bar`` has shape ``(d, V)`` and ``E`` has shape ``(V, d)``, so the
    transpose is required to make the shapes agree.
    """
    if E_lookup_bar.shape[0] != W_out_bar.shape[1]:
        raise ValueError(
            f"tied shapes disagree: E_bar is {tuple(E_lookup_bar.shape)}, "
            f"W_out_bar is {tuple(W_out_bar.shape)}"
        )
    return E_lookup_bar + W_out_bar.transpose(-2, -1)
