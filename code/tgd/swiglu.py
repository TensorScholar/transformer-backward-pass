"""SwiGLU — contract §2 IDs 16–22, and the ``G^pre`` / ``G`` / ``D`` discipline.

    G^pre = N^(2) W_g          gate pre-activation       (B, T, d_ff)
    U     = N^(2) W_u          up projection              (B, T, d_ff)
    G     = SiLU(G^pre)        activated gate             (B, T, d_ff)
    D     = G * U              GATED PRODUCT              (B, T, d_ff)
    F     = D W_d              down projection            (B, T, d)

Naming, from contract §7, which this module is the reference for:

* ``G^pre`` is the gate pre-activation. The verification scripts of the earlier
  revision named this tensor ``P`` in their local variables; the name ``P`` is
  **forbidden in prose** and permitted only as a script-local variable.
* ``D`` is the gated product. ``H`` is the head count and is never used for this
  tensor. ``D`` is also not "depth" and not "data".
* ``U`` is the up pre-activation, not the feed-forward network as a whole.

Reverse
-------
    D_bar = F_bar W_d^T
    G_bar = D_bar * U
    U_bar = D_bar * G
    Gpre_bar = G_bar * SiLU'(G^pre)
    W_g_bar = N^(2)T Gpre_bar        N^(2)bar_g = Gpre_bar W_g^T
    W_u_bar = N^(2)T U_bar           N^(2)bar_u = U_bar     W_u^T
    N^(2)bar  = N^(2)bar_g + N^(2)bar_u          <- a merge, contract A.3
"""
from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F


def silu(x: torch.Tensor) -> torch.Tensor:
    return F.silu(x)


def silu_prime(x: torch.Tensor) -> torch.Tensor:
    """``d/dx [x * sigmoid(x)] = sigma + x * sigma * (1 - sigma)``."""
    sig = torch.sigmoid(x)
    return sig + x * sig * (1.0 - sig)


class SwiGLUCache(NamedTuple):
    N2: torch.Tensor        # (B, T, d)   normalized FFN input
    Gpre: torch.Tensor      # (B, T, d_ff)
    U: torch.Tensor         # (B, T, d_ff)
    G: torch.Tensor         # (B, T, d_ff)


def swiglu_forward(N2: torch.Tensor, W_g: torch.Tensor, W_u: torch.Tensor,
                   W_d: torch.Tensor) -> tuple[torch.Tensor, SwiGLUCache]:
    """Feed-forward branch. Returns ``(F, cache)``."""
    Gpre = N2 @ W_g
    U = N2 @ W_u
    G = silu(Gpre)
    D = G * U
    F = D @ W_d
    return F, SwiGLUCache(N2=N2, Gpre=Gpre, U=U, G=G)


def _rows(t: torch.Tensor) -> torch.Tensor:
    """Flatten every leading token/batch axis into one matrix axis.

    The weight gradients of this block are summed over **all** rows: with
    ``(B, T, ...)`` inputs, ``W_g_bar`` must be ``(d, d_ff)``, not the
    ``(B, d, d_ff)`` a batched GEMM over rank-3 operands would return. The
    same reduction is why the canonical path in ``reverse.py`` flattens
    before its parameter contractions.
    """
    return t.reshape(-1, t.shape[-1])


def swiglu_backward(F_bar: torch.Tensor, W_g: torch.Tensor,
                    W_u: torch.Tensor, W_d: torch.Tensor,
                    cache: SwiGLUCache
                    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Reverse of :func:`swiglu_forward`.

    Returns ``(parameter_grads, N2_bar)`` where ``N2_bar`` is the **merged**
    adjoint of the normalized FFN input. Both summands are named explicitly
    because this is merge ``N^(2)`` in contract §3. The parameter gradients
    are rank-agnostic: any leading ``(B, T)`` (or just ``T``) axes are
    flattened so each ``W_*_bar`` sums over every token and returns the
    parameter's own 2-D shape.
    """
    N2, Gpre, U, G = cache.N2, cache.Gpre, cache.U, cache.G

    D_bar = F_bar @ W_d.transpose(-2, -1)        # (..., d_ff)
    G_bar = D_bar * U
    U_bar = D_bar * G
    Gpre_bar = G_bar * silu_prime(Gpre)

    # named summands of the N^(2) merge, contract A.3
    N2_bar_g = Gpre_bar @ W_g.transpose(-2, -1)
    N2_bar_u = U_bar @ W_u.transpose(-2, -1)
    N2_bar = N2_bar_g + N2_bar_u

    # Parameter grads: flatten leading token/batch axes so each contraction
    # sums over rows and yields the weight's shape, matching reverse.py.
    # (``N2.transpose(-2,-1) @ Gpre_bar`` on rank-3 inputs would instead run
    # as B separate GEMMs and return ``(B, d, d_ff)``.)
    Fb_rows, N2_rows = _rows(F_bar), _rows(N2)
    Gpre_rows, U_rows = _rows(Gpre_bar), _rows(U_bar)
    D_rows = _rows(G * U)
    grads = {
        "W_g": N2_rows.transpose(-2, -1) @ Gpre_rows,
        "W_u": N2_rows.transpose(-2, -1) @ U_rows,
        "W_d": D_rows.transpose(-2, -1) @ Fb_rows,
    }
    return grads, N2_bar
