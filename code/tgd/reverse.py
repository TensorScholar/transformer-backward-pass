"""The complete reverse pass of the canonical block — contract §8.

This module closes the 23 contract IDs that Chapter 4 reported as forward-only.
It is the reverse of :func:`tgd.graph.block_forward`, written out by hand from
``tgd`` primitives, with every merge summand named as contract A.3 requires.

Order of computation, which is forced by the graph and not chosen for
convenience:

    1. vocabulary projection and the masked-mean loss
    2. final RMSNorm
    3. second residual  (merge R_ell)
    4. SwiGLU           (merge N^(2))
    5. attention-side RMSNorm
    6. first residual   (merge X_ell)
    7. output projection
    8. per-head attention reverse, then the concat adjoint
    9. RoPE reverse
   10. dense projection gradients and the three-way merge (merge N^(1))
   11. attention-side RMSNorm reverse
   12. embedding scatter-add

Works for any ``H``: with ``H == 1`` the head split and concat are identity
reshuffles, which is exactly why the multi-head case is exercised separately by
``ch08_two_head_verify.py``.

Five deletion mutations are supported, and contract A.6 requires a harness to
prove each one is detectable:

    drop_v_merge             delete the value path into the N^(1) merge
    drop_first_residual_identity
    drop_second_residual_identity
    drop_head_k_v_path       delete one head's value path (H > 1)
    sum_instead_of_concat    reshape the concat adjoint as a sum
"""
from __future__ import annotations

import math
from typing import NamedTuple

import torch

from . import attention as att
from . import loss as loss_mod
from . import residual as res
from . import rope as rope_mod
from . import rmsnorm as rn
from . import swiglu as glu
from .dims import Dims

#: Every supported mutation, for the ledger's DELETION_MUTATIONS registry.
MUTATIONS = (
    "drop_v_merge",
    "drop_first_residual_identity",
    "drop_second_residual_identity",
    "drop_head_k_v_path",
    "sum_instead_of_concat",
    # merge N^(2) has two summands; until v1.2 only merge N^(1) had a dedicated
    # control, which meant the register documented a merge it did not test.
    "drop_u_path",
    # and the same for the query and key paths of merge N^(1)
    "drop_qk_path",
)


class ReverseTrace(NamedTuple):
    """Named intermediates, so a test can assert a specific summand is nonzero.

    Without this, "the merge is present" is only checkable by deleting it and
    hoping the error moves.
    """

    Z_bar: torch.Tensor
    Nf_bar: torch.Tensor
    Xp_bar: torch.Tensor
    F_bar: torch.Tensor
    R_bar: torch.Tensor
    N2_bar_g: torch.Tensor
    N2_bar_u: torch.Tensor
    Y_bar: torch.Tensor
    O_bar: torch.Tensor
    S_bar: torch.Tensor
    A_bar: torch.Tensor
    Q_bar: torch.Tensor
    K_bar: torch.Tensor
    V_bar: torch.Tensor
    Qt_bar: torch.Tensor
    Kt_bar: torch.Tensor
    Vt_bar: torch.Tensor
    N1_bar_q: torch.Tensor
    N1_bar_k: torch.Tensor
    N1_bar_v: torch.Tensor
    N1_bar: torch.Tensor
    X_attn_bar: torch.Tensor
    X_bar: torch.Tensor


def _rows(z: torch.Tensor) -> torch.Tensor:
    """Reshape ``(B, T, d)`` to ``(B*T, d)``.

    The row convention says a linear map is ``X W``, so a parameter gradient is
    the contraction over the *token* axis. Doing that with ``transpose(-2, -1)``
    is wrong whenever a batch axis is present: it yields ``(B, d, T)`` and the
    subsequent matmul silently broadcasts into a batched outer product. Flattening
    the token axis first is correct for any ``B`` and is what the convention
    actually says.
    """
    return z.reshape(-1, z.shape[-1])


def _detach_norm_cache(c: rn.RMSNormCache) -> rn.RMSNormCache:
    """Detach a norm cache; the manual reverse never needs the graph."""
    return rn.RMSNormCache(x=c.x.detach(), r=c.r.detach(), u=c.u.detach(),
                           g=c.g.detach())


def output_backward(Nf_y: torch.Tensor, Z: torch.Tensor, W_out: torch.Tensor,
                    y: torch.Tensor, mask: torch.Tensor | None = None):
    """Contract IDs 29 -> 25 -> 24. Returns ``(W_out_bar, Nf_bar, n_sup)``."""
    Z_bar, n_sup = loss_mod.ce_backward(Z, y, mask)
    W_out_bar = _rows(Nf_y.detach()).T @ _rows(Z_bar)
    Nf_bar = Z_bar @ W_out.detach().transpose(-2, -1)
    return W_out_bar, Nf_bar, n_sup


def block_backward(Xp_bar: torch.Tensor, cache, p, dims: Dims,
                   tokens: torch.Tensor, *,
                   mutation: str | None = None
                   ) -> tuple[dict[str, torch.Tensor], ReverseTrace]:
    """Reverse of :func:`tgd.graph.block_forward`.

    Parameters
    ----------
    Xp_bar
        adjoint of the block's output residual stream, ``(B, T, d)``.
    cache
        the :class:`tgd.graph.BlockCache` from the forward pass.
    p
        the :class:`tgd.graph.BlockParams` used in the forward pass.
    tokens
        ``(B, T)`` token ids, needed for the embedding scatter-add.
    mutation
        one of :data:`MUTATIONS`, for tests only.

    Returns
    -------
    ``(grads, trace)`` where ``grads`` is keyed by the contract's parameter
    names (``g1``, ``W_Q``, ``W_K``, ``W_V``, ``W_O``, ``g2``, ``W_g``,
    ``W_u``, ``W_d``, ``gf``) and ``trace`` carries the named merge summands.
    """
    H, d_h, d = dims.H, dims.d_h, dims.d
    grads: dict[str, torch.Tensor] = {}

    # --- 3/4. second residual: X' = R + F (merge R_ell, contract 3) -----
    # The feed-forward branch arrives at R only after the SwiGLU and the second
    # RMSNorm have been reversed, so the merge is completed below; here we take
    # the identity contribution and the immediate F adjoint.
    R_bar_identity = torch.zeros_like(Xp_bar) \
        if mutation == "drop_second_residual_identity" else Xp_bar
    F_bar = Xp_bar

    # --- 4. SwiGLU (merge N^(2)) ----------------------------------------
    D_bar = F_bar @ p.W_d.detach().transpose(-2, -1)
    G_bar = D_bar * cache.glu.U.detach()
    U_bar = D_bar * cache.glu.G.detach()
    Gpre_bar = G_bar * glu.silu_prime(cache.glu.Gpre.detach())
    N2_y = cache.glu.N2.detach()          # the norm OUTPUT, already cached
    grads["W_g"] = _rows(N2_y).T @ _rows(Gpre_bar)
    grads["W_u"] = _rows(N2_y).T @ _rows(U_bar)
    grads["W_d"] = _rows(cache.glu.G.detach() * cache.glu.U.detach()).T \
        @ _rows(F_bar)
    N2_bar_g = Gpre_bar @ p.W_g.detach().transpose(-2, -1)   # named summand
    N2_bar_u = U_bar @ p.W_u.detach().transpose(-2, -1)     # named summand
    if mutation == "drop_u_path":
        N2_bar_u = torch.zeros_like(N2_bar_u)
    N2_bar = N2_bar_g + N2_bar_u

    # --- 5. second RMSNorm reverse ---------------------------------------
    R_ffn_bar, grads["g2"] = rn.rmsnorm_backward(
        N2_bar, _detach_norm_cache(cache.N2))

    # --- complete merge R_ell --------------------------------------------
    R_bar = R_bar_identity + R_ffn_bar

    # --- 6. first residual: R = X + Y (merge X_ell) ---------------------
    X_identity = torch.zeros_like(R_bar) \
        if mutation == "drop_first_residual_identity" else R_bar
    Y_bar = R_bar

    # --- 7. output projection -------------------------------------------
    grads["W_O"] = _rows(cache.O.detach()).T @ _rows(Y_bar)
    O_bar = Y_bar @ p.W_O.detach().transpose(-2, -1)

    # --- 8. per-head attention reverse, then the concat adjoint ---------
    scale = math.sqrt(d_h)          # contract C2: d_h, never d
    A_bar_head, V_bar_head, S_bar_head = [], [], []
    for h in range(H):
        O_h_bar = O_bar[:, :, h * d_h:(h + 1) * d_h]
        A_h = cache.A.A[h]
        V_h = cache.A.V[h]
        A_bar_h = O_h_bar @ V_h.detach().transpose(-2, -1)
        V_bar_h = A_h.detach().transpose(-2, -1) @ O_h_bar
        if mutation == "sum_instead_of_concat":
            # the classic bug: sum the head adjoints instead of reshaping
            O_h_bar = O_bar[:, :, :d_h] + O_bar[:, :, d_h:2 * d_h]
            A_bar_h = O_h_bar @ V_h.detach().transpose(-2, -1)
            V_bar_h = A_h.detach().transpose(-2, -1) @ O_h_bar
        if mutation == "drop_head_k_v_path" and h == 0:
            V_bar_h = torch.zeros_like(V_bar_h)
        S_bar_h = att.softmax_backward_rows(A_h.detach(), A_bar_h)
        A_bar_head.append(A_bar_h)
        V_bar_head.append(V_bar_h)
        S_bar_head.append(S_bar_h)

    A_bar = torch.stack(A_bar_head, dim=0)
    S_bar = torch.stack(S_bar_head, dim=0)

    # Q_bar needs K and K_bar needs Q, the rotated (post-RoPE) versions.
    Q_bar = torch.stack([S_bar_head[h] @ cache.K[h].detach() / scale
                         for h in range(H)], dim=0)
    K_bar = torch.stack([S_bar_head[h].transpose(-2, -1) @ cache.Q[h].detach()
                         / scale for h in range(H)], dim=0)

    # --- 9. RoPE reverse (row convention: y -> y R_t) --------------------
    angles = cache.angles.detach()
    Qt_bar = torch.stack([rope_mod.rope_backward_rows(Q_bar[h], angles)
                          for h in range(H)], dim=0)
    Kt_bar = torch.stack([rope_mod.rope_backward_rows(K_bar[h], angles)
                          for h in range(H)], dim=0)

    # --- the concat adjoint, for Q, K and V alike ------------------------
    # reverse of split_heads is a reshape of the adjoint, NOT a sum
    def concat_back(z):
        return att.merge_heads(z)

    Qt_dense = concat_back(Qt_bar)
    Kt_dense = concat_back(Kt_bar)
    Vt_dense = concat_back(torch.stack(V_bar_head, dim=0))

    # --- 10. dense projection gradients and the N^(1) three-way merge ---
    # The norm OUTPUT is rebuilt from the cache; under the v1.2.1 promotion
    # the cache's u/r live in fp32 for half-precision inputs, so cast the
    # rebuilt output to the adjoint stream's dtype before the GEMMs (the
    # mixed-dtype matmul would otherwise raise, and silently downcasting the
    # adjoints instead would lose the promotion's whole point).
    N1_y = (cache.N1.u.detach() * cache.N1.g.detach()).to(Qt_dense.dtype)
    grads["W_Q"] = _rows(N1_y).T @ _rows(Qt_dense)
    grads["W_K"] = _rows(N1_y).T @ _rows(Kt_dense)
    grads["W_V"] = _rows(N1_y).T @ _rows(Vt_dense)

    N1_bar_q = Qt_dense @ p.W_Q.detach().transpose(-2, -1)   # named summand
    N1_bar_k = Kt_dense @ p.W_K.detach().transpose(-2, -1)   # named summand
    N1_bar_v = Vt_dense @ p.W_V.detach().transpose(-2, -1)   # named summand
    if mutation == "drop_qk_path":
        N1_bar_q = torch.zeros_like(N1_bar_q)
        N1_bar_k = torch.zeros_like(N1_bar_k)
    if mutation == "drop_v_merge":
        N1_bar_v = torch.zeros_like(N1_bar_v)
    N1_bar = N1_bar_q + N1_bar_k + N1_bar_v

    # --- 11. attention-side RMSNorm reverse ------------------------------
    X_attn_bar, grads["g1"] = rn.rmsnorm_backward(
        N1_bar, _detach_norm_cache(cache.N1))

    # --- complete merge X_ell --------------------------------------------
    X_bar = X_identity + X_attn_bar

    trace = ReverseTrace(
        Xp_bar=Xp_bar, F_bar=F_bar, R_bar=R_bar,
        N2_bar_g=N2_bar_g, N2_bar_u=N2_bar_u, Y_bar=Y_bar, O_bar=O_bar,
        S_bar=S_bar, A_bar=A_bar, Q_bar=Q_bar, K_bar=K_bar, V_bar=Vt_dense,
        Qt_bar=Qt_dense, Kt_bar=Kt_dense, Vt_bar=Vt_dense,
        N1_bar_q=N1_bar_q, N1_bar_k=N1_bar_k, N1_bar_v=N1_bar_v, N1_bar=N1_bar,
        X_attn_bar=X_attn_bar, X_bar=X_bar, Z_bar=None, Nf_bar=None,
    )
    return grads, trace


def model_backward(Z_bar: torch.Tensor, X_bar: torch.Tensor, E: torch.Tensor,
                   W_out: torch.Tensor, tokens: torch.Tensor, V: int
                   ) -> torch.Tensor:
    """Contract ID 1: the embedding scatter-add, and the tied variant.

    ``E_bar = E_bar_lookup + W_out_bar^T`` when ``W_out = E^T``; the transpose is
    required to make the shapes agree (contract C6). Not applied here, because
    tying is an explicit fork and applying it silently would be exactly the kind
    of quiet change the variant policy forbids.
    """
    return loss_mod.embedding_scatter_add(tokens, X_bar, V)
