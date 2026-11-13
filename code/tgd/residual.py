"""The two residual additions and invariant I3.

    R_ell     = X_ell + Y_ell          first residual
    X_{ell+1} = R_ell + F_ell          second residual

Invariant I3: the contribution of the residual-stream argument to the adjoint of
a sum is **exactly** the upstream adjoint. There is no Jacobian, no factor, no
reshape:

    contribution of R_ell to Xp_bar  ==  Xp_bar

This is the most commonly dropped contribution in hand-derived backward passes,
because nothing about it is interesting in isolation — the bug is invisible in
the forward, invisible in the loss, and invisible to a reader skimming for the
interesting parts. It is caught only by a deletion mutation (contract A.6),
which is why both identity-path mutations are mandatory in every harness.

Reverse, with each summand named as contract A.3 requires:

    Xp_bar  ->  R_bar identity path :  R_bar += Xp_bar
    Xp_bar  ->  F_bar               :  F_bar = Xp_bar
    R_bar   ->  X_ell identity path :  X_bar += R_bar
    R_bar   ->  Y_ell (attention)   :  Y_bar = R_bar
"""
from __future__ import annotations

import torch


def residual_first(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """``R = X + Y``. Contract §2 IDs 13, 14."""
    return X + Y


def residual_second(R: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    """``Xp = R + F``. Contract §2 IDs 22, 23."""
    return R + F


def residual_second_backward(Xp_bar: torch.Tensor, R_ffn_bar: torch.Tensor, *,
                             drop_identity: bool = False
                             ) -> tuple[torch.Tensor, torch.Tensor]:
    """Reverse of ``Xp = R + F`` (contract ID 23). Returns ``(R_bar, F_bar)``.

    ``R_bar`` is merge ``R_ell`` of contract §3: the identity path carries
    ``Xp_bar`` unchanged (invariant I3), and the feed-forward branch contributes
    ``R_ffn_bar`` after the second RMSNorm has been reversed.

    ``drop_identity`` exists only so a mutation test can prove the suite is
    sensitive to the identity path. It must never be set outside a test.
    """
    if Xp_bar.shape != R_ffn_bar.shape:
        raise ValueError(
            f"shape mismatch: Xp_bar {tuple(Xp_bar.shape)} vs "
            f"R_ffn_bar {tuple(R_ffn_bar.shape)}"
        )
    F_bar = Xp_bar
    identity = torch.zeros_like(Xp_bar) if drop_identity else Xp_bar
    return identity + R_ffn_bar, F_bar


def residual_first_backward(R_bar: torch.Tensor, X_attn_bar: torch.Tensor, *,
                            drop_identity: bool = False
                            ) -> tuple[torch.Tensor, torch.Tensor]:
    """Reverse of ``R = X + Y`` (contract ID 14). Returns ``(X_bar, Y_bar)``.

    ``X_bar`` is merge ``X_ell`` of contract §3: the identity path carries
    ``R_bar`` unchanged (invariant I3), plus the attention branch.
    ``Y_bar`` is the attention branch's immediate adjoint and equals ``R_bar``.
    """
    if R_bar.shape != X_attn_bar.shape:
        raise ValueError(
            f"shape mismatch: R_bar {tuple(R_bar.shape)} vs "
            f"X_attn_bar {tuple(X_attn_bar.shape)}"
        )
    identity = torch.zeros_like(R_bar) if drop_identity else R_bar
    return identity + X_attn_bar, R_bar


def residual_backward(Xp_bar: torch.Tensor, X_attn_bar: torch.Tensor,
                      R_ffn_bar: torch.Tensor, *,
                      drop_first_identity: bool = False,
                      drop_second_identity: bool = False
                      ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Chain both residual reverses. Returns ``(X_bar, R_bar, Y_bar)``.

    This is the convenience form used when both transformed branches are already
    available. The computation itself proceeds one step at a time, because
    ``R_bar`` is needed to reach the attention branch; use
    :func:`residual_second_backward` and :func:`residual_first_backward` when
    following the actual order, as ``tgd.reverse`` does.
    """
    R_bar, _ = residual_second_backward(
        Xp_bar, R_ffn_bar, drop_identity=drop_second_identity)
    X_bar, Y_bar = residual_first_backward(
        R_bar, X_attn_bar, drop_identity=drop_first_identity)
    return X_bar, R_bar, Y_bar
