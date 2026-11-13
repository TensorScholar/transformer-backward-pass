"""RMSNorm — contract §2 IDs 3, 15, 24 and conventions C3, C4.

    r = sqrt( mean(x^2) + eps )        eps INSIDE the root, AFTER the mean
    u = x / r
    y = u * g

Reverse, for one position with width d, writing ``r`` for the same denominator:

    u_bar     = y_bar * g
    r_bar     = sum( u_bar * x ) / r
    x_bar     = u_bar / r  -  x * r_bar / r^2
              = u_bar / r  -  x * (x . u_bar) / (d * r^3)

    g_bar     = sum over TOKENS of ( y_bar * u )

The gain gradient sums over the **token** axis, not the feature axis. A norm
gain therefore has a different reduction axis from every other parameter, which
is a routine source of silent bugs, and it is called out as contract C4.

Verified numerically by ``ch04_graph_ledger.py`` against autograd.
"""
from __future__ import annotations

from typing import NamedTuple

import torch

#: Reduced-precision storage formats whose *arithmetic* must be promoted to
#: ``float32`` before any reduction or power is taken. Half-precision inputs are
#: legitimate (training at fp16/bf16 activations is the production case), but
#: two intermediate quantities of this rule overflow those formats outright:
#: ``x^2`` in the forward when a channel spikes (``300^2 = 9e4 > 65504``), and
#: ``d * r^3`` in the reverse, which exceeds 65504 for ``d = 128`` as soon as
#: ``r > (65504/128)^(1/3) = 8.0`` -- a value any real activation stream with
#: scale ~10 produces at every position. The silent failure is worse than a
#: NaN: the projection term ``x * dot / (d * r^3)`` underflows to exactly 0 and
#: the gradient is quietly wrong by the full size of that term.
_LOW_PRECISION = (torch.float16, torch.bfloat16)


def _work_dtype(x: torch.Tensor) -> torch.dtype:
    """``float64`` stays ``float64``; fp16/bf16 is promoted to ``float32``."""
    if x.dtype in _LOW_PRECISION:
        return torch.float32
    return x.dtype


class RMSNormCache(NamedTuple):
    """Everything the reverse rule needs, and nothing it does not."""

    x: torch.Tensor       # (..., T, d)
    r: torch.Tensor       # (..., T, 1)  the shared denominator
    u: torch.Tensor       # (..., T, d)  x / r
    g: torch.Tensor       # (d,)


def rmsnorm_forward(x: torch.Tensor, g: torch.Tensor,
                   eps: float = 1e-6) -> tuple[torch.Tensor, RMSNormCache]:
    """Contract C3. ``g`` broadcasts over the last axis only.

    The mean of squares is evaluated in ``float32`` whenever ``x`` is stored in
    fp16/bf16, so a channel spike (``x = 300`` in fp16) does not take ``x^2``
    to ``inf`` before the reduction. The cache stores the promoted-precision
    ``r`` and ``u``; the emitted ``y`` is cast back to the input dtype.
    """
    if g.shape != x.shape[-1:]:
        raise ValueError(
            f"gain must have shape ({x.shape[-1]},), got {tuple(g.shape)}"
        )
    wd = _work_dtype(x)
    xw = x.to(wd)
    ms = xw.pow(2).mean(dim=-1, keepdim=True)
    r = torch.sqrt(ms + eps)
    u = xw / r
    y = (u * g.to(wd)).to(x.dtype)
    return y, RMSNormCache(x=x, r=r, u=u, g=g)


def rmsnorm_backward(bar_y: torch.Tensor,
                     cache: RMSNormCache) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(x_bar, g_bar)``. ``g_bar`` sums over every non-feature axis.

    All internal arithmetic runs in ``float32`` (or ``float64``, matching a
    ``float64`` input); ``x_bar`` and ``g_bar`` are cast back to the input and
    gain dtypes at return. This keeps ``d * r^3`` inside the format's range:
    at ``d = 128`` the fp16 evaluation overflows for ``r > 8.0`` and silently
    zeroes the projection term.
    """
    x, r, u, g = cache.x, cache.r, cache.u, cache.g
    d = x.shape[-1]
    wd = _work_dtype(x)

    bar_y_w = bar_y.to(wd)
    xw, rw, uw, gw = x.to(wd), r.to(wd), u.to(wd), g.to(wd)
    u_bar = bar_y_w * gw
    # sum_i x_i * u_bar_i, kept as a trailing singleton for broadcasting
    dot = (xw * u_bar).sum(dim=-1, keepdim=True)     # (..., T, 1)
    x_bar = (u_bar / rw - xw * dot / (d * rw.pow(3))).to(bar_y.dtype)

    # contract C4: the gain gradient reduces over tokens (and batch), not features
    g_bar = (bar_y_w * uw).sum(dim=tuple(range(bar_y_w.dim() - 1))).to(g.dtype)

    return x_bar, g_bar
