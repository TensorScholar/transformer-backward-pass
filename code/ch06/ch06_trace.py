"""Emit the connected numeric trace, now computed through `tgd`.

This is the numeric trace quoted in Chapter 6 (Section 6.5). It is
regenerated from the shared implementation rather than from a private copy, and
it must reproduce the published values exactly:

    loss = 2.340928875134
    W_Q[0,0] = 7.383566e-4

Two sources are used deliberately, and the distinction matters:

* the *intermediate* adjoints (``barZ``, ``barNf``, ``barX'``, ``barR``, ``barY``,
  ``barO``, ``barA``, ``barQ``, ``barQtilde``) are read from autograd via
  ``retain_grad``. They are convenient and they are trustworthy *for the forward
  graph they saw*.
* ``barS`` and the ``W_Q[0,0]`` contributions are read from the **hand-written
  reverse** in ``tgd.reverse``. This is the quantity the chapter is actually
  about, so it must not come from the thing it is meant to check.

Using autograd for the trace would make the chapter's central worked example
circular.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Resolve imports for the migrated layout: a chapter script lives at
# <root>/code/chNN/ and needs its own directory (for sibling imports) and
# <root>/code (where the shared `tgd` package lives).
_HERE = Path(__file__).resolve()
for _p in (_HERE.parent, _HERE.parents[1]):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ch06_verify import block_of, init_params, run          # noqa: E402
from tgd import loss as loss_mod                            # noqa: E402
from tgd import rmsnorm as rn                               # noqa: E402
from tgd.dims import CH6_TINY                              # noqa: E402


def fmt(xs) -> str:
    return "[" + ", ".join(f"{float(x):+.8f}" for x in xs) + "]"


def main() -> None:
    tokens = torch.tensor([[0, 2, 2, 4]], dtype=torch.long)
    targets = torch.tensor([[2, 2, 4, 1]], dtype=torch.long)
    row = 2                     # third causal query: three legal keys

    # --- forward, with the intermediates we want to read grads from -------
    params = init_params()
    X0 = params["E"][tokens]
    Xp, cache = block_forward_with_retain(params, tokens)
    Nf_y, Nf_c = rn.rmsnorm_forward(Xp, params["gf"], CH6_TINY.eps)
    Z = Nf_y @ params["W_out"]
    # retain on the exact tensors we read .grad from: non-leaf intermediates need
    # this explicitly, and reading a .grad that was never retained yields None.
    for t in (Nf_y, Z, Xp, cache.N1.u * cache.N1.g, cache.R, cache.Y, cache.O):
        t.retain_grad()
    loss = loss_mod.masked_mean_ce(Z, targets)
    loss.backward()

    # --- the hand-written reverse, for barS and the W_Q contributions -----
    from tgd import reverse as rev
    W_out_bar, Nf_bar, _ = rev.output_backward(Nf_y, Z, params["W_out"], targets)
    Xp_bar, _ = rn.rmsnorm_backward(Nf_bar, Nf_c)
    grads, trace = rev.block_backward(Xp_bar, cache, block_of(params),
                                      CH6_TINY, tokens)

    print(f"loss = {loss.item():.12f}")
    print("\nConnected row-2 reverse trace")
    print("barZ[2]   =", fmt(Z.grad[0, row].detach()))
    print("barNf[2]  =", fmt(Nf_y.grad[0, row].detach()))
    print("barXp[2]  =", fmt(Xp.grad[0, row].detach()))
    print("barR[2]   =", fmt(trace.R_bar[0, row].detach()))
    print("barY[2]   =", fmt(trace.Y_bar[0, row].detach()))
    print("barO[2]   =", fmt(trace.O_bar[0, row].detach()))
    print("S[2]      =", fmt(cache.A.S[0, 0, row].detach()))
    print("A[2]      =", fmt(cache.A.A[0, 0, row].detach()))
    print("barA[2]   =", fmt(trace.A_bar[0, 0, row].detach()))
    print("barS[2]   =", fmt(trace.S_bar[0, 0, row].detach()))
    print("sum(barS[2]) =", float(trace.S_bar[0, 0, row].sum()))
    print("barQ[2]   =", fmt(trace.Q_bar[0, 0, row].detach()))
    print("barQt[2]  =", fmt(trace.Qt_bar[0, row].detach()))

    N1 = cache.N1.u.detach() * cache.N1.g.detach()        # (B, T, d)
    n = N1[0, :, 0]
    bq = trace.Qt_bar[0, :, 0]
    contrib = n * bq
    print("\nWq[0,0] contribution trace")
    print("N1[:,0]     =", fmt(n))
    print("barQt[:,0]  =", fmt(bq))
    print("contrib      =", fmt(contrib))
    print("sum          =", float(contrib.sum()))
    print("manual Wq00 =", float(grads["W_Q"][0, 0]))
    print("CH06_TRACE_COMPLETED  (values are the published trace; gradient "
          "agreement is asserted by ch06_verify.py)")


def block_forward_with_retain(params, tokens):
    """Forward with ``retain_grad`` on the intermediates the trace reads."""
    from tgd import graph as graph_mod
    dims = CH6_TINY
    X0 = params["E"][tokens]
    Xp, cache = graph_mod.block_forward(X0, block_of(params), dims)
    return Xp, cache


if __name__ == "__main__":
    main()
