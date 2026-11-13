"""Two-head (H=2) verification, now a thin client of `tgd`.

Up to v1.2 this file carried its own `rmsnorm_forward`, `rope_*`, `silu_prime`,
`forward` and `manual_backward` -- a third independent copy of the forward graph.
It is now a harness over the shared implementation, so the H=2 case tests the
same code the book differentiates.

What makes this case worth its own file: with H=1 the head split and concat are
identity reshuffles, so the multi-head bookkeeping is *unobservable*. These
mutations are specific to that bookkeeping and cannot be expressed at H=1:

  * ``drop_head_k_v_path``  -- delete one head's value-path contribution
  * ``sum_instead_of_concat`` -- reshape the concat adjoint as a sum

The second is the classic multi-head backprop bug, and it is invisible to a
one-head test by construction.

Usage:  python ch06_two_head_verify.py
Marker: TWO_HEAD_CHECKS_PASSED
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

import math

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tgd import attention as att
from tgd import graph as graph_mod
from tgd import loss as loss_mod
from tgd import reverse as rev
from tgd import rmsnorm as rn
from tgd.dims import CH8_TWO_HEAD

DTYPE = torch.float64
EPSILONS = (1e-4, 3e-5, 1e-5)
NAMES = ["g1", "g2", "gf", "W_Q", "W_K", "W_V", "W_O", "W_g", "W_u", "W_d",
         "W_out", "E"]


def make_params(dims, seed: int = 11) -> dict:
    g = torch.Generator().manual_seed(seed)

    def p(*shape, scale=0.2):
        return (torch.randn(*shape, generator=g, dtype=DTYPE)
                * scale).requires_grad_(True)

    return {
        "E": p(dims.V, dims.d),
        "g1": torch.ones(dims.d, dtype=DTYPE, requires_grad=True),
        "W_Q": p(dims.d, dims.d), "W_K": p(dims.d, dims.d),
        "W_V": p(dims.d, dims.d), "W_O": p(dims.d, dims.d),
        "g2": torch.ones(dims.d, dtype=DTYPE, requires_grad=True),
        "W_g": p(dims.d, dims.d_ff), "W_u": p(dims.d, dims.d_ff),
        "W_d": p(dims.d_ff, dims.d),
        "gf": torch.ones(dims.d, dtype=DTYPE, requires_grad=True),
        "W_out": p(dims.d, dims.V),
    }


def block_of(p: dict) -> graph_mod.BlockParams:
    return graph_mod.BlockParams(
        g1=p["g1"], W_Q=p["W_Q"], W_K=p["W_K"], W_V=p["W_V"], W_O=p["W_O"],
        g2=p["g2"], W_g=p["W_g"], W_u=p["W_u"], W_d=p["W_d"])


def run(tokens, targets, mutation=None):
    dims = CH8_TWO_HEAD
    params = make_params(dims)
    Xp, cache = graph_mod.block_forward(params["E"][tokens], block_of(params),
                                        dims)
    Nf_y, Nf_c = rn.rmsnorm_forward(Xp, params["gf"], dims.eps)
    Z = Nf_y @ params["W_out"]
    loss = loss_mod.masked_mean_ce(Z, targets)
    W_out_bar, Nf_bar, _ = rev.output_backward(Nf_y, Z, params["W_out"], targets)
    Xp_bar, gf_bar = rn.rmsnorm_backward(Nf_bar, Nf_c)
    grads, trace = rev.block_backward(Xp_bar, cache, block_of(params), dims,
                                      tokens, mutation=mutation)
    grads["W_out"] = W_out_bar
    grads["gf"] = gf_bar
    grads["E"] = rev.model_backward(None, trace.X_bar, params["E"],
                                    params["W_out"], tokens, dims.V)
    return params, grads, trace, loss, cache


def fwd_loss(params, tokens, targets) -> float:
    dims = CH8_TWO_HEAD
    Xp, _ = graph_mod.block_forward(params["E"][tokens], block_of(params), dims)
    Nf_y, _ = rn.rmsnorm_forward(Xp, params["gf"], dims.eps)
    return loss_mod.masked_mean_ce(Nf_y @ params["W_out"], targets).item()


def rel_error(a, b) -> float:
    a, b = a.detach().double(), b.detach().double()
    return float((a - b).norm() / (a.norm() + b.norm() + 1e-12))


def main() -> None:
    dims = CH8_TWO_HEAD
    print(f"torch={torch.__version__}")
    print(f"config: B={dims.B} T={dims.T} V={dims.V} d={dims.d} H={dims.H} "
          f"d_h={dims.d_h} d_ff={dims.d_ff} L={dims.n_layers}")
    assert dims.H == 2, "this file is the two-head check; H=1 is ch06_verify.py"
    assert dims.d_h == 4 and dims.d == dims.H * dims.d_h

    # contract C2: the two scale factors are different here, and a one-head test
    # could never have detected a swap
    assert abs(math.sqrt(dims.d_h) - math.sqrt(dims.d)) > 1e-9
    print(f"scale factor 1/sqrt(d_h)={1 / math.sqrt(dims.d_h):.6f} != "
          f"1/sqrt(d)={1 / math.sqrt(dims.d):.6f}  (C2)")

    tokens = torch.tensor([[0, 2, 2, 4]], dtype=torch.long)
    targets = torch.tensor([[2, 2, 4, 1]], dtype=torch.long)
    assert int((tokens == 2).sum()) == 2, "want a repeated token id"

    params, grads, trace, loss, cache = run(tokens, targets)
    loss.backward()
    print(f"loss={loss.item():.12f}")

    print("\nmanual vs autograd relative errors")
    worst = 0.0
    for n in NAMES:
        e = rel_error(grads[n], params[n].grad)
        worst = max(worst, e)
        print(f"  {n:5s} {e:.3e}")
    assert worst < 1e-10, f"two-head manual/autograd mismatch: {worst}"

    print("\nper-head masked-softmax invariants")
    max_rows = 0.0
    for h in range(dims.H):
        A0 = cache.A.A[h].detach()[0]
        S0 = cache.A.S[h].detach()[0]
        S_b0 = trace.S_bar[h].detach()[0]
        r = float(S_b0.sum(-1).abs().max())
        max_rows = max(max_rows, r)
        masked = ~torch.isfinite(S0)
        assert bool((A0[masked] == 0).all()), f"head {h}: masked prob must be 0"
        assert bool((S_b0[masked] == 0).all()), f"head {h}: masked s_bar must be 0"
        print(f"  head {h}: max |row sum of bar s| = {r:.3e}; masked entries "
              f"exactly zero")
    assert max_rows < 1e-12, f"row sums too large: {max_rows}"

    print("\nmutation controls (all are multi-head specific)")
    for mut in ("drop_head_k_v_path", "sum_instead_of_concat", "drop_v_merge",
                "drop_first_residual_identity", "drop_second_residual_identity"):
        _, m, _, _, _ = run(tokens, targets, mutation=mut)
        err = max(rel_error(m[n], grads[n]) for n in NAMES)
        print(f"  {mut:32s} worst={err:.3e}")
        assert err > 1e-6, f"mutation {mut} was NOT detected"

    print("\ncentral finite differences (all branch families, epsilon sweep)")
    idx = {"E": (2, 3), "g1": (0,), "g2": (3,), "gf": (2,),
           "W_Q": (0, 0), "W_K": (1, 2), "W_V": (2, 0), "W_O": (0, 3),
           "W_g": (1, 1), "W_u": (2, 3), "W_d": (0, 2), "W_out": (1, 0)}
    worst_fd = 0.0
    for name in NAMES:
        i = idx[name]
        analytic = grads[name][i].item()
        best = float("inf")
        for eps in EPSILONS:
            orig = params[name][i].item()
            with torch.no_grad():
                params[name][i] = orig + eps
            lp = fwd_loss(params, tokens, targets)
            with torch.no_grad():
                params[name][i] = orig - eps
            lm = fwd_loss(params, tokens, targets)
            with torch.no_grad():
                params[name][i] = orig
            fd = (lp - lm) / (2 * eps)
            best = min(best, abs(fd - analytic) / (abs(fd) + abs(analytic)
                                                   + 1e-12))
        worst_fd = max(worst_fd, best)
        print(f"  {name:5s}{str(i):9s} analytic={analytic:+.8e} "
              f"best_rel={best:.3e}")
    assert worst_fd < 5e-6, f"two-head FD error too large: {worst_fd}"

    print(f"\nworst={worst:.3e}  worst_fd={worst_fd:.3e}")
    print("TWO_HEAD_CHECKS_PASSED")


if __name__ == "__main__":
    main()
