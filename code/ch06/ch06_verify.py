"""Chapter 6 verification harness — now a thin client of `tgd`.

Up to v1.2 this file carried its own copy of the forward graph: its own
`rmsnorm_forward`, `rope_*`, `silu_prime`, `forward` and `manual_backward`. That
duplication was one of three independent forward graphs in the project, and the
three disagreed (the missing final RMSNorm was the root of the entire P0 defect
class). The implementation now lives in `tgd/` and this file only checks it.

**The published numeric trace is unchanged.** `init_params` here preserves the
historical RNG draw order (E, W_Q, W_K, W_V, W_O, W_g, W_u, W_d, W_out, with
gains as ones and no draw), so `ch06_trace.py` still prints the same values
quoted in Chapter 6.

What is asserted:
  * every trainable tensor's manual gradient against autograd, float64;
  * the masked-softmax row-sum invariant;
  * central finite differences across all twelve tensors with an epsilon sweep;
  * three deletion mutations, each of which must grow the error.

Usage:  python ch06_verify.py
Marker: ALL_CHECKS_PASSED
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


from tgd import graph as graph_mod
from tgd import loss as loss_mod
from tgd import reverse as rev
from tgd import rmsnorm as rn
from tgd.dims import CH6_TINY

DTYPE = torch.float64
EPSILONS = (1e-4, 3e-5, 1e-5)

NAMES = ["E", "g1", "W_Q", "W_K", "W_V", "W_O", "g2", "W_g", "W_u", "W_d",
         "gf", "W_out"]


def init_params(seed: int = 7) -> dict:
    """Historical draw order, preserved so the published trace is unchanged."""
    dims = CH6_TINY
    g = torch.Generator().manual_seed(seed)

    def p(*shape, scale=0.2):
        return (torch.randn(*shape, generator=g, dtype=DTYPE) * scale
                ).requires_grad_(True)

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


def block_of(params: dict) -> graph_mod.BlockParams:
    return graph_mod.BlockParams(
        g1=params["g1"], W_Q=params["W_Q"], W_K=params["W_K"],
        W_V=params["W_V"], W_O=params["W_O"], g2=params["g2"],
        W_g=params["W_g"], W_u=params["W_u"], W_d=params["W_d"])


def run(tokens: torch.Tensor, targets: torch.Tensor, mutation: str | None = None):
    """Full canonical forward + hand-written reverse, via `tgd` only."""
    dims = CH6_TINY
    params = init_params()
    X0 = params["E"][tokens]
    Xp, cache = graph_mod.block_forward(X0, block_of(params), dims)
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
    return params, grads, trace, loss, cache, Z, Nf_y


def rel_error(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.detach().double(), b.detach().double()
    return float((a - b).norm() / (a.norm() + b.norm() + 1e-12))


def finite_difference(params, tokens, targets, name, idx, eps) -> float:
    orig = params[name][idx].item()
    with torch.no_grad():
        params[name][idx] = orig + eps
    lp, _, _ = forward_loss(params, tokens, targets)
    with torch.no_grad():
        params[name][idx] = orig - eps
    lm, _, _ = forward_loss(params, tokens, targets)
    with torch.no_grad():
        params[name][idx] = orig
    return (lp - lm) / (2 * eps)


def forward_loss(params, tokens, targets):
    dims = CH6_TINY
    Xp, _ = graph_mod.block_forward(params["E"][tokens], block_of(params), dims)
    Nf_y, _ = rn.rmsnorm_forward(Xp, params["gf"], dims.eps)
    Z = Nf_y @ params["W_out"]
    return loss_mod.masked_mean_ce(Z, targets).item(), Xp, Z


def main() -> None:
    print(f"torch={torch.__version__}")
    # tgd is written for (B, T, ...) tensors, contract section 1. B = 1 here, so
    # every quantity equals the 2-D (T, ...) form the earlier revision used, and
    # the published trace is unchanged. Rows are indexed with [:, t].
    tokens = torch.tensor([[0, 2, 2, 4]], dtype=torch.long)
    targets = torch.tensor([[2, 2, 4, 1]], dtype=torch.long)
    assert int((tokens == 2).sum()) == 2, "want a repeated token id"

    params, grads, trace, loss, cache, Z, Nf_y = run(tokens, targets)
    loss.backward()

    print(f"loss={loss.item():.12f}")
    print("manual vs autograd relative errors")
    worst = 0.0
    for n in NAMES:
        e = rel_error(grads[n], params[n].grad)
        worst = max(worst, e)
        print(f"  {n:5s} {e:.3e}")
    assert worst < 1e-10, f"manual/autograd mismatch: {worst}"

    # --- masked-softmax row-sum invariant, per head -----------------------
    # Uses the adjoints the reverse already produced, so this checks the real
    # computation rather than a separately reconstructed one.
    max_rows = 0.0
    for h in range(CH6_TINY.H):
        S_b = trace.S_bar[h].detach()            # (B, T, T)
        A0 = cache.A.A[h].detach()[0]            # (T, T)  probabilities
        S0 = cache.A.S[h].detach()[0]            # (T, T)  masked scores
        S_b0 = S_b[0]                            # (T, T)
        r = float(S_b.sum(-1).abs().max())
        max_rows = max(max_rows, r)
        masked = ~torch.isfinite(S0)              # (T, T), True where illegal
        assert bool(masked.any()), "the causal mask should mask something"
        assert bool((A0[masked] == 0).all()), "masked probability must be 0"
        assert bool((S_b0[masked] == 0).all()), "masked score adjoint must be 0"
        print(f"head {h}: max |row sum of bar s| = {r:.3e}; "
              f"masked probability and masked score adjoint exactly zero")
    assert max_rows < 1e-12, f"row sums too large: {max_rows}"

    # --- finite differences, all twelve tensors, epsilon sweep -----------
    print("finite-difference checks")
    worst_fd = 0.0
    for name in NAMES:
        idx = {
            "E": (2, 3), "g1": (2,), "g2": (3,), "gf": (0,),
            "W_Q": (0, 0), "W_K": (1, 2), "W_V": (2, 0), "W_O": (0, 3),
            "W_g": (1, 1), "W_u": (2, 3), "W_d": (0, 2), "W_out": (1, 0),
        }[name]
        analytic = grads[name][idx].item()
        best = float("inf")
        for eps in EPSILONS:
            fd = finite_difference(params, tokens, targets, name, idx, eps)
            best = min(best, abs(fd - analytic) / (abs(fd) + abs(analytic)
                                                   + 1e-12))
        worst_fd = max(worst_fd, best)
        print(f"  {name:5s}{str(idx):9s} analytic={analytic:+.8e} "
              f"best_rel={best:.3e}")
    assert worst_fd < 5e-6, f"finite-difference error too large: {worst_fd}"

    # --- deletion mutations ---------------------------------------------
    print("mutation controls (each must fail loudly)")
    for mut, expect in [("drop_v_merge", 1e-2),
                        ("drop_first_residual_identity", 1e-2),
                        ("drop_second_residual_identity", 1e-2)]:
        _, m, _, _, _, _, _ = run(tokens, targets, mutation=mut)
        err = max(rel_error(m[n], grads[n]) for n in NAMES)
        print(f"  {mut:32s} worst={err:.3e}")
        assert err > expect, f"mutation {mut} not detected: {err:.3e}"

    # --- reduced-precision adversarial conditioning (v1.2.1) --------------
    # The fp64 toy oracle above cannot see the failures that only appear at
    # real activation scales: the d*r^3 term of rmsnorm_backward overflows
    # fp16 for r > (65504/d)^(1/3) = 8.0 at d = 128, silently zeroing the
    # projection term. Checked here against an autograd oracle built on the
    # SAME promoted graph (cast -> fp32 arithmetic, working-precision loss,
    # so autograd differentiates exactly what the manual rule computes).
    print("reduced-precision adversarial conditioning (fp32 and fp16)")
    torch.manual_seed(1234)
    for dt, tol in ((torch.float32, 2e-5), (torch.float16, 5e-3)):
        d = 128
        # 100x channel outliers: columns 0,1 carry 100x the scale, which
        # drives r past the fp16 d*r^3 overflow threshold at d=128
        x = torch.randn(4, 16, d) * 0.5
        x[..., 0] *= 100.0
        x[..., 1] *= -100.0
        g = torch.ones(d)
        xw = x.to(dt).detach().clone().requires_grad_(True)
        gw = g.to(dt).detach().clone().requires_grad_(True)
        wd = torch.float32 if dt in (torch.float16, torch.bfloat16) else dt
        # autograd oracle on the promoted graph: cast is a graph node, loss
        # stays in working precision (no output downcast in the path)
        xw32 = xw.to(wd)
        ms = xw32.pow(2).mean(-1, keepdim=True)
        r32 = torch.sqrt(ms + 1e-6)
        u32 = xw32 / r32
        y32 = u32 * gw.to(wd)
        y32.sum().backward()
        # manual rule on the same inputs
        y, c = rn.rmsnorm_forward(xw.detach(), gw.detach())
        xb, gb = rn.rmsnorm_backward(torch.ones_like(y), c)
        e_x = float((xb.to(wd) - xw.grad.to(wd)).norm()
                    / (xw.grad.to(wd).norm() + 1e-30))
        e_g = float((gb.to(wd) - gw.grad.to(wd)).norm()
                    / (gw.grad.to(wd).norm() + 1e-30))
        print(f"  rmsnorm {str(dt):16s} outliers: dX {e_x:.3e}  dG {e_g:.3e}")
        assert e_x < tol and e_g < tol, (
            f"rmsnorm {dt} adversarial conditioning failed: dX {e_x:.3e}, "
            f"dG {e_g:.3e}"
        )
        # ill-conditioned input: per-channel scales spanning 1e-3..1e3
        scales = torch.logspace(-3, 3, d)
        x2 = (torch.randn(4, 16, d) * scales).to(dt)
        x2w = x2.detach().clone().requires_grad_(True)
        g2w = torch.ones(d, dtype=dt).detach().clone().requires_grad_(True)
        x2w32 = x2w.to(wd)
        ms2 = x2w32.pow(2).mean(-1, keepdim=True)
        u2 = x2w32 / torch.sqrt(ms2 + 1e-6)
        (u2 * g2w.to(wd)).sum().backward()
        y2, c2 = rn.rmsnorm_forward(x2.detach(), g2w.detach())
        xb2, gb2 = rn.rmsnorm_backward(torch.ones_like(y2), c2)
        e_x2 = float((xb2.to(wd) - x2w.grad.to(wd)).norm()
                     / (x2w.grad.to(wd).norm() + 1e-30))
        print(f"  rmsnorm {str(dt):16s} cond~1e6: dX {e_x2:.3e}")
        assert e_x2 < tol, f"rmsnorm {dt} cond=1e6 failed: dX {e_x2:.3e}"

    # --- chunked fused cross-entropy VJP: bitwise vs dense (v1.2.1) -------
    print("chunked cross-entropy VJP (ce_backward_chunked vs dense)")
    torch.manual_seed(1235)
    B, T, d_ce, V = 2, 32, 64, 97
    Nf = torch.randn(B, T, d_ce, dtype=torch.float64) * 0.3
    W_out = torch.randn(d_ce, V, dtype=torch.float64) * 0.3
    y_ce = torch.randint(0, V, (B, T))
    Z_ce = Nf @ W_out
    Z_bar, n_sup = loss_mod.ce_backward(Z_ce, y_ce)
    dense_Nf_bar = Z_bar @ W_out.transpose(-2, -1)
    dense_W_bar = Nf.reshape(-1, d_ce).T @ Z_bar.reshape(-1, V)
    for chunk in (7, 8, 16, 64):
        nb, wb = loss_mod.ce_backward_chunked(Nf, W_out, y_ce, n_sup,
                                              chunk=chunk)
        if B * T % chunk == 0:
            # row-aligned chunks run the same GEMM M-blocks as the dense
            # path, so bitwise identity is achievable and required
            assert torch.equal(nb, dense_Nf_bar), (
                f"chunked Nf_bar differs from dense at chunk={chunk}"
            )
        else:
            # a misaligned tail block runs a different GEMM M and may differ
            # in the last ULP; require fp64-tight agreement instead
            drift = float((nb - dense_Nf_bar).norm()
                          / (dense_Nf_bar.norm() + 1e-30))
            assert drift < 1e-12, (
                f"chunked Nf_bar drifts from dense at chunk={chunk}: "
                f"{drift:.3e}"
            )
        w_rel = float((wb - dense_W_bar).norm()
                      / (dense_W_bar.norm() + 1e-30))
        assert w_rel < 1e-12, (
            f"chunked W_out_bar drifts from dense at chunk={chunk}: {w_rel:.3e}"
        )
    print(f"  Nf_bar bitwise-identical at aligned chunks (8/16/64); "
          f"misaligned chunk within fp64 GEMM-blocking rounding; W_out_bar "
          f"within {w_rel:.1e}; dense peak = 4x [B,T,V]")

    print(f"worst={worst:.3e}  worst_fd={worst_fd:.3e}")
    print("ALL_CHECKS_PASSED")


if __name__ == "__main__":
    main()
