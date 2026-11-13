"""Verification of the shared reverse in `tgd.reverse` against autograd.

Chapter 4 reported 23 contract IDs as forward-only. This script checks that the
reverse now written in `tgd/reverse.py` is correct, at H=1 and at H=2, and that
all five deletion mutations are detectable, closing contract ID completeness
across all canonical backward operations.

Nothing here is chapter-specific: it tests the library, not the prose. The
chapter that *explains* this reverse is Chapter 6, and its own harness
(`ch06_verify.py`, re-pointed at `tgd`) must agree with this.

Usage:  python ch06_reverse_verify.py
Marker: CH06_REVERSE_VERIFY_PASSED
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


from tgd import graph as graph_mod
from tgd import loss as loss_mod
from tgd import reverse as rev
from tgd import rope as rope_mod
from tgd.dims import CH6_TINY, CH8_TWO_HEAD, Dims

DTYPE = torch.float64
FAILURES: list[str] = []


def check(cond: bool, msg: str) -> None:
    if cond:
        print(f"  ok    {msg}")
    else:
        print(f"  FAIL  {msg}")
        FAILURES.append(msg)


def rel_error(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.detach().double(), b.detach().double()
    return float((a - b).norm() / (a.norm() + b.norm() + 1e-12))


def make_params(dims: Dims, seed: int):
    """Parameter draw order must match the historical ch06_verify.py order so the
    published numeric trace stays bit-identical after the port:
    E, W_Q, W_K, W_V, W_O, W_g, W_u, W_d, W_out (gains are ones, no draw)."""
    g = torch.Generator().manual_seed(seed)

    def p(*shape, scale=0.2):
        return torch.randn(*shape, generator=g, dtype=DTYPE) * scale

    params = {
        "E": p(dims.V, dims.d),
        "g1": torch.ones(dims.d, dtype=DTYPE),
        "W_Q": p(dims.d, dims.d), "W_K": p(dims.d, dims.d),
        "W_V": p(dims.d, dims.d), "W_O": p(dims.d, dims.d),
        "g2": torch.ones(dims.d, dtype=DTYPE),
        "W_g": p(dims.d, dims.d_ff), "W_u": p(dims.d, dims.d_ff),
        "W_d": p(dims.d_ff, dims.d),
        "gf": torch.ones(dims.d, dtype=DTYPE),
        "W_out": p(dims.d, dims.V),
    }
    for v in params.values():
        v.requires_grad_(True)
    return params


def block_params_of(params: dict) -> graph_mod.BlockParams:
    return graph_mod.BlockParams(
        g1=params["g1"], W_Q=params["W_Q"], W_K=params["W_K"],
        W_V=params["W_V"], W_O=params["W_O"], g2=params["g2"],
        W_g=params["W_g"], W_u=params["W_u"], W_d=params["W_d"],
    )


def run(dims: Dims, seed: int, tokens: torch.Tensor, targets: torch.Tensor,
        mutation: str | None = None):
    """Full manual reverse, seeded correctly.

    The order matters and is forced: the vocabulary head must be reversed FIRST
    to obtain the true adjoint of the block's output, and that adjoint is what
    seeds ``block_backward``. Seeding it with ones -- or with anything other than
    the real upstream adjoint - compares the manual reverse against a different
    objective and produces large, meaningless relative errors.
    """
    from tgd import rmsnorm as rn
    params = make_params(dims, seed)
    X0 = params["E"][tokens]
    Xp, cache = graph_mod.block_forward(X0, block_params_of(params), dims)
    Nf_y, Nf_c = rn.rmsnorm_forward(Xp, params["gf"], dims.eps)
    Z = Nf_y @ params["W_out"]
    loss = loss_mod.masked_mean_ce(Z, targets)

    W_out_bar, Nf_bar, n_sup = rev.output_backward(Nf_y, Z, params["W_out"],
                                                   targets)
    Xp_bar, gf_bar = rn.rmsnorm_backward(Nf_bar, Nf_c)

    grads, trace = rev.block_backward(Xp_bar, cache, block_params_of(params),
                                      dims, tokens, mutation=mutation)
    grads["W_out"] = W_out_bar
    grads["gf"] = gf_bar
    grads["E"] = rev.model_backward(None, trace.X_bar, params["E"],
                                    params["W_out"], tokens, dims.V)
    return params, grads, trace, loss, cache, Z


def rev_output_head(Xp, params, dims) -> torch.Tensor:
    from tgd import rmsnorm as rn
    Nf, _ = rn.rmsnorm_forward(Xp, params["gf"], dims.eps)
    return Nf @ params["W_out"]


print("=" * 74)
print("tgd.reverse -- verification against autograd")
print(f"torch {torch.__version__}   dtype {DTYPE}")
print("=" * 74)

# tokens deliberately repeat id 2, so the embedding scatter-add is exercised
TOK = torch.tensor([[0, 2, 2, 4]], dtype=torch.long)
TGT = torch.tensor([[2, 2, 4, 1]], dtype=torch.long)
assert int((TOK == 2).sum()) == 2, "want a repeated token id"

for dims, seed in ((CH6_TINY, 7), (CH8_TWO_HEAD, 11)):
    print(f"\n[1] manual vs autograd, all parameters -- {dims}")
    params, grads, trace, loss, cache, Z = run(dims, seed, TOK, TGT)
    loss.backward()
    E_bar = grads["E"]

    names = ["g1", "g2", "gf", "W_Q", "W_K", "W_V", "W_O", "W_g", "W_u",
             "W_d", "W_out"]
    worst = 0.0
    for n in names:
        e = rel_error(grads[n], params[n].grad)
        worst = max(worst, e)
        print(f"        {n:6s} {e:.3e}")
    check(worst < 1e-10, f"worst manual/autograd relative error {worst:.3e} < 1e-10")
    check(rel_error(E_bar, params["E"].grad) < 1e-10,
          f"embedding scatter-add matches autograd "
          f"({rel_error(E_bar, params['E'].grad):.3e})")

    # the two-headed case must actually differ from the one-headed scale factor
    if dims.H > 1:
        import math
        check(abs(math.sqrt(dims.d_h) - math.sqrt(dims.d)) > 1e-9,
              f"the two scale factors differ at H={dims.H} "
              f"({1 / math.sqrt(dims.d_h):.4f} vs {1 / math.sqrt(dims.d):.4f})")

print("\n[2] central finite differences, every parameter, epsilon sweep")
for dims, seed in ((CH6_TINY, 7), (CH8_TWO_HEAD, 11)):
    params, grads, trace, loss, cache, Z = run(dims, seed, TOK, TGT)

    probes = {
        "W_Q": (0, 0), "W_K": (1, 2), "W_V": (2, 0), "W_O": (0, 3),
        "W_g": (1, 1), "W_u": (2, 3), "W_d": (0, 2), "W_out": (1, 0),
        "g1": (0,), "g2": (3,), "gf": (2,), "E": (2, 3),
    }
    worst = 0.0
    for name, idx in probes.items():
        analytic = grads[name][idx].item()
        best = float("inf")
        for eps in (1e-4, 3e-5, 1e-5):
            orig = params[name][idx].item()
            with torch.no_grad():
                params[name][idx] = orig + eps
            lp, _ = graph_mod.block_forward(
                params["E"][TOK], block_params_of(params), dims)
            lp = loss_mod.masked_mean_ce(rev_output_head(lp, params, dims), TGT)
            with torch.no_grad():
                params[name][idx] = orig - eps
            lm, _ = graph_mod.block_forward(
                params["E"][TOK], block_params_of(params), dims)
            lm = loss_mod.masked_mean_ce(rev_output_head(lm, params, dims), TGT)
            with torch.no_grad():
                params[name][idx] = orig
            fd = (lp.item() - lm.item()) / (2 * eps)
            best = min(best, abs(fd - analytic) / (abs(fd) + abs(analytic) + 1e-12))
        worst = max(worst, best)
    check(worst < 5e-6,
          f"worst best-epsilon FD relative error {worst:.3e} < 5e-6 ({dims})")

print("\n[3] deletion mutations must be detected (contract A.6)")
print(f"    registered: {rev.MUTATIONS}")
for dims, seed in ((CH6_TINY, 7), (CH8_TWO_HEAD, 11)):
    ref_params, ref, ref_trace, loss, cache, Z = run(dims, seed, TOK, TGT)
    loss.backward()
    params = ref_params

    for mut in rev.MUTATIONS:
        if mut in ("drop_head_k_v_path", "sum_instead_of_concat") and dims.H == 1:
            # both are degenerate at H=1: with a single head the split and concat
            # are identity reshuffles, so there is no second head to drop and
            # summing one head is indistinguishable from reshaping it.
            print(f"    skip   {mut} (degenerate at H=1)")
            continue
        _, m, _, _, _, _ = run(dims, seed, TOK, TGT, mutation=mut)
        # E MUST be in this set. The first residual identity path feeds X_bar,
        # and X_bar feeds *only* the embedding gradient -- no block parameter
        # depends on it. Detecting that mutation therefore requires checking E
        # specifically, which is precisely the parameter a reader forgets. An
        # earlier revision of this harness compared only the block parameters
        # and reported the mutation as undetectable; it was undetectable *there*.
        err = max(rel_error(m[n], ref[n]) for n in
                  ("E", "g1", "g2", "W_Q", "W_K", "W_V", "W_O", "W_g", "W_u",
                   "W_d"))
        flag = "ok   " if err > 1e-6 else "FAIL "
        print(f"    {flag}{mut:32s} worst={err:.3e}")
        if err <= 1e-6:
            FAILURES.append(f"mutation not detected: {mut} at {dims}")

print("\n[4] merge summands are individually nonzero")
params, grads, trace, loss, cache, Z = run(CH8_TWO_HEAD, 11, TOK, TGT)
for nm, t in (("N1_bar_q", trace.N1_bar_q), ("N1_bar_k", trace.N1_bar_k),
              ("N1_bar_v", trace.N1_bar_v), ("N2_bar_g", trace.N2_bar_g),
              ("N2_bar_u", trace.N2_bar_u), ("X_attn_bar", trace.X_attn_bar),
              ("R_bar", trace.R_bar)):
    check(float(t.abs().max()) > 0, f"{nm} is nonzero "
                                    f"(max {float(t.abs().max()):.3e})")
# and each merge equals the sum of its named summands
check(torch.allclose(trace.N1_bar,
                     trace.N1_bar_q + trace.N1_bar_k + trace.N1_bar_v,
                     atol=1e-14),
      "N^(1) merge equals the sum of its three named summands")
check(torch.allclose(trace.X_bar, trace.R_bar + trace.X_attn_bar, atol=1e-14),
      "X merge equals R_bar (identity) + X_attn_bar (attention)")

print("\n[5] contract completeness")
pending = [i for i, v in graph_mod.BACKWARD_STATUS.items() if v == "pending"]
n_impl = sum(1 for v in graph_mod.BACKWARD_STATUS.values() if v == "implemented")
n_na = sum(1 for v in graph_mod.BACKWARD_STATUS.values() if v == "n/a")
check(not pending, f"no contract ID is forward-only (implemented={n_impl}, "
                   f"n/a={n_na}, total={len(graph_mod.BACKWARD_STATUS)})")

print("\n" + "=" * 74)
if FAILURES:
    print(f"CH06 REVERSE VERIFY FAILED: {len(FAILURES)} problem(s)")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print("CH06_REVERSE_VERIFY_PASSED")
print("=" * 74)
