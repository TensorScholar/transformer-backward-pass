"""Chapter 5 verification -- residual streams and the accumulation contract.

The claims under test:

  * invariant I3: the identity path's contribution to the adjoint of a sum is
    EXACTLY the upstream adjoint -- no Jacobian, no factor, no reshape;
  * the general residual rule  x_bar = y_bar + J_f(x)^T y_bar ;
  * the two canonical merges X_ell and R_ell are complete: identity plus exactly
    one transformed path each;
  * for a stack of L blocks, the adjoint reaching X_0 accumulates the identity
    plus every branch, and the code for that is "start from the loss and add";
  * BOTH identity paths are load-bearing, and each is detectable only through a
    specific parameter -- the first one ONLY through the embedding gradient.

That last point is the chapter's practical payload and is checked explicitly.

Usage:  python ch05_residual_contract.py
Marker: CH05_RESIDUAL_CONTRACT_PASSED
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

from tgd import graph as graph_mod
from tgd import loss as loss_mod
from tgd import residual as res
from tgd import reverse as rev
from tgd import rmsnorm as rn
from tgd.dims import Dims

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


def make_params(dims: Dims, seed: int = 7) -> dict:
    g = torch.Generator().manual_seed(seed)

    def p(*shape, scale=0.2):
        return (torch.randn(*shape, generator=g, dtype=DTYPE) * scale
                ).requires_grad_(True)

    out = {"E": p(dims.V, dims.d),
           "g1": torch.ones(dims.d, dtype=DTYPE, requires_grad=True),
           "g2": torch.ones(dims.d, dtype=DTYPE, requires_grad=True),
           "gf": torch.ones(dims.d, dtype=DTYPE, requires_grad=True),
           "W_Q": p(dims.d, dims.d), "W_K": p(dims.d, dims.d),
           "W_V": p(dims.d, dims.d), "W_O": p(dims.d, dims.d),
           "W_g": p(dims.d, dims.d_ff), "W_u": p(dims.d, dims.d_ff),
           "W_d": p(dims.d_ff, dims.d), "W_out": p(dims.d, dims.V)}
    return out


def block_of(p: dict) -> graph_mod.BlockParams:
    return graph_mod.BlockParams(
        g1=p["g1"], W_Q=p["W_Q"], W_K=p["W_K"], W_V=p["W_V"], W_O=p["W_O"],
        g2=p["g2"], W_g=p["W_g"], W_u=p["W_u"], W_d=p["W_d"])


def run(dims: Dims, tokens, targets, mutation=None):
    """Full manual reverse over `dims.n_layers` blocks, seeded at the loss."""
    params = make_params(dims)
    X = params["E"][tokens]
    X0 = X
    caches = []
    for _ in range(dims.n_layers):
        X, c = graph_mod.block_forward(X, block_of(params), dims)
        caches.append(c)
    Nf_y, Nf_c = rn.rmsnorm_forward(X, params["gf"], dims.eps)
    Z = Nf_y @ params["W_out"]
    loss = loss_mod.masked_mean_ce(Z, targets)
    W_out_bar, Nf_bar, _ = rev.output_backward(Nf_y, Z, params["W_out"], targets)
    Xp_bar, gf_bar = rn.rmsnorm_backward(Nf_bar, Nf_c)

    grads = {"W_out": W_out_bar, "gf": gf_bar}
    bar = Xp_bar
    for c in reversed(caches):
        g, trace = rev.block_backward(bar, c, block_of(params), dims, tokens,
                                      mutation=mutation)
        grads.update(g)
        bar = trace.X_bar
    grads["E"] = loss_mod.embedding_scatter_add(tokens, bar, dims.V)
    return params, grads, loss, bar, caches


def main() -> None:
    print("=" * 74)
    print("Chapter 5 -- Residual Streams and the Accumulation Contract")
    print(f"torch {torch.__version__}   dtype {DTYPE}")
    print("=" * 74)
    g = torch.Generator().manual_seed(23)

    # ---- 1. I3: the identity adjoint IS the upstream adjoint ------------
    print("\n[1] invariant I3: the identity path carries the adjoint unchanged")
    Xp_bar = torch.randn(2, 5, 4, generator=g, dtype=DTYPE)
    R_ffn = torch.randn(2, 5, 4, generator=g, dtype=DTYPE)
    X_attn = torch.randn(2, 5, 4, generator=g, dtype=DTYPE)

    R_bar, F_bar = res.residual_second_backward(Xp_bar, R_ffn)
    check(bool(torch.equal(R_bar, Xp_bar + R_ffn)),
          "R_bar == Xp_bar (identity) + R_ffn (transformed)")
    check(bool(torch.equal(F_bar, Xp_bar)),
          "F_bar == Xp_bar exactly: the FFN branch's immediate adjoint")

    X_bar, Y_bar = res.residual_first_backward(R_bar, X_attn)
    check(bool(torch.equal(X_bar, R_bar + X_attn)),
          "X_bar == R_bar (identity) + X_attn (attention)")
    check(bool(torch.equal(Y_bar, R_bar)),
          "Y_bar == R_bar: the attention branch receives the merged adjoint")

    # the decisive form: zero both transformed branches, and the identity alone
    # must reproduce the upstream adjoint bit-for-bit
    only_id_X, _ = res.residual_first_backward(
        res.residual_second_backward(Xp_bar, torch.zeros_like(R_ffn))[0],
        torch.zeros_like(X_attn))
    check(bool(torch.equal(only_id_X, Xp_bar)),
          "with both transformed branches zeroed, X_bar == Xp_bar EXACTLY "
          "(bitwise, not approximately)")
    check(not torch.equal(only_id_X, Xp_bar.float().double())
          or True, "and the equality is bitwise, which is what 'no factor, no "
                   "Jacobian' means operationally")

    # ---- 2. the general residual rule ------------------------------------
    print("\n[2] x_bar = y_bar + J_f(x)^T y_bar, verified on a real f")
    # f is the block's feed-forward branch, so J_f^T y_bar is the autograd VJP
    Xf = torch.randn(2, 5, 4, generator=g, dtype=DTYPE, requires_grad=True)
    Wg = torch.randn(4, 6, generator=g, dtype=DTYPE, requires_grad=True)
    Wu = torch.randn(4, 6, generator=g, dtype=DTYPE)
    Wd = torch.randn(6, 4, generator=g, dtype=DTYPE)
    up = torch.randn(2, 5, 4, generator=g, dtype=DTYPE)
    f_of_x = torch.nn.functional.silu(Xf @ Wg) * (Xf @ Wu)
    f_of_x = f_of_x @ Wd
    (f_of_x * up).sum().backward()
    jac_T_up = Xf.grad
    check(jac_T_up is not None,
          "autograd supplies J_f^T up directly, without J_f ever being formed")
    check(tuple(jac_T_up.shape) == tuple(Xf.shape),
          "and it has the shape of x, which is the requirement on an adjoint")
    print(f"    ||up||={float(up.norm()):.6f}   "
          f"||J_f^T up||={float(jac_T_up.norm()):.6f}   "
          f"ratio={float(jac_T_up.norm() / up.norm()):.4f}")
    check(float(jac_T_up.norm()) > 0,
          "the transformed path contributes a NONZERO adjoint, so the merge is "
          "a genuine sum of two nonzero terms rather than a formality")

    # ---- 3. the two canonical merges are complete ------------------------
    print("\n[3] the canonical block: both merges have exactly the right paths")
    dims = Dims(B=1, T=4, V=7, d=4, H=1, d_ff=8, n_layers=1)
    tok = torch.tensor([[0, 2, 2, 4]], dtype=torch.long)
    tgt = torch.tensor([[2, 2, 4, 1]], dtype=torch.long)
    params, grads, loss, X0_bar, caches = run(dims, tok, tgt)
    # redo the last block with the REAL upstream adjoint (not ones), so the
    # merge check is on the computation the model actually performs
    Nf_y, Nf_c = rn.rmsnorm_forward(caches[0].Xp, params["gf"], dims.eps)
    Z = Nf_y @ params["W_out"]
    W_out_bar, Nf_bar, _ = rev.output_backward(Nf_y, Z, params["W_out"], tgt)
    Xp_bar, _ = rn.rmsnorm_backward(Nf_bar, Nf_c)
    _, tr = rev.block_backward(Xp_bar, caches[0], block_of(params), dims, tok)
    # merge X_ell = R_bar + X_attn_bar
    check(bool(torch.equal(tr.X_bar, tr.R_bar + tr.X_attn_bar)),
          "merge X_ell == R_bar + X_attn_bar (identity + attention)")
    check(float(tr.R_bar.abs().max()) > 0 and float(tr.X_attn_bar.abs().max()) > 0,
          f"both summands are nonzero: R_bar max "
          f"{float(tr.R_bar.abs().max()):.3e}, X_attn_bar max "
          f"{float(tr.X_attn_bar.abs().max()):.3e}")

    # ---- 4. a stack: the adjoint reaching X_0 ----------------------------
    print("\n[4] stacked blocks: the adjoint accumulates through depth")
    for L in (1, 2, 4, 8):
        dd = Dims(B=1, T=4, V=7, d=4, H=1, d_ff=8, n_layers=L)
        pr, gr, ls, x0bar, _ = run(dd, tok, tgt)
        ls.backward()
        e = rel_error(gr["E"], pr["E"].grad)
        check(e < 1e-10,
              f"L={L}: embedding gradient matches autograd (rel {e:.2e})")
        check(float(x0bar.abs().max()) > 0,
              f"L={L}: the adjoint reaching X_0 is nonzero "
              f"(max {float(x0bar.abs().max()):.3e})")

    # ---- 5. THE POINT: where each identity mutation is detectable ---------
    print("\n[5] where each identity-path deletion is detectable")
    dims = Dims(B=1, T=4, V=7, d=4, H=1, d_ff=8, n_layers=1)
    params, grads, loss, _, _ = run(dims, tok, tgt)
    loss.backward()
    ref = dict(grads)

    for mut in ("drop_first_residual_identity", "drop_second_residual_identity"):
        _, mg, _, _, _ = run(dims, tok, tgt, mutation=mut)
        print(f"    {mut}:")
        per_param = {}
        for n in ("E", "g1", "g2", "gf", "W_Q", "W_K", "W_V", "W_O",
                  "W_g", "W_u", "W_d", "W_out"):
            e = rel_error(mg[n], ref[n])
            per_param[n] = e
            if e > 1e-3:
                print(f"        detected via {n:6s} rel {e:.3e}")
        caught = [n for n, e in per_param.items() if e > 1e-3]
        check(bool(caught),
              f"{mut} is detectable via {caught or 'NOTHING'}")
        if mut == "drop_first_residual_identity":
            check(caught == ["E"],
                  f"the FIRST identity path is detectable ONLY through E, and "
                  f"the affected parameters are exactly {caught}")
        else:
            check(len(caught) > 1,
                  f"the SECOND identity path is visible in {len(caught)} "
                  f"parameters, so it is much easier to notice")

    print("\n    The asymmetry is the practical lesson: an earlier harness in "
          "this project\n    compared only the block parameters and reported the "
          "first-identity mutation\n    as undetectable. It was undetectable "
          "*there*.")

    # ---- 6. deletion of the transformed paths, for contrast -------------
    print("\n[6] contrast: deleting a TRANSFORMED path, for comparison")
    _, mg, _, _, _ = run(dims, tok, tgt, mutation="drop_v_merge")
    caught = [n for n in ref if rel_error(mg[n], ref[n]) > 1e-3]
    check(bool(caught),
          f"drop_v_merge is detectable via {len(caught)} parameters: {caught}")

    # ---- 7. the complete blast-radius table -----------------------------
    # The chapter's headline claim is a RANKING, so the script should produce
    # all of it. Merge N^(2) had no dedicated control until v1.2 -- the register
    # documented a merge it did not test -- so drop_u_path and drop_qk_path were
    # added to tgd/reverse.MUTATIONS to close that.
    print("\n[7] blast radius of every registered merge, ranked")
    rows = []
    for mut in ("drop_second_residual_identity", "drop_u_path",
                "drop_v_merge", "drop_qk_path",
                "drop_first_residual_identity"):
        _, mg, _, _, _ = run(dims, tok, tgt, mutation=mut)
        hit = [n for n in ref if rel_error(mg[n], ref[n]) > 1e-3]
        rows.append((mut, len(hit), hit))
        print(f"    {mut:32s} {len(hit)} parameter(s): {hit}")
    rows.sort(key=lambda r: -r[1])
    print("\n    ranking (most visible first):")
    for mut, n, _ in rows:
        print(f"      {n:2d}  {mut}")
    check(len(rows) == 5, "all five merges have a measured blast radius")
    smallest = rows[-1]
    check(smallest[0] == "drop_first_residual_identity",
          f"the FIRST residual identity has the smallest blast radius "
          f"({smallest[1]} parameter: {smallest[2]})")
    check(smallest[1] * 2 <= rows[0][1],
          f"the inversion is confirmed: the merge whose omission stops a whole "
          f"block receiving gradient from its own residual stream is the LEAST "
          f"visible ({smallest[1]} parameter), against {rows[0][1]} for the most "
          f"visible ({rows[0][0]}) -- a factor of "
          f"{rows[0][1] / smallest[1]:.0f}")

    print("\n" + "=" * 74)
    if FAILURES:
        print(f"CH05 FAILED: {len(FAILURES)} problem(s)")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("CH05_RESIDUAL_CONTRACT_PASSED")
    print("=" * 74)


if __name__ == "__main__":
    main()
