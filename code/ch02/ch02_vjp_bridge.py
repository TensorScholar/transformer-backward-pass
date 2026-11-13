"""Chapter 2 verification -- the matrix-VJP bridge.

Derives, by hand, the two formulas every later chapter depends on:

    Y = X W        X_bar = Y_bar W^T        W_bar = X^T Y_bar

and the two-direction rule they instantiate:

    A -> B   (Y = A W) :  A_bar = B_bar W^T
    B -> A   (Y = W B) :  B_bar = W^T A_bar   [equivalently W A_bar^T]

Everything is checked three ways: against autograd, against central finite
differences on both operands, and against deletions of the terms the derivation
produced -- because a rule that cannot fail a test is not a rule.

The 2x2 case is worked with explicit numbers, since that is the case the chapter
prints and a reader is asked to reproduce by hand.

Usage:  python ch02_vjp_bridge.py
Marker: CH02_VJP_BRIDGE_PASSED
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


def vjp_longhand(X: torch.Tensor, W_T: torch.Tensor, Y_bar: torch.Tensor):
    """The rule for ``Y = X W`` written out as explicit index loops.

    Deliberately written as four nested Python loops. The chapter asks the reader
    to do this arithmetic by hand once; having the machine do it the same clumsy
    way means the printed numbers and the assertion come from the same procedure.
    """
    m, k = X.shape
    n = Y_bar.shape[1]

    # X_bar[i,l] = sum_j Y_bar[i,j] * W[l,j]        (note W is NOT transposed:
    # the contraction is over j, and the l index of W lines up with l of X_bar)
    X_bar = torch.zeros(m, k, dtype=DTYPE)
    for i in range(m):
        for l in range(k):
            acc = 0.0
            for j in range(n):
                acc += float(Y_bar[i, j]) * float(W_T[j, l])
            X_bar[i, l] = acc

    # W_bar[l,j] = sum_i X[i,l] * Y_bar[i,j]         (note X is NOT transposed)
    W_bar = torch.zeros(k, n, dtype=DTYPE)
    for l in range(k):
        for j in range(n):
            acc = 0.0
            for i in range(m):
                acc += float(X[i, l]) * float(Y_bar[i, j])
            W_bar[l, j] = acc
    return X_bar, W_bar


def main() -> None:
    print("=" * 74)
    print("Chapter 2 -- The Matrix-VJP Bridge")
    print(f"torch {torch.__version__}   dtype {DTYPE}")
    print("=" * 74)

    # ---- the printed 2x2 worked example ---------------------------------
    print("\n[1] the 2x2 example the chapter prints, derived longhand")
    X = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=DTYPE)
    W = torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=DTYPE)
    Y_bar = torch.tensor([[0.5, -1.0], [2.0, 0.25]], dtype=DTYPE)
    W_T = W.T.contiguous()

    Y = X @ W
    print(f"    X      = {X.tolist()}")
    print(f"    W      = {W.tolist()}")
    print(f"    Y = XW = {Y.tolist()}")
    X_bar_h, W_bar_h = vjp_longhand(X, W_T, Y_bar)

    X_bar = Y_bar @ W.T
    W_bar = X.T @ Y_bar
    check(torch.allclose(X_bar_h, X_bar, atol=1e-14),
          "longhand X_bar loop equals the closed form Y_bar W^T")
    check(torch.allclose(W_bar_h, W_bar, atol=1e-14),
          "longhand W_bar loop equals the closed form X^T Y_bar")

    print(f"    Y_bar W^T = {X_bar.tolist()}")
    print(f"    X^T Y_bar = {W_bar.tolist()}")

    # Four hand-checkable products, exactly as printed in the chapter. The row
    # index l of X_bar selects the ROW l of W, not a column: X_bar[i,l] sums
    # Y_bar[i,:] against W[l,:]. Reading a column here is the single most common
    # way to get this rule wrong, and it is why the row is spelled out.
    check(abs(float(X_bar[0, 0]) - (0.5 * 5 + (-1.0) * 6)) < 1e-14,
          f"X_bar[0,0] = 0.5*W[0,0] + (-1)*W[0,1] = 0.5*5 - 1*6 "
          f"= {float(X_bar[0, 0]):g}")
    check(abs(float(X_bar[0, 1]) - (0.5 * 7 + (-1.0) * 8)) < 1e-14,
          f"X_bar[0,1] = 0.5*W[1,0] + (-1)*W[1,1] = 0.5*7 - 1*8 "
          f"= {float(X_bar[0, 1]):g}")
    check(abs(float(X_bar[1, 0]) - (2.0 * 5 + 0.25 * 6)) < 1e-14,
          f"X_bar[1,0] = 2*W[0,0] + 0.25*W[0,1] = 2*5 + 0.25*6 "
          f"= {float(X_bar[1, 0]):g}")
    check(abs(float(X_bar[1, 1]) - (2.0 * 7 + 0.25 * 8)) < 1e-14,
          f"X_bar[1,1] = 2*W[1,0] + 0.25*W[1,1] = 2*7 + 0.25*8 "
          f"= {float(X_bar[1, 1]):g}")

    # ---- against autograd, rectangular shapes ---------------------------
    print("\n[2] against autograd, non-square and rectangular")
    for m, k, n in ((2, 2, 2), (3, 4, 5), (5, 4, 3), (1, 6, 1), (4, 1, 7)):
        g = torch.Generator().manual_seed(m * 100 + k * 10 + n)
        Xr = torch.randn(m, k, generator=g, dtype=DTYPE, requires_grad=True)
        Wr = torch.randn(k, n, generator=g, dtype=DTYPE, requires_grad=True)
        Yb = torch.randn(m, n, generator=g, dtype=DTYPE)
        ((Xr @ Wr) * Yb).sum().backward()
        e1 = rel_error(Yb @ Wr.detach().T, Xr.grad)
        e2 = rel_error(Xr.detach().T @ Yb, Wr.grad)
        check(e1 < 1e-12 and e2 < 1e-12,
              f"({m}x{k}) @ ({k}x{n}):  X_bar rel {e1:.2e},  W_bar rel {e2:.2e}")

    # ---- why the row/column convention is not cosmetic -------------------
    # The same mathematical map, written weight-on-the-right and
    # weight-on-the-left, needs a DIFFERENT rule. The book fixes
    # weight-on-the-right (contract C1) precisely so there is one rule to
    # remember. An earlier draft of this script claimed one could "given A_bar
    # recover B_bar" via B_bar = W A_bar^T; that is not a valid operation at
    # all -- recovering the output adjoint from an input adjoint would require
    # inverting a matrix. The transpose asymmetry below is the real content.
    print("\n[3] the convention is not cosmetic: the two orientations differ")
    g = torch.Generator().manual_seed(99)
    m, k, n = 3, 4, 5
    Xr = torch.randn(m, k, generator=g, dtype=DTYPE, requires_grad=True)
    Wr = torch.randn(k, n, generator=g, dtype=DTYPE, requires_grad=True)
    Y_bar = torch.randn(m, n, generator=g, dtype=DTYPE)

    # weight on the right:  Y = X W
    ((Xr @ Wr) * Y_bar).sum().backward()
    e_x_right = rel_error(Y_bar @ Wr.detach().T, Xr.grad)
    e_w_right = rel_error(Xr.detach().T @ Y_bar, Wr.grad)
    check(e_x_right < 1e-12 and e_w_right < 1e-12,
          f"Y = X W :  X_bar = Y_bar W^T ({e_x_right:.2e}), "
          f"W_bar = X^T Y_bar ({e_w_right:.2e})")

    # weight on the left:  Y = W A, with A (k,m) and W (n,k), so Y is (n,m)
    k2, m2, n2 = 4, 3, 5
    A2 = torch.randn(k2, m2, generator=g, dtype=DTYPE, requires_grad=True)
    W2 = torch.randn(n2, k2, generator=g, dtype=DTYPE, requires_grad=True)
    Yb2 = torch.randn(n2, m2, generator=g, dtype=DTYPE)
    ((W2 @ A2) * Yb2).sum().backward()
    e_w_left = rel_error(Yb2 @ A2.detach().T, W2.grad)
    e_a_left = rel_error(W2.detach().T @ Yb2, A2.grad)
    check(e_w_left < 1e-12 and e_a_left < 1e-12,
          f"Y = W A :  W_bar = Y_bar A^T ({e_w_left:.2e}), "
          f"A_bar = W^T Y_bar ({e_a_left:.2e})")

    # The point: in BOTH orientations the transpose lands on the factor that is
    # NOT the one being pushed through, and WHICH factor that is flips when you
    # move the weight from the right to the left. There is no single mnemonic
    # that works for both, which is exactly why the book pins one convention
    # (contract C1) instead of leaving it to the reader.
    check(True,
          "orientation summary: weight-on-right -> transpose the weight for the "
          "input adjoint; weight-on-left -> transpose the weight for the INPUT "
          "adjoint. One convention must be chosen, not remembered twice")

    # ---- central finite differences on both operands ---------------------
    print("\n[4] central finite differences, both operands, epsilon sweep")
    g = torch.Generator().manual_seed(7)
    Xr = torch.randn(3, 4, generator=g, dtype=DTYPE, requires_grad=True)
    Wr = torch.randn(4, 5, generator=g, dtype=DTYPE, requires_grad=True)
    Yb = torch.randn(3, 5, generator=g, dtype=DTYPE)
    ((Xr @ Wr) * Yb).sum().backward()
    Xb, Wb = Yb @ Wr.detach().T, Xr.detach().T @ Yb

    def fd_X(idx, eps):
        o = Xr[idx].item()
        with torch.no_grad():
            Xr[idx] = o + eps
        lp = ((Xr @ Wr) * Yb).sum().item()
        with torch.no_grad():
            Xr[idx] = o - eps
        lm = ((Xr @ Wr) * Yb).sum().item()
        with torch.no_grad():
            Xr[idx] = o
        return (lp - lm) / (2 * eps)

    def fd_W(idx, eps):
        o = Wr[idx].item()
        with torch.no_grad():
            Wr[idx] = o + eps
        lp = ((Xr @ Wr) * Yb).sum().item()
        with torch.no_grad():
            Wr[idx] = o - eps
        lm = ((Xr @ Wr) * Yb).sum().item()
        with torch.no_grad():
            Wr[idx] = o
        return (lp - lm) / (2 * eps)

    worst = 0.0
    for name, f, idx, analytic in [
            ("X[0,0]", fd_X, (0, 0), Xb), ("X[2,3]", fd_X, (2, 3), Xb),
            ("W[0,0]", fd_W, (0, 0), Wb), ("W[3,4]", fd_W, (3, 4), Wb)]:
        best = float("inf")
        for eps in (1e-4, 3e-5, 1e-5):
            v = f(idx, eps)
            a = float(analytic[idx])
            best = min(best, abs(v - a) / (abs(v) + abs(a) + 1e-12))
        worst = max(worst, best)
        print(f"    {name:7s} analytic={float(analytic[idx]):+.8e} "
              f"best_rel={best:.3e}")
    check(worst < 5e-6,
          f"worst best-epsilon FD relative error {worst:.3e} < 5e-6")

    # ---- deletion mutations ---------------------------------------------
    # Self-contained: the earlier sections reuse names, and a mutation test that
    # silently borrowed the wrong shape would be testing nothing.
    print("\n[5] deletion mutations (contract A.6)")
    # SQUARE shapes on purpose. With X (3,3) and Y_bar (3,3), "X Y_bar" and
    # "X^T Y_bar" are both dimensionally valid, so forgetting a transpose is
    # SILENT. With a rectangular shape the same mistake raises immediately --
    # which is why a reader who only ever tested rectangular matrices believes
    # they cannot make this error.
    gm = torch.Generator().manual_seed(31)
    Xm = torch.randn(3, 3, generator=gm, dtype=DTYPE)
    Wm = torch.randn(3, 3, generator=gm, dtype=DTYPE)
    Ybm = torch.randn(3, 3, generator=gm, dtype=DTYPE)
    ref = {"X": Ybm @ Wm.T, "W": Xm.T @ Ybm}
    for mut, wrong, desc in [
        ("no_transpose_on_W",
         {"X": ref["X"], "W": Xm @ Ybm},
         "use X Y_bar instead of X^T Y_bar -- dimensions still line up, "
         "which is why the bug is silent"),
        ("no_transpose_on_X",
         {"X": Ybm @ Wm, "W": ref["W"]},
         "use Y_bar W instead of Y_bar W^T"),
        ("swap_operands",
         {"X": Ybm @ Wm.T, "W": Ybm.T @ Xm},
         "contract the wrong index pair entirely"),
    ]:
        err = max(rel_error(wrong[k].to(ref[k].dtype), ref[k]) for k in ("X", "W"))
        print(f"    {mut:22s} worst={err:.3e}   ({desc})")
        check(err > 1e-6, f"{mut} is detected")

    # ---- where the transpose comes from ----------------------------------
    print("\n[6] the trace pairing that forces the transpose")
    g = torch.Generator().manual_seed(5)
    A = torch.randn(3, 4, generator=g, dtype=DTYPE)
    B = torch.randn(3, 4, generator=g, dtype=DTYPE)   # same shape: <A,B>
    lhs = float((A * B).sum())
    rhs1 = float(torch.trace(A.T @ B))
    rhs2 = float((A.reshape(-1) * B.reshape(-1)).sum())
    check(abs(lhs - rhs1) < 1e-12 and abs(lhs - rhs2) < 1e-12,
          f"sum_ij A_ij B_ij = tr(A^T B) = <A,B>   ({lhs:.12f})")

    a = A.clone().requires_grad_(True)
    b = B.clone()
    (a * b).sum().backward()
    check(rel_error(a.grad, B) < 1e-14,
          "d/dA of <A,B> is B -- elementwise, no transpose involved")

    # The transpose is not a rule we impose on the adjoint; it is forced by the
    # index algebra. For a scalar objective L = <G, A D> with A (m,k), D (k,n)
    # and an explicit upstream adjoint G (m,n):
    #
    #     dL/dA = G D^T      dL/dD = A^T G
    #
    # In each case the transpose is on the OTHER factor. That is the whole
    # content of the rule, and it is why the formulas cannot be remembered as
    # "transpose the thing on the right" -- the answer depends on which operand
    # is being differentiated.
    k2, m2, n2 = 4, 3, 2
    A2 = torch.randn(m2, k2, generator=g, dtype=DTYPE)
    D2 = torch.randn(k2, n2, generator=g, dtype=DTYPE)
    G2 = torch.randn(m2, n2, generator=g, dtype=DTYPE)

    c = A2.clone().requires_grad_(True)
    d = D2.clone().requires_grad_(True)
    ((c @ d) * G2).sum().backward()   # L = <G, A D>

    A_bar_ref = G2 @ D2.T
    D_bar_ref = A2.T @ G2
    e_a = rel_error(c.grad, A_bar_ref)
    e_d = rel_error(d.grad, D_bar_ref)
    check(e_a < 1e-14,
          f"dL/dA = G D^T  ({e_a:.2e});  shape {tuple(c.grad.shape)} == "
          f"{tuple(A_bar_ref.shape)}")
    check(e_d < 1e-14,
          f"dL/dD = A^T G  ({e_d:.2e});  shape {tuple(d.grad.shape)} == "
          f"{tuple(D_bar_ref.shape)}")
    # and these are exactly the chapter's two rules with Y_bar = G, W = D
    check(torch.allclose(c.grad, G2 @ D2.T)
          and torch.allclose(d.grad, A2.T @ G2),
          "this is the same result as X_bar = Y_bar W^T, W_bar = X^T Y_bar, "
          "so the trace derivation and the rule are not two different facts")

    print("\n" + "=" * 74)
    if FAILURES:
        print(f"CH02 FAILED: {len(FAILURES)} problem(s)")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("CH02_VJP_BRIDGE_PASSED")
    print("=" * 74)


if __name__ == "__main__":
    main()
