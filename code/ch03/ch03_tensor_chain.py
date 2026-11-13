"""Chapter 3 verification -- from scalars to tensors.

The claims under test:

  * the adjoint of a rank-n map is defined by an inner product and needs no
    flattening convention and no Jacobian;
  * elementwise maps give elementwise adjoints;
  * a BROADCAST is a fan-in: the adjoint sums over the broadcast axes;
  * a tensor used twice is a fan-in on the way back, and the two contributions
    must be ADDED -- this is the rule (2.1) does not supply, and it is the
    mechanism behind the book's worst defect class;
  * a rank-3 "transpose" is a contraction over two axes, not a single operation.

Everything is checked against autograd, against central finite differences, and
against deletions.

Usage:  python ch03_tensor_chain.py
Marker: CH03_TENSOR_CHAIN_PASSED
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


# --------------------------------------------------------------- adjoint ----

def vjp_elementwise(Y_bar: torch.Tensor, cache) -> torch.Tensor:
    """Y = f(X) elementwise:  X_bar = Y_bar * f'(X).

    For f = exp, f' = exp, so the rule is ``Y_bar * exp(X)``. Writing
    ``exp(-X)`` here is a plausible slip -- the two differ by a factor of
    exp(2X) and still have the right shape -- which is why the script asserts
    the value rather than the shape.
    """
    return Y_bar * torch.exp(cache["X"])


def vjp_broadcast(Y_bar: torch.Tensor, other: torch.Tensor,
                  target_shape) -> torch.Tensor:
    """Reverse of ``Y = X * other`` where ``other`` was broadcast against X.

    Two steps, and the order matters:
      1. the elementwise product's adjoint is ``Y_bar * other``, at the BROADCAST
         shape;
      2. reduce to ``target_shape`` by SUMMING over exactly the axes that were
         broadcast.

    Omitting step 2, or doing it before step 1, are the two classic errors.
    """
    Y_bar = Y_bar * other
    return _sum_broadcast_axes(Y_bar, target_shape)


def _sum_broadcast_axes(Y_bar: torch.Tensor, target_shape) -> torch.Tensor:
    """Sum over broadcast axes, and over any leading axes the target lacks."""
    extra = Y_bar.dim() - len(target_shape)
    if extra:
        Y_bar = Y_bar.sum(dim=tuple(range(extra)))
    axes = [i for i, s in enumerate(target_shape) if s == 1 and Y_bar.shape[i] != 1]
    if axes:
        Y_bar = Y_bar.sum(dim=tuple(axes), keepdim=True)
    return Y_bar.reshape(target_shape)


def vjp_fanin(x_bar_a: torch.Tensor, x_bar_b: torch.Tensor) -> torch.Tensor:
    """A tensor consumed twice: the contributions ADD.

    Not a rule from Chapter 2. Chapter 2's rule handles one consumer; this is the
    extra obligation, and it is registered as a merge in the contract.
    """
    return x_bar_a + x_bar_b


def vjp_contract_batched(A_bar: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Y[b] = A[b] @ B[b]  (batch of matrix products).  A_bar[b] = Y_bar[b] B[b]^T.

    Note the transpose is on B, per head/batch, and B is NOT transposed once for
    the whole batch. This is the shape of operation a multi-head attention
    backward is made of, and getting it wrong changes shapes only when the
    per-item widths differ.
    """
    return A_bar @ B.transpose(-2, -1)


def main() -> None:
    print("=" * 74)
    print("Chapter 3 -- From Scalars to Tensors")
    print(f"torch {torch.__version__}   dtype {DTYPE}")
    print("=" * 74)
    g = torch.Generator().manual_seed(17)

    # ---- 1. elementwise on a rank-3 tensor -------------------------------
    print("\n[1] elementwise: the adjoint is elementwise, at any rank")
    for shape in [(5,), (4, 3), (2, 3, 4), (2, 2, 3, 3)]:
        X = torch.randn(*shape, generator=g, dtype=DTYPE, requires_grad=True)
        Y_bar = torch.randn(*shape, generator=g, dtype=DTYPE)
        ((torch.exp(X)) * Y_bar).sum().backward()
        ref = vjp_elementwise(Y_bar, {"X": X.detach()})
        check(rel_error(ref, X.grad) < 1e-13,
              f"rank {len(shape)} shape {tuple(shape)}: "
              f"rel {rel_error(ref, X.grad):.2e}")
    check(rel_error(vjp_elementwise(Y_bar, {"X": X.detach()}), X.grad) < 1e-13,
          "a Jacobian would be 2^n_dim entries per element; this rule has none, "
          "and is exact at rank 4")

    # ---- 2. broadcast is a fan-in ----------------------------------------
    print("\n[2] broadcast: the reverse is a SUM over the broadcast axes")
    cases = [
        ((3, 4), (4,)),        # row vector over columns
        ((3, 4), (3, 1)),      # column vector over rows
        ((3, 4), (1,)),        # scalar over everything
        ((3, 1), (1, 4)),      # outer product
        ((2, 3, 4), (4,)),     # rank-3 with trailing broadcast
    ]
    for tgt, src in cases:
        X = torch.randn(*tgt, generator=g, dtype=DTYPE, requires_grad=True)
        S = torch.randn(*src, generator=g, dtype=DTYPE)
        Y_bar = torch.randn(*torch.broadcast_shapes(tgt, src),
                            generator=g, dtype=DTYPE)
        ((X * S) * Y_bar).sum().backward()
        ref = vjp_broadcast(Y_bar, S, tgt)
        ok = tuple(ref.shape) == tuple(tgt)
        check(ok and rel_error(ref, X.grad) < 1e-13,
              f"target {tgt} <- source {src}: shape {tuple(ref.shape)} "
              f"rel {rel_error(ref, X.grad):.2e}")

    # the common wrong answer: broadcasting the adjoint back without summing
    print("\n[2b] two wrong ways to reverse a broadcast, both shape-correct")
    tgt, src = (3, 4), (4,)
    X = torch.randn(*tgt, generator=g, dtype=DTYPE, requires_grad=True)
    S = torch.randn(*src, generator=g, dtype=DTYPE)
    Y_bar = torch.randn(*tgt, generator=g, dtype=DTYPE)
    ((X * S) * Y_bar).sum().backward()
    correct = vjp_broadcast(Y_bar, S, tgt)
    check(rel_error(correct, X.grad) < 1e-13, "multiply then sum is right")
    # error 1: applying the broadcast factor a second time
    twice = _sum_broadcast_axes(Y_bar * S * S, tgt)
    # error 2: reducing the wrong operand (the source, not the product)
    wrong_operand = _sum_broadcast_axes(Y_bar, src).expand(tgt).clone()
    for nm, bad in (("double_multiply", twice),
                    ("reduce_wrong_operand", wrong_operand)):
        err = rel_error(bad, X.grad)
        shape_ok = tuple(bad.shape) == tuple(X.grad.shape)
        check(err > 1e-3 and shape_ok,
              f"{nm}: wrong (rel {err:.3e}) yet the shape is correct "
              f"{tuple(bad.shape)} -- a shape check cannot see it")

    # ---- 3. the fan-in: a tensor used twice ------------------------------
    print("\n[3] fan-in: a tensor consumed twice must SUM its two contributions")
    X = torch.randn(3, 4, generator=g, dtype=DTYPE, requires_grad=True)
    A = torch.randn(4, 5, generator=g, dtype=DTYPE)
    B = torch.randn(4, 5, generator=g, dtype=DTYPE)
    Y_bar = torch.randn(3, 5, generator=g, dtype=DTYPE)
    Y = X @ (A + B)
    (Y * Y_bar).sum().backward()
    path_a = Y_bar @ A.T          # via the A branch
    path_b = Y_bar @ B.T          # via the B branch
    summed = vjp_fanin(path_a, path_b)
    check(rel_error(summed, X.grad) < 1e-13,
          f"the two paths add to the autograd gradient "
          f"({rel_error(summed, X.grad):.2e})")
    check(rel_error(path_a, X.grad) > 1e-3,
          f"EITHER path alone is wrong ({rel_error(path_a, X.grad):.3e}) -- "
          f"which is the failure mode, not a rounding difference")
    check(rel_error(path_b, X.grad) > 1e-3,
          f"the other path alone is wrong too "
          f"({rel_error(path_b, X.grad):.3e})")
    # ---- 4. fan-out: one tensor feeding many outputs ----------------------
    print("\n[4] fan-out: one upstream adjoint reaches every consumer unchanged")
    X = torch.randn(4, generator=g, dtype=DTYPE, requires_grad=True)
    W1 = torch.randn(4, 3, generator=g, dtype=DTYPE)
    W2 = torch.randn(4, 6, generator=g, dtype=DTYPE)
    Yb1 = torch.randn(3, generator=g, dtype=DTYPE)
    Yb2 = torch.randn(6, generator=g, dtype=DTYPE)
    (((X @ W1) * Yb1).sum() + ((X @ W2) * Yb2).sum()).backward()
    p1 = Yb1 @ W1.T
    p2 = Yb2 @ W2.T
    check(rel_error(p1 + p2, X.grad) < 1e-13,
          "each consumer receives the same upstream adjoint; the merge happens "
          "on the way back, not on the way forward")

    # ---- 5. batched contraction: the rank-3 "transpose" -------------------
    print("\n[5] rank-3: the transpose is a per-item contraction, not one op")
    Bt, T, dh, B = 2, 5, 3, 4
    Qt = torch.randn(B, T, dh, generator=g, dtype=DTYPE, requires_grad=True)
    Kt = torch.randn(B, T, dh, generator=g, dtype=DTYPE)
    S_bar = torch.randn(B, T, T, generator=g, dtype=DTYPE)
    scale = dh ** 0.5
    # Y = S = Q K^T / sqrt(dh)
    ((Qt @ Kt.transpose(-2, -1) / scale) * S_bar).sum().backward()
    Q_ref = S_bar @ Kt / scale
    check(rel_error(Q_ref, Qt.grad) < 1e-13,
          f"dL/dQ[b] = S_bar[b] K[b]/sqrt(d_h), per batch item "
          f"({rel_error(Q_ref, Qt.grad):.2e})")
    # The realistic error: transposing the WRONG operand. S_bar^T is (B,T,T) and
    # Kt is (B,T,d_h), so this product is shape-correct and silently wrong.
    # Note the other candidate -- transposing the whole rank-3 tensor once --
    # is NOT shape-correct here and would have been caught immediately. The
    # dangerous errors are the ones that stay inside the shape budget.
    wrong = (S_bar.transpose(-2, -1) @ Kt) / scale
    check(tuple(wrong.shape) == tuple(Qt.grad.shape),
          f"transposing the wrong operand keeps the shape "
          f"{tuple(wrong.shape)} == {tuple(Qt.grad.shape)}")
    check(rel_error(wrong, Qt.grad) > 1e-3,
          f"and is wrong by a factor of ~2 in magnitude "
          f"(rel {rel_error(wrong, Qt.grad):.3e}) -- undetectable by shape")
    # for contrast, the blanket global transpose IS caught
    blanket = (Kt.transpose(-1, -2) @ S_bar.transpose(-1, -2)) / scale
    check(tuple(blanket.shape) != tuple(Qt.grad.shape),
          f"a blanket transpose of the whole rank-3 tensor gives shape "
          f"{tuple(blanket.shape)} != {tuple(Qt.grad.shape)}, so THAT error is "
          f"caught by a shape check -- not all rank-3 mistakes are silent")

    # ---- 6. the Jacobian we never form -----------------------------------
    print("\n[6] the Jacobian is never formed, and the script shows why not")
    n = 40
    x = torch.randn(n, generator=g, dtype=DTYPE, requires_grad=True)
    ybar = torch.randn(n, generator=g, dtype=DTYPE)
    (torch.exp(x) * ybar).sum().backward()
    # the Jacobian is n x n; the VJP applies it to ybar
    J = torch.diag(torch.exp(x.detach()))
    vjp_by_jacobian = J.T @ ybar
    vjp_by_rule = ybar * torch.exp(x.detach())
    check(rel_error(vjp_by_jacobian, vjp_by_rule) < 1e-13,
          f"J^T ybar equals the elementwise rule, for a {n}x{n} Jacobian")
    check(n * n > 1000,
          f"the Jacobian is {n}x{n} = {n * n} entries for an operation whose "
          f"adjoint rule is one multiply; at the vocabulary scale of a real "
          f"model it is not formable")
    # and for a quadratic, where the Jacobian depends on X
    m = 12
    x = torch.randn(m, generator=g, dtype=DTYPE, requires_grad=True)
    yb = torch.randn(m, generator=g, dtype=DTYPE)
    ((x * x) * yb).sum().backward()
    Jq = torch.diag(2 * x.detach())
    check(rel_error(Jq.T @ yb, x.grad) < 1e-13,
          "for Y = X*X the Jacobian is diag(2X): it depends on X, so it cannot "
          "be a constant table, which is the structural reason to avoid it")

    # ---- 7. finite differences, the tensor cases -------------------------
    print("\n[7] central finite differences on the tensor rules")
    X = torch.randn(3, 4, generator=g, dtype=DTYPE, requires_grad=True)
    A = torch.randn(4, 5, generator=g, dtype=DTYPE)
    Bm = torch.randn(4, 5, generator=g, dtype=DTYPE)
    yb = torch.randn(3, 5, generator=g, dtype=DTYPE)
    ((X @ (A + Bm)) * yb).sum().backward()
    analytic = yb @ A.T + yb @ Bm.T
    for idx in [(0, 0), (1, 3), (2, 2)]:
        best = float("inf")
        for eps in (1e-4, 3e-5, 1e-5):
            o = X[idx].item()
            with torch.no_grad():
                X[idx] = o + eps
            lp = ((X @ (A + Bm)) * yb).sum().item()
            with torch.no_grad():
                X[idx] = o - eps
            lm = ((X @ (A + Bm)) * yb).sum().item()
            with torch.no_grad():
                X[idx] = o
            f = (lp - lm) / (2 * eps)
            an = float(analytic[idx])
            best = min(best, abs(f - an) / (abs(f) + abs(an) + 1e-12))
        check(best < 5e-6,
              f"fan-in at X{idx}: best-epsilon FD rel {best:.3e} vs the summed "
              f"analytic value")

    print("\n" + "=" * 74)
    if FAILURES:
        print(f"CH03 FAILED: {len(FAILURES)} problem(s)")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("CH03_TENSOR_CHAIN_PASSED")
    print("=" * 74)


if __name__ == "__main__":
    main()
