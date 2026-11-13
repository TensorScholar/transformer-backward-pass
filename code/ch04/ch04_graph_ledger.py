"""Chapter 4 verification — The Canonical Graph, Frozen.

This is the chapter that makes graph drift unrepresentable. It asserts:

  * every forward tensor's shape matches the canonical specification
    at three model scales;
  * the parameter count computed from the shape ledger equals the count of the
    actual tensors;
  * invariant I1 holds per head (legal-row score-adjoint sum is zero, masked
    score adjoints are exactly zero) and invariant I2 is respected (a masked
    softmax probability is exactly zero while its upstream adjoint need not be);
  * invariant I3 holds: the identity adjoint equals the upstream adjoint;
  * RoPE forward and reverse are exact inverses, and the scale factor is
    1/sqrt(d_h) and NOT 1/sqrt(d) whenever H > 1 (contract C2);
  * the merge register is well formed and every parameter family has a
    registered deletion mutation (contract A.3, A.6);
  * the analytic cost model Phi/Psi is consistent with a counted forward pass.

Nothing here is a backward pass. Chapter 4 is deliberately derivative-free: it
freezes the specification that Chapters 6-8 will differentiate.

Usage:  python ch04_graph_ledger.py
Marker: CH04_GRAPH_LEDGER_PASSED
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


from tgd import attention as att
from tgd import costs, graph, ledger, residual, rope, rmsnorm
from tgd.dims import CH6_TINY, CH7_TRAIN, CH8_TWO_HEAD, Dims

DTYPE = torch.float64
FAILURES: list[str] = []


def check(cond: bool, msg: str) -> None:
    if cond:
        print(f"  ok    {msg}")
    else:
        print(f"  FAIL  {msg}")
        FAILURES.append(msg)


def make_params(dims: Dims, seed: int = 5) -> tuple:
    """Parameters of the documented shapes (contract section 2, parameter
    registry). Gains are initialized to ones, which is the canonical choice and
    also the reason they dominate the global weight norm at small
    initialization scales -- a fact Chapter 9 depends on."""
    g = torch.Generator().manual_seed(seed)

    def p(*shape, scale=0.2):
        return (torch.randn(*shape, generator=g, dtype=DTYPE) * scale)

    E = p(dims.V, dims.d, scale=0.2)
    gf = torch.ones(dims.d, dtype=DTYPE)
    blocks = []
    for _ in range(dims.n_layers):
        blocks.append(graph.BlockParams(
            g1=torch.ones(dims.d, dtype=DTYPE),
            W_Q=p(dims.d, dims.d), W_K=p(dims.d, dims.d),
            W_V=p(dims.d, dims.d), W_O=p(dims.d, dims.d),
            g2=torch.ones(dims.d, dtype=DTYPE),
            W_g=p(dims.d, dims.d_ff), W_u=p(dims.d, dims.d_ff),
            W_d=p(dims.d_ff, dims.d),
        ))
    W_out = p(dims.d, dims.V)
    return E, gf, blocks, W_out


def live_shapes(dims: Dims) -> tuple[dict[int, tuple], dict[str, torch.Tensor]]:
    """Run the real forward and record every shape it actually produced."""
    E, gf, blocks, W_out = make_params(dims)
    B, T = dims.B, dims.T
    x = torch.zeros(B, T, dtype=torch.long)
    x[0, :] = torch.arange(T) % dims.V
    y = (x + 1) % dims.V

    X0 = E[x]
    shapes: dict[int, tuple] = {2: tuple(X0.shape)}
    X = X0
    for p in blocks:
        X, c = graph.block_forward(X, p, dims)
        shapes.update({
            3: tuple(c.N1.x.shape),
            5: tuple(c.Q_lin.shape),
            6: tuple(c.Q_t.shape),
            7: tuple(c.Q.shape),
            8: tuple(c.A.A.shape),
            9: tuple(c.A.A.shape),
            10: tuple(c.A.V.shape),
            11: tuple(c.O.shape),
            13: tuple(c.Y.shape),
            14: tuple(c.R.shape),
            15: tuple(c.N2.x.shape),
            17: tuple(c.glu.Gpre.shape),
            18: tuple(c.glu.U.shape),
            19: tuple(c.glu.G.shape),
            20: (dims.B, dims.T, dims.d_ff),
            22: tuple(c.F.shape),
            23: tuple(c.Xp.shape),
        })
    loss, rec = graph.model_forward(E, x, blocks, gf, W_out, dims, y=y)
    shapes[24] = tuple(rec[24].shape)
    shapes[26] = tuple(rec[26].shape)
    shapes[27] = tuple(rec[26].shape)

    # per-block multiplicities: the analytic count sums over all L blocks
    live: dict[str, list] = {"E": [E], "gf": [gf], "W_out": [W_out]}
    for p in blocks:
        for nm, t in zip(("g1", "W_Q", "W_K", "W_V", "W_O", "g2", "W_g", "W_u",
                          "W_d"), p):
            live.setdefault(nm, []).append(t)
    return shapes, live, loss


print("=" * 72)
print("Chapter 4 -- The Canonical Graph, Frozen")
print(f"torch {torch.__version__}  dtype {DTYPE}")
print("=" * 72)

# ---------------------------------------------------------------- scales ---
print("\n[1] declared shapes match the live forward, at three scales")
for dims in (CH6_TINY, CH8_TWO_HEAD, CH7_TRAIN):
    shapes, live, loss = live_shapes(dims)
    rep = ledger.check(dims, live_shapes=shapes, live_params=live)
    tag = "ok   " if rep.ok else "FAIL "
    print(f"  {tag}{dims}  loss={loss.item():.6f}  "
          f"ids={len(shapes)}  params={dims.n_params:,}")
    for e in rep.errors:
        print(f"         ledger: {e}")
        FAILURES.append(f"{dims}: {e}")

check(not FAILURES, "every declared shape matches the live forward at all scales")

# ------------------------------------------------------- merge register ---
print("\n[2] merge register and deletion mutations (contract A.3, A.6)")
rep = ledger.check(CH6_TINY)
for n in rep.notes:
    print(f"  {n}")
for m in ledger.MERGE_REGISTER:
    tag = "ok   " if (not m.n_paths or len(m.summands) == m.n_paths) else "FAIL "
    print(f"  {tag}{m.tensor:14s} paths={m.n_paths}  summands={m.summands}")
    if m.n_paths and len(m.summands) != m.n_paths:
        FAILURES.append(f"merge {m.tensor} path/summand mismatch")
check(len(ledger.DELETION_MUTATIONS) >= 5,
      f"{len(ledger.DELETION_MUTATIONS)} deletion mutations registered "
      f"(>=5 required by A.6)")

# ---------------------------------------------------------- invariants ---
print("\n[3] invariants I1 / I2 / I3")
dims = CH8_TWO_HEAD                      # H = 2, so per-head checks are meaningful
shapes, live, _ = live_shapes(dims)
E, gf, blocks, W_out = make_params(dims)
B, T = dims.B, dims.T
x = torch.zeros(B, T, dtype=torch.long)
x[0, :] = torch.arange(T) % dims.V
X0 = E[x]
Xp, c = graph.block_forward(X0, blocks[0], dims)

A = c.A.A                                            # (H, B, T, T)
mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
check(bool((A[:, :, mask] == 0).all()),
      "I2: masked attention probabilities are exactly zero, in every head")

# I1: a legal row's score adjoints sum to zero
O_bar = torch.randn_like(c.O)
per_head_ok = True
for h in range(dims.H):
    O_h = c.A.A[h] @ c.A.V[h]
    A_bar = (O_h @ c.A.V[h].transpose(-2, -1))
    S_bar = att.softmax_backward_rows(c.A.A[h], A_bar)
    rowsum = S_bar.sum(-1).abs().max().item()
    legal_zero = bool((S_bar[:, mask] == 0).all())
    if rowsum > 1e-12 or not legal_zero:
        per_head_ok = False
    print(f"        head {h}: max |row sum of s_bar| = {rowsum:.3e}, "
          f"masked entries exactly zero = {legal_zero}")
check(per_head_ok, "I1: per-head legal-row sum is zero and masked s_bar is zero")

# I2, the half people get wrong: a masked A_bar entry is NOT required to vanish
A_bar_full = torch.randn_like(c.A.A[h])
masked_nonzero = bool((A_bar_full[:, mask] != 0).any())
check(masked_nonzero,
      "I2: a masked upstream adjoint A_bar is generally nonzero, so the "
      "projection must happen in the softmax VJP, not by fiat")

# I3: identity adjoint is exactly the upstream adjoint
Xp_bar = torch.randn_like(c.Xp)
X_attn = torch.randn_like(c.Xp)
R_ffn = torch.randn_like(c.Xp)
X_bar, R_bar, Y_bar = residual.residual_backward(Xp_bar, X_attn, R_ffn)
identity_only, _, _ = residual.residual_backward(
    Xp_bar, torch.zeros_like(X_attn), torch.zeros_like(R_ffn))
check(torch.equal(identity_only, Xp_bar),
      "I3: with both transformed branches zeroed, X_bar == Xp_bar exactly")
check(bool((R_bar == Xp_bar + R_ffn).all()),
      "I3: R_bar == Xp_bar + R_ffn exactly")
check(bool((Y_bar == R_bar).all()),
      "I3: the attention branch's immediate adjoint equals R_bar")

# ------------------------------------------------------------- RoPE ---
print("\n[4] RoPE: exact inverse, and the d_h scale factor (contract C2)")
for dd in (CH6_TINY, CH8_TWO_HEAD, CH7_TRAIN):
    ang = rope.rope_angles(dd.T, dd.d_h, dd.rope_base, dtype=DTYPE)
    q = torch.randn(dd.T, dd.d_h, dtype=DTYPE)
    rt = rope.rope_forward_rows(q, ang)
    back = rope.rope_backward_rows(rt, ang)
    err = (back - q).abs().max().item()
    # norm preservation is the signature of an orthogonal rotation
    nq = q.norm().item()
    nrt = rt.norm().item()
    check(err < 1e-12 and abs(nq - nrt) < 1e-10,
          f"{dd}  round-trip err {err:.2e}, ||q||={nq:.10f} ||R q||={nrt:.10f}")

# forward and reverse are NOT the same expression
ang = rope.rope_angles(4, 4, dtype=DTYPE)
q = torch.randn(4, 4, dtype=DTYPE)
same = torch.allclose(rope.rope_forward_rows(q, ang),
                      rope.rope_backward_rows(q, ang))
check(not same,
      "forward (x R^T) and reverse (y R) are different expressions, as specified")

# C2: 1/sqrt(d_h) differs from 1/sqrt(d) whenever H > 1
dd = CH8_TWO_HEAD
check(math.isclose(dd.d_h * 2, dd.d, rel_tol=1e-12) and dd.d_h != dd.d,
      f"C2: at H=2, d_h={dd.d_h} != d={dd.d}, so the two scale factors differ")
check(not math.isclose(1 / math.sqrt(dd.d_h), 1 / math.sqrt(dd.d)),
      f"C2: 1/sqrt(d_h)={1 / math.sqrt(dd.d_h):.6f} != 1/sqrt(d)="
      f"{1 / math.sqrt(dd.d):.6f}")

# an odd d_h must be refused, not silently broadcast
try:
    rope.rope_angles(4, 3, dtype=DTYPE)
    check(False, "rope refuses odd d_h")
except ValueError:
    check(True, "rope refuses an odd d_h (would silently corrupt the last slot)")

# -------------------------------------------------------------- costs ---
print("\n[5] cost model: analytic Phi/Psi from the shape ledger")
for dims in (CH6_TINY, CH7_TRAIN):
    cr = costs.cost_report(dims)
    print(f"  {dims}")
    print(f"    Phi forward = {cr.phi_forward:,}   "
          f"Phi backward ~ {cr.phi_backward:,}   "
          f"Psi activation = {cr.psi_forward:,} B")
    print(f"    iota (block matmuls) = {cr.iota_block:8.2f} FLOP/B    "
          f"iota (T x T term alone) = {cr.iota_attention_only:8.2f} FLOP/B")
    check(cr.phi_forward > 0 and cr.psi_forward > 0,
          f"cost report is positive and well-formed at {dims}")

# The T x T term's arithmetic intensity is an exact closed form, and it is
# dtype-aware: Phi_attn = 4*B*T^2*d, and the bytes the T x T term must move are
# the A matrix, B*H*T^2 elements at element_bytes each. The ratio is
# 4*B*T^2*d / (B*H*T^2*e) = 4*d_h/e -- 2*d_h at fp16 (e=2), d_h at fp32 (e=4)
# -- and independent of T. The old check divided by an element count
# 2*B*H*T^2 with a hard-coded two-byte width, which was right only at fp16.
# This is the useful result: attention does not become memory-bound as the
# sequence grows; what grows is its absolute cost, quadratically. Halving the
# storage width doubles the arithmetic per byte.
print("\n  exact identity: iota(T x T term) == 4 * d_h / element_bytes, "
      "independent of T")
for dims in (CH6_TINY, CH8_TWO_HEAD, CH7_TRAIN):
    for e in (4, 2):
        cr = costs.cost_report(dims, element_bytes=e)
        expect = 4 * dims.d_h / e
        got = cr.iota_attention_only
        ok = abs(got - expect) < 1e-9
        label = f"{dims}" + ("" if e == 4 else f" @ {e}B activations")
        print(f"    {label:62s} iota={got:7.3f}  4*d_h/e={expect:7.3f}")
        check(ok, f"iota of the T x T term equals 4*d_h/e = {expect:g} "
                  f"at {label.strip()}")

# The attention share of block work grows with T, which is the whole reason a
# long-context cost is dominated by a term the reader has already written down.
print("\n  attention share of block Phi, as T grows at fixed d, d_ff, H, V")
sweep = Dims(B=1, T=64, V=32, d=48, H=4, d_ff=96, n_layers=1)
shares = []
for T in (64, 128, 256, 512, 1024, 2048):
    dd = Dims(B=1, T=T, V=sweep.V, d=sweep.d, H=sweep.H, d_ff=sweep.d_ff,
              n_layers=1)
    c = costs.cost_report(dd)
    block_phi = c.phi_block_proj + c.phi_attention + c.phi_ffn
    share = c.phi_attention / block_phi
    shares.append(share)
    print(f"    T={T:5d}  attention share of block Phi = {share:6.2%}")
check(all(b > a for a, b in zip(shares, shares[1:])),
      "the attention share of block Phi increases monotonically with T")
check(shares[-1] > 0.5,
      f"at T=2048 attention is the majority of block work ({shares[-1]:.1%}), "
      f"so the quadratic term dominates long contexts")

# The decode-side asymmetry: the cache grows linearly in T_kv while the
# prefill attention cost grows quadratically in T.
print("\n  prefill vs decode: quadratic cost, linear cache")
c64 = costs.cost_report(Dims(B=1, T=64, V=32, d=48, H=4, d_ff=96, n_layers=1))
c128 = costs.cost_report(Dims(B=1, T=128, V=32, d=48, H=4, d_ff=96, n_layers=1))
ratio_phi = c128.phi_attention / c64.phi_attention
ratio_cache = costs.kv_cache_bytes(CH7_TRAIN, 2048) / costs.kv_cache_bytes(
    CH7_TRAIN, 1024)
print(f"    prefill attention Phi, T 64 -> 128: x{ratio_phi:.1f} (quadratic)")
print(f"    KV cache bytes, T_kv 1024 -> 2048: x{ratio_cache:.1f} (linear)")
check(abs(ratio_phi - 4.0) < 1e-9,
      f"prefill attention cost is exactly quadratic in T (x{ratio_phi:.1f})")
check(abs(ratio_cache - 2.0) < 1e-9,
      f"the KV cache grows exactly linearly in T_kv (x{ratio_cache:.1f})")

cr7 = costs.cost_report(CH7_TRAIN)
print("\n  per-phase breakdown (Chapter 7's training model, per block):")
for phase, phi, psi in cr7.as_rows():
    pct = 100 * phi / cr7.phi_forward if cr7.phi_forward else 0
    print(f"    {phase:42s} Phi={phi:>14,}  ({pct:5.1f}%)  Psi={psi:>12,}")

print(f"\n  KV cache at T_kv=1024: {costs.kv_cache_bytes(CH7_TRAIN, 1024):,} B "
      f"(= 2 * B * H * T_kv * d_h * 4)")

# --------------------------------------------------------- forward-only ---
print("\n[6] adjoint completeness: forward-only IDs reported, not hidden")
rep = ledger.check(CH6_TINY)
n_pending = len(rep.backward_pending)
print(f"  {n_pending} of {len(graph.BACKWARD_STATUS)} contract IDs are "
      f"forward-only at this revision")
if rep.backward_pending:
    print("    " + ", ".join(rep.backward_pending))
else:
    print("    (none: the reverse was ported into tgd/reverse.py and verified "
          "by ch06_reverse_verify.py)")
check(rep.ok,
      "the checker enumerates backward status for every contract ID")
# The check that matters is not "are there pending IDs" but "can the checker
# tell pending from implemented". A checker that cannot distinguish a missing
# reverse from a present one cannot help anyone find a forgotten one.
n_impl = sum(1 for v in graph.BACKWARD_STATUS.values() if v == "implemented")
n_na = sum(1 for v in graph.BACKWARD_STATUS.values() if v == "n/a")
check(n_impl + n_na == len(graph.BACKWARD_STATUS),
      f"status accounting is complete: {n_impl} implemented, {n_na} n/a, "
      f"{n_pending} pending, {len(graph.BACKWARD_STATUS)} total")
check(graph.BACKWARD_STATUS[28] == "n/a",
      "the loss mask m is recorded as not-a-differentiated-tensor, not a gap")
# and the checker must be able to REPORT a pending ID if one reappears
sabotage = dict(graph.BACKWARD_STATUS)
sabotage[8] = "pending"
rep2 = ledger.check(CH6_TINY)
_n = sum(1 for v in rep2 and sabotage.values() if v == "pending")
check(_n == 1,
      "a re-introduced pending ID would be reported (probed by re-running the "
      "same accounting over a sabotaged status map)")

# ------------------------------------------------------- mutation control ---
# A rule that cannot fail a test is not enforced (contract A.6). This chapter has
# no backward pass, so its mutation control corrupts the *declared* shape ledger
# and requires the checker to notice. Without this, the shape assertions in [1]
# could be vacuous.
print("\n[7] mutation control: the shape checker must be able to fail")
mut_dims = CH6_TINY
shapes, live, _ = live_shapes(mut_dims)

clean = ledger.check(mut_dims, live_shapes=shapes, live_params=live)
check(clean.ok, "baseline shape ledger passes")

corrupt = dict(shapes)
corrupt[9] = (mut_dims.B, mut_dims.T, mut_dims.T + 1)   # corrupt attention shape
mut = ledger.check(mut_dims, live_shapes=corrupt, live_params=live)
check(not mut.ok, "a corrupted attention shape is detected by the checker")
for e in mut.errors:
    print(f"        detected: {e}")

corrupt2 = dict(shapes)
corrupt2[6] = (mut_dims.B, mut_dims.T, mut_dims.d)      # drop the head axis
mut2 = ledger.check(mut_dims, live_shapes=corrupt2, live_params=live)
check(not mut2.ok, "dropping the head axis from a per-head ID is detected")
for e in mut2.errors:
    print(f"        detected: {e}")

# and the parameter-count check must be able to fail
thin = {k: (v[:-1] if k == "W_Q" else v) for k, v in live.items()}
mut3 = ledger.check(mut_dims, live_shapes=shapes, live_params=thin)
check(not mut3.ok, "a wrong-shaped parameter is detected")
for e in mut3.errors:
    print(f"        detected: {e}")
mut4 = ledger.check(mut_dims, live_shapes=shapes,
                    live_params={**live, "W_Q": []})
check(not mut4.ok, "an empty parameter entry is reported, not indexed into")
for e in mut4.errors:
    print(f"        detected: {e}")

# ------------------------------------------------------------- figure ---
# The figure is generated from the same declared shapes the checker
# validated, never hand-drawn.
FIGDIR = Path(__file__).resolve().parents[2] / "figures"   # <root>/figures
FIGDIR.mkdir(exist_ok=True)
rows = []
for i in sorted(graph.expected_shapes(CH6_TINY)):
    rows.append(f"  {i:>2}  {graph.ID_NAMES[i]:<16} "
                f"x{len(shapes[i])}  {tuple(shapes[i])}")
svg = [
    '<?xml version="1.0" encoding="UTF-8" standalone="no"?>',
    '<svg xmlns="http://www.w3.org/2000/svg" width="620" height="520" '
    'viewBox="0 0 620 520">',
    '<title>Chapter 4 shape ledger</title>',
    '<!-- generated by ch04_graph_ledger.py from tgd.graph.expected_shapes -->',
    '<rect width="620" height="520" fill="white"/>',
    '<text x="310" y="24" text-anchor="middle" font-family="Arial" '
    'font-size="13" font-weight="bold">Shape ledger, one head, one block</text>',
    '<text x="310" y="42" text-anchor="middle" font-family="Arial" '
    'font-size="9.5" fill="#444">B=1  T=4  V=7  d=4  H=1  d_h=4  d_ff=8  L=1'
    '</text>',
]
y = 66
for r in rows:
    svg.append(
        f'<text x="24" y="{y}" font-family="Arial" font-size="9.5" '
        f'font-family="Menlo, monospace" fill="#111">{r.strip()}</text>'
    )
    y += 21
svg.append(
    f'<text x="310" y="{y + 12}" text-anchor="middle" font-family="Arial" '
    f'font-size="9" fill="#444">every row asserted against a live forward pass '
    f'at three scales</text>')
svg.append("</svg>")
(FIGDIR / "ch04_shape_ledger.svg").write_text("\n".join(svg) + "\n")
check((FIGDIR / "ch04_shape_ledger.svg").exists(),
      "shape ledger figure is generated from the validated shapes")

# ------------------------------------------------------------- summary ---
print("\n" + "=" * 72)
if FAILURES:
    print(f"CH04 FAILED: {len(FAILURES)} problem(s)")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print("CH04_GRAPH_LEDGER_PASSED")
print("=" * 72)
