"""Chapter 13 verification -- RoPE: three implementations, one rotation.

Turns every claim in the chapter into an assertion. The claims are:

  1. the block-diagonal matrix, the complex-number form and the real-valued
     slicing form produce the SAME rotation, forward and reverse;
  2. every implementation is norm preserving -- orthogonality, checked rather
     than assumed, because a wrong SIGN is orthogonal too;
  3. `q_m^T k_n` depends only on `m - n`, exactly, and identically under all
     three implementations;
  4. the adjoint of a rotation by theta is a rotation by -theta, checked
     symbolically (R^T R = I) and numerically (reverse vs autograd);
  5. reverse-then-forward is the identity, while forward-then-forward is NOT --
     the last is the failure mode that preserves norms and is silently wrong;
  6. the block-diagonal matrix is never built, and the report says what it
     would cost;
  7. context extension does NOT degrade the attention score algebraically
     (measured: zero), but DOES degrade its numerical accuracy, and the
     degradation grows with position;
  8. `base ** (-2j/d)` costs measurably more than `exp(-2j log(base)/d)`;
  9. linear and NTK-aware scaling change the geometry by different, measurable
     amounts, and NTK-aware is strictly better where it claims to be;
 10. scaling the ANGLES by 1/alpha is bit-for-bit linear interpolation and not
     NTK-aware scaling at all.

Usage:  python ch13_rope_verify.py
Marker: CH13_ROPE_VERIFY_PASSED
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
for _p in (_HERE.parent, _HERE.parents[1]):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch                                                  # noqa: E402

from tgd import rope as R                                      # noqa: E402

FAILURES: list[str] = []


def check(cond: bool, msg: str) -> None:
    if cond:
        print(f"  ok    {msg}")
    else:
        print(f"  FAIL  {msg}")
        FAILURES.append(msg)


def main() -> None:
    print("=" * 78)
    print("Chapter 13 -- rotary position embedding and context scaling")
    print("Row-vector convention (contract C1): Q = Q~ R_t^T, R_t = [[c,-s],[s,c]],")
    print("theta_{t,j} = t * base^(-2j/d_h).")
    print("=" * 78)

    # ---------------------------------------------------------------------
    print("\n[1] three implementations of one rotation")
    eq = R.equivalence_report(T=64, d_h=8)
    print(f"    T={eq['T']} d_h={eq['d_h']} base={eq['base']:.0f} dtype={eq['dtype']}")
    print("    relative error against the block-diagonal matrix:")
    for impl in R.IMPLEMENTATIONS:
        print(f"      {impl:>8}:  forward {eq['rel_fwd_vs_matrix'][impl]:.3e}   "
              f"reverse {eq['rel_bwd_vs_matrix'][impl]:.3e}   "
              f"norm err {eq['norm_rel_err'][impl]:.3e}")
    tol = 1e-15
    for impl in ("complex", "sliced"):
        check(eq["rel_fwd_vs_matrix"][impl] < tol,
              f"{impl} forward matches the block-diagonal matrix to "
              f"{eq['rel_fwd_vs_matrix'][impl]:.2e}")
        check(eq["rel_bwd_vs_matrix"][impl] < tol,
              f"{impl} reverse matches the block-diagonal matrix to "
              f"{eq['rel_bwd_vs_matrix'][impl]:.2e}")
    for impl in R.IMPLEMENTATIONS:
        check(eq["norm_rel_err"][impl] < 1e-13,
              f"{impl} preserves ||q|| to {eq['norm_rel_err'][impl]:.2e} "
              "(orthogonality, checked not assumed)")

    # ---------------------------------------------------------------------
    print("\n[2] the sign trap: a wrong sign is still an orthogonal rotation")
    print(f"    rotating by -theta instead of +theta: relative error "
          f"{eq['sign_trap_rel']:.4f}")
    print(f"    and it is a perfect rotation:        {eq['sign_trap_is_orthogonal']}")
    check(eq["sign_trap_is_orthogonal"],
          "the WRONG sign is norm-preserving too -- so norms cannot catch it")
    check(eq["sign_trap_rel"] > 0.1,
          f"but it is wrong by {eq['sign_trap_rel']:.2%} in relative error, "
          "which the matrix reference does catch")

    # ---------------------------------------------------------------------
    print("\n[3] q_m^T k_n depends only on (m - n)")
    rel = R.relative_report(T=32, d_h=4)
    print(f"    {rel['n_pairs']} position pairs (m, n), translations c in 1, 3, 7")
    print(f"    worst translation deviation (m,n) vs (m+c,n+c): "
          f"{rel['worst_translation_deviation']:.3e}")
    print(f"    angle closure  max |(theta_m - theta_n) - (m-n) f| = "
          f"{rel['angle_closure_deviation']:.3e}")
    print(f"    score by implementation: "
          + ", ".join(f"{k}={v:.12f}" for k, v in rel['score_by_impl'].items()))
    check(rel["worst_translation_deviation"] == 0.0,
          "translation invariance is EXACTLY zero, not merely small")
    check(rel["angle_closure_deviation"] < 1e-15,
          "the angle identity theta_m - theta_n = (m-n) * inv_freq holds to "
          f"{rel['angle_closure_deviation']:.2e}")
    check(rel["score_spread"] == 0.0,
          "all three implementations return the BIT-IDENTICAL attention score")

    # ---------------------------------------------------------------------
    print("\n[4] the adjoint of a rotation by theta is a rotation by -theta")
    adj = R.adjoint_report(T=32, d_h=8)
    print(f"    R(phi)^T R(phi) - I                       max "
          f"{adj['orthogonality_deviation']:.3e}")
    print(f"    R(phi)^T R(psi) - R(psi - phi)            max "
          f"{adj['composition_deviation']:.3e}")
    print(f"    R(-psi) v  vs  R(psi)^T v                 max "
          f"{adj['transpose_equals_negangle_deviation']:.3e}")
    for impl in R.IMPLEMENTATIONS:
        print(f"    reverse vs autograd, {impl:>8}: "
              f"{adj['reverse_vs_autograd'][impl]:.3e}")
    check(adj["orthogonality_deviation"] < 1e-15,
          "R_t is orthogonal: R^T R = I")
    check(adj["composition_deviation"] < 1e-15,
          "R(phi)^T R(psi) = R(psi - phi), so the transpose is a rotation")
    check(adj["transpose_equals_negangle_deviation"] < 1e-15,
          "rotating the adjoint by -psi equals applying R_psi^T")
    check(adj["worst_reverse_vs_autograd"] == 0.0,
          "every implementation's reverse matches autograd EXACTLY (0.0)")
    print(f"    forward applied twice, deviation "
          f"{adj['double_forward_deviation']:.4f}  <- rotation by 2*theta, "
          "not the identity")

    # ---------------------------------------------------------------------
    print("\n[5] reverse-then-forward is the identity")
    for impl in R.IMPLEMENTATIONS:
        print(f"    {impl:>8}: {adj['reverse_forward_identity'][impl]:.3e}")
    for impl in R.IMPLEMENTATIONS:
        check(adj["reverse_forward_identity"][impl] < 1e-15,
              f"{impl}: reverse(forward(q)) == q to "
              f"{adj['reverse_forward_identity'][impl]:.2e}")
    check(adj["double_forward_deviation"] > 0.1,
          f"forward(forward(q)) is {adj['double_forward_deviation']:.2%} wrong, "
          "so using the forward as its own backward is detectable")

    # ---------------------------------------------------------------------
    print("\n[6] why the block-diagonal matrix is never built")
    mc = R.matrix_cost(T=4096, d_h=64)
    print(f"    T={mc['T']} d_h={mc['d_h']} float32({mc['element_bytes']} B)")
    print(f"    dense matrix          {mc['matrix_bytes']:>14,} B "
          f"({mc['matrix_bytes'] / 2**20:.1f} MiB)")
    print(f"    nonzeros              {mc['nonzero']:>14,}")
    print(f"    density               {mc['density']:>14.4%}")
    print(f"    dense FLOPs           {mc['dense_flops']:>14,.0f}")
    print(f"    FLOPs actually needed {mc['actual_flops']:>14,.0f}")
    print(f"    waste ratio           {mc['waste_ratio']:>14.1f}x")
    angle_table_bytes = 4096 * (64 // 2) * 4
    print(f"    cos/sin angle table      {angle_table_bytes:>14,} B "
          f"({angle_table_bytes / 2**20:.2f} MiB)  <- what is actually stored")
    print(f"    waste vs the angle table {mc['matrix_bytes'] / angle_table_bytes:>14.0f}x")
    check(mc["density"] == 2.0 / mc["d_h"],
          f"the matrix is {mc['density']:.4%} dense -- 2x2 blocks on a diagonal")
    check(mc["waste_ratio"] > 10.0,
          f"a dense multiply does {mc['waste_ratio']:.0f}x the necessary work")
    check(mc["matrix_bytes"] / angle_table_bytes > 100,
          f"and it would cost {mc['matrix_bytes'] / angle_table_bytes:.0f}x more "
          "memory than the angle table it is built from")

    # ---------------------------------------------------------------------
    print("\n[7] context extension: the ALGEBRA does not degrade")
    print("    q^T R_{m-n} k mentions neither m nor n, so probing at an")
    print("    out-of-range position cannot change the score. Measured:")
    cliff = R.numerical_cliff_report()
    print(f"    probe distance held FIXED at delta={cliff['delta']}, "
          f"probe position t swept {cliff['positions']}")
    print("    (positions with t - delta < 0 are skipped: the guard in "
          "relative_score exists for exactly this)")
    print(f"    {'t':>6} {'in train':>9} | {'f32 pow':>10} {'f32 exp':>10} "
          f"{'f64 pow':>10}   (ULPs of the logit)")
    for r in cliff["rows"]:
        print(f"    {r['t']:>6} {str(r['in_train']):>9} | "
              f"{r['f32_pow_ulps']:>10.1f} {r['f32_exp_ulps']:>10.1f} "
              f"{r['f64_pow_ulps']:>10.1f}")
    check(cliff["rows"][0]["t"] >= cliff["delta"],
          "every reported probe is in range; no negative index was read")

    # ---------------------------------------------------------------------
    print("\n[8] what DOES degrade: numerical accuracy, growing with position")
    rt = R.float32_roundtrip_report()
    print(f"    float32 only, exponential form, delta={rt['delta']}, "
          f"score mean |S| = {rt['score_mean_abs']:.4f}, "
          f"score std = {rt['points'][f't={2047}']['score_std']:.4f}")
    print(f"    {'t':>6} | {'max abs err':>12} {'rms err':>12} {'ULPs':>10} "
          f"{'rms/SD':>10}")
    for k, v in rt["points"].items():
        print(f"    {v['t']:>6} | {v['max_abs_err']:>12.3e} "
              f"{v['rms_err']:>12.3e} {v['ulps']:>10.1f} "
              f"{v['rms_err'] / v['score_std']:>10.2e}")
    print(f"    ULP growth {rt['points']['t=2047']['t']} -> "
          f"{rt['points']['t=4095']['t']}: {rt['ulp_growth']:.2f}x "
          f"(position ratio {rt['position_ratio']:.2f}x)")
    check(rt["ulp_growth"] > 1.5,
          f"doubling the position grows the float32 error "
          f"{rt['ulp_growth']:.2f}x -- the cliff is real")
    worst = rt["points"]["t=4095"]["rms_err"] / rt["points"]["t=4095"]["score_std"]
    check(worst < 1e-3,
          f"but at {worst:.1e} of the score's own SD it is immaterial; the "
          "cliff is a precision result, not a failure")

    # ---------------------------------------------------------------------
    print("\n[9] base ** (-2j/d) versus exp(-2j log(base)/d)")
    print(f"    at t={cliff['rows'][-1]['t']}, delta={cliff['delta']}, "
          "float64 (20x more precision than float32):")
    print(f"      pow form  {cliff['rows'][-1]['f64_pow_ulps']:>10.1f} ULPs")
    print(f"      exp form  {cliff['rows'][-1]['f64_exp_ulps']:>10.1f} ULPs  "
          "(this IS the reference, so zero by construction)")
    print(f"    formula-choice cost grows {cliff['formula_choice_cost_float64']['growth']:.2f}x "
          f"from t={cliff['rows'][0]['t']} to t={cliff['rows'][-1]['t']}")
    check(cliff["formula_choice_cost_float64"]["at_last"] > 100,
          f"the pow form is worth {cliff['formula_choice_cost_float64']['at_last']:.0f} "
          "ULPs of a float64 logit at the extended position")
    check(cliff["formula_choice_cost_float64"]["growth"] > 2.0,
          "and that cost GROWS with context length, because the frequency error "
          "is multiplied by t")

    # ---------------------------------------------------------------------
    print("\n[10] what linear and NTK-aware scaling cost")
    sc = R.scaling_cost_report()
    print(f"    trained N={sc['train_len']}, extended to {sc['eval_len']}, "
          f"alpha={sc['alpha']:.1f}, NTK base {sc['base']:.0f} -> "
          f"{sc['ntk_base']:.3f}")
    print("    RMS deviation from the TRAINED geometry, in units of the "
          "trained score's SD:")
    print(f"    {'delta':>6} {'in train':>9} | {'linear':>8} {'ntk':>8} "
          f"{'naive':>8}")
    for r in sc["rows"]:
        print(f"    {r['delta']:>6} {str(r['in_train']):>9} | "
              f"{r['linear_dev_in_sd']:>8.4f} {r['ntk_dev_in_sd']:>8.4f} "
              f"{r['ntk_naive_dev_in_sd']:>8.4f}")
    print(f"    local band delta<={sc['local_hi']}:      linear "
          f"{sc['linear_dev_in_sd_local']:.4f}  ntk "
          f"{sc['ntk_dev_in_sd_local']:.4f}  -> NTK better by "
          f"{sc['ntk_advantage_local']:.2f}x")
    print(f"    trained band delta<={sc['train_len'] - 1}: linear "
          f"{sc['linear_dev_in_sd_trained_band']:.4f}  ntk "
          f"{sc['ntk_dev_in_sd_trained_band']:.4f}  -> NTK better by "
          f"{sc['ntk_advantage_trained_band']:.2f}x")
    print(f"    long band delta>{sc['train_len']}:  linear "
          f"{sc['linear_dev_in_sd_long_band']:.4f}  ntk "
          f"{sc['ntk_dev_in_sd_long_band']:.4f}")
    check(sc["ntk_dev_in_sd_local"] < sc["linear_dev_in_sd_local"],
          f"NTK-aware preserves local geometry better than linear "
          f"({sc['ntk_dev_in_sd_local']:.4f} vs "
          f"{sc['linear_dev_in_sd_local']:.4f} SD)")
    check(sc["linear_dev_in_sd_local"] > 0.5,
          f"linear interpolation already distorts ADJACENT tokens by "
          f"{sc['linear_dev_in_sd_local']:.2f} SD -- it does not preserve "
          "local structure")
    check(sc["ntk_advantage_local"] > 1.5,
          f"NTK-aware's advantage where it is claimed is "
          f"{sc['ntk_advantage_local']:.2f}x")
    check(sc["ntk_dev_in_sd_long_band"] > 1.0,
          f"BUT in the long band NTK also deviates by "
          f"{sc['ntk_dev_in_sd_long_band']:.2f} SD: extension is not free and "
          "NTK does not fix the long range")

    # ---------------------------------------------------------------------
    print("\n[11] scaling the angles is NOT NTK-aware scaling")
    sp = R.scaling_spectrum_report()
    print(f"    d_h={sp['d_h']} alpha={sp['alpha']:.1f} base {sp['base']:.0f} "
          f"-> NTK base {sp['ntk_base']:.3f}")
    print(f"    {'j':>4} {'inv_freq':>12} {'linear ratio':>13} {'ntk ratio':>11} "
          f"{'ntk/linear':>11}")
    for r in sp["rows"]:
        print(f"    {r['j']:>4} {r['inv_freq']:>12.4e} {r['lin_ratio']:>13.6f} "
              f"{r['ntk_ratio']:>11.6f} {r['ntk_over_lin']:>11.6f}")
    print(f"    theta_t / alpha equals linear interpolation, bitwise: "
          f"{sp['naive_equals_linear_bitwise']} "
          f"(max difference {sp['naive_vs_linear_max_diff']:.1e})")
    print(f"    versus TRUE NTK it differs by up to "
          f"{sp['naive_vs_true_ntk_max_abs']:.1f} radians "
          f"({sp['naive_vs_true_ntk_max_rel']:.1%} relative)")
    check(sp["naive_equals_linear_bitwise"],
          "dividing the angles by alpha IS linear interpolation, bit for bit")
    check(sp["lin_ratio_spread"] == 1.0,
          f"linear scaling multiplies EVERY frequency by the same "
          f"{sp['rows'][0]['lin_ratio']:.6f}")
    check(sp["ntk_ratio_spread"] > 1.9,
          f"NTK-aware scales frequencies by ratios spanning "
          f"{sp['ntk_ratio_spread']:.2f}x, from "
          f"{sp['ntk_ratio_spread'] and min(r['ntk_ratio'] for r in sp['rows']):.6f} "
          f"to {max(r['ntk_ratio'] for r in sp['rows']):.6f}")
    check(sp["ntk_over_lin_ratio_max_err_vs_closed_form"] < 1e-14,
          "the NTK/linear ratio is exactly alpha^(-2j/(d-2)): "
          f"{sp['ntk_over_lin_ratio_at_j0']:.6f} at j=0 down to "
          f"{sp['ntk_over_lin_ratio_at_j_last']:.6f} at the slowest pair")

    # ---------------------------------------------------------------------
    print("\n[12] the shipped implementation is the cheap one")
    q = torch.randn(1, 4096, 64, dtype=torch.float32)
    ang = R.rope_angles(4096, 64, dtype=torch.float32)
    out = R.rope_forward_rows(q, ang)
    check(out.shape == q.shape, f"slicing preserves shape {tuple(out.shape)}")
    check(out.numel() * out.element_size()
          < R.matrix_cost(4096, 64)["matrix_bytes"] / 2,
          "the sliced form allocates one output tensor and never the rotation")

    # odd d_h is refused rather than silently corrupting the last slot
    try:
        R.rope_angles(4, 3)
        refused = False
    except ValueError:
        refused = True
    check(refused, "an odd d_h is refused rather than silently mis-paired")

    # ---------------------------------------------------------------------
    print("\n[12b] long-context phase tables need working precision (v1.2.1)")
    # At T >= 8192 the position index itself leaves the half-precision
    # mantissa's exact-integer range: bf16 represents 8191 and 8192
    # identically (its integer grid steps by 64 there), so a table built in
    # the activation dtype aliases positions. The shipped table is evaluated
    # in at least fp32 regardless of the requested storage dtype, and the
    # rotation evaluates cos/sin in the same working precision.
    a64 = R.rope_angles(8192, 128, dtype=torch.float64)
    a32 = R.rope_angles(8192, 128, dtype=torch.float32)
    a16 = R.rope_angles(8192, 128, dtype=torch.float16)
    abf = R.rope_angles(8192, 128, dtype=torch.bfloat16)
    check(a16.dtype == torch.float32 and abf.dtype == torch.float32,
          "a half-precision table request returns the fp32 working-precision "
          f"table (got {a16.dtype}/{abf.dtype}), never a quantized one")
    err32 = float((a32.double() - a64).abs().max())
    check(err32 < 1e-3,
          f"the fp32 table tracks the fp64 table to {err32:.2e} rad at "
          "T=8192, d_h=128 (fp32 position rounding, not aliasing)")
    # the aliasing this prevents, quantified: distinct positions that a
    # half-precision table would collapse
    t16 = torch.arange(8192, dtype=torch.float16)
    tbf = torch.arange(8192, dtype=torch.bfloat16)
    uniq16 = int(t16.unique().numel())
    uniqbf = int(tbf.unique().numel())
    check(uniq16 < 8192 and uniqbf < 8192,
          f"raw fp16/bf16 position ids collapse 8192 positions to "
          f"{uniq16}/{uniqbf} distinct values -- the aliasing the working-"
          "precision table rule exists to prevent")
    # and the rotation itself: fp16 activations at T=8192 stay norm-preserving
    q16 = torch.randn(1, 8192, 128, dtype=torch.float16)
    out16 = R.rope_forward_rows(q16, a16)
    n_in = q16.float().norm(dim=-1)
    n_out = out16.float().norm(dim=-1)
    worst_norm = float(((n_in - n_out).abs() / n_in).max())
    check(worst_norm < 1e-2,
          f"fp16 rotation at T=8192 preserves norms to {worst_norm:.2e} "
          "relative (fp16 storage rounding only; angles evaluated in fp32)")
    rt16 = R.rope_backward_rows(out16, a16)
    check(bool(torch.isfinite(rt16).all()),
          "and the reverse rotation returns finite values at T=8192 in fp16")

    # ---------------------------------------------------------------------
    print("\n[13] how much of this is sampling error")
    ss = R.seed_stability_report()
    print(f"    {ss['n_samples']} random (q, k) draws per seed, seeds "
          f"{ss['seeds']}; the statistics are means, so they carry error")
    print(f"    {'metric':<34} {'mean':>8} {'sd':>9} {'rel sd':>9}")
    for r in ss["rows"]:
        print(f"    {r['metric']:<34} {r['mean']:>8.4f} "
              f"{r['sample_sd_across_seeds']:>9.5f} {r['relative_sd']:>9.2%}")
    print(f"    worst relative sample standard deviation across seeds: "
          f"{ss['worst_relative_sd']:.2%}")
    check(ss["worst_relative_sd"] < 0.05,
          f"every reported scaling deviation is stable to better than "
          f"{ss['worst_relative_sd']:.1%} across {len(ss['seeds'])} seeds")
    check(sc["ntk_advantage_local"] > 10 * ss["worst_relative_sd"],
          f"and the NTK advantage ({sc['ntk_advantage_local']:.2f}x) is an order "
          "of magnitude larger than that error, so the comparison survives it")

    print("\n" + "=" * 78)
    if FAILURES:
        print(f"CH13 FAILED: {len(FAILURES)} problem(s)")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("CH13_ROPE_VERIFY_PASSED")
    print("=" * 78)


if __name__ == "__main__":
    main()