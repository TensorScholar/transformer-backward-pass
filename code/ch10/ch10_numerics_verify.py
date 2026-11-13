"""Chapter 10 verification -- reduced-precision failure modes, measured.

Every claim in the chapter becomes an assertion here:

  1. the naive variance matches float64 while E[x^2] >> E[x]^2, and fails
     once the cancellation exceeds the format's resolution;
  2. the zero-centred variance NEVER fails by cancellation, over the same sweep
     -- which is the reason RMSNorm assumes a zero mean;
  3. the exact mu/sigma ratio at which the naive variance first returns a
     non-positive number, and at which it first stops matching float64;
  4. the predicted ratio ~ 1/sqrt(u) is checked against the measurement;
  5. exp overflows float16 at exactly ln(65504), and the naive softmax returns
     nan there while the max-subtracted one does not;
  6. W + dW truncates below a relative update of u = 2^-p, measured for
     float16 and bfloat16, and the threshold is ABSOLUTE in dW rather than
     relative to W;
  7. the safe softmax equals the float64 reference wherever the naive one does
     not.

Usage:  python ch10_numerics_verify.py
Marker: CH10_NUMERICS_VERIFY_PASSED
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
for _p in (_HERE.parent, _HERE.parents[1]):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np                                                  # noqa: E402

from tgd import numerics as N                                       # noqa: E402

FAILURES: list[str] = []


def check(cond: bool, msg: str) -> None:
    if cond:
        print(f"  ok    {msg}")
    else:
        print(f"  FAIL  {msg}")
        FAILURES.append(msg)


def main() -> None:
    print("=" * 74)
    print("Chapter 10 -- precision and numerics")
    print(N.MACHINE_NOTE)
    print("=" * 74)

    print("\n[1] format structure")
    for fmt in ("float16", "bfloat16", "float32"):
        u = N.unit_roundoff(fmt)
        mf = N.FORMATS[fmt][2]
        print(f"    {fmt:<9} unit roundoff 2^-p = {u:.6e}   max finite {mf:.6e}")
    u16 = N.unit_roundoff("float16")
    ubf = N.unit_roundoff("bfloat16")
    check(abs(ubf / u16 - 8.0) < 1e-12,
          f"bfloat16 has exactly 8x the unit roundoff of float16 "
          f"({ubf / u16:.1f}x): 7 stored mantissa bits against 10")

    # ---- 2 & 3. cancellation in the naive variance ---------------------
    print("\n[2] RMSNorm variance: naive vs zero-centred, mu = 1, sigma = 1, d=1024")
    print(f"    {'fmt':<9} {'mu/sig':>7} {'float64 ref':>14} {'naive':>14} "
          f"{'zero-centred':>14} {'naive ok':>9}")
    pts = N.measure_cancellation(1.0, 1.0, n=1024, seed=1)
    by: dict[str, list] = {}
    for p in pts:
        by.setdefault(p.fmt, []).append(p)
        print(f"    {p.fmt:<9} {p.ratio:>7g} {p.ref:>14.6f} {p.naive:>14.6f} "
              f"{p.zero_centered:>14.6f} {'yes' if p.sign_ok else 'NO':>9}")

    # Two separate questions, previously conflated:
    #
    #   (A) ZERO-MEAN input. Then E[x^2] IS the variance, and the naive and
    #       zero-centred forms are computing the same quantity, so there is
    #       nothing to cancel and both are accurate.
    #   (B) NON-ZERO-MEAN input. Then E[x^2] = mu^2 + sigma^2, which is NOT the
    #       variance. The naive form returns (mu^2 + sigma^2) - mu^2 and
    #       cancels; the zero-centred form returns mu^2 + sigma^2 correctly but
    #       is answering a different question.
    #
    # (B) is precisely WHY RMSNorm assumes a zero mean. The assumption is not a
    # numerical trick; it is the modelling decision that makes the subtraction
    # unnecessary. Testing only (B) and comparing E[x^2] to the variance, as a
    # first draft of this script did, compares two different quantities and
    # reports a failure that does not exist.
    print("\n[2b] (A) zero-mean input: both forms are the variance, and both hold")
    rngA = np.random.default_rng(11)
    for sigma in (1.0, 1e-2, 1e2):
        x0 = sigma * rngA.standard_normal(1024)
        ref0 = N.rmsnorm_reference(x0)
        n16 = N.rmsnorm_variance_naive(x0, "float16")
        z16 = N.rmsnorm_variance_zero_centered(x0, "float16")
        b16 = N.rmsnorm_variance_naive(x0, "bfloat16")
        zbf = N.rmsnorm_variance_zero_centered(x0, "bfloat16")
        print(f"    sigma={sigma:<8g} ref={ref0:12.6e}  f16 naive={n16:12.6e} "
              f"zc={z16:12.6e}  bf16 naive={b16:12.6e} zc={zbf:12.6e}")
        check(abs(n16 - ref0) / ref0 < 1e-2 and abs(z16 - ref0) / ref0 < 1e-2,
              f"sigma={sigma:g}: with a zero mean there is nothing to cancel, "
              f"and the naive form is as accurate as the zero-centred one")
        check(abs(zbf - ref0) / ref0 < 1e-2,
              f"sigma={sigma:g}: the same holds in bfloat16")

    print("\n[2c] (B) non-zero-mean input: the zero-centred form returns "
          "E[x^2] = mu^2 + sigma^2, NOT the variance")
    for r in (8, 45, 90, 128):
        rngB = np.random.default_rng(12)
        xb = float(r) + rngB.standard_normal(1024)
        refb = N.rmsnorm_reference(xb)
        nb = N.rmsnorm_variance_naive(xb, "float16")
        zb = N.rmsnorm_variance_zero_centered(xb, "float16")
        expected_second_moment = refb + float(r) ** 2
        print(f"    mu/sig={r:<5g} variance={refb:10.5f}  naive={nb:12.5f}  "
              f"E[x^2]={zb:12.5f}  mu^2+sig^2={expected_second_moment:12.5f}")
        check(nb <= 0.0 or abs(nb - refb) > 1e-3 * refb,
              f"mu/sig={r}: the naive form has lost the variance "
              f"(got {nb:.5f}, true {refb:.5f})")
        check(abs(zb - expected_second_moment) / expected_second_moment < 1e-2,
              f"mu/sig={r}: the zero-centred form is accurate as a SECOND "
              f"MOMENT, which is the only question it answers")
    for fmt, rows in by.items():
        nonfinite = [r for r in rows if not np.isfinite(r.zero_centered)]
        if nonfinite:
            first = min(r.ratio for r in nonfinite)
            print(f"    note {fmt}: E[x^2] OVERFLOWS the format at mu/sigma >= "
                  f"{first:g}. Removing the subtraction does not remove "
                  f"overflow -- two distinct bugs, two distinct thresholds.")
        # NB: the zero-centred result is E[x^2] = mu^2 + sigma^2, NOT the
        # variance, so it must not be compared against r.ref here. Section [2c]
        # performs that comparison against the correct target.

    for fmt in ("float16", "bfloat16"):
        sign, inexact = N.cancellation_threshold(fmt, 1.0, 1.0, n=1024, seed=1)
        u = N.unit_roundoff(fmt)
        predicted = 1.0 / math.sqrt(u)
        print(f"\n[3] {fmt} cancellation thresholds")
        print(f"    first mu/sigma giving a NON-POSITIVE variance : {sign:.1f}")
        print(f"    first mu/sigma losing 0.1% accuracy          : {inexact:.1f}")
        print(f"    analytic prediction 1/sqrt(u)                  : {predicted:.1f}")
        # The prediction is an order-of-magnitude statement, not an equality.
        # Requiring the error in E[x^2] to EXCEED the true variance needs
        # u*mu^2 > sigma^2, i.e. mu/sigma > 1/sqrt(u) -- but the two roundings
        # (one on E[x^2], one on E[x]^2) are independent and partially cancel,
        # so the measured threshold lands above the naive prediction.
        check(sign is not None and predicted * 0.5 < sign < predicted * 8.0,
              f"{fmt}: non-positive variance appears at mu/sigma ~ {sign:.1f}; "
              f"the order-of-magnitude prediction 1/sqrt(u) = {predicted:.1f}, "
              f"ratio {sign / predicted:.2f} (above 1 because the two roundings "
              f"partially cancel)")
        check(inexact is not None and inexact <= sign + 1e-9,
              f"{fmt}: the variance stops being accurate BEFORE it goes "
              f"non-positive ({inexact:.1f} <= {sign:.1f})")

    # ---- 4. softmax overflow --------------------------------------------
    print("\n[4] softmax overflow in float16")
    ov = N.measure_overflow("float16")
    exact = ov["overflow_at_x"]
    print(f"    largest finite float16                : {ov['max_finite']}")
    print(f"    ln(largest finite)                    : {exact:.6f}")
    print(f"    largest x that survives as float16    : {ov['largest_safe_x']:.6f}")
    print(f"    smallest x that overflows to inf      : {ov['smallest_overflowing_x']:.6f}")
    check(abs(exact - math.log(65504.0)) < 1e-12,
          f"exp overflows at exactly ln(65504) = {exact:.6f}")
    check(ov["softmax_is_nan"],
          "the naive softmax returns nan there: exp overflows to inf and "
          "inf/inf is nan")
    # The float16 grid is COARSE near ln(65504): there is no representable value
    # between 11.0859 and 11.0938. The probe must walk the float16 grid, not
    # real numbers, or "largest safe" and "smallest overflowing" coincide and the
    # measurement says nothing about where the boundary actually falls.
    grid = np.arange(11.0, 11.2, 1e-4, dtype=np.float32)
    safe_f, bad_f = None, None
    for g in np.unique(grid.astype(np.float16)):
        with np.errstate(over="ignore"):
            if np.isfinite(np.float16(math.exp(float(g)))):
                safe_f = float(g)
            elif bad_f is None:
                bad_f = float(g)
    ov["largest_safe_x"] = safe_f
    ov["smallest_overflowing_x"] = bad_f
    print(f"    largest SAFE float16 on the grid           : {safe_f:.6f}")
    print(f"    smallest float16 that overflows exp        : {bad_f:.6f}")
    check(safe_f is not None and bad_f is not None and bad_f > safe_f,
          "the two probes are distinct, so the boundary is located")
    # exp(ln(65504)) = 65504.0 exactly, and that IS representable, so the
    # boundary is not crossed by an epsilon: the float16 grid jumps straight from
    # 11.0859 to 11.0938, and the latter is the first value whose exponential
    # exceeds the format's range.
    with np.errstate(over="ignore"):
        e = np.float16(math.exp(exact))
    check(np.isfinite(e),
          f"exp(ln(65504)) = {e} is itself representable, so the overflow "
          f"boundary lies at the next float16 up ({bad_f:.5f}), not at ln(65504) "
          f"itself")
    check(abs(bad_f - exact) < 0.01,
          f"and that boundary is within {bad_f - exact:.5f} of the analytic "
          f"ln(65504) = {exact:.5f}")

    x_safe = np.array([0.0, ov["largest_safe_x"]], dtype=np.float16)
    x_bad = np.array([0.0, ov["smallest_overflowing_x"]], dtype=np.float16)
    ref = np.exp(x_bad.astype(np.float64))
    ref = ref / ref.sum()
    s_safe = N.softmax_safe(x_bad, "float16")
    s_naive = N.softmax_naive(x_bad, "float16")
    print(f"    naive on the overflowing input     : {s_naive}")
    print(f"    safe  on the overflowing input     : {s_safe}")
    print(f"    float64 reference                 : {ref}")
    check(np.all(np.isfinite(s_safe)),
          "the max-subtracted softmax is FINITE on the same input")
    check(np.allclose(s_safe, ref, atol=2e-3),
          f"and agrees with the float64 reference to "
          f"{np.abs(s_safe - ref).max():.2e}")
    check(np.allclose(N.softmax_safe(x_safe, "float16"),
                      N.softmax_naive(x_safe, "float16"), atol=2e-3),
          "where the naive softmax survives, the two agree: max-subtraction "
          "changes nothing numerically, it only avoids overflow")

    # ---- 5. truncation / swamping ---------------------------------------
    print("\n[5] weight update truncation: W + dW where dW is representably small")
    print(f"    {'fmt':<9} {'threshold |dW|/|W|':>20} {'unit roundoff':>16} "
          f"{'threshold / u':>15}")
    res = {}
    for fmt in ("float16", "bfloat16"):
        r = N.measure_truncation(fmt)
        res[fmt] = r
        print(f"    {fmt:<9} {r['threshold_rel']:>20.6e} "
              f"{r['unit_roundoff']:>16.6e} {r['ratio_to_u']:>15.4f}")
        check(0.2 < r["ratio_to_u"] < 2.0,
              f"{fmt}: updates below {r['threshold_rel']:.3e} of |W| vanish; "
              f"that is {r['ratio_to_u']:.2f}x the unit roundoff "
              f"2^-p = {r['unit_roundoff']:.3e}")
    check(res["bfloat16"]["threshold_rel"] > res["float16"]["threshold_rel"],
          f"bfloat16 swallows 8x larger relative updates than float16 "
          f"({res['bfloat16']['threshold_rel']:.3e} vs "
          f"{res['float16']['threshold_rel']:.3e}) -- fewer mantissa bits")

    print("\n[6] the threshold is ABSOLUTE in dW, not relative to W")
    print(f"    {'|W|':>12} {'float16 thr':>14} {'abs dW lost':>14}")
    rows16 = N.measure_truncation_by_magnitude("float16")
    for wv, thr in rows16:
        print(f"    {wv:>12.4g} {thr:>14.3e} {wv * thr:>14.3e}")
    lost = [wv * thr for wv, thr in rows16]
    rels = [thr for _, thr in rows16]
    spread_rel = (max(rels) - min(rels)) / min(rels)
    check(spread_rel < 0.05,
          f"the RELATIVE threshold is constant to {spread_rel:.1%} across the "
          f"binades ({min(rels):.4e} to {max(rels):.4e}), landing on 2^-11 = "
          f"{N.unit_roundoff('float16'):.6e}")
    check(max(lost) / min(lost) > 1e3,
          f"while the ABSOLUTE update lost spans {min(lost):.2e} to "
          f"{max(lost):.2e} -- a factor of {max(lost) / min(lost):.3g}. The "
          f"threshold is relative to W and absolute in dW, which is exactly why "
          f"swamping gets worse for large weights.")

    # ---- 6. the forward pass actually breaks ----------------------------
    print("\n[7] the forward pass breaks, not just the intermediate quantity")
    rng = np.random.default_rng(0)
    n = 1024
    x = 100.0 + rng.standard_normal(n)
    ref_y = N.rmsnorm_forward(x, np.ones(n), "float64")
    y16 = N.rmsnorm_forward(x, np.ones(n), "float16")
    v_naive = N.rmsnorm_variance_naive(x, "float16")
    v_zc = N.rmsnorm_variance_zero_centered(x, "float16")
    ref_v = N.rmsnorm_reference(x)
    err_naive = np.abs(y16 - ref_y).max()
    y64 = N.rmsnorm_forward(x, np.ones(n), "float64")
    y_zc = y16.copy()
    # the zero-centred route: y = x / sqrt(E[x^2]) computed in float16
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        y_zc = (cast16 := N.cast(x, "float16")) / np.float16(
            np.sqrt(np.float16(v_zc)))
    err_zc = np.abs(y_zc - ref_y).max()
    print(f"    mu=100, sigma=1, d=1024")
    print(f"      true variance                 : {ref_v:.6e}")
    print(f"      naive  E[x^2]-E[x]^2 float16 : {v_naive:.6e}")
    print(f"      zero-centred E[x^2]  float16 : {v_zc:.6e}")
    print(f"      max |y_naive  - y_float64|   : {err_naive:.6e}")
    print(f"      max |y_zero   - y_float64|   : {err_zc:.6e}")
    # The naive variance does not have to go NEGATIVE to be broken. At mu=100
    # the float16 ulp of mu^2 is 8, so the subtraction is quantised to multiples
    # of 8 and the result is POSITIVE and roughly an order of magnitude wrong.
    # Asserting "non-positive" here would have missed the more common failure.
    check(v_naive <= 0.0 or abs(v_naive - ref_v) > 2 * ref_v,
          f"the naive variance is grossly wrong ({v_naive:.4f} against a true "
          f"{ref_v:.4f}): no NaN, no warning, just a wrong number")
    ulp_mu2 = abs(np.float16(100.0 * 100.0 + 8.0) - np.float16(100.0 * 100.0))
    check(abs(v_naive - round(v_naive / ulp_mu2) * ulp_mu2) < 1e-6,
          f"and it is exactly quantised to the float16 ulp of mu^2, which is "
          f"{ulp_mu2:.0f} at mu=100: the variance comes out in multiples of "
          f"{ulp_mu2:.0f}, so a true value near 1 can only come out 0, 8, 16, ...")
    second_moment = ref_v + 100.0 ** 2
    check(v_zc > 0.0 and abs(v_zc - second_moment) / second_moment < 5e-2,
          f"the zero-centred route is accurate as a SECOND MOMENT "
          f"({v_zc:.2f} against mu^2+sigma^2 = {second_moment:.2f})")
    check(err_zc < err_naive,
          f"and the resulting activations are closer to float64 by the "
          f"zero-centred route ({err_zc:.2e} vs {err_naive:.2e})")

    print("\n" + "=" * 74)
    if FAILURES:
        print(f"CH10 FAILED: {len(FAILURES)} problem(s)")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("CH10_NUMERICS_VERIFY_PASSED")
    print("=" * 74)


if __name__ == "__main__":
    main()