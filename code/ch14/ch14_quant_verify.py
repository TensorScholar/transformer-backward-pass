r"""Chapter 14 verification -- quantization: the outlier, and the block that traps it.

Turns every claim in the chapter into an assertion:

  1. the bandwidth arithmetic: int8 halves the bytes, so it doubles the speed,
     and group scales forfeit part of that back (counted, not asserted);
  2. absmax quantization round-trips a clean matrix to a small relative MSE,
     bounded in closed form by the int8 step s/2;
  3. ONE element at 50.0 raises the absmax MSE by more than two orders of
     magnitude -- the outlier problem, measured;
  4. the float16 control, so "bad" is measured against the right baseline;
  5. block-wise quantization at g=64 restores the MSE to within a small factor
     of the clean baseline;
  6. the isolation claim, in its correct differential form: introducing the
     outlier changes exactly ONE block, perturbs at most g elements, and leaves
     every other block bit-identical;
  7. group-wise quantization beats absmax for every group size, and the
     penalty ratio is monotone in g;
  8. the casts: round-half-even vs truncate, the all-zero guard, and the
     multiplier/divisor direction of the scale;
  9. a check that is deliberately able to FAIL: a reconstruction with
     relative MSE > 0.5 must raise, so the "scale applied backwards" bug
     cannot be reintroduced silently.

Usage:  python ch14_quant_verify.py
Marker: CH14_QUANT_VERIFY_PASSED
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
for _p in (_HERE.parent, _HERE.parents[1]):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np                                                  # noqa: E402

from tgd import quantize as Q                                       # noqa: E402

FAILURES: list[str] = []


def check(cond: bool, msg: str) -> None:
    if cond:
        print(f"  ok    {msg}")
    else:
        print(f"  FAIL  {msg}")
        FAILURES.append(msg)


def main() -> None:
    print("=" * 78)
    print("Chapter 14 -- quantization: absmax, outliers, and group-wise scales")
    print("Scale is a MULTIPLIER s = 127/max|W|;  q = round(s*w),  w_hat = q/s.")
    print("=" * 78)

    cfg = Q.QuantConfig(rows=64, cols=4096, group_size=64, seed=0)

    # ---------------------------------------------------------------------
    print("\n[1] the bandwidth arithmetic")
    b = Q.bandwidth_report()
    print(f"    {'variant':<26}{'bytes/token':>13}{'speedup':>10}{'intensity':>11}")
    for k, v in b.items():
        if k.startswith("_") or k == "kv_cache_note":
            continue
        print(f"    {k:<26}{v['bytes_per_token']:>13,.0f}"
              f"{v['speedup_vs_float16']:>9.3f}x{v['intensity_vs_float16']:>10.3f}x")
    fp16 = b["float16"]
    i8 = b["int8"]
    check(abs(i8["speedup_vs_float16"] - 2.0) < 1e-12,
          f"int8 is exactly 2.000x the float16 speed "
          f"({i8['speedup_vs_float16']:.3f}x)")
    check(i8["intensity_vs_float16"] > fp16["intensity_vs_float16"],
          f"and arithmetic intensity RISES with it "
          f"({i8['intensity_vs_float16']:.3f}x), which is the direction that "
          "matters for an IO-bound kernel")
    g64 = b["int8_g64_fp32scales"]
    check(g64["speedup_vs_float16"] < 2.0,
          f"group scales forfeit part of the 2x: g=64 with fp32 scales gives "
          f"{g64['speedup_vs_float16']:.3f}x, not 2x")
    check(b["int8_g128_fp16scales"]["speedup_vs_float16"]
          > g64["speedup_vs_float16"],
          f"and larger groups forfeit less: g=128 with fp16 scales "
          f"{b['int8_g128_fp16scales']['speedup_vs_float16']:.3f}x")
    check(b["float16_plus_scales"]["bytes_per_token"] > fp16["bytes_per_token"],
          "even float16 + scales is LARGER than plain float16 -- the control "
          "that stops 'we added scales' being read as free")

    # ---------------------------------------------------------------------
    print("\n[2] absmax on a clean matrix")
    r = Q.outlier_report(cfg)
    clean = r["absmax_clean"]
    w = Q._matrix(cfg)
    s = Q.absmax_scale(w)
    print(f"    matrix {cfg.rows}x{cfg.cols} = {cfg.n_elements:,} elements, "
          f"max|W| = {float(np.abs(w).max()):.4f}")
    print(f"    scale s = {s:.4f}  ->  int8 step 1/s = {1/s:.6f}")
    print(f"    MSE {clean['mse']:.6e}   relative MSE {clean['rel_mse']:.3e}   "
          f"max abs err {clean['max_abs_err']:.4e}")
    # closed form: uniform quantization error is +- step/2, so MSE <= step^2/12
    step = 1.0 / s
    bound = step ** 2 / 12.0
    print(f"    closed-form bound step^2/12 = {bound:.6e}")
    check(clean["rel_mse"] < 1e-3,
          f"clean absmax relative MSE {clean['rel_mse']:.2e} is a real "
          "quantizer, not a no-op and not a zeroer")
    check(clean["mse"] <= bound * 1.5,
          f"clean MSE {clean['mse']:.3e} respects the +-step/2 uniform bound "
          f"{bound:.3e} (the 1.5 slack is for the gaussian tail)")

    # ---------------------------------------------------------------------
    print("\n[3] ONE element at 50.0")
    out = r["absmax_outlier"]
    print(f"    outlier at {r['outlier_index']} (flat {r['outlier_flat_index']}), "
          f"block {r['outlier_block']} of {r['n_blocks']}")
    print(f"    scale collapses {r['absmax_scale_clean']:.4f} -> "
          f"{r['absmax_scale_outlier']:.4f}   "
          f"(shrinks {r['absmax_scale_clean']/r['absmax_scale_outlier']:.2f}x)")
    print(f"    int8 levels left for a 3-sigma weight: "
          f"{r['levels_clean']:.2f} -> {r['levels_outlier']:.2f}")
    print(f"    MSE {out['mse']:.6e}   relative MSE {out['rel_mse']:.3e}   "
          f"max abs err {out['max_abs_err']:.4e}")
    print(f"    ratio to the clean baseline: {r['absmax_mse_ratio']:.1f}x")
    check(r["absmax_scale_outlier"] < r["absmax_scale_clean"] / 10,
          f"the scale SHRINKS {r['absmax_scale_clean']/r['absmax_scale_outlier']:.1f}x, "
          "which is the mechanism (the multiplier s gets smaller, so the int8 "
          "step 1/s gets proportionally coarser)")
    check(r["absmax_mse_ratio"] > 100,
          f"MSE rises {r['absmax_mse_ratio']:.0f}x from ONE element in "
          f"{cfg.n_elements:,} -- the outlier problem, measured")
    check(out["rel_mse"] > 100 * clean["rel_mse"],
          f"relative MSE {clean['rel_mse']:.2e} -> {out['rel_mse']:.2e}")
    check(r["levels_outlier"] < 1.0,
          f"the bulk of the tensor is left {r['levels_outlier']:.2f} of an int8 "
          "level at 3 sigma: most weights now round to a single integer")

    # ---------------------------------------------------------------------
    print("\n[4] the float16 control")
    print(f"    float16 cast of the same outlier tensor: MSE "
          f"{r['float16_mse']:.4e}")
    print(f"    absmax WITH outlier:                     MSE "
          f"{out['mse']:.4e}")
    check(r["float16_mse"] < out["mse"],
          f"absmax with an outlier is {out['mse']/r['float16_mse']:.0f}x worse "
          "than simply casting to float16 -- the comparison that matters")
    check(r["float16_mse"] < clean["mse"],
          "float16 beats clean absmax too, which is why int8 was never free")

    # ---------------------------------------------------------------------
    print("\n[5] block-wise quantization traps the damage")
    bc, bo = r["blocks_clean"], r["blocks_outlier"]
    print(f"    {'scheme':<30}{'MSE':>14}{'rel MSE':>12}{'max abs err':>13}")
    for lbl, m in (("absmax, no outlier", clean), ("absmax, WITH outlier", out),
                   (f"blockwise g={cfg.group_size}, no outlier", bc),
                   (f"blockwise g={cfg.group_size}, WITH outlier", bo)):
        print(f"    {lbl:<30}{m['mse']:>14.6e}{m['rel_mse']:>12.3e}"
              f"{m['max_abs_err']:>13.4e}")
    print(f"    blockwise outlier/clean penalty: {r['blocks_mse_ratio']:.3f}x")
    print(f"    recovery vs absmax-with-outlier:  {r['blocks_recovery_ratio']:.1f}x")
    check(r["blocks_mse_ratio"] < 1.5,
          f"block-wise barely notices the outlier: {r['blocks_mse_ratio']:.3f}x")
    check(r["blocks_recovery_ratio"] > 100,
          f"and recovers {r['blocks_recovery_ratio']:.0f}x of the absmax damage")
    check(bo["mse"] < out["mse"],
          f"block-wise WITH the outlier beats absmax WITH it by "
          f"{out['mse']/bo['mse']:.0f}x")
    check(bc["mse"] < clean["mse"],
          f"and block-wise also beats absmax on the CLEAN tensor "
          f"({clean['mse']:.2e} -> {bc['mse']:.2e}), because a per-block max is "
          "a tighter scale than a per-tensor max")

    # ---------------------------------------------------------------------
    print("\n[6] the isolation claim, in its correct differential form")
    iso = Q.block_isolation_report(cfg)
    print(f"    blocks whose reconstruction CHANGED when the outlier was added: "
          f"{iso['n_blocks_changed_by_outlier']} "
          f"(the outlier is in block {iso['outlier_block']})")
    print(f"    elements perturbed: {iso['n_elements_perturbed']} of "
          f"{iso['n_elements']:,}  (bound g = {iso['perturbation_bound']})")
    print(f"    largest change OUTSIDE that block: "
          f"{iso['max_change_outside_outlier_block']:.1f}")
    print(f"    for contrast, blocks with ANY rounding error at all: "
          f"{iso['blocks_with_any_rounding_error']} of {iso['n_blocks']}")
    print(f"    the outlier block's own error is "
          f"{iso['outlier_block_err_inflation']:.1f}x a median block's")
    check(iso["outlier_changes_exactly_one_block"],
          f"the outlier changes EXACTLY one block, and it is the right one "
          f"({iso['changed_blocks']})")
    check(iso["max_change_outside_outlier_block"] == 0.0,
          "every other block is BIT-IDENTICAL with and without the outlier")
    check(iso["bound_respected"],
          f"at most g={iso['perturbation_bound']} elements are perturbed "
          f"({iso['n_elements_perturbed']}), independent of the outlier's size")
    check(iso["n_elements_perturbed"] <= cfg.group_size,
          "so the damage fraction is bounded by g/N = "
          f"{cfg.group_size/cfg.n_elements:.2e} no matter how large 50.0 is")
    check(iso["blocks_with_any_rounding_error"] == iso["n_blocks"],
          "and note the honest control: rounding error exists in EVERY block, "
          "which is why 'one block damaged' is a differential claim only")

    # ---------------------------------------------------------------------
    print("\n[7] group size sweep")
    rows = Q.group_size_sweep(cfg)
    print(f"    {'g':>5} {'scales':>7} {'overhead':>9} {'clean MSE':>12} "
          f"{'outlier MSE':>13} {'penalty':>9}")
    for row in rows:
        print(f"    {row['group_size']:>5} {row['n_scales']:>7} "
              f"{row['storage_overhead']:>8.2%} {row['clean_mse']:>12.4e} "
              f"{row['outlier_mse']:>13.4e} {row['outlier_penalty_ratio']:>8.3f}x")
    ratios = [r["outlier_penalty_ratio"] for r in rows]
    check(all(rows[i]["outlier_mse"] <= rows[i]["clean_mse"] * 3
              for i in range(len(rows))),
          "no group size lets the outlier do catastrophic damage")
    check(all(ratios[i] <= ratios[i + 1] + 1e-9 for i in range(len(ratios) - 1)),
          f"the outlier penalty is MONOTONE in g "
          f"({ratios[0]:.3f}x at g=1 rising to {ratios[-1]:.3f}x at "
          f"g={rows[-1]['group_size']})")
    check(rows[0]["outlier_mse"] < rows[-1]["outlier_mse"],
          f"and smaller groups are strictly safer: g=1 gives "
          f"{rows[0]['outlier_mse']:.3e} vs g={rows[-1]['group_size']} "
          f"{rows[-1]['outlier_mse']:.3e}")
    check(rows[-1]["outlier_penalty_ratio"] > 1.5,
          f"while paying for it: g={rows[-1]['group_size']} costs "
          f"{rows[-1]['storage_overhead']:.2%} in scale bytes and still lets the "
          f"outlier cost {rows[-1]['outlier_penalty_ratio']:.2f}x -- so 'smaller "
          "groups are better' is a trade, not a free win")

    # ---------------------------------------------------------------------
    print("\n[8] the casts")
    e = Q.exactness_checks()
    r0 = e["rounding"]
    print(f"    halfway values {r0['half_way_values']}")
    print(f"      round-half-even  -> {r0['round_half_even']}")
    print(f"      astype(int8)     -> {r0['truncate_toward_zero']}")
    check(r0["differ"],
          "astype(int8) TRUNCATES toward zero; it does not round, and the "
          "two disagree by a full level on half-way values")
    z = e["zero_tensor"]
    check(not z["has_nan"] and z["roundtrip_is_zero"],
          f"an all-zero tensor round-trips to exactly zero with scale "
          f"{z['scale']}, producing no NaN")
    s3 = e["saturation"]
    check(s3["within_int8"],
          f"values beyond int8 clip rather than wrap "
          f"({s3['quantized']})")
    idem = e["idempotent"]
    check(not idem["second_pass_adds_error"],
          f"the round trip is a fixed point: a second pass adds no error "
          f"({idem['mse_1']:.6e} -> {idem['mse_2']:.6e}), so repeated "
          "quantize/dequantize cycles do not degrade")
    d = e["definition"]
    check(d["max_dev_from_integer_multiple"] == 0.0,
          f"every reconstruction is an exact integer multiple of 1/s "
          f"(max deviation {d['max_dev_from_integer_multiple']:.1f}) -- the "
          "definition itself, checked")

    # ---------------------------------------------------------------------
    print("\n[9] how much of this depends on the particular draw")
    ss = Q.seed_stability_report()
    print(f"    n = {ss['shape'][0] * ss['shape'][1]:,} elements per draw, "
          f"{ss['n_seeds']} seeds {ss['seeds']}")
    print(f"    {'quantity':<26}{'mean':>12}{'sd':>10}{'rel sd':>9}"
          f"{'min':>11}{'max':>11}")
    for k, v in ss["stats"].items():
        print(f"    {k:<26}{v['mean']:>12.5g}{v['sample_sd_across_seeds']:>10.3g}"
              f"{v['relative_sd']:>8.2%}{v['min']:>11.4g}{v['max']:>11.4g}")
    for k in ("absmax_mse_ratio", "blocks_mse_ratio", "blocks_recovery_ratio"):
        lo, hi = {"absmax_mse_ratio": ss["absmax_penalty_range"],
                  "blocks_mse_ratio": ss["block_penalty_range"],
                  "blocks_recovery_ratio": ss["recovery_range"]}[k]
        print(f"    {k:<26} range [{lo:.4g}, {hi:.4g}]")
    check(ss["penalty_still_over_50x"],
          f"the absmax outlier penalty exceeds 50x at ALL {ss['n_seeds']} seeds "
          f"(range {ss['absmax_penalty_range'][0]:.0f}-"
          f"{ss['absmax_penalty_range'][1]:.0f}x), so 111.4x is not a lucky draw")
    check(ss["block_penalty_always_under_2x"],
          f"the block-wise penalty stays below 2x at ALL {ss['n_seeds']} seeds "
          f"(max {ss['block_penalty_range'][1]:.3f}x)")
    check(ss["recovery_always_over_100x"],
          f"the recovery factor exceeds 100x at ALL {ss['n_seeds']} seeds "
          f"(min {ss['recovery_range'][0]:.0f}x)")
    z = (r["absmax_mse_ratio"] - ss["stats"]["absmax_mse_ratio"]["mean"]) / \
        ss["stats"]["absmax_mse_ratio"]["sample_sd_across_seeds"]
    check(abs(z) < 1.0,
          f"and the quoted seed-0 penalty {r['absmax_mse_ratio']:.1f}x sits "
          f"{abs(z):.2f} SD from the 8-seed mean, i.e. inside the error bar")

    # ---------------------------------------------------------------------
    print("\n[10] a guard that must be able to FAIL")
    zeros_backwards = np.zeros(16, dtype=np.float32)   # the scale applied wrong
    raised = False
    try:
        Q._metrics(w[:16, :16], zeros_backwards)
    except ValueError:
        raised = True
    check(raised,
          "a reconstruction with no information RAISES rather than reporting a "
          "plausible MSE -- this is the check that makes the scale-direction bug "
          "unreintroducible")

    print("\n" + "=" * 78)
    if FAILURES:
        print(f"CH14 FAILED: {len(FAILURES)} problem(s)")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("CH14_QUANT_VERIFY_PASSED")
    print("=" * 78)


if __name__ == "__main__":
    main()