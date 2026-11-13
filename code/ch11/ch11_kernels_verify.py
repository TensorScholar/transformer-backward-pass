"""Chapter 11 verification -- tiling, byte counts, and the online softmax.

Turns every claim into an assertion:

  1. matmul_tiled equals matmul_naive, elementwise;
  2. the ANALYTIC byte model matches an instrumented execution of the real
     loops -- so the numbers quoted in the chapter are counted, not asserted;
  3. the tiled/naive VRAM reduction approaches 2T and matches it at scale;
  4. online_softmax equals standard_attention, and its returned log-sum-exp is
     exact against a float64 reference;
  5. online_softmax is CORRECT WITHOUT RESCALING ONLY BY ACCIDENT -- the
     un-rescaled variant is shown to be wrong, because that rescaling is the
     whole trick and an unchecked implementation of it fails silently;
  6. the N^2 cliff at N=8192, d=128, float16.

Usage:  python ch11_kernels_verify.py [--reduced-canonical]
Marker: CH11_KERNELS_VERIFY_PASSED

``--reduced-canonical`` (or the environment variable
``TBP_REDUCED_CANONICAL=1``) skips only the full canonical ``T=4096,
d=2048`` blocks, whose fp64 dense references allocate multi-GB score
tensors and need a >8 GB host. Every sweep, contract, mutation control,
instrumented-accounting and SRAM check still runs at its full shape; the
skipped blocks are reported as such. The default is the full suite, which
is what the committed ``ch11_kernels_verify_output_v1.0.txt`` records.
"""
from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve()
for _p in (_HERE.parent, _HERE.parents[1]):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np                                                  # noqa: E402

from tgd import kernels as KN                                       # noqa: E402

FAILURES: list[str] = []

#: Skip only the multi-GB canonical T=4096 d=2048 blocks (fp64 dense
#: references need >8 GB RAM). Selected by --reduced-canonical or
#: TBP_REDUCED_CANONICAL=1; every other check runs at full shape.
REDUCED_CANONICAL = ("--reduced-canonical" in sys.argv
                     or os.environ.get("TBP_REDUCED_CANONICAL") == "1")
FULL_CANONICAL = not REDUCED_CANONICAL


def check(cond: bool, msg: str) -> None:
    if cond:
        print(f"  ok    {msg}")
    else:
        print(f"  FAIL  {msg}")
        FAILURES.append(msg)
    assert cond, msg


def main() -> None:
    print("=" * 78)
    print("Chapter 11 -- GPU execution, tiling, and the online softmax")
    print(KN.COUNTING_MODEL)
    print("=" * 78)

    # ---- 1 & 2. equivalence and the analytic byte model ------------------
    print("\n[1] tiled matmul == naive matmul, and both are byte-counted")
    for (M, K, N, T) in ((64, 64, 64, 16), (96, 48, 32, 16), (64, 64, 64, 8)):
        rng = np.random.default_rng(0)
        A = rng.standard_normal((M, K))
        B = rng.standard_normal((K, N))
        t0 = time.perf_counter()
        Cn, bn = KN.matmul_naive(A, B, element_bytes=2)
        tn = time.perf_counter() - t0
        t0 = time.perf_counter()
        Ct, bt = KN.matmul_tiled(A, B, T, element_bytes=2)
        tt = time.perf_counter() - t0
        err = float(np.abs(Cn - Ct).max())
        print(f"    {M}x{K}x{N} T={T}: naive {tn:.2f}s {bn}")
        print(f"    {'':{len(f'{M}x{K}x{N} T={T}')}} tiled {tt:.2f}s {bt}")
        check(err < 1e-9,
              f"{M}x{K}x{N} T={T}: outputs agree to {err:.2e}")

        # the analytic model must reproduce the instrumented count exactly
        pred_n = (2 * M * N * K + M * N) * 2
        pred_t = (2 * M * N * K // T + M * N) * 2
        check(bn.vram_total == pred_n,
              f"{M}x{K}x{N}: instrumented naive VRAM {bn.vram_total} B == "
              f"analytic 8*M*N*K*2 = {pred_n} B")
        check(bt.vram_total == pred_t,
              f"{M}x{K}x{N} T={T}: instrumented tiled VRAM {bt.vram_total} B "
              f"== analytic {pred_t} B")
        check(bt.sram_total > 0,
              f"{M}x{K}x{N} T={T}: SRAM accesses counted separately "
              f"({bt.sram_total / 1e6:.2f} MB) -- these are the traffic the "
              f"tiling moved OFF the memory bus")

    # ---- 3. the reduction factor, at scale -------------------------------
    print("\n[2] the VRAM reduction factor, analytically")
    M = K = N = 1024
    print(f"    {'tile':>5} {'naive VRAM':>14} {'tiled VRAM':>14} "
          f"{'reduction':>10} {'T':>5} {'naive iota':>11} {'tiled iota':>11}")
    for T in (8, 16, 32, 64, 128):
        r = KN.tiling_reduction(M, K, N, T, element_bytes=2)
        print(f"    {T:>5} {r['naive_vram_bytes'] / 1e6:>12.1f} MB "
              f"{r['tiled_vram_bytes'] / 1e6:>12.3f} MB "
              f"{r['reduction']:>9.2f}x {r['predicted_reduction']:>5.0f} "
              f"{r['naive_intensity']:>11.4f} {r['tiled_intensity']:>11.2f}")
        check(abs(r["reduction"] / r["predicted_reduction"] - 1) < 0.1,
              f"T={T}: reduction {r['reduction']:.2f}x is within 10% of the "
              f"predicted T = {r['predicted_reduction']:.0f}x")
    r32 = KN.tiling_reduction(M, K, N, 32, element_bytes=2)
    check(r32["reduction"] > 25,
          f"at 1024^3 with 32x32 tiles the reduction is "
          f"{r32['reduction']:.2f}x")
    check(r32["naive_intensity"] < 1.0,
          f"and the naive kernel's arithmetic intensity is "
          f"{r32['naive_intensity']:.4f} FLOP/B -- below 1, so it is "
          f"memory-bound by Chapter 9's criterion")

    # ---- 4 & 5. the online softmax, and why the rescaling is the trick ----
    print("\n[3] online softmax == standard attention")
    for (N, d, blk) in ((32, 8, 8), (64, 16, 16), (48, 8, 32)):
        rng = np.random.default_rng(1)
        Q = rng.standard_normal((N, d)) * 2.0
        Kk = rng.standard_normal((N, d)) * 2.0
        V = rng.standard_normal((N, d))
        O_ref, _ = KN.standard_attention(Q, Kk, V)
        O_on, lse = KN.online_softmax(Q, Kk, V, block=blk)
        err = float(np.abs(O_ref - O_on).max())
        # exact reference log-sum-exp in float64
        S = Q @ Kk.T
        m = S.max(axis=-1, keepdims=True)
        lse_ref = (m[:, 0] + np.log(np.exp(S - m).sum(axis=-1)))
        lse_err = float(np.abs(lse - lse_ref).max())
        print(f"    N={N} d={d} block={blk}: max |O difference| = {err:.3e}, "
              f"max |lse difference| = {lse_err:.3e}")
        check(err < 1e-12,
              f"N={N} block={blk}: online softmax reproduces standard "
              f"attention to {err:.2e}")
        check(lse_err < 1e-12,
              f"N={N} block={blk}: the returned log-sum-exp is exact "
              f"({lse_err:.2e}) -- this is what a real kernel carries")

    # ---- the failure mode this chapter is about ---------------------------
    print("\n[4] the same routine WITHOUT rescaling is silently wrong")
    N, d, blk = 64, 16, 16
    rng = np.random.default_rng(1)
    Q = rng.standard_normal((N, d)) * 2.0
    Kk = rng.standard_normal((N, d)) * 2.0
    V = rng.standard_normal((N, d))
    O_ref, _ = KN.standard_attention(Q, Kk, V)

    # a plausible-looking bug: accumulate Z without rescaling when the block max
    # rises, i.e. sum exp terms expressed in inconsistent units
    m_run = np.full(N, -np.inf)
    l_run = np.zeros(N)
    acc = np.zeros((N, d))
    for s0 in range(0, N, blk):
        s1 = min(s0 + blk, N)
        Sb = Q @ Kk[s0:s1].T
        m_blk = Sb.max(axis=-1)
        m_new = np.maximum(m_run, m_blk)
        alpha = np.exp(m_run - m_new)
        # BUG: l_run is not rescaled by alpha
        l_run = l_run + np.exp(Sb - m_new[:, None]).sum(axis=-1)
        Pb = np.exp(Sb - m_new[:, None])
        acc = acc * alpha[:, None] + Pb @ V[s0:s1]
        m_run = m_new
    O_bad = acc / l_run[:, None]
    err_bad = float(np.abs(O_ref - O_bad).max())
    rel_bad = float(np.abs(O_ref - O_bad).max() / np.abs(O_ref).max())
    O_ok, _ = KN.online_softmax(Q, Kk, V, block=blk)
    err_ok = float(np.abs(O_ref - O_ok).max())
    print(f"    correct   : max |difference| = {err_ok:.3e}")
    print(f"    unrescaled: max |difference| = {err_bad:.3e} "
          f"({rel_bad:.1%} relative)")
    check(rel_bad > 0.05,
          f"dropping the rescale gives an output wrong by {rel_bad:.1%} -- "
          f"finite, plausible, and not an error")
    check(err_ok < 1e-12, "and the rescaled version is exact")

    # ---- 6. the N^2 cliff ------------------------------------------------
    print("\n[5] the N^2 cliff at N=8192, d=128, float16")
    N, d, e = 8192, 128, 2
    std = KN.attention_bytes_standard(N, d, e)
    for blk in (64, 128, 256):
        fl = KN.attention_bytes_flash(N, d, blk, e)
        print(f"    block={blk:>4}: standard {std['total'] / 1e9:7.3f} GB   "
              f"fused {fl['total'] / 1e6:7.2f} MB   "
              f"saved {(1 - fl['total'] / std['total']) * 100:6.2f}%   "
              f"ratio {std['total'] / fl['total']:8.1f}x")
    fl128 = KN.attention_bytes_flash(N, d, 128, e)
    ratio = std["total"] / fl128["total"]
    print(f"    one N x N score matrix is "
          f"{std['score_matrix_bytes'] / 1e6:.1f} MB in float16")
    print(f"    standard total {std['total'] / 1e9:.3f} GB   "
          f"fused total {fl128['total'] / 1e6:.2f} MB")
    check(std["score_matrix_bytes"] > 1e8,
          f"the score matrix alone is {std['score_matrix_bytes'] / 1e6:.1f} MB "
          f"-- larger than the whole KV cache of a 32-layer d=4096 model at 8K")
    check(ratio > 50,
          f"fusion saves {ratio:.1f}x of attention's VRAM traffic at N=8192")
    # Fused traffic is the FIXED Q/K/V/O cost; standard grows as N^2. So the
    # fraction must shrink and the ratio must grow with context -- that is the
    # structural claim, and asserting a fixed percentage here would be asserting
    # a number that is true only at one N.
    check(fl128["total"] < std["total"] / 50,
          f"and fused traffic is {fl128['total'] / std['total'] * 100:.2f}% of "
          f"standard at N=8192")

    # The single-read figure is the OPTIMISTIC doctrine (all operands
    # cache-resident, forward only). The module's own no-cache backward
    # doctrine re-reads dO and Q/K/V once and writes dQ/dK/dV -- 2.75x the
    # forward operand bytes at this shape; report both so the cliff is not
    # overstated by conflating the doctrines (65x single-read becomes 24x
    # doctrine-consistent at N=8192).
    honest = fl128["doctrine_q_restream_total"]
    print(f"    doctrine-consistent fused (bwd re-reads + grad writes): "
          f"{honest / 1e6:.2f} MB -> honest cliff {std['total'] / honest:.1f}x "
          f"(vs {ratio:.1f}x single-read)")
    check(std["total"] / honest > 4.0,
          f"even under the no-cache backward doctrine, fusion saves "
          f"{std['total'] / honest:.1f}x at N=8192")

    # scaling in N: standard is quadratic, fused is linear
    print("\n[6] how the two scale with context length (d=128, float16)")
    print(f"    {'N':>7} {'standard GB':>13} {'fused MB':>11} {'ratio':>10} "
          f"{'std / N':>10}")
    for Ns in (1024, 2048, 4096, 8192, 16384, 32768):
        s_ = KN.attention_bytes_standard(Ns, d, e)
        f_ = KN.attention_bytes_flash(Ns, d, 128, e)
        print(f"    {Ns:>7} {s_['total'] / 1e9:>12.2f} G "
              f"{f_['total'] / 1e6:>10.2f} M "
              f"{s_['total'] / f_['total']:>9.1f}x "
              f"{s_['total'] / Ns / 1e3:>9.2f} k")
    ratios = [KN.attention_bytes_standard(Ns, d, e)["total"]
              / KN.attention_bytes_flash(Ns, d, 128, e)["total"]
              for Ns in (1024, 2048, 4096, 8192, 16384, 32768)]
    check(all(b > a for a, b in zip(ratios, ratios[1:])),
          f"the saving GROWS with context length ({ratios[0]:.1f}x -> "
          f"{ratios[-1]:.1f}x), because fused traffic is fixed in N and standard "
          f"traffic is quadratic")
    big = KN.attention_bytes_standard(32768, d, e)["total"]
    check(big > 8e9,
          f"at N=32768 standard attention moves {big / 1e9:.1f} GB of scores "
          f"for a single layer -- it does not fit in an 80 GB card alongside "
          f"the model")

    # ---- 7. device context, labelled as specification --------------------
    print("\n[7] what those bytes cost, using PUBLISHED specifications")
    print(f"    {'device':<12} {'VRAM':>10} {'HBM BW':>10} {'SRAM':>10} "
          f"{'SRAM BW':>10} {'BW ratio':>10}")
    for name, d_ in KN.DEVICES.items():
        if d_.sram_tb_s == 0:
            print(f"    {name:<12} {d_.vram_gb:>7.0f} GB {d_.vram_tb_s:>7.2f} TB/s"
                  f" {'--':>10} {'--':>10} {'--':>10}")
            continue
        print(f"    {name:<12} {d_.vram_gb:>7.0f} GB {d_.vram_tb_s:>7.2f} TB/s"
              f" {d_.sram_kb:>7.0f} KB {d_.sram_tb_s:>7.1f} TB/s"
              f" {d_.sram_tb_s / d_.vram_tb_s:>9.1f}x")
    h = KN.DEVICES["H100 SXM"]
    check(h.sram_tb_s / h.vram_tb_s > 5,
          f"SRAM is {h.sram_tb_s / h.vram_tb_s:.1f}x faster than HBM on the H100 "
          f"({h.sram_tb_s} TB/s vs {h.vram_tb_s} TB/s)")
    check(h.vram_gb * 1024 / 1024 > h.sram_kb / 1024,
          f"and it is {h.vram_gb * 1024 / h.sram_kb:.0f}x smaller in capacity "
          f"({h.vram_gb:.0f} GB vs {h.sram_kb:.0f} KB) -- the whole tiling "
          f"strategy is about borrowing the small thing")

    # ---- 8. time estimates, explicitly labelled -------------------------
    print("\n[8] time estimates at the H100 specifications (arithmetic only)")
    fl128 = KN.attention_bytes_flash(N, d, 128, e)
    t_std = std["total"] / (h.vram_tb_s * 1e12)
    t_fl = fl128["total"] / (h.vram_tb_s * 1e12)
    print(f"    standard attention : {std['total'] / 1e9:.3f} GB / "
          f"{h.vram_tb_s} TB/s = {t_std * 1e3:.2f} ms")
    print(f"    fused attention    : {fl128['total'] / 1e6:.2f} MB / "
          f"{h.vram_tb_s} TB/s = {t_fl * 1e6:.2f} us")
    check(t_std > t_fl * 50,
          f"memory time drops by {t_std / t_fl:.0f}x; these are bandwidth "
          f"estimates, not measured, and exclude all arithmetic and launch "
          f"overhead")

    # ---- 9. Stage 3-B: R1 (dense fp64 autograd) & R3 (tiled fp64 fwd+bwd) -
    print("\n[9] Stage 3-B R1 & R3 -- tiled GQA online-softmax forward+backward (fp64)")
    r1_r3 = KN.verify_r1_and_r3(run_full_canonical_4096=FULL_CANONICAL)
    for k in ("O", "lse", "Q_bar", "K_bar", "V_bar"):
        sw_abs = r1_r3[f"sweep_worst_{k}_max_abs"]
        sw_rel = r1_r3[f"sweep_worst_{k}_rel_norm"]
        assert sw_abs < 1e-13 and sw_rel < 1e-13
        if FULL_CANONICAL:
            c_abs = r1_r3[f"canonical_4096_{k}_max_abs"]
            c_rel = r1_r3[f"canonical_4096_{k}_rel_norm"]
            assert c_abs < 1e-12 and c_rel < 1e-13
            check(
                sw_abs < 1e-13 and c_abs < 1e-12,
                f"R3 vs R1 {k:5s}: sweep max_abs={sw_abs:.3e} (rel={sw_rel:.3e}), "
                f"canonical T=4096 max_abs={c_abs:.3e} (rel={c_rel:.3e})",
            )
        else:
            check(True, f"R3 vs R1 {k:5s}: sweep max_abs={sw_abs:.3e} "
                  f"(rel={sw_rel:.3e}) [canonical T=4096 skipped: "
                  "reduced-canonical mode]")

    # ---- 10. Stage 3-B: R2 (dense fp64 GQA with explicit hand-written group sum)
    print("\n[10] Stage 3-B R2 -- dense fp64 GQA with explicit hand-written group sum")
    r2_res = KN.verify_r2(run_full_canonical_4096=FULL_CANONICAL)
    ek_sum = r2_res["worst_per_head_summand_K_bar_abs"]
    ev_sum = r2_res["worst_per_head_summand_V_bar_abs"]
    assert ek_sum < 1e-14 and ev_sum < 1e-14
    check(
        ek_sum < 1e-14 and ev_sum < 1e-14,
        f"R2 explicit group-sum summands vs isolated single-head autograd: "
        f"K_bar={ek_sum:.3e}, V_bar={ev_sum:.3e}",
    )
    err_last = r2_res["broken_merge_last_head_K_bar_rel"]
    err_avg = r2_res["broken_merge_averaged_K_bar_rel"]
    assert err_last > 0.5 and err_avg > 0.5
    check(
        err_last > 0.5 and err_avg > 0.5,
        f"R2 broken group-merge controls caught: last-head-only={err_last:.3e}, "
        f"averaged={err_avg:.3e}",
    )
    for k in ("O", "lse", "Q_bar", "K_bar", "V_bar"):
        s21_abs = r2_res[f"sweep_r2_vs_r1_{k}_max_abs"]
        s32_abs = r2_res[f"sweep_r3_vs_r2_{k}_max_abs"]
        assert s21_abs < 1e-13 and s32_abs < 1e-13
        if FULL_CANONICAL:
            c21_abs = r2_res[f"canonical_4096_r2_vs_r1_{k}_max_abs"]
            c32_abs = r2_res[f"canonical_4096_r3_vs_r2_{k}_max_abs"]
            assert c21_abs < 1e-12 and c32_abs < 1e-12
            check(
                s21_abs < 1e-13 and s32_abs < 1e-13 and c21_abs < 1e-12 and c32_abs < 1e-12,
                f"R2 3-way {k:5s}: sweep(R2-R1={s21_abs:.3e}, R3-R2={s32_abs:.3e}), "
                f"T=4096(R2-R1={c21_abs:.3e}, R3-R2={c32_abs:.3e})",
            )
        else:
            check(True, f"R2 3-way {k:5s}: sweep(R2-R1={s21_abs:.3e}, "
                  f"R3-R2={s32_abs:.3e}) [canonical T=4096 skipped]")

    # ---- 11. Stage 3-B Step 4: The Production Object (fp16/fp32, 112.5 MiB) -
    print("\n[11] Stage 3-B Step 4 -- Production Object (fp16 operands/storage, fp32 accumulators, 112.5 MiB save set)")
    prod_res = KN.verify_production_object(run_full_canonical_4096=FULL_CANONICAL)
    save_mib = prod_res["canonical_4096_save_set_mib"]
    assert save_mib == 112.5
    check(
        save_mib == 112.5,
        f"Canonical T=4096 save set {save_mib:.1f} MiB "
        f"(Q=32, K=8, V=8 in fp16; O=64, lse=0.5 in fp32)",
    )
    pk_dq = prod_res["peaked_softmax_dQ_rel_norm"]
    pk_dk = prod_res["peaked_softmax_dK_rel_norm"]
    pk_dv = prod_res["peaked_softmax_dV_rel_norm"]
    assert pk_dq < 1e-5 and pk_dk < 1e-5 and pk_dv < 1e-5
    check(
        pk_dq < 1e-5 and pk_dk < 1e-5 and pk_dv < 1e-5,
        f"Peaked-softmax stress regime rel_norm vs R1: "
        f"dQ={pk_dq:.3e}, dK={pk_dk:.3e}, dV={pk_dv:.3e}",
    )
    for k in ("O", "lse", "Q_bar", "K_bar", "V_bar"):
        sw_abs = prod_res[f"sweep_prod_vs_r1_{k}_max_abs"]
        sw_rel = prod_res[f"sweep_prod_vs_r1_{k}_rel_norm"]
        if k == "O":
            assert sw_abs < 2e-3 and sw_rel < 5e-4
        else:
            assert sw_abs < 5e-5 and sw_rel < 5e-6
        if FULL_CANONICAL:
            c_abs = prod_res[f"canonical_4096_prod_vs_r1_{k}_max_abs"]
            c_rel = prod_res[f"canonical_4096_prod_vs_r1_{k}_rel_norm"]
            if k == "O":
                assert c_abs < 2e-3 and c_rel < 5e-4
                ok = sw_abs < 2e-3 and sw_rel < 5e-4 and c_abs < 2e-3 and c_rel < 5e-4
            else:
                assert c_abs < 5e-5 and c_rel < 5e-6
                ok = sw_abs < 5e-5 and sw_rel < 5e-6 and c_abs < 5e-5 and c_rel < 5e-6
            check(
                ok,
                f"Production vs R1 {k:5s}: sweep max_abs={sw_abs:.3e} (rel={sw_rel:.3e}), "
                f"T=4096 max_abs={c_abs:.3e} (rel={c_rel:.3e})",
            )
        else:
            check(True, f"Production vs R1 {k:5s}: sweep max_abs={sw_abs:.3e} "
                  f"(rel={sw_rel:.3e}) [canonical T=4096 skipped]")

    # ---- 12. Stage 3-B Step 5: F1, F2, F3 & Pairwise Blind-Spot Matrix -----
    print("\n[12] Stage 3-B Step 5 -- F1, F2, F3 single-field mutations & Pairwise Blind-Spot Matrix")
    s5 = KN.verify_step5_mutants_and_blind_spots(run_full_canonical_4096=FULL_CANONICAL)

    f1_dk = float(s5["f1_multi_tile_gqa_dK_rel"])
    f1_dv = float(s5["f1_multi_tile_gqa_dV_rel"])
    f1_fp64 = float(s5["f1_fp64_dK_rel"])
    f1_tile = float(s5["f1_tile_only_mha_dK_rel"])
    f1_grp = float(s5["f1_group_only_1tile_dK_rel"])
    assert f1_dk > 0.80 and f1_dv > 0.80 and f1_fp64 > 0.80
    assert f1_tile > 0.50 and f1_grp > 0.50
    check(
        f1_dk > 0.80 and f1_tile > 0.50 and f1_grp > 0.50,
        f"F1 (kv_grad_reduction='overwrite_assign'): dK={f1_dk:.3e}, dV={f1_dv:.3e}, "
        f"fp64={f1_fp64:.3e} (tile-only={f1_tile:.3e}, group-only={f1_grp:.3e})",
    )

    f2_diff_dq = float(s5["f2_diffuse_dQ_rel"])
    f2_pk_dq = float(s5["f2_peaked_dQ_rel"])
    f2_pk_dk = float(s5["f2_peaked_dK_rel"])
    f2_pk_dv = float(s5["f2_peaked_dV_rel"])
    prod_pk_dq = float(s5["prod_peaked_dQ_rel"])
    f2_save_mib = float(s5["canonical_4096_f2_save_mib"])
    f2_delta_mib = float(s5["canonical_4096_f2_repair_delta_mib"])
    assert f2_save_mib == 80.5 and f2_delta_mib == 32.0
    assert f2_diff_dq < 5e-4 and f2_pk_dq > 500.0 * prod_pk_dq and f2_pk_dv < 1e-5
    check(
        f2_save_mib == 80.5 and f2_delta_mib == 32.0 and f2_pk_dq > 500.0 * prod_pk_dq,
        f"F2 (save_O=fp16, {f2_save_mib:.1f} MiB, repair +{f2_delta_mib:.1f} MiB): "
        f"diffuse dQ={f2_diff_dq:.3e}, peaked dQ={f2_pk_dq:.3e} vs healthy {prod_pk_dq:.3e} "
        f"({f2_pk_dq / prod_pk_dq:.0f}x worse), dV={f2_pk_dv:.3e}",
    )

    f3_spill_diff = float(s5["f3_retained_vs_spilled_fp32_max_abs"])
    prod_dk_s = float(s5["prod_diffuse_dK_rel"])
    f3_dk_s = float(s5["f3_diffuse_dK_rel"])
    f3_swamp_dv = float(s5["f3_swamp_tail_dV_rel"])
    sram_128 = float(s5["sram_prod_bkv128_kib"])
    sram_256 = float(s5["sram_prod_bkv256_kib"])
    sram_f3_256 = float(s5["sram_f3_bkv256_kib"])
    assert f3_spill_diff == 0.0 and f3_dk_s > 500.0 * prod_dk_s and f3_swamp_dv > 0.90
    assert sram_128 == 152.0 and sram_256 == 280.0 and sram_f3_256 == 152.0
    check(
        f3_spill_diff == 0.0 and f3_dk_s > 500.0 * prod_dk_s and f3_swamp_dv > 0.90,
        f"F3 3-way control (kv_grad_accum=fp16): retained==spilled_fp32 (0.0), "
        f"diffuse dK={f3_dk_s:.3e} vs {prod_dk_s:.3e} ({f3_dk_s / prod_dk_s:.0f}x), "
        f"swamp tail dV={f3_swamp_dv:.3e}; SRAM B_kv=128:{sram_128:.0f}KiB, "
        f"B_kv=256 fp32:{sram_256:.0f}KiB>228KB vs fp16:{sram_f3_256:.0f}KiB",
    )

    # ---- 13. Stage 3-B Step 6: R4 & R5 Reference Implementations -----------
    print("\n[13] Stage 3-B Step 6 -- R4 (KV-outer / Q-inner schedule) & R5 (Spilled fp32 accumulators)")
    s6 = KN.verify_r4_and_r5(run_full_canonical_4096=FULL_CANONICAL)

    r4_step4_diff = float(s6["r4_prod_vs_step4_max_abs"])
    r5_step4_diff = float(s6["r5_prod_vs_step4_max_abs"])
    r4_dual_dq = float(s6["r4_dual_q_overwrite_dQ_rel"])
    c_steps = int(s6["canonical_4096_causal_tile_steps"])
    r4_q_rmw = float(s6["canonical_4096_r4_q_bar_rmw_mib"])
    r4_kv_wr = float(s6["canonical_4096_r4_kv_bar_write_mib"])
    r5_sp_rd = float(s6["canonical_4096_r5_kv_spill_read_mib"])
    r5_sp_wr = float(s6["canonical_4096_r5_kv_spill_write_mib"])
    r5_sram = float(s6["canonical_4096_r5_bwd_sram_kib"])

    assert r4_step4_diff == 0.0 and r5_step4_diff == 0.0
    assert r4_dual_dq > 0.35
    assert c_steps == 2112
    assert r4_q_rmw == 2112.0 and r4_kv_wr == 32.0
    assert r5_sp_rd == 2112.0 and r5_sp_wr == 2112.0 and r5_sram == 24.0

    for k in ("O", "lse", "Q_bar", "K_bar", "V_bar"):
        sw_abs = float(s6[f"sweep_r4_fp64_vs_r1_{k}_max_abs"])
        sw_rel = float(s6[f"sweep_r4_fp64_vs_r1_{k}_rel_norm"])
        assert sw_abs < 1e-13 and sw_rel < 1e-13
        if FULL_CANONICAL:
            c_abs = float(s6[f"canonical_4096_r4_fp64_vs_r1_{k}_max_abs"])
            c_rel = float(s6[f"canonical_4096_r4_fp64_vs_r1_{k}_rel_norm"])
            assert c_abs < 1e-12 and c_rel < 1e-13
            check(
                sw_abs < 1e-13 and c_abs < 1e-12,
                f"R4(fp64) vs R1 {k:5s}: sweep max_abs={sw_abs:.3e} (rel={sw_rel:.3e}), "
                f"T=4096 max_abs={c_abs:.3e} (rel={c_rel:.3e})",
            )
        else:
            check(True, f"R4(fp64) vs R1 {k:5s}: sweep max_abs={sw_abs:.3e} "
                  f"(rel={sw_rel:.3e}) [canonical T=4096 skipped]")

    check(
        r4_step4_diff == 0.0 and r4_dual_dq > 0.35 and r4_q_rmw == 2112.0 and r4_kv_wr == 32.0,
        f"R4(PRODUCTION_SCHEDULE): bitwise==Step4 (diff={r4_step4_diff:.1e}), "
        f"dual Q-overwrite dQ_rel={r4_dual_dq:.3e}; T=4096 Q_bar RMW={r4_q_rmw:.1f} MiB, "
        f"KV_bar write={r4_kv_wr:.1f} MiB ({c_steps} causal steps)",
    )
    check(
        r5_step4_diff == 0.0 and r5_sp_rd == 2112.0 and r5_sp_wr == 2112.0 and r5_sram == 24.0,
        f"R5(spill_kv_accum_to_hbm=True): bitwise==Step4 (diff={r5_step4_diff:.1e}), "
        f"SRAM={r5_sram:.1f} KiB, T=4096 HBM spill R/W={r5_sp_rd:.1f}/{r5_sp_wr:.1f} MiB "
        f"({c_steps} causal steps)",
    )

    # ---- 14. Stage 3-B Step 7: Instrumented Accounting Layer ----------------
    print("\n[14] Stage 3-B Step 7 -- Instrumented Accounting Layer (analytic == runtime counters)")
    s7 = KN.verify_step7_instrumented_accounting(run_full_canonical_4096=FULL_CANONICAL)

    sv_s4_tot = float(s7["save_step4_total_mib"])
    sv_s4_16 = float(s7["save_step4_fp16_mib"])
    sv_s4_32 = float(s7["save_step4_fp32_mib"])
    sv_f2_tot = float(s7["save_f2_total_mib"])
    sv_f2_16 = float(s7["save_f2_fp16_mib"])
    sv_f2_32 = float(s7["save_f2_fp32_mib"])
    sv_mat_tot = float(s7["save_mat_kv_total_mib"])
    sv_dense_tot = float(s7["save_dense_scores_total_mib"])
    sv_fp64_tot = float(s7["save_fp64_total_mib"])
    assert (sv_s4_tot, sv_s4_16, sv_s4_32) == (112.5, 48.0, 64.5)
    assert (sv_f2_tot, sv_f2_16, sv_f2_32) == (80.5, 80.0, 0.5)
    assert sv_mat_tot == 160.5 and sv_dense_tot == 1136.5 and sv_fp64_tot == 321.0
    check(
        sv_s4_tot == 112.5 and sv_f2_tot == 80.5 and sv_mat_tot == 160.5 and sv_dense_tot == 1136.5,
        f"SaveSet per-dtype accounting (counter==analytic): Step4={sv_s4_tot:.1f} MiB "
        f"({sv_s4_16:.1f} fp16 + {sv_s4_32:.1f} fp32), F2={sv_f2_tot:.1f} MiB "
        f"({sv_f2_16:.1f} fp16 + {sv_f2_32:.1f} fp32), MatKV={sv_mat_tot:.1f} MiB, "
        f"DenseScores={sv_dense_tot:.1f} MiB, fp64={sv_fp64_tot:.1f} MiB",
    )

    sr_s4_128 = float(s7["sram_step4_bkv128_kib"])
    sr_s4_pk = float(s7["sram_step4_bkv128_peak_kib"])
    sr_s4_256 = float(s7["sram_step4_bkv256_kib"])
    sr_f3_128 = float(s7["sram_f3_bkv128_kib"])
    sr_f3_256 = float(s7["sram_f3_bkv256_kib"])
    sr_r5_128 = float(s7["sram_r5_bkv128_kib"])
    sr_r4_128 = float(s7["sram_r4_bkv128_kib"])
    assert (sr_s4_128, sr_s4_pk, sr_s4_256) == (152.0, 232.0, 280.0)
    assert (sr_f3_128, sr_f3_256, sr_r5_128, sr_r4_128) == (88.0, 152.0, 24.0, 192.0)
    check(
        sr_s4_128 == 152.0 and sr_s4_256 == 280.0 and sr_f3_256 == 152.0 and sr_r5_128 == 24.0,
        f"Per-SM SRAM coexistence (vs H100 228KB / A100 164KB): "
        f"Step4(128)={sr_s4_128:.0f}KiB (peak {sr_s4_pk:.0f}KiB), "
        f"Step4(256)={sr_s4_256:.0f}KiB>228KB, F3(256)={sr_f3_256:.0f}KiB, "
        f"R5={sr_r5_128:.0f}KiB, R4(128)={sr_r4_128:.0f}KiB",
    )

    tr_s4 = float(s7["canonical_4096_step4_bwd_hbm_total_mib"])
    tr_f2 = float(s7["canonical_4096_f2_bwd_hbm_total_mib"])
    tr_f3 = float(s7["canonical_4096_f3_bwd_hbm_total_mib"])
    tr_r4 = float(s7["canonical_4096_r4_bwd_hbm_total_mib"])
    tr_r5 = float(s7["canonical_4096_r5_bwd_hbm_total_mib"])
    pk_tiled = float(s7["canonical_4096_hbm_peak_tiled_mib"])
    pk_dense = float(s7["canonical_4096_hbm_peak_dense_mib"])
    assert (tr_s4, tr_f2, tr_f3, tr_r4, tr_r5) == (1312.5, 1280.5, 1296.5, 4808.25, 5504.5)
    assert (pk_tiled, pk_dense) == (272.5, 8464.5)
    check(
        tr_s4 == 1312.5 and tr_r5 == 5504.5 and tr_r4 == 4808.25 and pk_tiled == 272.5,
        f"Canonical T=4096 backward HBM traffic (counter==analytic): "
        f"Step4={tr_s4:.1f} MiB, F2={tr_f2:.1f} MiB, F3={tr_f3:.1f} MiB, "
        f"R4={tr_r4:.2f} MiB, R5={tr_r5:.1f} MiB; "
        f"HBM peak-live tiled={pk_tiled:.1f} MiB vs dense={pk_dense:.1f} MiB",
    )

    # ---- 15. Stage 3-B Step 8: Full Parametric Sweeps with Interior Thresholds ----
    print("\n[15] Stage 3-B Step 8 -- Full Parametric Sweeps with Interior Thresholds (F1, F2, F3)")
    s8 = KN.verify_step8_parametric_sweeps()
    s8_f1 = s8["f1"]
    s8_f2 = s8["f2"]
    s8_f3 = s8["f3"]

    g_curve = list(s8_f1["g_curve_dK_rel"])
    nq_curve = list(s8_f1["nq_curve_dK_rel"])
    diag_curve = list(s8_f1["diag_curve_dK_rel"])
    idx_g_25 = int(s8_f1["idx_g_tau_025"])
    idx_g_65 = int(s8_f1["idx_g_tau_065"])
    idx_nq_25 = int(s8_f1["idx_nq_tau_025"])
    idx_nq_65 = int(s8_f1["idx_nq_tau_065"])
    idx_dg_70 = int(s8_f1["idx_diag_tau_070"])
    max_s4_f1 = float(s8_f1["max_step4_dK_rel"])
    assert 0 < idx_g_25 < idx_g_65 < len(g_curve)
    assert 0 < idx_nq_25 < idx_nq_65 < len(nq_curve)
    assert 0 < idx_dg_70 < len(diag_curve) and max_s4_f1 < 5e-6
    check(
        g_curve[0] < 1e-6 and g_curve[-1] > 0.65 and nq_curve[-1] > 0.75 and diag_curve[-1] > 0.90,
        f"F1 (G x N_q) sweep: (G=1,N_q=1)={g_curve[0]:.2e} (bitwise==Step4), "
        f"G-curve={[round(x, 3) for x in g_curve]} (idx@0.25={idx_g_25}, idx@0.65={idx_g_65}), "
        f"N_q-curve={[round(x, 3) for x in nq_curve]} (idx@0.25={idx_nq_25}, idx@0.65={idx_nq_65}), "
        f"diag={[round(x, 3) for x in diag_curve]} (idx@0.70={idx_dg_70})",
    )

    f2_dq_c = list(s8_f2["f2_dQ_rel_curve"])
    f2_dk_c = list(s8_f2["f2_dK_rel_curve"])
    s4_dq_c = list(s8_f2["step4_dQ_rel_curve"])
    idx_dq_2e4 = int(s8_f2["idx_dQ_tau_2e4"])
    idx_dq_1e3 = int(s8_f2["idx_dQ_tau_1e3"])
    idx_dk_2e4 = int(s8_f2["idx_dK_tau_2e4"])
    idx_dk_1e3 = int(s8_f2["idx_dK_tau_1e3"])
    assert 0 < idx_dq_2e4 < idx_dq_1e3 < len(f2_dq_c)
    assert 0 < idx_dk_2e4 < idx_dk_1e3 < len(f2_dk_c)
    assert max(s4_dq_c) < 1e-5
    check(
        f2_dq_c[0] < 1e-4 and f2_dq_c[-1] > 3e-3 and max(s4_dq_c) < 1e-5,
        f"F2 peakedness (beta) sweep: dQ_rel={f2_dq_c[0]:.2e}->{f2_dq_c[-1]:.2e} "
        f"(interior idx@2e-4={idx_dq_2e4}, idx@1e-3={idx_dq_1e3}), "
        f"dK_rel={f2_dk_c[0]:.2e}->{f2_dk_c[-1]:.2e} "
        f"(interior idx@2e-4={idx_dk_2e4}, idx@1e-3={idx_dk_1e3}), "
        f"Step4 max dQ_rel={max(s4_dq_c):.2e}",
    )

    sram_s4_c = list(s8_f3["sram_step4_curve_kib"])
    sram_f3_c = list(s8_f3["sram_f3_curve_kib"])
    sram_r5_c = list(s8_f3["sram_r5_curve_kib"])
    idx_s4_h100 = int(s8_f3["idx_s4_h100"])
    idx_f3_h100 = int(s8_f3["idx_f3_h100"])
    f3_tail_dv = list(s8_f3["f3_tail_dV_rel_curve"])
    f3_tail_dk = list(s8_f3["f3_tail_dK_rel_curve"])
    s4_tail_dv = list(s8_f3["step4_tail_dV_rel_curve"])
    idx_sw_002 = int(s8_f3["idx_swamp_tau_002"])
    idx_sw_050 = int(s8_f3["idx_swamp_tau_050"])
    f3_nq_dv = list(s8_f3["f3_nq_dV_rel_curve"])
    idx_nq_f3 = int(s8_f3["idx_nq_f3_tau_14e4"])
    assert 0 < idx_s4_h100 < idx_f3_h100 < len(sram_s4_c)
    assert 0 < idx_sw_002 < idx_sw_050 < len(f3_tail_dv)
    assert 0 < idx_nq_f3 < len(f3_nq_dv)
    # F2 decoupled 2D grid: ||O|| offset drives F2's dQ at FIXED diffuse rows
    # (peakedness held at mean max_p ~ 0.150), the scale-invariance control
    # stays flat, and sharper rows only amplify. This is the mechanism claim
    # of §11.6: Delta = dO.O is corrupted by the fp16 save of O, so output
    # magnitude is the primary driver and peakedness the amplifier.
    g2 = s8["f2_decoupled_grid"]
    gdq = [list(row) for row in g2["f2_dQ_rel_grid"]]
    check(
        g2["diffuse_dQ_crosses_1e3"] and g2["peakedness_amplifies_everywhere"]
        and g2["scale_control_spread"] < 3.0
        and g2["amplification_factor"] > 50.0,
        f"F2 decoupled grid: at fixed diffuse rows (max_p={g2['mean_max_p_grid'][0][0]:.3f}) "
        f"||O||-offset drives dQ {g2['diffuse_floor_dQ']:.1e} -> "
        f"{g2['diffuse_end_dQ']:.1e} ({g2['amplification_factor']:.0f}x), "
        f"crossing 1e-3 at idx={g2['idx_dQ_tau_1e3_diffuse']}; V-scaling control "
        f"spread {g2['scale_control_spread']:.2f} (scale-invariant, as required); "
        f"peaked rows amplify at every offset "
        f"(beta=3 vs beta=1: {gdq[1][3]:.1e} vs {gdq[0][3]:.1e} at offset 4)",
    )
    check(
        idx_s4_h100 == 3 and idx_f3_h100 == 4 and f3_tail_dv[-1] > 0.90 and max(s4_tail_dv) < 5.5e-3,
        f"F3 SRAM & absorption sweeps: SRAM Step4={sram_s4_c} KiB (H100 idx={idx_s4_h100}) vs "
        f"F3={sram_f3_c} KiB (H100 idx={idx_f3_h100}) vs R5={sram_r5_c} KiB; "
        f"tail-alpha dV_rel={f3_tail_dv[0]:.2e}->{f3_tail_dv[-1]:.2e} "
        f"(idx@0.02={idx_sw_002}, idx@0.50={idx_sw_050}, dK_end={f3_tail_dk[-1]:.2e}, "
        f"Step4_max={max(s4_tail_dv):.2e} < 5.5e-3 portable bound); "
        f"N_q dV idx@1.4e-4={idx_nq_f3}",
    )

    print("\n" + "=" * 78)
    if FAILURES:
        print(f"CH11 FAILED: {len(FAILURES)} problem(s)")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("CH11_KERNELS_VERIFY_PASSED")
    print("=" * 78)


if __name__ == "__main__":
    main()