"""Chapter 12 verification -- GQA: cache footprint, views, and the merge.

Turns every claim in the chapter into an assertion:

  1. MHA, GQA and MQA all compute the same attention shape, and the cache
     footprint is PROPORTIONAL to H_kv with d_h held fixed;
  2. expand allocates ZERO extra bytes -- it is a stride-0 view sharing storage
     with the source -- while repeat_interleave allocates exactly G times;
  3. the two broadcast mechanisms are numerically IDENTICAL, so the choice is
     purely about memory;
  4. reshape() of an expanded view SILENTLY materialises the copy, defeating the
     saving -- the most common way GQA fails;
  5. the gradient at one KV head equals the sum over its G query heads, to
     float32 rounding, against autograd;
  6. the same merge holds for V.

Usage:  python ch12_gqa_verify.py
Marker: CH12_GQA_VERIFY_PASSED
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
for _p in (_HERE.parent, _HERE.parents[1]):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch                                                  # noqa: E402

from tgd import gqa as G                                      # noqa: E402

FAILURES: list[str] = []


def check(cond: bool, msg: str) -> None:
    if cond:
        print(f"  ok    {msg}")
    else:
        print(f"  FAIL  {msg}")
        FAILURES.append(msg)


def main() -> None:
    print("=" * 78)
    print("Chapter 12 -- grouped-query attention")
    print("d_h = d_model / n_q_heads is held FIXED across all three variants;")
    print("the cache saving comes entirely from narrowing H_kv.")
    print("=" * 78)

    # ---- 1. the cache footprint ----------------------------------------
    print("\n[1] KV cache footprint at B=8, N=32768, d=128, H_q=32, float16")
    rows = G.kv_cache_report(batch=8, seq=32768, d_model=128, n_q=32, n_layers=1)
    print(f"    {'variant':>7} {'H_kv':>5} {'d_h':>4} {'G':>5} "
          f"{'KV width':>9} {'bytes':>14} {'MiB':>9} {'saving':>8}")
    for r in rows:
        print(f"    {r['name']:>7} {r['n_kv']:>5} {r['d_h']:>4} {r['group']:>5} "
              f"{r['kv_width']:>9} {r['bytes']:>14,} {r['MiB']:>9.2f} "
              f"{r['saving_vs_mha']:>7.1f}x")
    mha, gqa, mqa = rows
    check(mha["kv_width"] == 128 and gqa["kv_width"] == 32 and mqa["kv_width"] == 4,
          "KV width narrows as H_kv falls: 128 -> 32 -> 4")
    check(abs(gqa["saving_vs_mha"] - 4.0) < 1e-9,
          f"GQA(H_kv=8) saves exactly 4x over MHA "
          f"({gqa['saving_vs_mha']:.2f}x)")
    check(abs(mqa["saving_vs_mha"] - 32.0) < 1e-9,
          f"MQA(H_kv=1) saves exactly 32x over MHA "
          f"({mqa['saving_vs_mha']:.2f}x)")
    check(mqa["bytes"] == 4 * 2**20,
          f"MQA footprint is exactly 4 MiB ({mqa['bytes']:,} bytes)")
    print(f"    for reference: 8 batched 32K sequences of KV at d=128 need "
          f"{mha['MiB']:.1f} MiB under MHA")
    print("    and Chapter 9's formula gives the same number:")
    print(f"      2 * L * d_kv * B * N * e = 2 * 1 * {mha['kv_width']} * 8 * "
          f"32768 * 2 = {mha['bytes']:,}")

    # multi-layer, for scale
    for L in (32, 80):
        c = G.cache_bytes(G.AttnConfig(8, 32768, 128, 32, 32), L, 32768)
        cg = G.cache_bytes(G.AttnConfig(8, 32768, 128, 32, 8), L, 32768)
        print(f"    L={L:>3}: MHA {c / 2**30:8.2f} GB    GQA {cg / 2**30:8.2f} GB"
              f"    saving {c / cg:.1f}x")

    # ---- 2 & 3. views versus copies -------------------------------------
    print("\n[2] the broadcast: expand allocates nothing, repeat_interleave "
          "allocates G times")
    for n_kv in (32, 8, 1):
        a = G.broadcast_allocation(B=8, T=512, d_model=128, n_q=32, n_kv=n_kv)
        cfg = a["config"]
        print(f"    H_kv={n_kv:>2} (G={cfg.group:>2}): source "
              f"{a['source_bytes'] / 1e6:7.3f} MB  expand "
              f"{a['expand_bytes'] / 1e6:7.3f} MB  repeat_interleave "
              f"{a['repeat_bytes'] / 1e6:7.3f} MB  ratio {a['ratio']:5.1f}x")
        check(a["extra_from_expand"] == 0,
              f"H_kv={n_kv}: expand allocates EXACTLY 0 extra bytes "
              f"({a['extra_from_expand']})")
        check(a["extra_from_repeat"] == a["source_bytes"] * (cfg.group - 1),
              f"H_kv={n_kv}: repeat_interleave allocates G-1 extra copies "
              f"({a['extra_from_repeat']:,} B = source x {cfg.group - 1})")
        check(a["identical_values"],
              f"H_kv={n_kv}: both mechanisms are numerically IDENTICAL, so "
              f"the choice is purely about memory")
        check(a["expand_is_view"],
              f"H_kv={n_kv}: expand shares storage with the source (it is a "
              f"view, not a tensor)")

    # ---- 4. the silent trap ---------------------------------------------
    print("\n[3] reshape() of an expanded view SILENTLY materialises the copy")
    cfg = G.AttnConfig(batch=8, seq=512, d_model=128, n_q_heads=32,
                       n_kv_heads=8)
    K = torch.randn(8, 8, 512, cfg.d_h)
    view = G.broadcast_kv(K, cfg, "expand")
    reshaped = view.reshape(8, 32, 512, cfg.d_h)
    print(f"    expanded view storage     : {G.storage_nbytes(view):,} B")
    print(f"    after .reshape()          : {G.storage_nbytes(reshaped):,} B")
    print(f"    logical size of both      : {reshaped.numel() * 2:,} B")
    check(G.storage_nbytes(view) == G.storage_nbytes(K),
          "the expanded view costs nothing")
    check(G.storage_nbytes(reshaped) > G.storage_nbytes(K),
          f"but reshape() copies it: {G.storage_nbytes(reshaped) / G.storage_nbytes(K):.0f}x "
          f"the source, silently undoing the entire GQA saving")

    # ---- 5 & 6. the gradient merge --------------------------------------
    print("\n[4] the gradient at a KV head is the SUM over its G query heads")
    for (B, T, d_model, n_q, n_kv) in ((2, 16, 32, 8, 2), (2, 16, 64, 8, 4),
                                       (1, 32, 64, 16, 4)):
        r = G.gradient_merge_check(B, T, d_model, n_q, n_kv)
        cfg, Gp = r["config"], r["group"]
        print(f"    B={B} T={T} d={d_model} H_q={n_q} H_kv={n_kv} (G={Gp})")
        print(f"      manual vs autograd:  dQ {r['err_dQ_manual_vs_autograd']:.3e}"
              f"   dK {r['err_dK_manual_vs_autograd']:.3e}"
              f"   dV {r['err_dV_manual_vs_autograd']:.3e}")
        for ph in r["per_head"]:
            print(f"      KV head {ph['kv_head']} serves Q heads {ph['q_heads']}: "
                  f"sum of per-head adjoints deviates by "
                  f"{ph['max_dev_from_sum']:.3e}")
        tol = 1e-5
        check(r["err_dK_manual_vs_autograd"] < tol,
              f"G={Gp}: hand-written dK matches autograd to "
              f"{r['err_dK_manual_vs_autograd']:.2e}")
        check(r["err_dV_manual_vs_autograd"] < tol,
              f"G={Gp}: hand-written dV matches autograd to "
              f"{r['err_dV_manual_vs_autograd']:.2e}")
        check(r["worst_sum_deviation"] < tol,
              f"G={Gp}: dK[head] == sum of its {Gp} query heads' contributions, "
              f"worst deviation {r['worst_sum_deviation']:.2e}")

    # ---- the capacity argument, made concrete ---------------------------
    print("\n[5] the capacity trade, measured")
    for n_kv in (32, 8, 4, 1):
        cfg = G.AttnConfig(batch=2, seq=64, d_model=128, n_q_heads=32,
                           n_kv_heads=n_kv)
        kv_params = 2 * cfg.kv_width * 128      # K and V projections, d_model=128
        q_params = cfg.n_q_heads * cfg.d_h * 128
        print(f"    H_kv={n_kv:>2} (G={cfg.group:>2}): KV projection params "
              f"{kv_params:>6}   Q projection params {q_params:>6}   "
              f"KV share {kv_params / (kv_params + q_params):.1%}")
    print("    the KV WIDTH, not the parameter count, is what the cache pays for")

    print("\n" + "=" * 78)
    if FAILURES:
        print(f"CH12 FAILED: {len(FAILURES)} problem(s)")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("CH12_GQA_VERIFY_PASSED")
    print("=" * 78)


if __name__ == "__main__":
    main()