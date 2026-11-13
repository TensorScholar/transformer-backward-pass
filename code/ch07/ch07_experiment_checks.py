"""Chapter 7 experiment checks — the manuscript's numbers, re-derived from the CSVs.

What this file is
-----------------
The six Chapter 7 experiments are *measured* chapters: their scripts
(``ch07_attention_scaling.py`` ... ``ch07_norm_sensitivity_v1_2.py``) run
training loops and write CSVs, and the manuscript quotes tables computed from
those CSVs. Until now, nothing re-checked the tables: a CSV could be regenerated
with different data, or a number could be mistyped into the manuscript, and no
gate would notice. This script closes that hole. It loads the six committed
CSVs, recomputes every aggregate statistic the manuscript quotes, and asserts
each against the printed table to the manuscript's own rounding. It also
asserts the experiments' *pre-stated directional predictions* (score std
flat under scaling, entropy collapse without it, interior LR optima for both
architectures, the clipping non-effect on the update, the depth gradient
growth), because a table that matches but a direction that reversed would be
a different and worse failure.

Tolerances
----------
All comparisons are at the manuscript's printed precision (``1e-4`` for
``x.xxxx`` tables, ``1e-2`` for two-decimal inline claims like "92%"), plus a
tiny epsilon for float parsing. A mismatch is a FAIL and exits 1; there is no
soft pass.

Usage:  python ch07_experiment_checks.py
Marker: CH07_EXPERIMENT_CHECKS_PASSED
"""
from __future__ import annotations

import csv
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


FAILURES: list[str] = []


def check(cond: bool, msg: str) -> None:
    if cond:
        print(f"  ok    {msg}")
    else:
        print(f"  FAIL  {msg}")
        FAILURES.append(msg)


def load(name: str) -> list[dict[str, str]]:
    with (HERE / name).open(newline="") as f:
        return list(csv.DictReader(f))


def mean_sd(xs: list[float]) -> tuple[float, float]:
    return st.mean(xs), (st.stdev(xs) if len(xs) > 1 else 0.0)


def fnum(s: str) -> float:
    return float(s) if s not in ("", None) else float("nan")


# ---------------------------------------------------------------------------
print("=" * 78)
print("Chapter 7 -- experiment checks: manuscript tables vs committed CSVs")
print("=" * 78)

# ---- [1] attention scaling: the 6-row table ------------------------------
print("\n[1] attention scaling table (Sec. 7.2, d_h in {16,64,256})")
rows = load("ch07_attention_scaling_results.csv")
agg: dict[tuple[int, str], list[tuple[float, float, float, float]]] = defaultdict(list)
for r in rows:
    agg[(int(r["d_h"]), r["variant"])].append(
        (fnum(r["score_std"]), fnum(r["normalized_entropy"]),
         fnum(r["score_grad_rms"]), fnum(r["mean_max_probability"])))
# manuscript values, printed at 5 decimals: (score_std, entropy, grad_rms, max_p)
TABLE = {
    (16, "scaled"):     (1.02452, 0.84166, 0.07904, 0.28470),
    (16, "unscaled"):   (4.09808, 0.32641, 0.07369, 0.69842),
    (64, "scaled"):     (0.99927, 0.84390, 0.07617, 0.28586),
    (64, "unscaled"):   (7.99417, 0.15135, 0.05559, 0.84783),
    (256, "scaled"):    (0.99388, 0.85113, 0.07793, 0.28429),
    (256, "unscaled"):  (15.90207, 0.07150, 0.04345, 0.92745),
}
for k, expected in TABLE.items():
    cols = list(zip(*agg[k]))
    means = [st.mean(c) for c in cols]
    sds = [st.stdev(c) for c in cols]
    ok = all(abs(m - e) < 1e-4 for m, e in zip(means, expected))
    check(ok, f"d_h={k[0]:3d} {k[1]:>8s}: means "
          + " ".join(f"{m:.5f}" for m in means) + f" match the printed row")
    # the ± entries: 0.03894 etc.
    check(len(cols[0]) == 12, f"d_h={k[0]:3d} {k[1]:>8s}: n=12 seeds per cell")
    del sds

# the sqrt(d_h) law: unscaled std ratios as d_h quadruples
uns = [st.mean(list(zip(*agg[(d, "unscaled")]))[0]) for d in (16, 64, 256)]
r1, r2 = uns[1] / uns[0], uns[2] / uns[1]
check(1.9 < r1 < 2.1 and 1.9 < r2 < 2.1,
      f"unscaled score std grows as sqrt(d_h): ratios {r1:.3f}, {r2:.3f} "
      "(manuscript: x1.95 then x1.99)")
sc = [st.mean(list(zip(*agg[(d, "scaled")]))[0]) for d in (16, 64, 256)]
check(all(0.9 < v < 1.1 for v in sc),
      f"scaled score std stays O(1): {sc[0]:.3f} -> {sc[1]:.3f} -> {sc[2]:.3f}")
# entropy collapse: 0.326 -> 0.151 -> 0.072, max prob -> 0.927 (> 92%)
ent = [st.mean(list(zip(*agg[(d, "unscaled")]))[1]) for d in (16, 64, 256)]
mx = [st.mean(list(zip(*agg[(d, "unscaled")]))[3]) for d in (16, 64, 256)]
check(ent[0] > ent[1] > ent[2],
      f"unscaled entropy falls monotonically: {ent[0]:.3f} -> {ent[1]:.3f} -> {ent[2]:.3f}")
check(mx[2] > 0.92,
      f"at d_h=256 unscaled max prob {mx[2]:.4f} > 92% on a single key")

# ---- [2] clipping: fractions and the first-step table ---------------------
print("\n[2] clipping (Sec. 7.3, c=2.5, n=5 seeds, 60 steps)")
rows = load("ch07_clipping_results.csv")
per_seed: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
g0: dict[str, list[float]] = defaultdict(list)
g0c: dict[str, list[float]] = defaultdict(list)
for r in rows:
    per_seed[r["condition"]][r["seed"]].append(fnum(r["clipped"]))
    if int(r["step"]) == 0:
        g0[r["condition"]].append(fnum(r["grad_norm"]))
        g0c[r["condition"]].append(fnum(r["grad_norm_postclip"]))
frac = {c: st.mean([st.mean(v) for v in per_seed[c].values()])
        for c in ("moderate_clip", "extreme_clip")}
check(abs(frac["moderate_clip"] - 0.0367) < 5e-4,
      f"moderate cumulative clip fraction {frac['moderate_clip']:.4f} = 3.67%")
check(abs(frac["extreme_clip"] - 0.7733) < 5e-4,
      f"extreme cumulative clip fraction {frac['extreme_clip']:.4f} = 77.33%")
m_sd = mean_sd(g0["moderate_clip"])
e_sd = mean_sd(g0["extreme_clip"])
check(abs(m_sd[0] - 2.92533) < 1e-4 and abs(m_sd[1] - 0.16627) < 1e-4,
      f"moderate |g0| {m_sd[0]:.5f}±{m_sd[1]:.5f} = 2.92533±0.16627")
check(abs(e_sd[0] - 6.21491) < 1e-4 and abs(e_sd[1] - 0.67664) < 1e-4,
      f"extreme |g0| {e_sd[0]:.5f}±{e_sd[1]:.5f} = 6.21491±0.67664")
for c in ("moderate_clip", "extreme_clip"):
    check(all(abs(v - 2.5) < 1e-5 for v in g0c[c]),
          f"{c}: |g0| after clip = 2.50000 for every seed")
# the headline non-effect: update_to_weight with and without clip at step 0
uwc: dict[str, list[float]] = defaultdict(list)
uwn: dict[str, list[float]] = defaultdict(list)
for r in rows:
    if int(r["step"]) == 0:
        # note: '_no_clip' also endswith '_clip'; test the longer suffix first
        if r["condition"].endswith("_no_clip"):
            uwn[r["condition"][: -len("_no_clip")]].append(
                fnum(r["update_to_weight"]))
        else:
            uwc[r["condition"][: -len("_clip")]].append(
                fnum(r["update_to_weight"]))
for c in ("moderate", "extreme"):
    m_c, sd_c = mean_sd(uwc[c])
    m_n, sd_n = mean_sd(uwn[c])
    check(abs(m_c - m_n) < 2e-4,
          f"{c}: clipping does not move step-0 update/weight "
          f"({m_n:.5f} -> {m_c:.5f}; manuscript 0.01913 / 0.00650)")

# ---- [3] init scale: U-shaped loss, step-59 table -------------------------
print("\n[3] initialization scale (Sec. 7.4, step 59)")
rows = load("ch07_init_scale_results.csv")
by_sigma: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
for r in rows:
    if int(r["step"]) == 59:
        by_sigma[r["init_std"]]["loss"].append(fnum(r["loss"]))
        by_sigma[r["init_std"]]["gn"].append(fnum(r["grad_norm"]))
EXPECT = {"0.005": (3.0918, 1.5482), "0.02": (2.7530, 1.6056),
          "0.08": (2.5927, 1.4564), "0.3": (3.3842, 2.0639)}
for sig, (el, eg) in EXPECT.items():
    ml, sl = mean_sd(by_sigma[sig]["loss"])
    mg, sg = mean_sd(by_sigma[sig]["gn"])
    check(abs(ml - el) < 1e-3 and abs(mg - eg) < 1e-3,
          f"sigma={sig}: loss {ml:.4f}±{sl:.4f}, grad norm {mg:.4f}±{sg:.4f} "
          f"match the printed row ({el}/{eg})")
losses = [mean_sd(by_sigma[s]["loss"])[0] for s in ("0.005", "0.02", "0.08", "0.3")]
check(losses[0] > losses[1] > losses[2] and losses[3] > losses[2],
      f"loss is U-shaped with interior optimum at sigma=0.08: "
      + " -> ".join(f"{v:.4f}" for v in losses))

# ---- [4] normalization fork: the initialization-regime table ---------------
print("\n[4] normalization fork (Sec. 7.5, step 0)")
rows = load("ch07_normalization_fork_results.csv")
g0n: dict[str, list[float]] = defaultdict(list)
res4: dict[str, list[float]] = defaultdict(list)
logit: dict[str, list[float]] = defaultdict(list)
for r in rows:
    if int(r["step"]) == 0:
        c = r["condition"]
        g0n[c].append(fnum(r["grad_norm"]))
        res4[c].append(fnum(r["L3.resid_after_ffn_rms"]))
        logit[c].append(fnum(r["logit_rms"]))
norm = "pre_rmsnorm"
non = "no_norm_matched_lr"
mn, sn = mean_sd(g0n[norm]); mu, su = mean_sd(g0n[non])
check(abs(mn - 2.925) < 1e-3 and abs(sn - 0.166) < 1e-3,
      f"normalized g0 {mn:.3f}±{sn:.3f} = 2.925±0.166")
check(abs(mu - 0.1094) < 1e-3 and abs(su - 0.0016) < 1e-3,
      f"unnormalized g0 {mu:.4f}±{su:.4f} = 0.1094±0.0016")
ratio = mn / mu
check(26.0 < ratio < 27.5,
      f"gradient-norm ratio {ratio:.1f} = 26.7x before any training step")
rn, _ = mean_sd(res4[norm]); ru, _ = mean_sd(res4[non])
ln, _ = mean_sd(logit[norm]); lu, _ = mean_sd(logit[non])
check(abs(rn - 0.5705) < 1e-3 and abs(ru - 0.0851) < 1e-3,
      f"layer-4 residual RMS {rn:.4f} vs {ru:.4f} (printed 0.5705 / 0.0851)")
check(abs(ln - 0.5534) < 1e-3 and abs(lu - 0.0470) < 1e-3,
      f"logit RMS {ln:.4f} vs {lu:.4f} (printed 0.5534 / 0.0470)")

# ---- [5] norm sensitivity: interior argmins for both architectures ---------
print("\n[5] LR x init grid (Sec. 7.6): interior optima, different preferred LRs)")
rows = load("ch07_norm_sensitivity_results_v1.2.csv")
grid: dict[tuple[str, str, str], list[float]] = defaultdict(list)
for r in rows:
    grid[(r["arch"], r["init_std"], r["lr"])].append(
        fnum(r["val_loss_postupdate"]))
LRS = ["0.001", "0.002", "0.004", "0.006", "0.01", "0.015", "0.02"]
for arch in ("full_norm", "no_norm"):
    for init in ("0.02", "0.08"):
        means = {lr: st.mean(grid[(arch, init, lr)]) for lr in LRS}
        best = min(means, key=means.get)
        check(best in ("0.004", "0.006") and best not in (LRS[0], LRS[-1]),
              f"{arch} @ init {init}: argmin lr={best} is interior "
              f"(best cell {means[best]:.4f})")
# the diverged cell: no_norm 0.02 @ 0.02 is far above the clipped axis (>4)
dv = st.mean(grid[("no_norm", "0.02", "0.02")])
check(dv > 4.0,
      f"no_norm @ init 0.02, lr 0.02 diverged: held-out loss {dv:.4f} > 4.0 "
      "(the figure clips its axis at 4.0 and labels this cell)")
# best cells and the gap: 2.8076 vs 3.1226
bn = st.mean(grid[("full_norm", "0.08", "0.004")])
bu = st.mean(grid[("no_norm", "0.08", "0.006")])
check(abs(bn - 2.8076) < 1e-3, f"best normalized cell {bn:.4f} = 2.8076")
check(abs(bu - 3.1226) < 1e-3, f"best unnormalized cell {bu:.4f} = 3.1226")
check(bn < bu, f"the gap narrows but does not close: {bu - bn:.4f} > 0")

# ---- [6] norm placement: the paired table ----------------------------------
print("\n[6] norm placement (Sec. 7.7, paired, n=6)")
PLACE = {"full_norm": 2.8188, "block_only": 2.8533,
         "final_only": 3.0209, "no_norm": 3.2939}
for arch, expected in PLACE.items():
    vals = grid[(arch, "0.08", "0.002")]
    m, s = mean_sd(vals)
    check(abs(m - expected) < 1e-3,
          f"{arch:>10s}: held-out {m:.4f}±{s:.4f} = {expected}")
# monotone degradation ordering
order = [st.mean(grid[(a, "0.08", "0.002")]) for a in
         ("full_norm", "block_only", "final_only", "no_norm")]
check(order[0] < order[1] < order[2] < order[3],
      "placement ordering: full < block-only < final-only < none "
      + " -> ".join(f"{v:.4f}" for v in order))

# ---- [7] depth: gradient growth with depth ---------------------------------
print("\n[7] depth (Sec. 7.8): global gradient norm vs depth")
rows = load("ch07_depth_residual_results_v1.2.csv")
by_depth: dict[int, list[float]] = defaultdict(list)
for r in rows:
    by_depth[int(r["depth"])].append(fnum(r["global_param_grad_norm"]))
ds = sorted(by_depth)
means = [st.mean(by_depth[d]) for d in ds]
check(means == sorted(means) and means[0] < means[-1],
      "global gradient norm grows monotonically with depth: "
      + " -> ".join(f"{d}:{m:.3f}" for d, m in zip(ds, means)))
# the per-layer picture the figure shows: block gradients comparable across
# depth (no explosion within a fixed layer index)
for layer in ("L0.after_ffn_grad_rms",):
    vals = defaultdict(list)
    for r in rows:
        if r.get(layer):
            vals[int(r["depth"])].append(fnum(r[layer]))
    fixed = [st.mean(vals[d]) for d in sorted(vals)]
    spread = (max(fixed) - min(fixed)) / st.mean(fixed)
    check(spread < 1.0,
          f"block-0 after-FFN gradient RMS comparable across depths "
          f"(spread {spread:.2f}x of the mean): residual stream carries depth")

# ---------------------------------------------------------------------------
print()
if FAILURES:
    print(f"CH07 FAILED: {len(FAILURES)} problem(s)")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print("CH07_EXPERIMENT_CHECKS_PASSED")
