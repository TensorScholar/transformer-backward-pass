"""Normalization sensitivity and component isolation, v1.2.

Supersedes ch07_norm_sensitivity_v1_1.py. The v1.1 grid was
(0.001, 0.002, 0.006); both arms' optima were then at or near the grid edge and
the no-normalization arm's argmin was unresolved, so the "two-sided" claim was
not actually closed. v1.2 extends the grid upward until an interior argmin is
observed for every architecture, and raises the seed count from 3 to 6 so the
per-cell standard deviations are not read off a 3-sample spread.

What this does and does not do:

  * It is a sensitivity check over a declared grid, not a global hyperparameter
    search. No claim is made about the best achievable result for either
    architecture.
  * The no-normalization arm is expected to prefer a larger learning rate than
    the normalized arm, because removing the norms shrinks the forward scale and
    the gradient norm. The grid must therefore extend far enough to bracket the
    no-norm optimum, or the comparison is one-sided and the no-norm numbers are
    lower bounds rather than optima.
  * Every architecture receives exactly the same LR list, the same init list,
    the same step budget, and the same data generators. The only differences are
    the two declared architectural switches.

Metric convention: every printed value is mean ± sample standard deviation
across seeds (unbiased / Bessel-corrected). See Chapter 7 Section 7.2.

Usage:  python ch07_norm_sensitivity_v1_2.py
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

import csv
import statistics
import time

from ch07_training_harness import RunConfig, train_run

SEEDS = (0, 1, 2, 3, 4, 5)
STEPS = 30
INITS = (0.02, 0.08)
# Extended upward. v1.1 stopped at 0.006 with the no-norm argmin on the edge.
LRS = (0.001, 0.002, 0.004, 0.006, 0.010, 0.015, 0.020)

VOCAB = 32
CHANCE = __import__("math").log(VOCAB)

rows: list[dict] = []
t0 = time.time()
n_runs = 0


def run(kind, arch, use_block, use_final, init, lr, seed):
    global n_runs
    recs = train_run(
        RunConfig(
            seed=seed, init_std=init, steps=STEPS, lr=lr,
            use_norm=use_block, use_final_norm=use_final,
        )
    )
    n_runs += 1
    r = recs[-1]
    assert int(r["step"]) == STEPS - 1, (
        f"last record is step {r['step']}, expected {STEPS - 1}; "
        "the grid would silently report a non-terminal step"
    )
    assert r.get("val_loss_postupdate") not in (None, "", "nan"), (
        "val_loss_postupdate missing at the final step; "
        "the harness must have skipped a checkpoint"
    )
    rows.append(dict(
        kind=kind, arch=arch, use_block_norm=use_block,
        use_final_norm=use_final, init_std=init, lr=lr, seed=seed, **r,
    ))


# ---------------------------------------------------------------- grid ----
# Identical LR and init list for both architectures.
for arch, use_block, use_final in [
    ("full_norm", True, True),
    ("no_norm", False, False),
]:
    for init in INITS:
        for lr in LRS:
            for seed in SEEDS:
                run("grid", arch, use_block, use_final, init, lr, seed)

# ----------------------------------------------------------- isolation ----
# Block norms vs final RMSNorm at the declared matched recipe.
for arch, use_block, use_final in [
    ("full_norm", True, True),
    ("block_only", True, False),
    ("final_only", False, True),
    ("no_norm", False, False),
]:
    for seed in SEEDS:
        run("isolation", arch, use_block, use_final, 0.08, 0.002, seed)

# --------------------------------------------------------------- write ----
out = Path(__file__).with_name("ch07_norm_sensitivity_results_v1.2.csv")
fields: list[str] = []
for r in rows:
    for k in r:
        if k not in fields:
            fields.append(k)
with out.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)


def cell(kind, arch, init, lr):
    return [
        float(r["val_loss_postupdate"])
        for r in rows
        if r["kind"] == kind and r["arch"] == arch
        and abs(float(r["init_std"]) - init) < 1e-12
        and abs(float(r["lr"]) - lr) < 1e-12
    ]


elapsed = time.time() - t0

print(f"ch07_norm_sensitivity_v1_2  seeds={len(SEEDS)}  steps={STEPS}  "
      f"runs={n_runs}  elapsed={elapsed:.1f}s")
print(f"chance level ln(V={VOCAB}) = {CHANCE:.5f}\n")
print(f"Grid: step-{STEPS - 1} post-update held-out loss, mean±sd over "
      f"n={len(SEEDS)} seeds")

argmins: dict[tuple[str, float], tuple[float, float]] = {}
for arch in ["full_norm", "no_norm"]:
    print(f"\n{arch}")
    for init in INITS:
        best = (float("inf"), None)
        for lr in LRS:
            v = cell("grid", arch, init, lr)
            m, s = statistics.mean(v), statistics.stdev(v)
            flag = ""
            if m < best[0]:
                best = (m, lr)
            print(f"  init={init:.3f} lr={lr:.3f}: {m:.5f}±{s:.5f}{flag}")
        argmins[(arch, init)] = best
        interior = LRS.index(best[1]) not in (0, len(LRS) - 1)
        print(f"    argmin lr={best[1]:.3f} loss={best[0]:.5f} "
              f"({'INTERIOR' if interior else 'ON GRID EDGE - unresolved'})")

print("\nComponent isolation at init=0.08 lr=0.002:")
for arch in ["full_norm", "block_only", "final_only", "no_norm"]:
    rr = [r for r in rows if r["kind"] == "isolation" and r["arch"] == arch]
    v = [float(r["val_loss_postupdate"]) for r in rr]
    g = [float(r["grad_norm"]) for r in rr]
    print(f"  {arch:12s} val={statistics.mean(v):.5f}±{statistics.stdev(v):.5f} "
          f"grad={statistics.mean(g):.5f}±{statistics.stdev(g):.5f}")

# Paired t-statistic against the matched full_norm arm, on held-out loss.
base = [float(r["val_loss_postupdate"]) for r in rows
        if r["kind"] == "isolation" and r["arch"] == "full_norm"]
print("\nPaired comparison vs full_norm (held-out loss, same seed):")
for arch in ["block_only", "final_only", "no_norm"]:
    other = [float(r["val_loss_postupdate"]) for r in rows
             if r["kind"] == "isolation" and r["arch"] == arch]
    d = [o - b for o, b in zip(other, base)]
    md = statistics.mean(d)
    sd = statistics.stdev(d)
    t = md / (sd / len(d) ** 0.5) if sd > 0 else float("nan")
    print(f"  {arch:12s} delta={md:+.5f}  t({len(d) - 1})={t:+.2f}  "
          f"all_same_sign={all(x > 0 for x in d) or all(x < 0 for x in d)}")

unresolved = [k for k, (_, lr) in argmins.items()
              if LRS.index(lr) in (0, len(LRS) - 1)]
if unresolved:
    print(f"\nWARNING: argmin on grid edge for {unresolved} - "
          "grid does not bracket the optimum")
else:
    print("\nAll architectures have an interior argmin: the grid brackets "
          "every optimum in range.")

print(f"\nwrote {out.name} ({len(rows)} rows)")
