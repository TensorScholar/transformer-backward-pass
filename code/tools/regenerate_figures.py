"""Regenerate Chapter 7 manuscript figures from checked-in CSVs.

Run from any directory:
    python3 code/tools/regenerate_figures.py

No publisher art direction is imposed; these are semantic figures. Every axis
choice here is deliberate and commented where it could mislead:

  * depth adjoint figure starts at zero, so a ~1.8x growth is not visually
    inflated and the non-monotone 1->2 block interval is visible;
  * the update-to-weight figure plots only the steps at which the metric is
    actually defined, instead of drawing gaps through NaN;
  * the attention score/entropy figures use log-log axes, because the claim
    under test is a power law in d_h and a linear axis hides it;
  * the normalization-sensitivity figure gets its own dedicated output rather
    than overloading the fork figure, and marks the chance-level line.

No plotting style or fixed colors are specified; publisher art direction can
replace these later.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]   # <root>
CODE = ROOT / "code"
CH07 = CODE / "ch07"
FIG = ROOT / "figures"

WRITTEN: list[str] = []


def _finish(fig, filename):
    fig.tight_layout()
    fig.savefig(FIG / filename, format="svg")
    plt.close(fig)
    WRITTEN.append(filename)


def line_plot(x, series, labels, title, xlabel, ylabel, filename,
              marker="o", xscale=None, yscale=None, zero_base=False,
              axhline=None, axhline_label=None, xticks=None):
    fig = plt.figure(figsize=(7.2, 4.6))
    ax = fig.add_subplot(111)
    for y, label in zip(series, labels):
        ax.plot(x, y, marker=marker, label=label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xscale:
        ax.set_xscale(xscale)
    if yscale:
        ax.set_yscale(yscale)
    if zero_base:
        # Start the y axis at zero. A truncated baseline on a growth claim
        # visually amplifies the effect and can hide a non-monotone interval.
        lo, hi = ax.get_ylim()
        ax.set_ylim(0 if lo > 0 else lo, hi)
    if xticks is not None:
        ax.set_xticks(xticks)
    if axhline is not None:
        ax.axhline(axhline, color="0.35", linestyle="--", linewidth=1.1,
                   label=axhline_label)
    if len(labels) > 1 or axhline is not None:
        ax.legend()
    ax.grid(True, alpha=0.25, which="both")
    _finish(fig, filename)


# ---------------------------------------------- attention scaling (§7.4) ----
# The claim is std ~ sqrt(d_h): a power law. Linear axes put the three widths
# far apart and flatten the two "scaled" series onto each other, so use log-log.
att = pd.read_csv(CH07 / "ch07_attention_scaling_results.csv")
agg = att.groupby(["d_h", "variant"], as_index=False).agg(
    score_std=("score_std", "mean"),
    entropy=("normalized_entropy", "mean"),
)
widths = sorted(agg.d_h.unique())
assert set(agg.variant.unique()) == {"scaled", "unscaled"}, (
    f"unexpected variants {set(agg.variant.unique())}"
)
for metric, title, ylabel, filename in [
    ("score_std", "Attention score scale at initialization",
     "Mean score standard deviation", "ch07_attention_score_std.svg"),
    ("entropy", "Attention entropy at initialization",
     "Mean normalized entropy", "ch07_attention_entropy.svg"),
]:
    ys, labels = [], []
    for variant in ["scaled", "unscaled"]:
        sub = agg[agg.variant == variant].set_index("d_h").reindex(widths)
        assert sub[metric].notna().all(), f"{variant} missing a d_h value"
        ys.append(sub[metric].to_numpy())
        labels.append(variant)
    line_plot(widths, ys, labels, title, "Head width d_h", ylabel, filename,
              xscale="log", yscale="log", xticks=widths)

# ------------------------------------------------ init scale (§7.6, §7.10) ----
init = pd.read_csv(CH07 / "ch07_init_scale_results.csv")
steps = sorted(init.step.unique())
for metric, title, ylabel, filename in [
    ("train_loss_preupdate", "Training loss under initialization-scale interventions",
     "Training loss", "ch07_init_scale_loss.svg"),
    ("L3.resid_after_ffn_rms", "Residual-stream RMS under initialization-scale interventions",
     "Layer-3 residual RMS", "ch07_init_scale_residual_rms.svg"),
]:
    ys, labels = [], []
    for std in sorted(init.init_std.unique()):
        sub = init[init.init_std == std].groupby("step")[metric].mean()
        assert sub.notna().all(), f"{metric} has missing steps at init_std={std}"
        ys.append(sub.reindex(steps).to_numpy())
        labels.append(f"init std={std:g}")
    line_plot(steps, ys, labels, title, "Training step", ylabel, filename,
              marker=None)

# The optimizer-movement metrics are only defined at declared checkpoints, so
# the CSV holds NaN elsewhere. Plotting the full step axis produced a figure
# that was 55 gaps around 5 points. Restrict the x axis to the defined steps and
# label the sparsity instead of hiding it.
for metric, title, ylabel, filename, yscale in [
    ("update_to_weight", "Global update-to-weight ratio at measured checkpoints",
     "Update / weight norm", "ch07_init_scale_update_ratio.svg", "log"),
    ("update_norm", "Optimizer update norm at measured checkpoints",
     "Update norm", "ch07_init_scale_update_norm.svg", None),
]:
    ys, labels, defined_steps = [], [], None
    for std in sorted(init.init_std.unique()):
        sub = init[init.init_std == std].groupby("step")[metric].mean()
        sub = sub.dropna()
        if defined_steps is None:
            defined_steps = sorted(sub.index)
        else:
            assert sorted(sub.index) == defined_steps, (
                f"{metric} is measured at different steps for init_std={std}"
            )
        ys.append(sub.reindex(defined_steps).to_numpy())
        labels.append(f"init std={std:g}")
    line_plot(
        defined_steps, ys, labels,
        f"{title}\n(measured only at checkpoints 0, 1, 5, 20, 59; not a per-step curve)",
        "Training step", ylabel, filename, marker="o", yscale=yscale,
        xticks=defined_steps,
    )

# Per-group update-to-weight ratio: the interpretable view, because the global
# ratio's denominator is dominated by the unit-initialized norm gains.
group_cols = [c for c in init.columns if c.startswith("update_to_weight__")]
if group_cols:
    sel = [c for c in group_cols]
    xs, series, labels = [], [], []
    grp = init[init.step.isin(defined_steps)].groupby("step")
    for col in sel:
        series.append(grp[col].mean().reindex(defined_steps).to_numpy())
        labels.append(col.replace("update_to_weight__", ""))
    line_plot(
        defined_steps, series, labels,
        "Update-to-weight ratio by active parameter group at measured checkpoints\n"
        "the global ratio is dominated by the norm-gain denominator",
        "Training step", "Update / weight norm",
        "ch07_init_scale_update_ratio_by_group.svg",
        marker="o", yscale="log", xticks=defined_steps,
    )

# ------------------------------------------------------------ depth (§7.8) ----
depth = pd.read_csv(CH07 / "ch07_depth_residual_results_v1.2.csv")
d = depth.groupby("depth", as_index=False).agg(
    input_grad=("embedding_output_grad_rms", "mean"),
    block0_grad=("block0_after_ffn_grad_rms", "mean"),
    resid=("last_resid_rms", "mean"),
    input_sd=("embedding_output_grad_rms", "std"),
    block0_sd=("block0_after_ffn_grad_rms", "std"),
)
depths = d.depth.to_numpy()
# error bars: the seed spread is what makes the 1->2 block interval a tie, so it
# must be visible rather than implied.
fig = plt.figure(figsize=(7.2, 4.6))
ax = fig.add_subplot(111)
ax.errorbar(depths, d.input_grad, yerr=d.input_sd, marker="o", capsize=3,
            label="dL/dX0 RMS")
ax.errorbar(depths, d.block0_grad, yerr=d.block0_sd, marker="s", capsize=3,
            label="block-0 output adjoint RMS")
ax.set_title("Early adjoints under paired depth changes\n"
             "mean +/- sample SD over 8 paired seeds; y axis starts at zero")
ax.set_xlabel("Number of blocks")
ax.set_ylabel("Adjoint RMS")
ax.set_xticks(depths)
ax.set_xscale("log")
# Deliberately LINEAR and zero-based. A log axis cannot include zero, and a
# truncated non-zero baseline on a growth claim inflates the apparent effect
# and hides the non-monotone 1->2 block interval. The seed spread is drawn as
# error bars so the reader can see that interval is a tie.
ax.set_ylim(0, max(d.input_grad + d.input_sd) * 1.15)
ax.legend()
ax.grid(True, alpha=0.25)
_finish(fig, "ch07_depth_early_gradients.svg")

line_plot(depths, [d.resid.to_numpy()], ["last residual RMS"],
          "Residual scale under paired depth changes", "Number of blocks",
          "Residual RMS", "ch07_depth_residual.svg", xscale="log",
          xticks=depths)

# Normalized companion: the early adjoint per unit of residual scale. This is
# the quantity that actually declines across depth, and hiding it behind the
# unnormalized plot would be the misleading choice.
line_plot(
    depths,
    [(d.input_grad / d.resid).to_numpy(), (d.block0_grad / d.resid).to_numpy()],
    ["dL/dX0 / last residual RMS", "block-0 adjoint / last residual RMS"],
    "Early adjoint per unit of residual scale under paired depth changes\n"
    "the normalized view declines where the absolute view rises",
    "Number of blocks", "Adjoint RMS / residual RMS",
    "ch07_depth_normalized_adjoints.svg", xscale="log", xticks=depths,
    zero_base=True,
)

# --------------------------------------------------- normalization fork (§7.9) ----
fork = pd.read_csv(CH07 / "ch07_normalization_fork_results.csv")
f = fork[fork.step == fork.step.max()]
pivot = f.groupby("condition")["val_loss_postupdate"].mean().sort_index()
line_plot(
    list(range(len(pivot))), [pivot.to_numpy()], [pivot.index[0]],
    "Matched-recipe fork: held-out loss by condition at the final step",
    "condition (see table in the chapter)", "Held-out loss",
    "ch07_normalization_fork.svg", marker="o", zero_base=False,
)

# ------------------------------------------- normalization sensitivity (§7.9) ----
ns = pd.read_csv(CH07 / "ch07_norm_sensitivity_results_v1.2.csv")
grid = ns[ns.kind == "grid"]
chance = float(np.log(32))
fig = plt.figure(figsize=(7.6, 4.8))
ax = fig.add_subplot(111)
for arch, style in [("full_norm", "-o"), ("no_norm", "--s")]:
    for init_std in sorted(grid.init_std.unique()):
        sub = grid[(grid.arch == arch) & (grid.init_std == init_std)]
        g = sub.groupby("lr")["val_loss_postupdate"].agg(["mean", "std"])
        g = g.sort_index()
        ax.errorbar(g.index, g["mean"], yerr=g["std"], fmt=style, capsize=3,
                    label=f"{arch}, init={init_std:g}")
ax.axhline(chance, color="0.35", linestyle=":", linewidth=1.3,
           label=f"chance level ln(V=32) = {chance:.3f}")
ax.set_xscale("log")
# One grid cell diverges (held-out loss ~23 with a seed SD of ~47). Left
# unhandled it stretches the axis and flattens every non-diverged curve into an
# unreadable band. Bound the axis to the non-diverged range and label the
# off-scale point explicitly rather than dropping it.
finite_max = float(grid["val_loss_postupdate"].replace([np.inf, -np.inf], np.nan)
                   .dropna().max())
bounded = grid.groupby(["arch", "init_std", "lr"])["val_loss_postupdate"] \
              .mean().sort_values()
off_scale = bounded[bounded > 4.0]
ax.set_ylim(min(2.6, chance - 0.4), 4.0)
if len(off_scale):
    desc = "; ".join(
        f"{a}/init={i:g}/lr={lr:g} = {v:.1f}"
        for (a, i, lr), v in off_scale.items()
    )
    ax.text(0.02, 0.03,
            f"off scale (diverged): {desc}\ny axis clipped at 4.0; "
            f"chance = {chance:.3f}",
            transform=ax.transAxes, fontsize=7.5, va="bottom", color="0.25")
del finite_max
ax.set_title("Held-out loss across normalization, learning rate, and initialization\n"
             "mean +/- sample SD over 6 paired seeds; every optimum is interior")
ax.set_xlabel("Learning rate")
ax.set_ylabel("Held-out loss")
ax.legend()
ax.grid(True, alpha=0.25, which="both")
_finish(fig, "ch07_norm_sensitivity.svg")

iso = ns[ns.kind == "isolation"]
g = iso.groupby("arch")["val_loss_postupdate"].agg(["mean", "std"])
order = ["full_norm", "block_only", "final_only", "no_norm"]
g = g.reindex([o for o in order if o in g.index])
fig = plt.figure(figsize=(7.2, 4.4))
ax = fig.add_subplot(111)
ax.bar(range(len(g)), g["mean"].to_numpy(), yerr=g["std"].to_numpy(), capsize=4)
ax.axhline(chance, color="0.35", linestyle=":", linewidth=1.3)
ax.set_xticks(range(len(g)))
ax.set_xticklabels(g.index, rotation=12, ha="right")
ax.set_ylim(bottom=0)
ax.set_title("Where the normalization belongs: block norms vs final RMSNorm\n"
             "held-out loss, mean +/- sample SD over 6 seeds; dashed line = chance")
ax.set_ylabel("Held-out loss")
ax.grid(True, alpha=0.25, axis="y")
_finish(fig, "ch07_norm_placement.svg")

# ------------------------------------------------- clipping frequency (§7.10) ----
clip = pd.read_csv(CH07 / "ch07_clipping_results.csv")
freq = clip.groupby(["condition", "seed"], as_index=False)["clip_fraction"].max()
f = freq.groupby("condition", as_index=False)["clip_fraction"].mean()
fig = plt.figure(figsize=(7.2, 4.4))
ax = fig.add_subplot(111)
ax.bar(range(len(f)), f["clip_fraction"].to_numpy())
ax.set_xticks(range(len(f)))
ax.set_xticklabels(f.condition, rotation=12, ha="right", fontsize=8)
ax.set_ylim(0, 1.05)
ax.set_title("Cumulative fraction of steps whose global gradient exceeded the cap\n"
             "threshold 2.5, at the final recorded step")
ax.set_ylabel("Cumulative clipping frequency")
ax.grid(True, alpha=0.25, axis="y")
_finish(fig, "ch07_clipping_frequency.svg")

print(f"wrote {len(WRITTEN)} Chapter 7 figures to {FIG}")
for name in WRITTEN:
    print(f"  {name}")
