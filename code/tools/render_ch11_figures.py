"""Render Chapter 11 publication-quality SVG figures directly from verified sweeps.

Usage:
    python code/tools/render_ch11_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).resolve()
ROOT = _HERE.parents[2]
CODE = ROOT / "code"
FIG = ROOT / "figures"

if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from tgd.kernels import verify_step8_parametric_sweeps


def style_axis(ax: plt.Axes) -> None:
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.tick_params(direction="out", length=4, width=0.8)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def render_fig_11a(data: dict) -> Path:
    f1 = data["f1"]
    x = [1, 2, 4, 8]
    g_curve = f1["g_curve_dK_rel"]
    nq_curve = f1["nq_curve_dK_rel"]
    diag_curve = f1["diag_curve_dK_rel"]
    max_s4 = f1["max_step4_dK_rel"]

    fig, ax = plt.subplots(figsize=(6.8, 4.6), dpi=300)

    # Production reference baseline
    ax.axhspan(1e-9, max_s4 * 1.5, color="#2ca02c", alpha=0.15, label=f"Production kernel (≤ {max_s4:.1e})")
    ax.axhline(max_s4, color="#2ca02c", linestyle=":", linewidth=1.2)

    # F1 curves
    ax.plot(x, g_curve, marker="o", color="#d62728", linewidth=1.8, label="F1: G sweep ($N_q = 1$)")
    ax.plot(x, nq_curve, marker="s", color="#ff7f0e", linewidth=1.8, label="F1: $N_q$ sweep ($G = 1$)")
    ax.plot(x, diag_curve, marker="^", color="#9467bd", linewidth=1.8, label="F1: Diagonal ($G = N_q$)")

    # Threshold lines
    ax.axhline(0.25, color="#555555", linestyle="--", linewidth=1.0)
    ax.text(8.15, 0.25, r"$\tau_1 = 0.25$", verticalalignment="center", fontsize=8.5, color="#333333")

    ax.axhline(0.65, color="#555555", linestyle="--", linewidth=1.0)
    ax.text(8.15, 0.65, r"$\tau_2 = 0.65$", verticalalignment="center", fontsize=8.5, color="#333333")

    ax.axhline(0.70, color="#777777", linestyle=":", linewidth=1.0)
    ax.text(8.15, 0.70, r"$\tau = 0.70$", verticalalignment="bottom", fontsize=8.0, color="#555555")

    # Annotate blind spot
    ax.scatter([1], [g_curve[0]], color="#d62728", s=60, zorder=5)
    ax.annotate(
        "Bitwise equal to Production\nat (1, 1): exact 0.0 diff",
        xy=(1, g_curve[0]),
        xytext=(1.25, 5e-7),
        arrowprops=dict(arrowstyle="->", color="#333333", lw=0.8),
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.3", fc="#f8f9fa", ec="#cccccc", lw=0.6),
    )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(["1", "2", "4", "8"])
    ax.set_xlim(0.85, 10.5)
    ax.set_ylim(1e-8, 1.5)

    ax.set_xlabel("Fan-in dimension ($G$, $N_q$, or $G = N_q$)", fontsize=10)
    ax.set_ylabel("Relative $\\bar{K}$ error vs R1", fontsize=10)
    ax.set_title("Figure 11.A: F1 Fan-In Error Surface Slices & Interior Crossings", fontsize=11, fontweight="bold")
    style_axis(ax)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9)

    fig.tight_layout()
    out_path = FIG / "ch11_f1_fan_in.svg"
    fig.savefig(out_path, format="svg")
    plt.close(fig)
    return out_path


def render_fig_11b(data: dict) -> Path:
    f2 = data["f2"]
    betas = f2["betas"]
    mean_p = f2["mean_max_p_curve"]
    f2_dq = f2["f2_dQ_rel_curve"]
    f2_dk = f2["f2_dK_rel_curve"]
    s4_dq = f2["production_dQ_rel_curve"]
    s4_dk = f2["production_dK_rel_curve"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 5.8), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.0]}, dpi=300)

    # Top panel: Error curves
    ax1.plot(betas, f2_dq, marker="o", color="#d62728", linewidth=1.8, label=r"F2 $\bar{Q}$ error ($save\_O = \mathrm{fp16}$)")
    ax1.plot(betas, f2_dk, marker="s", color="#ff7f0e", linewidth=1.8, label=r"F2 $\bar{K}$ error ($save\_O = \mathrm{fp16}$)")
    ax1.plot(betas, s4_dq, marker="^", color="#2ca02c", linewidth=1.5, linestyle="--", label=r"Production $\bar{Q}$ error ($save\_O = \mathrm{fp32}$)")
    ax1.plot(betas, s4_dk, marker="v", color="#1f77b4", linewidth=1.5, linestyle="--", label=r"Production $\bar{K}$ error ($save\_O = \mathrm{fp32}$)")

    # Thresholds
    ax1.axhline(2e-4, color="#555555", linestyle="--", linewidth=1.0)
    ax1.text(0.26, 2.2e-4, r"$\tau_1 = 2\times 10^{-4}$ (crossed at $\beta=1.0$)", fontsize=8.5, color="#333333")

    ax1.axhline(1e-3, color="#555555", linestyle="--", linewidth=1.0)
    ax1.text(0.26, 1.1e-3, r"$\tau_2 = 1\times 10^{-3}$ (crossed at $\beta=4.0$)", fontsize=8.5, color="#333333")

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_ylim(5e-8, 1e-2)
    ax1.set_ylabel("Relative gradient error vs R1", fontsize=10)
    ax1.set_title("Figure 11.B: F2 Softmax Peakedness Cancellation Cliff", fontsize=11, fontweight="bold")
    style_axis(ax1)
    ax1.legend(loc="lower right", fontsize=8.0, framealpha=0.9)

    # Bottom panel: Mean max probability
    ax2.plot(betas, mean_p, marker="D", color="#4b5563", linewidth=1.5)
    ax2.set_xscale("log")
    ax2.set_ylim(0.0, 1.05)
    ax2.set_ylabel(r"$\mathbb{E}_i[\max_j P_{ij}]$", fontsize=9.5)
    ax2.set_xlabel(r"Inverse temperature / logit scale parameter $\beta = s^2$", fontsize=10)
    ax2.set_xticks(betas)
    ax2.set_xticklabels([f"{b:g}" for b in betas])
    style_axis(ax2)

    # Note on bitwise identical V_bar
    ax1.text(
        0.02, 0.95,
        r"F2 $\bar{V}$ & forward $(O, \mathrm{lse})$ bitwise identical to Production ($\mathrm{rel\_err} \leq 5.5\times 10^{-7}$)",
        transform=ax1.transAxes,
        fontsize=8.0,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.25", fc="#ffffff", ec="#cccccc", lw=0.5),
    )

    fig.tight_layout()
    out_path = FIG / "ch11_f2_peakedness.svg"
    fig.savefig(out_path, format="svg")
    plt.close(fig)
    return out_path


def render_fig_11c(data: dict) -> Path:
    f3 = data["f3"]
    bkv = f3["bkv_vals"]
    sram_s4 = f3["sram_step4_curve_kib"]
    sram_f3 = f3["sram_f3_curve_kib"]
    sram_r5 = f3["sram_r5_curve_kib"]

    alphas = f3["alphas"]
    f3_dv = f3["f3_tail_dV_rel_curve"]
    f3_dk = f3["f3_tail_dK_rel_curve"]
    s4_dv = f3["step4_tail_dV_rel_curve"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.4), dpi=300)

    # Left panel: SRAM Coexistence
    ax1.plot(bkv, sram_s4, marker="o", color="#d62728", linewidth=1.8, label="Production (fp32 accum)")
    ax1.plot(bkv, sram_f3, marker="s", color="#ff7f0e", linewidth=1.8, label="F3 (fp16 accum)")
    ax1.plot(bkv, sram_r5, marker="^", color="#2ca02c", linewidth=1.8, linestyle="--", label="R5 (spilled fp32)")

    ax1.axhline(164.0, color="#666666", linestyle=":", linewidth=1.1)
    ax1.text(33, 168.0, "A100 limit (164 KB)", fontsize=8.0, color="#444444")

    ax1.axhline(228.0, color="#333333", linestyle="--", linewidth=1.1)
    ax1.text(33, 232.0, "H100 limit (228 KB)", fontsize=8.0, color="#111111")

    # Highlight B_kv = 256
    ax1.scatter([256], [152.0], color="#ff7f0e", s=70, zorder=5)
    ax1.annotate(
        r"F3 fits at $B_{kv}=256$" "\n(152 KiB ≤ 228 KB)",
        xy=(256, 152.0),
        xytext=(80, 70),
        arrowprops=dict(arrowstyle="->", color="#333333", lw=0.8),
        fontsize=8.0,
        bbox=dict(boxstyle="round,pad=0.25", fc="#f8f9fa", ec="#cccccc", lw=0.6),
    )

    ax1.set_xscale("log", base=2)
    ax1.set_xticks(bkv)
    ax1.set_xticklabels([str(b) for b in bkv])
    ax1.set_ylim(0, 580)
    ax1.set_xlabel(r"Key/Value tile size $B_{kv}$", fontsize=9.5)
    ax1.set_ylabel("Persistent backward SRAM (KiB)", fontsize=9.5)
    ax1.set_title("Persistent SRAM vs Tile Size", fontsize=10.5, fontweight="bold")
    style_axis(ax1)
    ax1.legend(loc="upper left", fontsize=8.0, framealpha=0.9)

    # Right panel: Tail Gradient Swamping
    ax2.plot(alphas, f3_dv, marker="o", color="#d62728", linewidth=1.8, label=r"F3 tail $\bar{V}$ error (fp16)")
    ax2.plot(alphas, f3_dk, marker="s", color="#ff7f0e", linewidth=1.8, label=r"F3 tail $\bar{K}$ error (fp16)")
    ax2.plot(alphas, s4_dv, marker="^", color="#2ca02c", linewidth=1.8, linestyle="--", label="Production / R5 (fp32)")

    ax2.axhline(0.02, color="#555555", linestyle="--", linewidth=1.0)
    ax2.text(0.8, 0.023, r"$\tau_1 = 0.02$ (idx 2, $\alpha=10^{-2}$)", fontsize=8.0, color="#333333")

    ax2.axhline(0.50, color="#555555", linestyle="--", linewidth=1.0)
    ax2.text(0.8, 0.55, r"$\tau_2 = 0.50$ (idx 3, $\alpha=10^{-3}$)", fontsize=8.0, color="#333333")

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlim(1.5, 5e-5)  # decreasing to right
    ax2.set_ylim(1e-7, 2.0)
    ax2.set_xlabel(r"Tail query scale $\alpha$ (relative to leading tile)", fontsize=9.5)
    ax2.set_ylabel("Relative tail gradient error vs R1", fontsize=9.5)
    ax2.set_title("Cross-Tile Dynamic Range Absorption", fontsize=10.5, fontweight="bold")
    style_axis(ax2)
    ax2.legend(loc="lower left", fontsize=8.0, framealpha=0.9)

    fig.suptitle("Figure 11.C: F3 Two-Lens SRAM Illusion and Cross-Tile Absorption Cliff", fontsize=11.5, fontweight="bold", y=0.98)
    fig.tight_layout()
    out_path = FIG / "ch11_f3_sram_absorption.svg"
    fig.savefig(out_path, format="svg")
    plt.close(fig)
    return out_path


def main() -> None:
    print("Running verified sweeps from code/tgd/kernels.py...")
    sweeps = verify_step8_parametric_sweeps()

    p_11a = render_fig_11a(sweeps)
    print(f"Rendered Figure 11.A -> {p_11a.relative_to(ROOT)}")

    p_11b = render_fig_11b(sweeps)
    print(f"Rendered Figure 11.B -> {p_11b.relative_to(ROOT)}")

    p_11c = render_fig_11c(sweeps)
    print(f"Rendered Figure 11.C -> {p_11c.relative_to(ROOT)}")
    print("All Chapter 11 figures successfully rendered.")


if __name__ == "__main__":
    main()
