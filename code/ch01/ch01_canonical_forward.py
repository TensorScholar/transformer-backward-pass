"""Chapter 1 verification -- introduction and the canonical forward baseline.

Establishes and numerically verifies the canonical Pre-RMSNorm decoder block
used throughout the companion code:

    1. Exact parameter registry and live tensor count parity across scales.
    2. Canonical shape ledger (`tgd.graph.expected_shapes`), minimal-cut cache
       pruning, and causal invariance (future-token invariance for prefix logits).
    3. Pre-RMSNorm unit-RMS contract, SwiGLU gating, and the two residual
       stream additions per block (attention branch + MLP branch).
    4. Four-account training memory ledger (`tgd.costs.four_account_ledger`)
       comparing autograd saved memory, minimal-cut activations, optimizer
       state, and per-layer vs full-model KV cache footprints.
    5. Finite-precision forward parity across float64, float32, and bfloat16.

Usage:  python ch01_canonical_forward.py
Marker: CH01_CANONICAL_FORWARD_PASSED
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

# Resolve imports for the canonical layout: a chapter script lives at
# <root>/code/chNN/ and needs its own directory and <root>/code (where `tgd` lives).
_HERE = Path(__file__).resolve()
for _p in (_HERE.parent, _HERE.parents[1]):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch

from tgd import costs as cost_mod
from tgd import rmsnorm as rn_mod
from tgd.dims import CH6_TINY, CH7_TRAIN, CH8_TWO_HEAD, Dims
from tgd.graph import BlockParams, block_forward, expected_shapes, model_forward

DTYPE = torch.float64
FAILURES: list[str] = []


def check(cond: bool, msg: str) -> None:
    if cond:
        print(f"  ok    {msg}")
    else:
        print(f"  FAIL  {msg}")
        FAILURES.append(msg)


def rel_error(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.detach().double(), b.detach().double()
    return float((a - b).norm() / (a.norm() + b.norm() + 1e-12))


def make_model(dims: Dims, seed: int = 11, scale: float = 0.08,
               dtype: torch.dtype = DTYPE) -> dict:
    """Allocate all canonical model parameters matching `dims.n_params`."""
    g = torch.Generator().manual_seed(seed)

    def p(*shape: int, s: float = scale) -> torch.Tensor:
        return torch.randn(*shape, generator=g, dtype=dtype) * s

    E = p(dims.V, dims.d)
    blocks = []
    for _ in range(dims.L):
        blocks.append(BlockParams(
            g1=torch.ones(dims.d, dtype=dtype),
            W_Q=p(dims.d, dims.d),
            W_K=p(dims.d, dims.d),
            W_V=p(dims.d, dims.d),
            W_O=p(dims.d, dims.d),
            g2=torch.ones(dims.d, dtype=dtype),
            W_g=p(dims.d, dims.d_ff),
            W_u=p(dims.d, dims.d_ff),
            W_d=p(dims.d_ff, dims.d),
        ))
    gf = torch.ones(dims.d, dtype=dtype)
    W_out = p(dims.d, dims.V)
    return {"E": E, "blocks": blocks, "gf": gf, "W_out": W_out}


def count_model_elements(model: dict) -> int:
    total = model["E"].numel() + model["gf"].numel() + model["W_out"].numel()
    for blk in model["blocks"]:
        for t in (blk.g1, blk.W_Q, blk.W_K, blk.W_V, blk.W_O,
                  blk.g2, blk.W_g, blk.W_u, blk.W_d):
            total += t.numel()
    return total


def section_1_shapes_and_registry() -> None:
    print("\n[1] canonical parameter registry and forward shape contract")
    multi_layer = Dims(B=2, T=8, V=32, d=16, H=4, d_ff=32, n_layers=2)
    scales = [CH6_TINY, CH8_TWO_HEAD, multi_layer, CH7_TRAIN]
    for dims in scales:
        label = repr(dims)
        model = make_model(dims, seed=11 + dims.d)
        expected_blk = 4 * dims.d * dims.d + 3 * dims.d * dims.d_ff + 2 * dims.d
        expected_tot = (dims.V * dims.d + dims.L * expected_blk
                        + dims.d + dims.d * dims.V)
        live_tot = count_model_elements(model)
        check(dims.per_block_params == expected_blk,
              f"per_block_params formula holds for {label}: {expected_blk:,}")
        check(dims.n_params == expected_tot == live_tot,
              f"total params match registry & allocation for {label}: {live_tot:,}")

        x = torch.randint(0, dims.V, (dims.B, dims.T),
                          generator=torch.Generator().manual_seed(3))
        y = torch.randint(0, dims.V, (dims.B, dims.T),
                          generator=torch.Generator().manual_seed(5))
        loss, rec = model_forward(
            model["E"], x, model["blocks"], model["gf"], model["W_out"], dims, y=y
        )
        check(math.isfinite(float(loss)) and float(loss) > 0.0,
              f"{label}: finite CE loss = {float(loss):.6f} (ln V = {math.log(dims.V):.6f})")

        exp_shapes = expected_shapes(dims)
        X0 = rec[2]
        _, cache = block_forward(X0, model["blocks"][0], dims, minimal_cut=False)
        X1_min, min_cache = block_forward(X0, model["blocks"][0], dims, minimal_cut=True)
        check(exp_shapes[2] == tuple(X0.shape) == (dims.B, dims.T, dims.d),
              f"{label}: X_0 shape {tuple(X0.shape)} matches ID 2")
        check(tuple(cache.S.shape) == exp_shapes[8]
              and tuple(cache.A.A.shape) == exp_shapes[9]
              == (dims.H, dims.B, dims.T, dims.T),
              f"{label}: S, A shapes {tuple(cache.A.A.shape)} match IDs 8 & 9")
        check(tuple(rec[26].shape) == exp_shapes[26] == (dims.B, dims.T, dims.V),
              f"{label}: logit tensor Z shape {tuple(rec[26].shape)} matches ID 26")
        pruned_ok = (min_cache.Q_lin is None and min_cache.R is None
                     and min_cache.F is None and min_cache.Xp is None)
        check(tuple(X1_min.shape) == (dims.B, dims.T, dims.d) and pruned_ok,
              f"{label}: minimal_cut=True releases non-boundary forward tensors")


def section_2_structural_invariants() -> None:
    print("\n[2] causal invariance, Pre-RMSNorm scaling, and head-dim normalization")
    dims = Dims(B=1, T=8, V=32, d=16, H=4, d_ff=32, n_layers=2)
    model = make_model(dims, seed=19)
    x_a = torch.tensor([[3, 7, 12, 1, 9, 20, 4, 15]], dtype=torch.long)
    x_b = x_a.clone()
    cutoff = 4
    x_b[0, cutoff:] = torch.tensor([25, 2, 30, 11], dtype=torch.long)

    logits_a, _ = model_forward(
        model["E"], x_a, model["blocks"], model["gf"], model["W_out"], dims
    )
    logits_b, _ = model_forward(
        model["E"], x_b, model["blocks"], model["gf"], model["W_out"], dims
    )
    prefix_diff = float((logits_a[:, :cutoff, :] - logits_b[:, :cutoff, :]).abs().max())
    future_diff = float((logits_a[:, cutoff:, :] - logits_b[:, cutoff:, :]).abs().max())
    check(prefix_diff == 0.0,
          f"strict causality: editing t>={cutoff} gives max prefix logit delta = {prefix_diff:.3e}")
    check(future_diff > 1e-3,
          f"nontrivial downstream sensitivity: max delta on t>={cutoff} = {future_diff:.3e}")

    X0 = model["E"][x_a]
    blk0 = model["blocks"][0]
    X1, cache = block_forward(X0, blk0, dims, minimal_cut=False)
    N1_y, _ = rn_mod.rmsnorm_forward(X0, blk0.g1, dims.eps)
    N2_y, _ = rn_mod.rmsnorm_forward(cache.R, blk0.g2, dims.eps)
    rms_n1 = N1_y.pow(2).mean(dim=-1).sqrt()
    rms_n2 = N2_y.pow(2).mean(dim=-1).sqrt()
    check(float((rms_n1 - 1.0).abs().max()) < 1e-2,
          f"Pre-RMSNorm N^(1) maintains near-unit RMS (max |RMS-1| = {float((rms_n1 - 1.0).abs().max()):.3e})")
    check(float((rms_n2 - 1.0).abs().max()) < 1e-2,
          f"Pre-RMSNorm N^(2) maintains near-unit RMS (max |RMS-1| = {float((rms_n2 - 1.0).abs().max()):.3e})")
    resid_diff = float((X1 - (X0 + cache.Y + cache.F)).abs().max())
    check(resid_diff == 0.0,
          f"two-residual block sum identity X_1 == X_0 + Y + F exact (max diff = {resid_diff:.3e})")

    # Verify 1/sqrt(d_h) vs 1/sqrt(d) scaling on independent unit-variance Q, K
    g = torch.Generator().manual_seed(42)
    q = torch.randn(4096, dims.d_h, generator=g, dtype=DTYPE)
    k = torch.randn(4096, dims.d_h, generator=g, dtype=DTYPE)
    dots = (q * k).sum(dim=-1)
    std_dh = float((dots / math.sqrt(dims.d_h)).std())
    std_d = float((dots / math.sqrt(dims.d)).std())
    check(abs(std_dh - 1.0) < 0.05 and std_d < 0.6,
          f"score std with 1/sqrt(d_h={dims.d_h}) = {std_dh:.3f} vs collapsed 1/sqrt(d={dims.d}) = {std_d:.3f}")


def section_3_three_layer_accounting() -> None:
    print("\n[3] four-account memory contract and compute/bandwidth profile")
    ref_small = Dims(B=4, T=2048, V=32000, d=768, H=12, d_ff=2048, n_layers=12)
    ref_7b = Dims(B=1, T=4096, V=32000, d=4096, H=32, d_ff=11008, n_layers=32)

    for label, dims in [("125M-class (B=4, T=2048, L=12)", ref_small),
                        ("7B-class (B=1, T=4096, L=32)", ref_7b)]:
        rep4 = cost_mod.cost_report(dims, element_bytes=4)
        rep2 = cost_mod.cost_report(dims, element_bytes=2)
        ledger = cost_mod.four_account_ledger(dims, element_bytes=4, include_optimizer_state=True)
        kv_layer_2b = cost_mod.kv_cache_bytes(dims, T_kv=dims.T, element_bytes=2)
        kv_total_2b = dims.L * kv_layer_2b
        param_bytes = dims.n_params * 4
        print(f"  Scale {label}:")
        print(f"    params         = {dims.n_params:,} ({param_bytes / 1024**3:.3f} GiB fp32)")
        print(f"    Phi_forward    = {rep4.phi_forward:,} FLOPs | Phi_backward ~ {rep4.phi_backward:,} FLOPs")
        print(f"    Psi_act (fp32) = {rep4.psi_forward / 1024**3:.3f} GiB | Psi_act (2B) = {rep2.psi_forward / 1024**3:.3f} GiB")
        print(f"    A_autograd     = {ledger['A_autograd_saved_mib']:.2f} MiB | D_prod = {ledger['D_production_cut_mib']:.2f} MiB (A/D = {ledger['A_over_D']:.2f}x)")
        print(f"    Optim+weights  = {ledger['optimizer_state_mib']:.2f} MiB | Train total = {ledger['training_step_total_mib']:.2f} MiB")
        print(f"    KV cache (2B)  = {kv_layer_2b / 1024**2:.2f} MiB/layer ({kv_total_2b / 1024**2:.2f} MiB model) | iota_block(2B) = {rep2.iota_block:.2f} FLOP/B")
        check(rep4.phi_backward == 2 * rep4.phi_forward,
              f"{label}: analytical backward/forward FLOP ratio == 2:1")
        check(rep4.psi_forward == 2 * rep2.psi_forward,
              f"{label}: activation memory scales linearly with element byte width")
        check(ledger["A_over_D"] > 1.0 and ledger["A_autograd_saved_mib"] > ledger["D_production_cut_mib"],
              f"{label}: naive autograd saves {ledger['A_over_D']:.2f}x production-cut activation memory")
        check(kv_layer_2b == 2 * dims.B * dims.H * dims.T * dims.d_h * 2,
              f"{label}: per-layer KV-cache equals 2 * B * H * T * d_h * 2B exact formula")


def section_4_precision_baseline() -> None:
    print("\n[4] finite-precision forward divergence (float64 vs float32 vs bfloat16)")
    dims = Dims(B=2, T=16, V=64, d=32, H=4, d_ff=64, n_layers=2)
    m64 = make_model(dims, seed=101, scale=0.12, dtype=torch.float64)
    g = torch.Generator().manual_seed(77)
    x = torch.randint(0, dims.V, (dims.B, dims.T), generator=g)
    y = torch.randint(0, dims.V, (dims.B, dims.T), generator=g)

    def cast_model(src: dict, dt: torch.dtype) -> dict:
        return {
            "E": src["E"].to(dt),
            "gf": src["gf"].to(dt),
            "W_out": src["W_out"].to(dt),
            "blocks": [
                BlockParams(*(t.to(dt) for t in (
                    b.g1, b.W_Q, b.W_K, b.W_V, b.W_O, b.g2, b.W_g, b.W_u, b.W_d
                ))) for b in src["blocks"]
            ],
        }

    loss64, rec64 = model_forward(m64["E"], x, m64["blocks"], m64["gf"], m64["W_out"], dims, y=y)
    m32 = cast_model(m64, torch.float32)
    loss32, rec32 = model_forward(m32["E"], x, m32["blocks"], m32["gf"], m32["W_out"], dims, y=y)
    mbf = cast_model(m64, torch.bfloat16)
    loss_bf, rec_bf = model_forward(mbf["E"], x, mbf["blocks"], mbf["gf"], mbf["W_out"], dims, y=y)

    err_32 = rel_error(rec32[26], rec64[26])
    err_bf = rel_error(rec_bf[26], rec64[26])
    print(f"    loss float64 = {float(loss64):.8f}")
    print(f"    loss float32 = {float(loss32):.8f}  (rel logit err = {err_32:.3e})")
    print(f"    loss bfloat16= {float(loss_bf):.8f}  (rel logit err = {err_bf:.3e})")
    check(err_32 < 1e-5, f"float32 forward closely tracks float64 (rel err {err_32:.3e} < 1e-5)")
    check(err_bf > err_32 and err_bf < 5e-2,
          f"bfloat16 introduces measurable forward discretization (rel err {err_bf:.3e} > float32)")


def main() -> int:
    print("=" * 72)
    print("Chapter 1 -- Introduction and the Canonical Forward Baseline")
    print(f"torch {torch.__version__}  dtype {DTYPE}")
    print("=" * 72)

    section_1_shapes_and_registry()
    section_2_structural_invariants()
    section_3_three_layer_accounting()
    section_4_precision_baseline()

    print("\n" + "=" * 72)
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} checks failed:")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("CH01_CANONICAL_FORWARD_PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
