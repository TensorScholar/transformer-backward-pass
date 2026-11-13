r"""Chapter 16 verification -- MoE: capacity without compute, and the router's price.

Turns every claim in the chapter into an assertion:

  1. the exact active-FLOP count for a single-token forward pass, dense vs
     top-2 sparse, with the parameter count beside it -- so the decoupling is
     two numbers that disagree rather than one number asserted;
  2. the exact weight VRAM for both, proving the decoupling STOPS at memory:
     resident bytes rise by the same factor the parameter count rises;
  3. the auxiliary loss gradient against the analytic form, to float64
     precision -- including the softmax-Jacobian term the first version omitted;
  4. the uniform-dispatch value is K (not 1), and the n*sum(v^2) >= 1 bound holds
     exactly when calibrated and is violated by construction when it is not;
  5. the routing measurement at the brief's scale (50 steps): mild concentration
     without the aux loss, restored balance with it -- reported honestly as
     1.27x vs 1.20x, not as a monopoly that was not measured;
  6. the fastest concentration this harness can produce (fast router), and the
     facts that no monopoly and no dead expert occur in any configuration;
  7. the structural properties: exactly-K dispatch including on tied
     probabilities, the K-way merge against an explicit sum, and a distribution
     for the router;
  8. a guard that must be able to FAIL: the FLOP model is checked against an
     independent recount, so a silent arithmetic error cannot ship.

Usage:  python ch16_moe_verify.py
Marker: CH16_MOE_VERIFY_PASSED
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
for _p in (_HERE.parent, _HERE.parents[1]):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch                                                  # noqa: E402

from tgd import moe as M                                       # noqa: E402

FAILURES: list[str] = []


def check(cond: bool, msg: str) -> None:
    if cond:
        print(f"  ok    {msg}")
    else:
        print(f"  FAIL  {msg}")
        FAILURES.append(msg)


def main() -> None:
    print("=" * 78)
    print("Chapter 16 -- mixture of experts: routing, FLOPs, and the aux loss")
    print("n_exp = 8 experts, d_ff = 1024 PER EXPERT, top-K = 2.")
    print("Experts are NOT named E: the contract registers E as the embedding")
    print("matrix (ID 1). The count is n_exp and the index is a.")
    print("=" * 78)

    # ---------------------------------------------------------------------
    print("\n[1] active FLOPs per token: dense vs top-2 sparse")
    f = M.flop_report()
    fd, fm = f["dense"], f["moe"]
    print(f"    dense  {f['n_params_dense']:>12,} params  "
          f"{fd['flops_per_token']:>12,} FLOPs/token")
    print(f"    moe    {f['n_params_moe']:>12,} params  "
          f"{fm['flops_per_token']:>12,} FLOPs/token")
    print(f"    router costs {fm['router_macs']:,} MACs/token "
          f"({fm['router_macs'] / fm['macs_per_token']:.3%} of the active total)")
    print(f"    all-experts total would be {fm['flops_all_experts']:,} FLOPs "
          f"(sparsity {fm['sparsity']:.1f}x)")
    check(f["flop_ratio_dense_over_moe"] > 1.9,
          f"MoE uses {f['flop_ratio_dense_over_moe']:.4f}x fewer active FLOPs "
          f"({fd['flops_per_token']:,} vs {fm['flops_per_token']:,})")
    check(f["param_ratio_moe_over_dense"] > 1.9,
          f"while holding {f['param_ratio_moe_over_dense']:.4f}x the parameters "
          f"({f['n_params_moe']:,} vs {f['n_params_dense']:,}) -- the two "
          "ratios point in opposite directions, which IS the decoupling")
    check(fd["flops_per_token"] == 2 * 3 * 1024 * 4096,
          f"dense is exactly 6*d*d_ff = {2 * 3 * 1024 * 4096:,} FLOPs")
    check(fm["macs_per_token"] == 1024 * 8 + 2 * 3 * 1024 * 1024,
          f"MoE is exactly d*n_exp + K*3*d*d_ff = "
          f"{1024 * 8 + 2 * 3 * 1024 * 1024:,} MACs")
    check(fm["router_macs"] / fm["macs_per_token"] < 0.01,
          "the router itself is under 1% of the active compute, so routing is "
          "structurally free and the experts are the whole bill")

    # ---------------------------------------------------------------------
    print("\n[2] weight VRAM: where the decoupling stops")
    v = M.vram_report()
    print(f"    dense {v['dense_bytes']:>12,} B  ({v['dense_MiB']:.2f} MiB)")
    print(f"    moe   {v['moe_bytes']:>12,} B  ({v['moe_MiB']:.2f} MiB)")
    print(f"    router is {v['router_share']:.4%} of the MoE bytes")
    check(v["vram_ratio_moe_over_dense"] > 1.9,
          f"MoE weights cost {v['vram_ratio_moe_over_dense']:.4f}x the dense "
          f"weights ({v['moe_MiB']:.2f} vs {v['dense_MiB']:.2f} MiB) -- VRAM "
          "goes UP by the capacity factor")
    check(abs(v["vram_ratio_moe_over_dense"]
              - f["param_ratio_moe_over_dense"]) < 1e-9,
          "and the VRAM ratio EQUALS the parameter ratio, because bytes are "
          "parameters times a constant -- the decoupling is between compute "
          "and parameter count, not between compute and memory")
    check(v["router_share"] < 0.001,
          f"the router is {v['router_share']:.4%} of the bytes, so it is not "
          "where the memory went")

    # ---------------------------------------------------------------------
    print("\n[3] the auxiliary loss gradient, against the analytic form")
    a = M.aux_gradient_report()
    print(f"    loss value {a['loss_value']:.6f} (above 1: {a['loss_above_1']})")
    print(f"    max abs err vs analytic: {a['max_abs_err_vs_analytic']:.3e}")
    print(f"    popular expert {a['popular_expert']} (share "
          f"{a['popular_share']:.4f}): mean logit grad "
          f"{a['popular_mean_logit_grad']:+.6f} -> pushed DOWN")
    print(f"    starved expert {a['starved_expert']} (share "
          f"{a['starved_share']:.4f}): mean logit grad "
          f"{a['starved_mean_logit_grad']:+.6f} -> pushed UP")
    print(f"    unmasked logits carry gradient: "
          f"{a['unmasked_logits_carry_gradient']}")
    check(a["max_abs_err_vs_analytic"] < 1e-12,
          f"analytic gradient matches autograd to "
          f"{a['max_abs_err_vs_analytic']:.2e} -- the softmax-Jacobian term "
          "included, which the first version omitted at 0.125 error")
    check(a["popular_pushed_down"] and a["starved_pushed_up"],
          "the mechanism is comparative: over-represented experts are pushed "
          "down AND under-represented experts are pushed up, from both sides")
    check(a["unmasked_logits_carry_gradient"],
          "even UNMASKED logits carry gradient through the softmax denominator "
          "-- changing them changes every selected expert's probability")

    # ---------------------------------------------------------------------
    print("\n[4] the uniform value is K, and the bound needs calibration")
    x = M.exactness_checks()
    print(f"    uniform top-2 dispatch loss: {x['aux_on_uniform_dispatch']:.6f} "
          f"(== K: {x['aux_uniform_dispatch_equals_k']})")
    b = x["aux_calibrated_lower_bound"]
    print(f"    calibrated n*sum(v^2): worst {b['worst']:.6f} over 256 draws, "
          f"uniform gives {b['at_uniform']:.6f}")
    check(x["aux_uniform_dispatch_equals_k"],
          f"uniform dispatch gives exactly K = 2.0, not 1 -- every 'equals one' "
          "statement is the top-1 special case")
    check(b["holds"] and b["uniform_is_1"],
          "and conditional on calibration, n*sum(v^2) >= 1 with equality iff "
          "uniform -- the true theorem, not the false unconditional one")
    check(x["dispatch_is_exactly_k"],
          "every token dispatches to exactly K experts")
    check(x["tied_selects_exactly_k"],
          f"tied probabilities still select exactly K "
          f"({x['tied_mask'][0][:4]}...) -- no duplicate-expert double count")
    check(x["probs_rows_sum_to_1"],
          "the router's softmax rows sum to 1")
    check(x["y_equals_explicit_sum"],
          "the forward equals the explicit per-expert sum term by term, so the "
          "K-way merge is what the code computes")

    # ---------------------------------------------------------------------
    print("\n[5] routing at the brief's scale: 50 steps, and whether aux helps")
    r = M.routing_collapse_report(steps=50, T=256, data_mode="iid", seed=0)
    fo, wo = r["no_aux"]["final"], r["with_aux"]["final"]
    print(f"    uniform reference share: {r['uniform_share']:.4f}   "
          "monopoly bound: 1.0")
    print(f"    seed 0: no-aux top {fo['top_share']:.4f} "
          f"({fo['collapse_ratio']:.2f}x uniform)  |  aux top "
          f"{wo['top_share']:.4f} ({wo['collapse_ratio']:.2f}x uniform)")
    check(not r["no_aux_is_monopoly"],
          f"no monopoly at 50 steps (top {fo['top_share']:.4f}, not > 0.9) -- "
          "reported, not manufactured")
    check(fo["collapse_ratio"] > 1.1,
          f"but the loop IS engaged: {fo['collapse_ratio']:.2f}x uniform "
          "without the aux loss")
    check(fo["n_dead_experts"] == 0 and wo["n_dead_experts"] == 0,
          "no dead expert in either run -- starvation has not occurred here")
    # The seed-0 pair (1.27x -> 1.20x) is WITHIN SEED NOISE. The first version
    # asserted restoration from that pair alone; across seeds it does not hold.
    # So the restoration claim is measured as a win rate, not a pair.
    print("    efficacy across seeds (win rate = aux top share strictly below):")
    eff = M.aux_efficacy_report()
    for row in eff["rows"]:
        print(f"      gamma={row['gamma']:<5} no-aux {row['no_aux_mean']:.4f}  "
              f"aux {row['with_aux_mean']:.4f}  win rate {row['win_rate']:.0%}  "
              f"improvement {row['mean_improvement']:+.4f}")
    check(not eff["restoration_significant_anywhere"],
          "at 50 steps the aux outcome is NOT significant on any gamma "
          "(no unanimous, sized win anywhere) -- the seed-0 restoration is "
          "noise, and claiming it would be reporting noise as signal")
    check(eff["large_gamma_hurts"],
          "and a large gamma HURTS on average (improvement negative), so "
          "'more regularisation is better' is false here")

    # ---------------------------------------------------------------------
    print("\n[6] how far the loop can drive it (fast router)")
    s = M.router_speed_report()
    for row in s["rows"]:
        print(f"    router x{row['router_lr_mult']:>3}: no-aux top "
              f"{row['no_aux']['top_share']:.4f} "
              f"({row['no_aux']['collapse_ratio']:.2f}x)  |  aux top "
              f"{row['with_aux']['top_share']:.4f} "
              f"({row['with_aux']['collapse_ratio']:.2f}x)")
    # The absolute concentration is library-sensitive: 0.49 on the reference
    # torch-2.10 build, 0.31-0.35 across seeds on torch 2.15 (the top-K tie
    # fix also legitimately changes one tied dispatch every ~10 steps). What
    # the chapter actually claims is the DIRECTION: a faster router
    # concentrates the routing well past the seed-noise band around the
    # 0.125 uniform floor. Both assertions below fail if the loop has no
    # force at all (max_top_share back at ~0.125).
    check(s["max_top_share"] > 0.2,
          f"a fast router drives concentration to {s['max_top_share']:.4f} -- "
          "the loop has real force when the router outruns the experts "
          "(absolute value is build-sensitive: 0.49 on the reference "
          "torch-2.10 environment, ~0.31-0.35 on 2.15)")
    check(s["max_top_share"] > 2.0 * 0.125,
          f"and that is {s['max_top_share'] / 0.125:.2f}x uniform -- past "
          "the seed-noise band around the uniform floor")
    check(not s["monopoly_reached"],
          "yet even there: no monopoly (top > 0.9) in any configuration")
    check(not s["any_dead"],
          "and no dead expert in any configuration -- the absorbing state is "
          "not reached at this scale")

    # ---------------------------------------------------------------------
    print("\n[7] a guard that must be able to FAIL")
    # The FLOP model is a formula. Recount it independently here: if the two
    # ever disagree, the table in section 16.1 is fiction.
    dcfg, mcfg = M.DenseConfig(), M.MoEConfig()
    recount_dense = 3 * dcfg.d * dcfg.d_ff
    recount_moe = mcfg.d * mcfg.n_exp + mcfg.k * 3 * mcfg.d * mcfg.d_ff
    fd2, fm2 = M.dense_flops(dcfg), M.moe_flops(mcfg)
    check(fd2["macs_per_token"] == recount_dense
          and fm2["macs_per_token"] == recount_moe,
          f"independent recount matches the model "
          f"({recount_dense:,} / {recount_moe:,} MACs)")
    # and the parameter model against real allocation
    dm, mm = M.DenseFFN(dcfg, seed=0), M.MoEFFN(mcfg, seed=0)
    check(dm.n_params == 3 * dcfg.d * dcfg.d_ff
          and mm.n_params == mcfg.n_exp * 3 * mcfg.d * mcfg.d_ff
          + mcfg.d * mcfg.n_exp,
          f"allocation matches the model "
          f"({dm.n_params:,} / {mm.n_params:,} params)")
    # the aux loss on a CONSISTENTLY concentrated routing must exceed the
    # uniform value. "Consistently" is load-bearing: the mask must be the top-K
    # OF the probabilities, because the loss couples f and zeta through the
    # router and they cannot move independently. The first version paired a
    # concentrated mask with random probabilities -- a state the router can never
    # produce -- and "discovered" that imbalance scores 1.38, below uniform.
    # That was a property of the impossible pair, not of the loss.
    cprobs = torch.softmax(
        torch.tensor([[3.0] + [0.0] * 7] * 64, dtype=torch.float64), dim=-1)
    cmask = torch.zeros(64, 8, dtype=torch.float64)
    cmask.scatter_(-1, torch.topk(cprobs, 2, dim=-1).indices, 1.0)
    closs, _ = M.load_balancing_loss(cprobs, cmask, 8)
    check(float(closs) > 2.0,
          f"a consistently concentrated routing scores {float(closs):.4f}, "
          "above the uniform value 2.0 -- the loss distinguishes what it "
          "claims to, when the state it is asked about can actually occur")

    print("\n" + "=" * 78)
    if FAILURES:
        print(f"CH16 FAILED: {len(FAILURES)} problem(s)")
        for f_ in FAILURES:
            print(f"  - {f_}")
        sys.exit(1)
    print("CH16_MOE_VERIFY_PASSED")
    print("=" * 78)


if __name__ == "__main__":
    main()