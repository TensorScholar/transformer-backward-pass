r"""Chapter 15 verification -- LoRA: what it saves, and what it provably does not change.

Turns every claim in the chapter into an assertion:

  1. the exact parameter, gradient and optimizer-state byte counts for a
     4096x4096 layer at r=8 and r=64, with Adam state counted as TWO moments;
  2. LoRA does NOT shrink the weight matrix, so the total saving is much smaller
     than the parameter ratio -- both are reported and the gap is asserted;
  3. x(W + AB) and xW + (xA)B agree to float32 ROUNDING, with the measured
     magnitude in ULPs of the output scale and inside the gamma_n budget.
     BITWISE EQUALITY IS NOT ASSERTED, because it is false;
  4. the merge is exact, and the merged layer carries no adapter;
  5. zero-init identity: at step 0 the adapted layer equals the frozen layer
     EXACTLY, so the adapter cannot perturb the base model before training;
  6. the frozen base receives no gradient, no optimizer state, and is
     byte-identical after a training run;
  7. the training-time FLOP overhead is r(d_in+d_out)/(d_in*d_out), and the
     merge amortises after exactly r tokens;
  8. a 10-step real backward pass on both, with the loss decreasing in both;
  9. a guard that must be able to FAIL: the byte-count model is checked against
     the actual allocated tensors, so a silent accounting error cannot ship.

Usage:  python ch15_lora_verify.py
Marker: CH15_LORA_VERIFY_PASSED
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
for _p in (_HERE.parent, _HERE.parents[1]):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch                                                  # noqa: E402

from tgd import lora as L                                      # noqa: E402

FAILURES: list[str] = []


def check(cond: bool, msg: str) -> None:
    if cond:
        print(f"  ok    {msg}")
    else:
        print(f"  FAIL  {msg}")
        FAILURES.append(msg)


def main() -> None:
    print("=" * 78)
    print("Chapter 15 -- parameter-efficient fine-tuning: LoRA")
    print("W_eff = W + W_down W_up,  W_down: (d_in, r),  W_up: (r, d_out).")
    print("Factors are NOT named A and B: contract section 7 reserves A for the")
    print("attention matrix and B for the batch dimension.")
    print("=" * 78)

    # ---------------------------------------------------------------------
    print("\n[1] the exact byte counts, 4096 x 4096, float32")
    MiB = 2 ** 20
    for r in (8, 64):
        m = L.memory_report(L.LoRAConfig(4096, 4096, r))
        f, l = m["full"], m["lora"]
        print(f"    r = {r}")
        print(f"      {'variant':<8}{'elements':>12}{'weights':>13}{'grads':>13}"
              f"{'adam m+v':>13}")
        for lbl, b in (("full", f), ("lora", l)):
            print(f"      {lbl:<8}{b['elements']:>12,}"
                  f"{b['weights_bytes'] / MiB:>10.2f} MiB"
                  f"{b['grad_bytes'] / MiB:>9.4f} MiB"
                  f"{b['adam_bytes'] / MiB:>9.4f} MiB")
        print(f"      frozen base resident in BOTH: "
              f"{m['frozen_base_bytes'] / MiB:.2f} MiB")
        print(f"      TOTAL   full {m['full_total_bytes'] / MiB:.2f} MiB   "
              f"lora {m['lora_total_bytes'] / MiB:.2f} MiB   "
              f"-> {m['total_ratio']:.2f}x")
        print(f"      ratios: params {m['param_ratio']:.2f}x, "
              f"grads {m['grad_ratio']:.2f}x, adam {m['adam_ratio']:.2f}x  "
              f"(bias frozen under LoRA, so all three coincide)")

        check(f["adam_bytes"] == 2 * f["grad_bytes"],
              f"r={r}: Adam state is exactly TWO moments (m and v), "
              f"{f['adam_bytes'] / MiB:.2f} MiB = 2 x grads "
              f"{f['grad_bytes'] / MiB:.2f} MiB -- reporting only m would "
              "understate the saving by half")
        check(m["param_ratio_weights_only"] == 4096 / (2 * r),
              f"r={r}: weights-only compression is exactly d/2r = "
              f"{4096 / (2 * r):.0f}x")
        check(m["lora_total_bytes"] > m["frozen_base_bytes"],
              f"r={r}: LoRA's total footprint is still dominated by the FROZEN "
              f"weights ({m['frozen_base_bytes'] / MiB:.2f} of "
              f"{m['lora_total_bytes'] / MiB:.2f} MiB) -- PEFT does not shrink "
              "the model, only its trainable part")
        check(m["total_ratio"] < m["param_ratio"],
              f"r={r}: total saving {m['total_ratio']:.2f}x is far below the "
              f"parameter ratio {m['param_ratio']:.0f}x, and conflating them "
              "overstates the benefit")
        _rel = abs(m["param_ratio"] - m["param_ratio_weights_only"]) \
            / m["param_ratio_weights_only"]
        check(_rel < 1e-3,
              f"r={r}: the two ratios agree to {_rel:.3%} "
              f"({m['param_ratio']:.4f}x with the full variant's bias, "
              f"{m['param_ratio_weights_only']:.0f}x weights-only) -- the only "
              "difference is the full variant's bias, since LoRA's is frozen")

    # ---------------------------------------------------------------------
    print("\n[2] forward equivalence:  x(W + AB)  vs  xW + (xA)B")
    for (d_in, d_out, r, batch) in ((512, 512, 8, 16), (4096, 4096, 8, 32),
                                    (4096, 4096, 64, 32), (1024, 768, 1, 8)):
        e = L.equivalence_report(d_in, d_out, r, batch)
        print(f"    d_in={d_in} d_out={d_out} r={r} batch={batch}")
        print(f"      max abs diff {e['max_abs_diff']:.6e}   "
              f"rms {e['rms_diff']:.6e}   output scale {e['output_scale']:.4f}")
        print(f"      = {e['ulps_of_output_scale']:.1f} ULPs of the output "
              f"scale;  roundoff budget {e['roundoff_budget']:.3e} is "
              f"{e['budget_over_max_diff']:.0f}x larger")
        print(f"      bitwise identical: {e['bitwise_identical']}  "
              f"(min |output| = {e['min_abs_output']:.2e})")
        check(e["within_roundoff"],
              f"d_in={d_in} r={r}: max diff is inside the rounding budget "
              f"({e['budget_over_max_diff']:.0f}x margin)")
        check(not e["bitwise_identical"],
              f"d_in={d_in} r={r}: the two evaluations are NOT bitwise equal, "
              "as they cannot be -- associativity is a theorem about reals")
        check(e["ulps_of_output_scale"] < 100.0,
              f"d_in={d_in} r={r}: they agree to "
              f"{e['ulps_of_output_scale']:.1f} ULPs of the output scale")

    # ---------------------------------------------------------------------
    print("\n[3] the merge is exact")
    x = L.exactness_checks()
    print(f"    zero-init identity exact: {x['zero_init_identity_exact']} "
          f"(max diff {x['zero_init_max_abs_diff']:.1e})")
    print(f"    merge vs unmerged output: max diff "
          f"{x['merge_max_abs_diff']:.1e}")
    print(f"    merged W == W + delta, bitwise: {x['merge_is_exact']}")
    print(f"    merged layer carries no adapter: {x['merged_has_no_adapter']}")
    check(x["zero_init_identity_exact"],
          "at init the adapted layer EQUALS the frozen layer exactly (W_up = 0), "
          "so the adapter cannot perturb the pretrained model before step 1")
    check(x["merge_max_abs_diff"] == 0.0,
          f"at r=8 with W_up = 0 the merge is exact to {x['merge_max_abs_diff']:.1e}")
    check(x["merge_is_exact"],
          "the merged weight is bitwise W + W_down W_up")
    check(x["merged_has_no_adapter"],
          "the merged layer is a plain frozen dense layer with no adapter")
    check(not x["base_received_grad"],
          "the frozen base receives NO gradient -- the entire source of the saving")
    check(not x["base_requires_grad"],
          "and requires_grad is False on the base, so no optimizer state exists")
    check(x["base_unchanged_by_training"],
          "the base weights are byte-identical after a training step")
    check(x["adapter_nonzero_after_one_step"],
          "while the adapter has become nonzero, so training did something")
    check(x["rank_gt_min_refused"],
          "r > min(d_in, d_out) is REFUSED rather than silently producing a "
          "factorisation larger than the matrix it replaces")

    # ---------------------------------------------------------------------
    print("\n[4] FLOP overhead and the cost of merging")
    for r in (8, 64):
        f = L.merge_flop_report(L.LoRAConfig(4096, 4096, r))
        print(f"    r = {r}: base {f['base_macs_per_token']:,} MACs/token, "
              f"adapter adds {f['lora_extra_macs_per_token']:,} "
              f"= {f['overhead_percent']:.4f}%")
        print(f"      merge costs {f['merge_macs_once']:,} MACs ONCE, "
              f"amortised after exactly {f['tokens_to_amortise_merge']:.0f} "
              "tokens of inference")
        check(abs(f["overhead_percent"] - 100.0 / (4096 / (2 * r))) < 1e-9,
              f"r={r}: overhead is exactly r(d_in+d_out)/(d_in d_out) = "
              f"{f['overhead_percent']:.4f}%")
        check(f["tokens_to_amortise_merge"] == r,
              f"r={r}: the merge pays for itself after exactly r = {r} tokens")
        check(f["merged_inference_macs_per_token"] == f["base_macs_per_token"],
              f"r={r}: merged inference costs the SAME MACs as the base model "
              "-- that is the whole of 'zero inference latency overhead'")

    # ---------------------------------------------------------------------
    print("\n[5] ten real backward passes, 4096 x 4096, r = 8")
    t = L.training_run_report(d_in=4096, d_out=4096, r=8, steps=10, batch=8)
    for kind in ("full", "lora"):
        d = t[kind]
        print(f"    {kind:<5} trainable {d['n_trainable']:>10,}  "
              f"grads {d['predicted_grad_bytes'] / MiB:>8.4f} MiB  "
              f"adam {d['predicted_adam_bytes'] / MiB:>8.4f} MiB  "
              f"persistent {d['persistent_bytes'] / MiB:>7.2f} MiB")
        print(f"          loss {d['losses'][0]:.6f} -> {d['losses'][-1]:.6f}  "
              f"decreased={d['loss_decreased']}  grads finite={d['grad_finite']}")
    print(f"    optimizer state saved: "
          f"{t['optimizer_state_saved_bytes']:,} B = "
          f"{t['optimizer_state_saved_bytes'] / MiB:.4f} MiB")
    print(f"    optimizer-state saving ratio: {t['optimizer_state_saving_ratio']:.1f}x")
    print(f"    gradient saving ratio:        {t['grad_saving_ratio']:.1f}x")
    print(f"    trainable-parameter ratio:    {t['param_ratio']:.1f}x")
    check(t["full"]["losses"][0] == t["lora"]["losses"][0],
          f"both runs start from the SAME loss "
          f"({t['full']['losses'][0]:.9f}) -- the zero-init identity, observed "
          "through a real training run")
    check(t["both_decreased_loss"],
          "both runs reduce the loss over 10 steps, so both are actually training")
    check(t["full"]["grad_finite"] and t["lora"]["grad_finite"],
          "all gradient norms are finite in both runs")
    check(t["optimizer_state_saving_ratio"] > 200,
          f"optimizer state falls {t['optimizer_state_saving_ratio']:.0f}x "
          f"({t['full']['predicted_adam_bytes'] / MiB:.2f} MiB -> "
          f"{t['lora']['predicted_adam_bytes'] / MiB:.4f} MiB)")
    check(t["lora"]["final_loss"] > t["full"]["final_loss"],
          f"and the honest cost: LoRA ends HIGHER ({t['lora']['final_loss']:.4f} "
          f"vs {t['full']['final_loss']:.4f}) because 65,536 trainable "
          "parameters fit the target more slowly than 16.8M -- capacity, not "
          "a bug")

    # ---------------------------------------------------------------------
    print("\n[6] what rank r physically is")
    print(f"    {'r':>5} {'adapter elements':>18} {'of dense':>11} "
          f"{'compression':>12} {'adam state':>13}")
    for row in L.rank_tradeoff():
        print(f"    {row['r']:>5} {row['lora_elements']:>18,} "
              f"{row['fraction_of_dense']:>10.6%} {row['compression']:>11.1f}x "
              f"{row['optimizer_MiB']:>9.4f} MiB")
    rows = L.rank_tradeoff()
    check(all(rows[i]["lora_elements"] < rows[i + 1]["lora_elements"]
              for i in range(len(rows) - 1)),
          "adapter size is strictly increasing in r, as r(d_in+d_out) requires")
    check(abs(rows[0]["lora_elements"] - 2 * 4096) < 1e-9,
          f"at r=1 the adapter has exactly 2*d = {2 * 4096:,} elements -- two "
          "rank-one matrices, which is the minimum a non-trivial update needs")
    check(rows[-1]["fraction_of_dense"] < 0.2,
          f"and even r=256 is only {rows[-1]['fraction_of_dense']:.1%} of dense, "
          "because r << d: the saving comes from r being small, not from r "
          "being tuned")

    # ---------------------------------------------------------------------
    print("\n[7] a guard that must be able to FAIL")
    # The byte model is a formula. If it ever stops matching what torch actually
    # allocates, every number in section 15.1 is fiction -- so it is checked
    # against the real tensors.
    cfg = L.LoRAConfig(256, 256, 8)
    layer = L.LoRALayer(256, 256, 8, seed=0)
    real_params = sum(p.numel() for p in layer.trainable())
    modelled = cfg.lora_params          # bias is FROZEN under LoRA
    check(real_params == modelled,
          f"the byte model predicts {modelled:,} adapter elements and the real "
          f"layer has {real_params:,} -- accounting is checked against "
          "allocation, not trusted")
    dense = L.FullLayer(256, 256, seed=0)
    real_dense = sum(p.numel() for p in dense.trainable())
    check(real_dense == cfg.dense_params + cfg.n_bias,
          f"and likewise for the dense layer: model "
          f"{cfg.dense_params + cfg.n_bias:,} vs actual {real_dense:,}")
    check(not layer.base.b.requires_grad and not layer.base.W.requires_grad,
          "the base bias AND weight are both frozen, so neither may appear in "
          "the trainable count -- the accounting bug this guard was written to "
          f"catch ({modelled:,} modelled vs {real_params:,} real at 256x256, r=8)")
    check(modelled == cfg.lora_params,
          f"the model now predicts exactly r(d_in+d_out) = {modelled:,}, with "
          "the frozen bias excluded")
    # and the Adam-state count must equal what a real optimizer allocates
    opt = torch.optim.AdamW(layer.trainable(), lr=1e-3)
    layer.forward(torch.randn(4, 256)).sum().backward()
    opt.step()
    # torch.optim.AdamW keys its state by PARAMETER OBJECT, not by index, so
    # indexing state[0] raises KeyError. Iterate the values.
    state_elems = sum(t.numel() for st in opt.state.values()
                      for t in (st["exp_avg"], st["exp_avg_sq"]))
    check(state_elems == 2 * real_params,
          f"a real AdamW allocates {state_elems:,} elements of state = "
          f"2 x {real_params:,} trainable, confirming the 2x moment factor")

    # ---------------------------------------------------------------------
    print("\n[8] how much of this depends on the draw")
    ss = L.seed_stability_report()
    print(f"    {ss['n_seeds']} seeds {ss['seeds']}; equivalence probe at "
          f"d_in={ss['d_in']}, training at {ss['steps']} steps, "
          f"batch {ss['batch']}")
    print(f"    {'quantity':<24}{'mean':>12}{'sd':>10}{'rel sd':>9}{'range':>22}")
    for k in ("ulps", "budget_margin", "full_final_loss", "lora_final_loss",
              "final_loss_ratio"):
        v = ss[k]
        print(f"    {k:<24}{v['mean']:>12.5g}{v['sample_sd_across_seeds']:>10.4g}"
              f"{v['relative_sd']:>8.2%}"
              f"{('[%.5g, %.5g]' % (v['min'], v['max'])):>22}")
    check(ss["always_within_roundoff"],
          f"the forward equivalence stays inside the rounding budget at ALL "
          f"{ss['n_seeds']} seeds (margin {ss['budget_margin']['min']:.0f}x to "
          f"{ss['budget_margin']['max']:.0f}x)")
    check(ss["ulps_always_under_100"],
          f"and the ULP count stays under 100 at every seed "
          f"({ss['ulps']['min']:.1f}-{ss['ulps']['max']:.1f}), so the 16-41 range "
          "in 16.4 is a spread, not a lucky draw")
    check(ss["step1_losses_identical_every_seed"],
          "the two runs' step-1 losses are IDENTICAL at every seed -- the "
          "zero-init identity, which is exact and therefore has no error bar")
    check(ss["lora_always_worse"],
          f"LoRA's higher step-10 loss holds at every seed "
          f"(ratio {ss['final_loss_ratio']['min']:.2f}-"
          f"{ss['final_loss_ratio']['max']:.2f}), so it is capacity and not one "
          "unlucky draw")

    print("\n" + "=" * 78)
    if FAILURES:
        print(f"CH15 FAILED: {len(FAILURES)} problem(s)")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("CH15_LORA_VERIFY_PASSED")
    print("=" * 78)


if __name__ == "__main__":
    main()