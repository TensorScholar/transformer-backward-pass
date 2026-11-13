r"""Chapter 17 verification -- the integrated decoder, running end to end.

Turns every claim in the chapter into an assertion:

  1. the configuration is what it claims: ~91M parameters, GQA(10, 2), dense
     SwiGLU, with the parameter model matching real allocation exactly;
  2. end-to-end memory at N=2048: weights plus KV cache, with GQA's cache saving
     over MHA stated as a measured ratio;
  3. prefill and cached decode agree -- without which any latency number is
     timing a wrong computation quickly;
  4. a 20-token generation runs, terminates, stays in vocabulary, is
     deterministic, and is timed: wall seconds, tokens per second, achieved
     GB/s, and decode-phase arithmetic intensity;
  5. the MoE variant runs end to end and its capacity/compute trade matches the
     Chapter 16 machinery;
  6. guards that must be able to FAIL: the memory model against allocation, and
     the prefill/decode agreement against a threshold that wrong-position RoPE
     would violate by orders of magnitude.

Usage:  python ch17_model_verify.py [--strict-bench]
Marker: CH17_MODEL_VERIFY_PASSED

``--strict-bench`` promotes the machine-dependent wall-clock checks (the
achieved-GB/s floor and the <25% timing-spread bound) from warnings to hard
failures. By default they warn: both are properties of the HOST, not of the
code -- on a shared CPU, scheduler noise alone can push the measured spread
past 25% or the achieved bandwidth under 1 GB/s while every numerical check
still passes, and a numerically exact run must not exit 1 because someone
else's container was busy.
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
for _p in (_HERE.parent, _HERE.parents[1]):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch                                                  # noqa: E402

from tgd import model as M                                     # noqa: E402

STRICT_BENCH = "--strict-bench" in sys.argv
FAILURES: list[str] = []


def check(cond: bool, msg: str) -> None:
    if cond:
        print(f"  ok    {msg}")
    else:
        print(f"  FAIL  {msg}")
        FAILURES.append(msg)


def bench_check(cond: bool, msg: str) -> None:
    """A wall-clock check: hard failure only under ``--strict-bench``.

    Timing-dependent bounds measure the host as much as the code. Without the
    flag a violation is reported as a WARNING and the exit code stays 0, so a
    numerically exact run on a loaded shared machine is not a false failure.
    """
    if cond:
        print(f"  ok    {msg}")
    elif STRICT_BENCH:
        print(f"  FAIL  {msg}")
        FAILURES.append(msg)
    else:
        print(f"  WARN  {msg} (host-dependent; pass --strict-bench to enforce)")


def main() -> None:
    print("=" * 78)
    print("Chapter 17 -- final synthesis: the integrated decoder")
    print("RMSNorm + GQA + RoPE + SwiGLU/MoE, with a KV-cache generate loop.")
    print("Every primitive is imported from tgd/, not reimplemented.")
    print("=" * 78)

    cfg = M.CONFIG_100M
    MiB = 2 ** 20

    # ---------------------------------------------------------------------
    print("\n[1] the configuration and its parameter model")
    print(f"    d={cfg.d} L={cfg.n_layers} H={cfg.H} H_kv={cfg.H_kv} "
          f"d_h={cfg.d_h} d_ff={cfg.d_ff} V={cfg.V} ffn={cfg.ffn}")
    print(f"    n_params (model): {cfg.n_params:,}")
    P = M.init_decoder(cfg, seed=0)
    print(f"    n_params (actual): {P.count():,}")
    check(cfg.n_params == P.count(),
          f"the parameter model matches allocation exactly "
          f"({cfg.n_params:,} = {P.count():,})")
    check(85_000_000 < cfg.n_params < 100_000_000,
          f"{cfg.n_params / 1e6:.1f}M parameters is the brief's ~100M scale")
    check(cfg.H_kv < cfg.H,
          f"this is genuinely GQA (H_kv={cfg.H_kv} < H={cfg.H}), not MHA "
          "relabeled")
    check(cfg.d_h == 64,
          f"per-head width d_h = {cfg.d_h}, so the 1/sqrt(d_h) scale of "
          "contract C2 is exercised at production width")

    # ---------------------------------------------------------------------
    print("\n[2] end-to-end memory at N=2048")
    mem = M.memory_report(cfg, seq_len=2048, batch=1)
    print(f"    weights : {mem['weight_MiB']:.2f} MiB")
    print(f"    KV cache: {mem['kv_MiB']:.2f} MiB  "
          f"(MHA would be {mem['kv_MiB_mha']:.2f} MiB)")
    print(f"    total   : {mem['total_MiB']:.2f} MiB")
    print(f"    GQA cache saving vs MHA: {mem['kv_saving_vs_mha']:.1f}x")
    print(f"    KV share of total: {mem['kv_share']:.1%}  "
          f"weights: {mem['weight_share']:.1%}")
    check(abs(mem["kv_saving_vs_mha"] - cfg.H / cfg.H_kv) < 1e-9,
          f"GQA cache saving is exactly H/H_kv = {cfg.H / cfg.H_kv:.0f}x")
    check(mem["weight_bytes"] == cfg.n_params * 4,
          "weight bytes are exactly 4x params (float32, no hidden terms)")
    check(mem["kv_bytes"] == 2 * cfg.n_layers * cfg.kv_width * 2048 * 4,
          "KV bytes are exactly 2*L*d_kv*T*e -- the Chapter 9 formula")
    check(mem["weight_share"] > 0.9,
          f"weights are {mem['weight_share']:.1%} of the total: at batch 1 the "
          "model dominates and the cache is small, which is why batch-1 "
          "latency is a weights-bandwidth story")
    # MoE variant for contrast
    mc = M.CONFIG_100M_MOE
    mm = M.memory_report(mc, seq_len=2048, batch=1)
    print(f"    MoE variant ({mc.n_exp} experts, top-{mc.top_k}): "
          f"{mc.n_params:,} params, {mm['total_MiB']:.1f} MiB total")
    check(mc.n_params > cfg.n_params,
          f"the MoE variant holds {mc.n_params / cfg.n_params:.2f}x the "
          "parameters -- capacity up, as Chapter 16 promises")

    # ---------------------------------------------------------------------
    print("\n[3] prefill and cached decode agree")
    prompt = torch.randint(0, cfg.V, (1, 8),
                           generator=torch.Generator().manual_seed(7))
    Z = M.decoder_forward(P, prompt)
    seq, lg, _ = M.generate(P, prompt, 4, record=True)
    dev = 0.0
    for i in range(4):
        Zr = M.decoder_forward(P, seq[:, :8 + i])
        dev = max(dev, float((Zr[0, -1] - lg[i]).abs().max()))
    print(f"    max logit deviation over 4 decode steps: {dev:.3e}")
    check(dev < 1e-4,
          f"prefill and decode agree to {dev:.2e} -- the cache logic is "
          "proved, not restated")
    check(bool(torch.isfinite(Z).all()),
          "prefill logits are all finite on the 91M model")

    # ---------------------------------------------------------------------
    print("\n[4] 20-token generation, timed")
    lat = M.latency_report(P, n_prompt=8, n_new=20, seed=1, repeat=2)
    print(f"    machine: {lat['machine']}")
    print(f"    wall time: {lat['wall_s']:.4f} s for 20 tokens")
    print(f"    throughput: {lat['tok_s']:.2f} tok/s")
    print(f"    bytes moved: {lat['bytes_moved_GB']:.3f} GB")
    print(f"    achieved bandwidth: {lat['achieved_GBs']:.2f} GB/s")
    print(f"    decode intensity: "
          f"{lat['decode_intensity_flop_per_byte']:.3f} FLOP/byte")
    print(f"    prefill/decode max dev on 91M: "
          f"{lat['max_logit_dev_prefill_vs_decode']:.3e}")
    bench_check(lat["tok_s"] > 1.0,
                f"{lat['tok_s']:.1f} tok/s is a running system, not a trace")
    check(lat["decode_intensity_flop_per_byte"] < 1.0,
          f"intensity {lat['decode_intensity_flop_per_byte']:.3f} < 1 "
          "FLOP/byte: decode is memory-bound, as Chapters 9 and 11 require")
    check(lat["max_logit_dev_prefill_vs_decode"] < 1e-3,
          "and the timed run computed the right thing while being timed")
    check(lat["mean_kv_bytes_per_step"] < lat["weight_bytes"],
          f"per-step KV traffic ({lat['mean_kv_bytes_per_step'] / 2**20:.2f} "
          f"MiB) is below weight traffic "
          f"({lat['weight_bytes'] / 2**20:.0f} MiB) -- weights dominate at "
          "batch 1, which is the whole of the bandwidth story")
    bench_check(lat["achieved_GBs"] > 1.0,
                f"{lat['achieved_GBs']:.1f} GB/s achieved -- a measured "
                "bandwidth, not a specification peak")
    # timing stability: best-of-N hides the spread, so it is measured
    stab = M.timing_stability(P, prompt, 6, runs=4)
    print(f"    timing over {stab['runs']} runs: mean {stab['mean_s']:.4f}s  "
          f"sample SD {stab['sample_sd_s']:.4f}s  rel {stab['relative_sd']:.1%}  "
          f"best {stab['best_s']:.4f}s")
    bench_check(stab["relative_sd"] < 0.25,
                f"wall-clock spread is {stab['relative_sd']:.1%} (sample SD over "
                f"{stab['runs']} runs) -- below 25%, so the headline tok/s means "
                "something; above it the machine would be too noisy to quote")

    # ---------------------------------------------------------------------
    print("\n[5] determinism and vocabulary")
    s1 = M.generate(P, prompt, 6)
    s2 = M.generate(P, prompt, 6)
    check(bool(torch.equal(s1, s2)),
          "same (config, seed, prompt) gives bitwise-identical tokens")
    check(bool(((s1 >= 0) & (s1 < cfg.V)).all()),
          f"all generated ids in [0, {cfg.V})")
    check(s1.shape == (1, 8 + 6),
          f"generate returns prompt + n_new tokens {tuple(s1.shape)}")

    # ---------------------------------------------------------------------
    print("\n[6] the MoE path runs end to end")
    Pm = M.init_decoder(M.CONFIG_100M_MOE, seed=0)
    Zm = M.decoder_forward(Pm, prompt)
    sm = M.generate(Pm, prompt, 2)
    check(bool(torch.isfinite(Zm).all()),
          "MoE prefill logits finite at 4 experts, top-2")
    check(bool(((sm >= 0) & (sm < cfg.V)).all()),
          "MoE generated ids in vocabulary")
    check(M.CONFIG_100M_MOE.n_params != cfg.n_params,
          "the MoE variant is a different-sized model, honestly reported")

    # ---------------------------------------------------------------------
    print("\n[7] guards that must be able to FAIL")
    integ = M.integration_checks()
    check(integ["prefill_vs_decode_agree"],
          f"independent prefill/decode check agrees "
          f"({integ['prefill_vs_decode_max_dev']:.2e})")
    check(integ["gqa_G1_matches_mha"],
          "GQA at G=1 is bitwise the canonical MHA block "
          f"({integ['gqa_G1_vs_canonical_max_dev']:.1e})")
    check(integ["counts_match"],
          "parameter model matches allocation on the small config too")
    check(integ["deterministic_tokens"] and integ["tokens_in_vocab"],
          "small-config generation deterministic and in-vocabulary")
    # wrong-position RoPE must be detectable: decode at pos+100 must DIFFER
    with torch.no_grad():
        from tgd import model as _m
        c0 = _m.KVCacheDecoder(cfg, 1, 16, dtype=P.E.dtype)
        lg_right = _m._model_decode(P, prompt[:, 0:1], c0, 0)
        c1 = _m.KVCacheDecoder(cfg, 1, 116, dtype=P.E.dtype)
        lg_wrong = _m._model_decode(P, prompt[:, 0:1], c1, 100)
        wrong_dev = float((lg_right - lg_wrong).abs().max())
    check(wrong_dev > 1e-3,
          f"decoding at absolute position 100 instead of 0 changes logits by "
          f"{wrong_dev:.3e} -- the classic silent bug is detectable, so the "
          "agreement in [3] is testing something")

    print("\n" + "=" * 78)
    if FAILURES:
        print(f"CH17 FAILED: {len(FAILURES)} problem(s)")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("CH17_MODEL_VERIFY_PASSED")
    print("=" * 78)


if __name__ == "__main__":
    main()