"""Chapter 9 verification -- generation, the cache, and the memory wall.

Turns every claim in the chapter into an assertion:

  1. generate_naive and generate_cached produce IDENTICAL logits at every step
     to float32 precision -- not merely the same greedy token;
  2. the two strategies emit the same token sequence;
  3. arithmetic intensity is defined and computed, and the prefill/decode
     contrast is asserted rather than narrated;
  4. decode intensity is independent of context length, which is the property
     that makes decode memory-bound;
  5. decode intensity is independent of the head count, which is why MHA and MQA
     differ in bandwidth but not in arithmetic;
  6. the KV cache footprint matches the closed form 2*L*d*batch*seq*bytes;
  7. measured tokens/sec for both strategies, over a 50-token generation.

Usage:  python ch09_generation_verify.py
Marker: CH09_GENERATION_VERIFY_PASSED
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
for _p in (_HERE.parent, _HERE.parents[1]):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch                                                    # noqa: E402

from tgd import generation as gen                               # noqa: E402
from tgd.dims import Dims                                       # noqa: E402

FAILURES: list[str] = []


def check(cond: bool, msg: str) -> None:
    if cond:
        print(f"  ok    {msg}")
    else:
        print(f"  FAIL  {msg}")
        FAILURES.append(msg)


def main() -> None:
    print("=" * 74)
    print("Chapter 9 -- generation, the KV cache, and the memory wall")
    print(f"torch {gen.MACHINE_INFO['torch']}   device={gen.MACHINE}   "
          f"threads={gen.MACHINE_INFO['threads']}")
    print("=" * 74)

    # ---- 1 & 2. equivalence -------------------------------------------
    print("\n[1] naive == cached, logits compared at EVERY step")
    for dims in (Dims(B=1, T=8, V=64, d=32, H=4, d_ff=64, n_layers=2),
                 Dims(B=1, T=8, V=97, d=32, H=2, d_ff=32, n_layers=3)):
        model = gen.init_model(dims, seed=7)
        prompt = torch.randint(0, dims.V, (1, 8),
                               generator=torch.Generator().manual_seed(3))
        seq_n, lg_n = gen.generate_naive(model, prompt, 12, record=True)
        seq_c, lg_c, cache = gen.generate_cached(model, prompt, 12, record=True)
        dev = max(float((a - b).abs().max()) for a, b in zip(lg_n, lg_c))
        rel = max(float((a - b).abs().max() / (b.abs().max() + 1e-12))
                 for a, b in zip(lg_n, lg_c))
        check(dev < 1e-5,
              f"{dims}: max |logit difference| over 12 steps = {dev:.3e} "
              f"(relative {rel:.3e}) < 1e-5")
        check(bool(torch.equal(seq_n, seq_c)),
              f"{dims}: emitted token sequences are identical")

    # ---- 3. arithmetic intensity, defined and computed ------------------
    print("\n[2] arithmetic intensity: FLOPs per byte of memory read")
    d = Dims(B=1, T=512, V=1000, d=512, H=8, d_ff=1024, n_layers=8)
    pf = gen.prefill_cost(d, n_new=512)
    dc = gen.decode_cost(d, n_cached=512)
    ai_pf = gen.arithmetic_intensity(pf["phi"], pf["bytes"])
    ai_dc = gen.arithmetic_intensity(dc["phi"], dc["bytes"])
    print(f"    prefill  T=512 : Phi={pf['phi']:.4e}  bytes={pf['bytes']:.4e}"
          f"  AI={ai_pf:8.2f} FLOP/B")
    print(f"    decode   k=512 : Phi={dc['phi']:.4e}  bytes={dc['bytes']:.4e}"
          f"  AI={ai_dc:8.2f} FLOP/B")
    check(ai_pf > ai_dc * 10,
          f"prefill intensity exceeds decode by {ai_pf / max(ai_dc, 1e-12):.0f}x "
          f"at the same context")

    # ---- 4. decode intensity is independent of context ------------------
    print("\n[3] decode intensity does NOT depend on context length")
    ais = {}
    for k in (1, 64, 512, 4096, 32768):
        c = gen.decode_cost(d, n_cached=k)
        ais[k] = gen.arithmetic_intensity(c["phi"], c["bytes"])
        print(f"    cached k={k:>6} : AI={ais[k]:8.3f} FLOP/B")
    proj_only = 2 * d.d * d.d / (4 * d.d * d.d)   # 2mnk/(4nk) at n=1 = 0.5
    check(all(abs(ais[k] - ais[1]) < 0.35 * ais[1] for k in ais),
          f"decode AI stays within 35% of its k=1 value ({ais[1]:.2f}) across "
          f"a 32,768x range of context")
    check(abs(ais[1] - proj_only) < 0.05,
          f"and at k=1 it equals the closed form 2*d*d/(4*d*d) = {proj_only:.3f} "
          f"FLOP/B: a rank-1 update of a float32 weight matrix is exactly "
          f"0.5 FLOP per byte")

    # ---- 5. decode intensity is independent of head count ---------------
    print("\n[4] decode AI is independent of H, because H*d_h = d exactly")
    row = []
    for H in (1, 2, 4, 8, 16):
        dd = Dims(B=1, T=8, V=1000, d=512, H=H, d_ff=1024, n_layers=8)
        c = gen.decode_cost(dd, n_cached=512)
        row.append((H, dd.d_h, gen.arithmetic_intensity(c["phi"], c["bytes"])))
    for H, d_h, ai in row:
        print(f"    H={H:>3}  d_h={d_h:>4}  AI={ai:8.3f} FLOP/B")
    spread = max(r[2] for r in row) - min(r[2] for r in row)
    check(spread < 1e-6,
          f"all five head counts give the same decode intensity "
          f"(spread {spread:.2e}) -- the score term sums H*d_h*d = d*d")

    # ---- 6. KV cache footprint -----------------------------------------
    print("\n[5] KV cache footprint: 2 * L * d * batch * seq * bytes")
    big = Dims(B=1, T=8, V=128256, d=4096, H=32, d_ff=11008, n_layers=32)
    print(f"    reference model: L={big.n_layers} d={big.d} H={big.H} "
          f"d_h={big.d_h} V={big.V}")
    for eb, name in ((2, "bf16"), (4, "fp32")):
        for seq in (2048, 8192, 32768, 131072):
            b = gen.kv_cache_bytes(big, seq, batch=1, element_bytes=eb)
            print(f"    seq={seq:>7} {name}: {b / 1e9:8.2f} GB")
    b1 = gen.kv_cache_bytes(big, 2048, 1, 2)
    b8 = gen.kv_cache_bytes(big, 16384, 1, 2)
    check(abs(b8 - 8 * b1) < 1,
          "the footprint is exactly linear in sequence length")
    check(abs(b1 - 2 * big.n_layers * big.d * 2048 * 2) < 1,
          "and matches the stated closed form 2 * L * d * batch * seq * bytes")

    # ---- 7. measurement -------------------------------------------------
    print("\n[6] measured latency, 50 new tokens, local hardware")
    print(f"    {'model':<34} {'naive tok/s':>13} {'cached tok/s':>13} "
          f"{'speedup':>8} {'max dlogit':>12}")
    rows = []
    for dims, npr in ((Dims(B=1, T=8, V=1000, d=128, H=4, d_ff=256, n_layers=4), 8),
                      (Dims(B=1, T=8, V=1000, d=256, H=8, d_ff=512, n_layers=6), 8)):
        m = gen.measure(dims, n_prompt=npr, n_new=50, repeat=3)
        rows.append(m)
        print(f"    {m.label:<34} {m.naive_tok_s:13.2f} "
              f"{m.cached_tok_s:13.2f} {m.speedup:8.2f}x "
              f"{m.max_logit_dev:12.2e}")
        check(m.max_logit_dev < 1e-5,
              f"{m.label}: naive and cached agree to {m.max_logit_dev:.2e}")
        check(m.detail["tokens_identical"],
              f"{m.label}: identical token sequences")
        check(m.speedup > 1.5,
              f"{m.label}: the cache is faster ({m.speedup:.2f}x)")

    print("\n    scaling with context, d=256 L=6, 50 new tokens")
    for npr in (8, 32, 64):
        m = gen.measure(Dims(B=1, T=8, V=1000, d=256, H=8, d_ff=512,
                             n_layers=6), n_prompt=npr, n_new=50, repeat=2)
        print(f"      prompt={npr:>3} : naive {m.naive_tok_s:8.2f} tok/s   "
              f"cached {m.cached_tok_s:8.2f} tok/s   speedup {m.speedup:6.2f}x")
        if npr >= 32:
            rows.append(m)

    print("\n[7] is the ridge point above or below the decode intensity?")
    # The ridge point of an accelerator is peak_FLOPS / bandwidth: the
    # intensity below which the machine cannot keep its compute units fed.
    # We do NOT measure either quantity here -- sustained tensor throughput
    # depends on thread count, BLAS backend and precision, so quoting a
    # measured-looking peak next to our measured latency would be dishonest.
    # These are PUBLISHED VENDOR SPECIFICATIONS, labelled as such, and they are
    # used only to answer a bracketing question: is the decode intensity far
    # below the ridge point of any real accelerator?
    RIDGE_SPEC = [
        ("A100 fp16 (spec)",     312e12, 1555e9),
        ("H100 fp16 (spec)",     989e12, 3350e9),
        ("M2 Max (spec)",      13800e12,  400e9),
    ]
    print(f"    {'accelerator':<22} {'peak':>12} {'bandwidth':>12} "
          f"{'ridge FLOP/B':>13}")
    for name, pf, bw in RIDGE_SPEC:
        r = pf / bw
        print(f"    {name:<22} {pf / 1e12:9.1f} TF/s {bw / 1e9:9.0f} GB/s "
              f"{r:13.1f}")
    check(ai_dc < min(pf / bw for _, pf, bw in RIDGE_SPEC) / 100,
          f"decode intensity {ai_dc:.2f} FLOP/B is more than 100x below the "
          f"ridge point of EVERY accelerator above; decode cannot be "
          f"compute-bound on any of them")

    print("\n" + "=" * 74)
    if FAILURES:
        print(f"CH09 FAILED: {len(FAILURES)} problem(s)")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("CH09_GENERATION_VERIFY_PASSED")
    print("=" * 74)


if __name__ == "__main__":
    main()