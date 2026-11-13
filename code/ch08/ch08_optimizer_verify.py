"""Chapter 8 verification -- the optimizer, measured rather than asserted.

Turns every claim in the chapter into an assertion:

  1. the explicit AdamW in tgd.optim matches torch.optim.AdamW elementwise,
     at the hyperparameters the Chapter 7 harness actually used;
  2. the step-0 scale invariance holds to float64 precision for every scale c,
     which is the identity Chapter 8 proves;
  3. at a later step the same invariance is BROKEN, and by how much -- the
     state accumulation is what breaks it;
  4. a gradient spike at a late step is penalised relative to a consistent one;
  5. epsilon acts as an upper bound on the per-parameter step, and the bound is
     measurable rather than rhetorical;
  6. per-group clipping breaks the invariance that global clipping preserves.

Usage:  python ch08_optimizer_verify.py
Marker: CH08_OPTIMIZER_VERIFY_PASSED
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve()
for _p in (_HERE.parent, _HERE.parents[1]):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tgd.optim import (BETA_1, BETA_2, EPS_ADAM, WEIGHT_DECAY,  # noqa: E402
                        AdamW)

DTYPE = torch.float64
FAILURES: list[str] = []


def check(cond: bool, msg: str) -> None:
    if cond:
        print(f"  ok    {msg}")
    else:
        print(f"  FAIL  {msg}")
        FAILURES.append(msg)


def rel(a, b) -> float:
    a, b = a.detach().double(), b.detach().double()
    return float((a - b).norm() / (a.norm() + b.norm() + 1e-12))


def quadratic(seed: int = 0, n: int = 64) -> torch.nn.Parameter:
    g = torch.Generator().manual_seed(seed)
    return torch.nn.Parameter(
        torch.randn(n, generator=g, dtype=torch.float32) * 0.1)


def main() -> None:
    print("=" * 74)
    print("Chapter 8 -- the optimizer, verified")
    print(f"torch {torch.__version__}")
    print(f"harness hyperparameters: betas=({BETA_1}, {BETA_2})  "
          f"eps={EPS_ADAM}  weight_decay={WEIGHT_DECAY}")
    print("=" * 74)

    # ---- 1. equivalence with torch.optim.AdamW --------------------------
    print("\n[1] explicit AdamW == torch.optim.AdamW, elementwise")
    for lr in (1e-3, 2e-3, 6e-3):
        a = torch.nn.Parameter(torch.linspace(-0.5, 0.5, 32, dtype=torch.float32))
        b = copy.deepcopy(a)
        mine = AdamW([a], lr=lr)
        theirs = torch.optim.AdamW(
            [b], lr=lr, betas=(BETA_1, BETA_2), eps=EPS_ADAM,
            weight_decay=WEIGHT_DECAY)
        worst = 0.0
        for step in range(12):
            g = torch.linspace(-1.0, 1.0, 32, dtype=torch.float32) * (1 + 0.3 * step)
            a.grad = g.clone()
            b.grad = g.clone()
            mine.step()
            theirs.step()
            worst = max(worst, float((a.detach() - b.detach()).abs().max()))
        # float32 accumulation over 12 steps: an ABSOLUTE 1e-7 threshold sits
        # below float32 resolution for parameters of order 1, so the comparison
        # must be relative to the parameter scale.
        check(worst / 1.0 < 1e-5,
              f"lr={lr}: 12 steps, max abs parameter difference {worst:.2e} "
              f"(relative, float32)")

    # the same, but with clipping active -- the Chapter 7 configuration
    print("\n[1b] equivalence with global clipping at threshold 2.5")
    a = torch.nn.Parameter(torch.linspace(-0.5, 0.5, 32, dtype=torch.float32))
    b = copy.deepcopy(a)
    mine = AdamW([a], lr=2e-3)
    theirs = torch.optim.AdamW([b], lr=2e-3, betas=(BETA_1, BETA_2),
                                eps=EPS_ADAM, weight_decay=WEIGHT_DECAY)
    worst = 0.0
    for step in range(8):
        g = torch.linspace(-6.0, 6.0, 32, dtype=torch.float32)
        a.grad = g.clone()
        b.grad = g.clone()
        torch.nn.utils.clip_grad_norm_([a], 2.5)
        torch.nn.utils.clip_grad_norm_([b], 2.5)
        mine.step()
        theirs.step()
        worst = max(worst, float((a.detach() - b.detach()).abs().max()))
    check(worst < 1e-5,
          f"8 clipped steps, max abs parameter difference {worst:.2e} "
          f"(relative, float32)")

    # ---- 2. step-0 scale invariance, exactly ----------------------------
    print("\n[2] step-0 scale invariance (the identity of section 8.2)")
    print("    replacing g_0 by c*g_0 must leave ||m_hat/(sqrt(v_hat)+eps)|| fixed")
    g0 = torch.linspace(-3.0, 3.0, 64, dtype=torch.float32)
    ref = None
    worst_dev = 0.0
    print(f"    {'c':>8} {'||ratio||':>14} {'rel dev from c=1':>20}")
    for c in (0.01, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 100.0):
        p = torch.nn.Parameter(torch.zeros(64))
        opt = AdamW([p], lr=1e-3)
        p.grad = g0 * c
        tr = opt.step()
        rn = float(tr.ratio_norm)
        if ref is None:
            ref = rn
        dev = abs(rn - ref) / (ref + 1e-30)
        worst_dev = max(worst_dev, dev)
        print(f"    {c:>8g} {rn:>14.10f} {dev:>20.2e}")
    check(worst_dev < 1e-5,
          f"ratio norm constant to {worst_dev:.2e} across a 10,000x gradient range")

    # and the closed form: the ratio is sign(g) elementwise
    p = torch.nn.Parameter(torch.zeros(64))
    opt = AdamW([p], lr=1e-3)
    p.grad = g0.clone()
    opt.step()
    ratio = None
    # recompute the stored ratio from state to check sign()
    m, v = opt.state[0]
    m_hat = m / (1 - BETA_1)
    v_hat = v / (1 - BETA_2)
    ratio = m_hat / (v_hat.sqrt() + EPS_ADAM)
    dev = float((ratio - torch.sign(g0)).abs().max())
    check(dev < 1e-6,
          f"ratio elementwise equals sign(g_0) to {dev:.2e} (float32 precision)")

    # ---- 3. the SAME invariance breaks once state exists -----------------
    print("\n[3] at a later step the same invariance is broken")
    print("    history at a CONSISTENT scale 1; only the FINAL gradient is scaled")
    g_const = torch.linspace(-1.0, 1.0, 64, dtype=torch.float32)
    print(f"    {'k':>5} {'c=0.1':>12} {'c=1':>12} {'c=10':>12} {'max dev':>10}")
    devs = {}
    for k_target in (1, 2, 5, 20, 100):
        row = []
        for c in (0.1, 1.0, 10.0):
            p = torch.nn.Parameter(torch.zeros(64))
            opt = AdamW([p], lr=1e-3)
            for _ in range(k_target - 1):
                p.grad = g_const.clone()          # consistent history
                opt.step()
            p.grad = g_const * c                  # only the last step scales
            row.append(float(opt.step().ratio_norm))
        base = row[1]
        cmax = max(abs(x - base) / (base + 1e-30) for x in row)
        devs[k_target] = cmax
        print(f"    {k_target:>5} " + " ".join(f"{x:>12.6f}" for x in row)
              + f" {cmax:>10.2e}")
    check(devs[1] < 1e-4,
          f"k=1: invariance holds (max dev {devs[1]:.2e})")
    check(devs[100] > 1e-2,
          f"k=100: invariance is BROKEN by accumulated state "
          f"(max dev {devs[100]:.2e})")
    # My first guess here was that the deviation would keep GROWING with history
    # length. It does not: it jumps at k=2 and then SATURATES, because beta_2
    # fixes an effective memory horizon (~1/(1-beta_2) = 20 steps here) past
    # which v_k no longer changes appreciably. Assert the saturating shape,
    # which is what the mechanism actually predicts.
    check(devs[2] > 0.05,
          f"the break is immediate: k=1 {devs[1]:.1e} -> k=2 {devs[2]:.1e}")
    check(devs[5] > 0.05 and devs[100] > 0.05,
          f"and it stays broken far into training "
          f"(k=5 {devs[5]:.1e}, k=100 {devs[100]:.1e})")
    check(devs[100] <= devs[2] * 1.5,
          f"the asymmetry SATURATES rather than growing without bound "
          f"({devs[2]:.1e} -> {devs[5]:.1e} -> {devs[100]:.1e}); beta_2 = "
          f"{BETA_2} gives an effective memory of ~{1 / (1 - BETA_2):.0f} steps, "
          f"past which v_k stops changing appreciably")

    # ---- 4. a late spike is penalised -----------------------------------
    print("\n[4] a gradient spike at a late step, against a consistent gradient")
    spike_ratio, consist_ratio = [], []
    for scale in (1.0, 10.0, 100.0):
        p = torch.nn.Parameter(torch.zeros(64))
        opt = AdamW([p], lr=1e-3)
        for _ in range(99):
            p.grad = g_const.clone()
            opt.step()
        p.grad = g_const * scale
        spike_ratio.append(float(opt.step().ratio_norm))
    for scale in (1.0, 10.0, 100.0):
        p = torch.nn.Parameter(torch.zeros(64))
        opt = AdamW([p], lr=1e-3)
        for _ in range(100):
            p.grad = g_const * scale
            opt.step()
        consist_ratio.append(float(opt.trace[-1].ratio_norm))
    print("    consistent signal (100 steps at scale s):")
    for s, r in zip((1, 10, 100), consist_ratio):
        print(f"      s={s:<4} ||ratio||={r:10.6f}")
    print("    single spike at step 100, after 99 steps at scale 1:")
    for s, r in zip((1, 10, 100), spike_ratio):
        print(f"      s={s:<4} ||ratio||={r:10.6f}")
    # a 100x spike produces far less than 100x movement
    spike_gain = spike_ratio[-1] / spike_ratio[0]
    consist_gain = consist_ratio[-1] / consist_ratio[0]
    print(f"    spike:      {spike_ratio[0]:.6f} -> {spike_ratio[-1]:.6f} "
          f"= x{spike_gain:.3f} for a 100x gradient")
    print(f"    consistent: {consist_ratio[0]:.6f} -> {consist_ratio[-1]:.6f} "
          f"= x{consist_gain:.3f} for a 100x gradient")
    # I predicted a >10x penalty for a spike. The measurement is less extreme
    # and more interesting than that: a 100x spike produces 0.486x the movement
    # while a 100x CONSISTENT signal produces 1.000x. The spike is discounted by
    # roughly half, not annihilated, because the momentum term still absorbs a
    # (1-beta_1) fraction of the new gradient.
    check(spike_gain < 0.6 * consist_gain,
          f"a 100x spike moves the parameter {spike_gain:.3f}x while a 100x "
          f"consistent signal moves it {consist_gain:.3f}x: the spike is "
          f"discounted to roughly half, not annihilated, because m_k still "
          f"absorbs (1-beta_1)={1 - BETA_1:g} of the new gradient")

    # ---- 5. epsilon bounds the step -------------------------------------
    print("\n[5] epsilon as an upper bound on the per-parameter step")
    print(f"    |delta_i| = lr*|m_hat_i| / (sqrt(v_hat_i) + eps) <= lr*|m_hat_i|/eps")
    p = torch.nn.Parameter(torch.zeros(64))
    opt = AdamW([p], lr=1e-3)
    p.grad = g_const * 1e-6           # a gradient far below eps
    tr = opt.step()
    print(f"    with ||g|| = {float(g_const.norm())*1e-6:.3e} and eps={EPS_ADAM:g}:")
    print(f"      eps/sqrt(v_hat+eps) reaches {float(tr.eps_term_max):.6f}")
    print(f"      ||update|| = {float(tr.update_norm):.6e}")
    check(float(tr.eps_term_max) > 1e-3,
          f"epsilon contributes a measurable share of the denominator "
          f"({float(tr.eps_term_max):.2e})")
    # the bound must actually hold
    m, v = opt.state[0]
    m_hat = m / (1 - BETA_1)
    v_hat = v / (1 - BETA_2)
    bound = 1e-3 * (m_hat.abs() / EPS_ADAM)
    actual = 1e-3 * (m_hat.abs() / (v_hat.sqrt() + EPS_ADAM))
    check(bool((actual <= bound + 1e-18).all()),
          "the per-parameter step obeys lr*|m_hat|/eps on every coordinate")

    # ---- 6. per-group clipping: correct claim, measured -----------------
    # My first attempt here claimed that rescaling ONE group by 0.5 changes the
    # two groups' step ratio by 2x. It does not, and the reason is worth having
    # in the book: at step 0 every parameter has its OWN zero state, so each
    # group's first step is independently scale-invariant. Per-group clipping is
    # therefore harmless at step 0 too. It only bites once state exists, and
    # then it bites per group -- which is precisely what a *global* rescale
    # avoids, because it moves every group by the same factor.
    print("\n[6] per-group vs global rescaling, with accumulated state")

    def build(steps: int, pre_scale):
        w1 = torch.nn.Parameter(torch.ones(8))
        w2 = torch.nn.Parameter(torch.ones(8) * 100.0)
        opt = AdamW([w1, w2], lr=1e-3)
        g = torch.linspace(-1.0, 1.0, 8, dtype=torch.float32)
        for _ in range(steps):
            w1.grad = g.clone()
            w2.grad = g.clone()
            opt.step()
        w1.grad = g * pre_scale        # rescale ONLY group 1
        w2.grad = g.clone()
        tr = opt.step()
        m1, v1 = opt.state[0]
        m2, v2 = opt.state[1]
        r1 = (m1 / (1 - BETA_1)) / ((v1 / (1 - BETA_2)).sqrt() + EPS_ADAM)
        r2 = (m2 / (1 - BETA_1)) / ((v2 / (1 - BETA_2)).sqrt() + EPS_ADAM)
        return float(r2.norm() / r1.norm()), tr

    for steps in (0, 1, 20):
        r_k, _ = build(steps, 0.1)      # 10x drop on group 1 only
        print(f"    {steps:>3} history step(s): group2/group1 step ratio = {r_k:.6f}")
        if steps == 0:
            check(abs(r_k - 1.0) < 1e-4,
                  f"at step 0 a per-group rescale changes nothing "
                  f"({r_k:.6f}): each group has its own zero state and is "
                  f"independently scale-invariant")

    r_20, _ = build(20, 0.1)
    # Direction: damping group 1 shrinks m_1 while v_1 is set by history, so
    # group 1's ratio FALLS and the group2/group1 ratio rises above 1.
    check(r_20 > 1.01,
          f"with 20 steps of history the SAME rescale moves the groups "
          f"differently (ratio {r_20:.6f}, i.e. group 1's step is damped "
          f"relative to group 2's) -- a global rescale could never do this, "
          f"which is the precise sense in which per-group clipping escapes the "
          f"invariance global clipping preserves")

    print("\n" + "=" * 74)
    if FAILURES:
        print(f"CH08 FAILED: {len(FAILURES)} problem(s)")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("CH08_OPTIMIZER_VERIFY_PASSED")
    print("=" * 74)


if __name__ == "__main__":
    main()