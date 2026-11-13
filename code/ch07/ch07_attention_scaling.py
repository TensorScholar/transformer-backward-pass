"""Chapter 7 Experiment A: attention scaling at initialization.

Question
--------
How does the 1/sqrt(d_h) factor change the distribution of attention scores,
attention entropy, and the local softmax score-gradient RMS as head width grows?

Assumptions / controlled setup
------------------------------
- Q and K components are independently sampled N(0, 1).
- Causal masking is used.
- No training or learned adaptation is involved; this is an initialization-local test.
- The same Q/K samples and upstream dL/dA samples are reused for scaled/unscaled variants.

Pre-stated predictions
----------------------
1. Unscaled score std grows approximately sqrt(d_h).
2. Scaled score std remains approximately O(1).
3. Unscaled softmax becomes more concentrated as d_h grows, lowering normalized entropy.
4. The local softmax gradient statistics change as concentration increases; direction and
   exact magnitude are measured rather than asserted as a universal theorem.

Scope
-----
This experiment does NOT establish that an unscaled trained Transformer cannot adapt or
train. It isolates one local initialization effect under stated distributional assumptions.
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
import math
import torch

DTYPE = torch.float64
T = 32
HEAD_DIMS = (16, 64, 256)
SEEDS = tuple(range(12))


def masked_softmax_stats(d_h: int, seed: int, scaled: bool):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(T, d_h, generator=g, dtype=DTYPE)
    k = torch.randn(T, d_h, generator=g, dtype=DTYPE)
    raw = q @ k.T
    scores = raw / math.sqrt(d_h) if scaled else raw
    scores = scores.clone().requires_grad_(True)
    mask = torch.triu(torch.full_like(scores, float('-inf')), diagonal=1)
    probs = torch.softmax(scores + mask, dim=-1)

    # Fixed-distribution upstream signal to probe the local softmax VJP.
    upstream = torch.randn(T, T, generator=g, dtype=DTYPE)
    upstream = torch.tril(upstream)
    (probs * upstream).sum().backward()
    bar_s = scores.grad.detach()

    legal = torch.tril(torch.ones(T, T, dtype=torch.bool))
    # Score std only over legal logits; future entries were not part of softmax alternatives.
    score_std = scores.detach()[legal].std(unbiased=True).item()

    # Normalize row entropy by log(number of legal keys); skip row 0 where log(1)=0.
    entropies = []
    for i in range(1, T):
        p = probs[i, : i + 1].detach()
        h = -(p * p.clamp_min(1e-300).log()).sum().item()
        entropies.append(h / math.log(i + 1))
    entropy_norm = sum(entropies) / len(entropies)

    grad_rms = bar_s[legal].pow(2).mean().sqrt().item()
    max_prob = torch.stack([probs[i, : i + 1].max() for i in range(1, T)]).mean().item()
    return score_std, entropy_norm, grad_rms, max_prob


def mean_std(xs):
    x = torch.tensor(xs, dtype=DTYPE)
    return x.mean().item(), x.std(unbiased=True).item()


def main():
    out = Path(__file__).with_name('ch07_attention_scaling_results.csv')
    rows = []
    print('Experiment A — attention scaling at initialization')
    print('T=32, Q/K iid N(0,1), 12 paired seeds, causal mask')
    print()
    print(f"{'d_h':>5} {'variant':>10} {'score_std':>18} {'norm_entropy':>18} {'grad_rms':>18} {'mean_max_p':>18}")
    for d_h in HEAD_DIMS:
        for scaled in (True, False):
            vals = [masked_softmax_stats(d_h, s, scaled) for s in SEEDS]
            cols = list(zip(*vals))
            summaries = [mean_std(c) for c in cols]
            label = 'scaled' if scaled else 'unscaled'
            print(f"{d_h:5d} {label:>10} " + ' '.join(f"{m:.5f}±{sd:.5f}" for m, sd in summaries))
            for seed, v in zip(SEEDS, vals):
                rows.append(dict(d_h=d_h, variant=label, seed=seed,
                                 score_std=v[0], normalized_entropy=v[1],
                                 score_grad_rms=v[2], mean_max_probability=v[3]))
    with out.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(f'\nWrote {out}')
    print('CH07_ATTENTION_SCALING_COMPLETED  '
          '(quantitative claims checked by ch07_experiment_checks.py)')


if __name__ == '__main__':
    main()
