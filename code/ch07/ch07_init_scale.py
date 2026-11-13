"""Chapter 7 Experiment B: initialization-scale sweep.

Question
--------
How does the standard deviation used to initialize the canonical tiny decoder's
linear/embedding weights change activation scale, gradient scale, and early optimization?

Pre-stated predictions
----------------------
1. Larger init_std will increase early residual-stream RMS and initial logit/gradient scale.
2. Very small init_std will begin with small residual activations and may make small relative
   representational changes early, but AdamW can alter the simple scale story after updates.
3. We do not pre-state a monotonic ranking of final loss; that is measured.

Scope
-----
This is a tiny 4-layer copy-4 synthetic task on CPU. It tests local training dynamics in a
controlled model family; it does not identify a universally optimal Transformer initialization.
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
import torch
from ch07_training_harness import RunConfig, train_run

SCALES=(0.005,0.02,0.08,0.30)
SEEDS=tuple(range(5))
STEPS=60


def mean_sd(xs):
    x=torch.tensor(xs,dtype=torch.float64)
    return x.mean().item(), x.std(unbiased=True).item()


def main():
    rows=[]
    for seed in SEEDS:
        for scale in SCALES:
            recs=train_run(RunConfig(seed=seed,init_std=scale,steps=STEPS))
            for r in recs:
                rows.append({'seed':seed,'init_std':scale,**r})
    out=Path(__file__).with_name('ch07_init_scale_results.csv')
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    with out.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

    print('Experiment B — initialization-scale sweep')
    print('4-layer Pre-RMSNorm decoder; copy-4 synthetic task; AdamW; 5 paired seeds')
    print(f"{'init_std':>9} {'step':>5} {'loss':>18} {'grad_norm':>18} {'resid_L3':>18} {'update/weight':>18}")
    for scale in SCALES:
        for step in [0,5,20,STEPS-1]:
            subset=[r for r in rows if r['init_std']==scale and r['step']==step]
            vals=[]
            for key in ['loss','grad_norm','L3.resid_after_ffn_rms','update_to_weight']:
                vals.append(mean_sd([float(r[key]) for r in subset]))
            print(f"{scale:9.3f} {step:5d} " + ' '.join(f"{m:.5f}±{sd:.5f}" for m,sd in vals))
    print(f'\nWrote {out}')
    print('CH07_INIT_SCALE_COMPLETED  '
          '(quantitative claims checked by ch07_experiment_checks.py)')


if __name__=='__main__':
    main()
