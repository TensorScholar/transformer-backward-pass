"""Chapter 7 Experiment C: Pre-RMSNorm vs no-normalization fork.

Primary comparison holds architecture size, init_std, AdamW settings, data generator,
training budget, and seeds fixed. A secondary learning-rate sensitivity for the no-norm
variant demonstrates that matched hyperparameters do not imply equally tuned models.

No claim of universal RMSNorm superiority is licensed by this experiment.
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

SEEDS=tuple(range(5)); STEPS=60; INIT=0.08
CONDS=(
    ('pre_rmsnorm', True, 0.002),
    ('no_norm_matched_lr', False, 0.002),
    ('no_norm_lr_0.006', False, 0.006),
    ('no_norm_lr_0.010', False, 0.010),
)

def mean_sd(xs):
    x=torch.tensor(xs,dtype=torch.float64)
    return x.mean().item(),x.std(unbiased=True).item()

def main():
    rows=[]
    for seed in SEEDS:
        for name,use_norm,lr in CONDS:
            recs=train_run(RunConfig(seed=seed,init_std=INIT,steps=STEPS,use_norm=use_norm,lr=lr))
            for r in recs: rows.append({'condition':name,'use_norm':use_norm,'lr':lr,'seed':seed,**r})
    out=Path(__file__).with_name('ch07_normalization_fork_results.csv')
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    with out.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

    print('Experiment C — normalization fork')
    print('init_std=0.08, 4 layers, 5 paired seeds, 60 steps')
    print(f"{'condition':>22} {'step':>5} {'loss':>18} {'grad_norm':>18} {'resid_L3':>18} {'logit_rms':>18}")
    for name,_,_ in CONDS:
        for step in [0,5,20,STEPS-1]:
            sub=[r for r in rows if r['condition']==name and r['step']==step]
            vals=[mean_sd([float(r[k]) for r in sub]) for k in ['loss','grad_norm','L3.resid_after_ffn_rms','logit_rms']]
            print(f"{name:>22} {step:5d} "+' '.join(f'{m:.5f}±{sd:.5f}' for m,sd in vals))
    print(f'\nWrote {out}')
    print('CH07_NORMALIZATION_FORK_COMPLETED  '
          '(quantitative claims checked by ch07_experiment_checks.py)')

if __name__=='__main__': main()
