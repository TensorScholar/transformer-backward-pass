"""Chapter 7 Experiment D: clipping as symptom control versus chronic intervention."""
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

SEEDS=tuple(range(5)); STEPS=60; THRESH=2.5
CONDS=(
 ('moderate_no_clip',0.08,None),
 ('moderate_clip',0.08,THRESH),
 ('extreme_no_clip',0.30,None),
 ('extreme_clip',0.30,THRESH),
)

def mean_sd(xs):
    x=torch.tensor(xs,dtype=torch.float64); return x.mean().item(),x.std(unbiased=True).item()

def main():
    rows=[]
    for seed in SEEDS:
        for name,init,clip in CONDS:
            recs=train_run(RunConfig(seed=seed,init_std=init,steps=STEPS,clip_norm=clip))
            for r in recs: rows.append({'condition':name,'seed':seed,'init_std':init,'clip_norm':clip or '',**r})
    out=Path(__file__).with_name('ch07_clipping_results.csv')
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields:fields.append(k)
    with out.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    print('Experiment D — gradient clipping')
    print(f'global norm threshold={THRESH}, 5 paired seeds, 60 steps')
    print(f"{'condition':>20} {'step':>5} {'loss':>18} {'pre_grad':>18} {'post_grad':>18} {'update/weight':>18}")
    for name,_,_ in CONDS:
        for step in [0,5,20,STEPS-1]:
            sub=[r for r in rows if r['condition']==name and r['step']==step]
            vals=[mean_sd([float(r[k]) for r in sub]) for k in ['loss','grad_norm','grad_norm_postclip','update_to_weight']]
            print(f"{name:>20} {step:5d} "+' '.join(f'{m:.5f}±{sd:.5f}' for m,sd in vals))
    print(f'\nWrote {out}')
    print('CH07_CLIPPING_COMPLETED  '
          '(quantitative claims checked by ch07_experiment_checks.py)')

if __name__=='__main__':main()
