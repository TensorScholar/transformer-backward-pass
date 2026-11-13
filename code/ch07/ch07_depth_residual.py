"""Chapter 7 depth experiment v1.2: paired fixed-prefix gradient traces.

Purpose
-------
Measure what happens to adjoints at *corresponding activations* as depth grows.
Unlike the old experiment, total parameter-gradient norm is not used as evidence
of depth-wise gradient propagation.

Pairing
-------
For each seed we initialize one 12-block master model. Shallower models copy:
- the same embedding,
- the same first D blocks,
- the same final RMSNorm,
- the same output head.
Thus common prefixes and the head are genuinely paired across depths.
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
import copy, csv, statistics
import torch
import torch.nn.functional as F
from ch07_training_harness import TinyLM, make_copy4_batch, total_grad_norm, rms

SEEDS = list(range(8))
DEPTHS = [1, 2, 4, 8, 12]
INIT_STD = 0.02
BATCH = 16; T = 24; VOCAB = 32; D = 48; H = 4; DFF = 96


def make_paired_from_master(master, depth):
    m = TinyLM(VOCAB, D, H, DFF, depth, INIT_STD, True, True, True)
    m.emb = copy.deepcopy(master.emb)
    m.blocks = torch.nn.ModuleList([copy.deepcopy(master.blocks[i]) for i in range(depth)])
    m.final_norm = copy.deepcopy(master.final_norm)
    m.out = copy.deepcopy(master.out)
    return m


def state_map(states):
    return {name: tensor for name, tensor in states}


def grad_rms(t):
    return rms(t.grad) if t.grad is not None else float('nan')


rows = []
for seed in SEEDS:
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(10000 + seed)
    master = TinyLM(VOCAB, D, H, DFF, max(DEPTHS), INIT_STD, True, True, True)
    x, y = make_copy4_batch(BATCH, T, VOCAB, gen)

    for depth in DEPTHS:
        model = make_paired_from_master(master, depth)
        z, diag, states = model(x, return_diag=True, retain_states=True)
        loss = F.cross_entropy(z.reshape(-1, VOCAB), y.reshape(-1))
        loss.backward()
        sm = state_map(states)

        row = {
            'depth': depth,
            'seed': seed,
            'loss': loss.item(),
            'global_param_grad_norm': total_grad_norm(model),  # context only
            'embedding_output_grad_rms': grad_rms(sm['embedding_output']),
            'block0_after_attn_grad_rms': grad_rms(sm['L0.after_attn']),
            'block0_after_ffn_grad_rms': grad_rms(sm['L0.after_ffn']),
            'final_hidden_grad_rms': grad_rms(sm['final_hidden']),
            'embedding_rms': diag['embedding_rms'],
            'last_resid_rms': diag[f'L{depth-1}.resid_after_ffn_rms'],
        }
        # Per-block residual-output adjoints allow shape/profile inspection.
        for i in range(depth):
            row[f'L{i}.after_ffn_grad_rms'] = grad_rms(sm[f'L{i}.after_ffn'])
        rows.append(row)

out = Path(__file__).with_name('ch07_depth_residual_results_v1.2.csv')
fields = []
for r in rows:
    for k in r:
        if k not in fields:
            fields.append(k)
with out.open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader(); w.writerows(rows)

print('Experiment B2 — paired depth / fixed-prefix gradient traces')
print('Same seed => same embedding, common prefix blocks, final norm, output head, and batch.')
print(f"{'depth':>5} {'dL/dX0 rms':>18} {'dL/dL0out rms':>18} {'global param grad':>18} {'last resid rms':>18}")
for d in DEPTHS:
    rr = [r for r in rows if r['depth'] == d]
    def ms(k):
        vals = [r[k] for r in rr]
        return statistics.mean(vals), statistics.stdev(vals)
    a,b = ms('embedding_output_grad_rms')
    c,e = ms('block0_after_ffn_grad_rms')
    g,h = ms('global_param_grad_norm')
    q,z = ms('last_resid_rms')
    print(f'{d:5d} {a:.6e}±{b:.2e} {c:.6e}±{e:.2e} {g:.6f}±{h:.6f} {q:.6f}±{z:.6f}')
print(out)
print('CH07_DEPTH_RESIDUAL_COMPLETED  '
      '(quantitative claims checked by ch07_experiment_checks.py)')
