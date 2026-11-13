"""Shared tiny decoder training harness for Chapter 7 v0.5.

Canonical graph:
- decoder-only causal Transformer
- Pre-RMSNorm blocks (unless an explicit fork disables block norms)
- per-head RoPE on Q/K
- MHA
- SwiGLU
- final RMSNorm (unless an explicit fork disables it)
- untied output projection

Instrumentation records:
- pre-update train loss
- post-update held-out loss on a fixed validation batch
- global and parameter-group gradient norms
- global and parameter-group update norms / update-to-weight ratios
- clipping state
- residual/attention diagnostics

Timing convention:
- activation and gradient diagnostics are measured before the optimizer step;
- parameter updates and validation loss are measured after the optimizer step.
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
import math
import random
from dataclasses import dataclass
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F

DTYPE = torch.float32
DEVICE = "cpu"


def rms(x):
    return x.detach().float().pow(2).mean().sqrt().item()


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        r = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return x / r * self.weight


def rope(x):
    # x: B,H,T,D; D even
    B, H, T, D = x.shape
    idx = torch.arange(0, D, 2, device=x.device, dtype=x.dtype)
    inv = 10000.0 ** (-idx / D)
    pos = torch.arange(T, device=x.device, dtype=x.dtype)[:, None]
    ang = pos * inv[None, :]
    c, s = ang.cos()[None, None, :, :], ang.sin()[None, None, :, :]
    even, odd = x[..., 0::2], x[..., 1::2]
    y = torch.empty_like(x)
    y[..., 0::2] = even * c - odd * s
    y[..., 1::2] = even * s + odd * c
    return y


class CausalAttention(nn.Module):
    def __init__(self, d, h, init_std, scaled=True):
        super().__init__()
        assert d % h == 0
        self.h = h
        self.dh = d // h
        self.scaled = scaled
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        for m in [self.q, self.k, self.v, self.o]:
            nn.init.normal_(m.weight, std=init_std)

    def forward(self, x, diag=None, prefix=""):
        B, T, D = x.shape
        H, dh = self.h, self.dh

        def split(z):
            return z.view(B, T, H, dh).transpose(1, 2)

        q = rope(split(self.q(x)))
        k = rope(split(self.k(x)))
        v = split(self.v(x))
        raw = q @ k.transpose(-1, -2)
        s = raw / (math.sqrt(dh) if self.scaled else 1.0)
        mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device, dtype=x.dtype), diagonal=1)
        a = torch.softmax(s + mask, dim=-1)
        o = (a @ v).transpose(1, 2).contiguous().view(B, T, D)
        if diag is not None:
            legal = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))[None, None, :, :].expand(B, H, -1, -1)
            diag[prefix + "score_rms"] = rms(s[legal])
            diag[prefix + "attn_max"] = torch.stack([a[:, :, i, : i + 1].amax(dim=-1).mean() for i in range(1, T)]).mean().item()
        return self.o(o)


class Block(nn.Module):
    def __init__(self, d, h, dff, init_std, scaled=True, use_norm=True):
        super().__init__()
        self.use_norm = use_norm
        self.n1 = RMSNorm(d) if use_norm else nn.Identity()
        self.attn = CausalAttention(d, h, init_std, scaled)
        self.n2 = RMSNorm(d) if use_norm else nn.Identity()
        self.wg = nn.Linear(d, dff, bias=False)
        self.wu = nn.Linear(d, dff, bias=False)
        self.wd = nn.Linear(dff, d, bias=False)
        for m in [self.wg, self.wu, self.wd]:
            nn.init.normal_(m.weight, std=init_std)

    def forward(self, x, diag=None, i=0, retain_state=False, state_list=None):
        n1 = self.n1(x)
        x = x + self.attn(n1, diag, prefix=f"L{i}.")
        if retain_state:
            x.retain_grad()
            if state_list is not None:
                state_list.append((f"L{i}.after_attn", x))
        if diag is not None:
            diag[f"L{i}.resid_after_attn_rms"] = rms(x)
        n2 = self.n2(x)
        f = F.silu(self.wg(n2)) * self.wu(n2)
        x = x + self.wd(f)
        if retain_state:
            x.retain_grad()
            if state_list is not None:
                state_list.append((f"L{i}.after_ffn", x))
        if diag is not None:
            diag[f"L{i}.resid_after_ffn_rms"] = rms(x)
        return x


class TinyLM(nn.Module):
    def __init__(self, vocab=32, d=48, h=4, dff=96, layers=4, init_std=0.02, scaled=True, use_norm=True, use_final_norm=None):
        super().__init__()
        self.vocab = vocab
        self.d = d
        self.h = h
        self.layers = layers
        self.use_block_norm = use_norm
        self.use_final_norm = use_norm if use_final_norm is None else use_final_norm
        self.emb = nn.Embedding(vocab, d)
        nn.init.normal_(self.emb.weight, std=init_std)
        self.blocks = nn.ModuleList([Block(d, h, dff, init_std, scaled, use_norm) for _ in range(layers)])
        self.final_norm = RMSNorm(d) if self.use_final_norm else nn.Identity()
        self.out = nn.Linear(d, vocab, bias=False)
        nn.init.normal_(self.out.weight, std=init_std)

    def forward(self, tok, return_diag=False, retain_states=False):
        diag = {} if return_diag else None
        states = [] if retain_states else None
        x = self.emb(tok)
        if retain_states:
            x.retain_grad()
            states.append(("embedding_output", x))
        if diag is not None:
            diag["embedding_rms"] = rms(x)
        for i, b in enumerate(self.blocks):
            x = b(x, diag, i, retain_state=retain_states, state_list=states)
        x = self.final_norm(x)
        if retain_states:
            x.retain_grad()
            states.append(("final_hidden", x))
        z = self.out(x)
        if diag is not None:
            diag["final_hidden_rms"] = rms(x)
            diag["logit_rms"] = rms(z)
        if return_diag and retain_states:
            return z, diag, states
        if return_diag:
            return z, diag
        if retain_states:
            return z, states
        return z


def make_copy4_batch(batch, T, vocab, generator):
    motif = torch.randint(0, vocab, (batch, 4), generator=generator)
    seq = motif.repeat(1, (T + 4) // 4)[:, : T + 1]
    return seq[:, :T], seq[:, 1 : T + 1]


def _norm_of_tensors(ts):
    return math.sqrt(sum(t.detach().float().pow(2).sum().item() for t in ts)) if ts else 0.0


def total_grad_norm(model):
    return _norm_of_tensors([p.grad for p in model.parameters() if p.grad is not None])


def active_weight_norm(model):
    return _norm_of_tensors([p for p in model.parameters() if p.requires_grad])


def named_parameter_groups(model):
    groups = defaultdict(list)
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("emb."):
            g = "embedding"
        elif name.startswith("out."):
            g = "output"
        elif "attn." in name:
            g = "attention"
        elif any(k in name for k in [".wg.", ".wu.", ".wd."]):
            g = "ffn"
        elif "n1." in name or "n2." in name or name.startswith("final_norm."):
            g = "norm"
        else:
            g = "other"
        groups[g].append((name, p))
    return groups


def group_norms(model, kind="weight"):
    out = {}
    for group, items in named_parameter_groups(model).items():
        if kind == "weight":
            ts = [p for _, p in items]
        elif kind == "grad":
            ts = [p.grad for _, p in items if p.grad is not None]
        else:
            raise ValueError(kind)
        out[group] = _norm_of_tensors(ts)
    return out


@dataclass
class RunConfig:
    seed: int = 0
    init_std: float = 0.02
    steps: int = 80
    lr: float = 2e-3
    batch: int = 16
    T: int = 24
    vocab: int = 32
    d: int = 48
    h: int = 4
    dff: int = 96
    layers: int = 4
    scaled: bool = True
    use_norm: bool = True
    use_final_norm: bool | None = None
    clip_norm: float | None = None
    record_all_steps: bool = True


def train_run(cfg: RunConfig):
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    data_gen = torch.Generator().manual_seed(10000 + cfg.seed)
    val_gen = torch.Generator().manual_seed(20000 + cfg.seed)
    model = TinyLM(cfg.vocab, cfg.d, cfg.h, cfg.dff, cfg.layers, cfg.init_std, cfg.scaled, cfg.use_norm, cfg.use_final_norm)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.95), weight_decay=0.01)

    # Fixed held-out motifs for the entire run.
    x_val, y_val = make_copy4_batch(max(64, cfg.batch), cfg.T, cfg.vocab, val_gen)

    records = []
    clip_count = 0
    checkpoints = {0, 1, 5, 20, cfg.steps - 1}

    for step in range(cfg.steps):
        x, y = make_copy4_batch(cfg.batch, cfg.T, cfg.vocab, data_gen)
        full = step in checkpoints

        # Cloning every parameter every step is unnecessary. We measure actual
        # optimizer movement at the declared checkpoints, while keeping
        # pre-clipping gradient traces for every step.
        if full:
            named_before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
            weight_groups_before = group_norms(model, "weight")
            weight_norm_before = active_weight_norm(model)
        else:
            named_before = None
            weight_groups_before = {}
            weight_norm_before = float("nan")

        opt.zero_grad(set_to_none=True)
        z, diag = model(x, return_diag=True)
        train_loss = F.cross_entropy(z.reshape(-1, cfg.vocab), y.reshape(-1))
        train_loss.backward()

        grad_groups_pre = group_norms(model, "grad")
        gn = total_grad_norm(model)
        clipped = bool(cfg.clip_norm is not None and gn > cfg.clip_norm)
        clip_count += int(clipped)
        if cfg.clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_norm)
        grad_groups_post = group_norms(model, "grad")
        gn_post = total_grad_norm(model)

        opt.step()

        update_norm = float("nan")
        update_to_weight = float("nan")
        group_delta_sq = {}
        val_loss = float("nan")

        if full:
            group_delta_sq = defaultdict(float)
            for group, items in named_parameter_groups(model).items():
                group_delta_sq[group] = sum(
                    (p.detach() - named_before[n]).float().pow(2).sum().item()
                    for n, p in items
                )
            update_norm = math.sqrt(sum(group_delta_sq.values()))
            update_to_weight = update_norm / (weight_norm_before + 1e-12)

            model.eval()
            with torch.no_grad():
                z_val = model(x_val)
                val_loss = F.cross_entropy(z_val.reshape(-1, cfg.vocab), y_val.reshape(-1)).item()
            model.train()

        rec = dict(
            step=step,
            loss=train_loss.item(),
            train_loss_preupdate=train_loss.item(),
            val_loss_postupdate=val_loss,
            grad_norm=gn,
            grad_norm_postclip=gn_post,
            clipped=float(clipped),
            clip_fraction=clip_count / (step + 1),
            weight_norm=weight_norm_before,
            update_norm=update_norm,
            update_to_weight=update_to_weight,
        )
        rec.update(diag)

        for g, v in grad_groups_pre.items():
            rec[f"grad_norm_pre__{g}"] = v
        for g, v in grad_groups_post.items():
            rec[f"grad_norm_post__{g}"] = v

        if full:
            for g, v in weight_groups_before.items():
                rec[f"weight_norm__{g}"] = v
            for g, sq in group_delta_sq.items():
                up = math.sqrt(sq)
                rec[f"update_norm__{g}"] = up
                rec[f"update_to_weight__{g}"] = up / (weight_groups_before.get(g, 0.0) + 1e-12)

        # Basic gradient/activation traces are retained for every step; the
        # expensive update/validation measurements are checkpointed.
        if cfg.record_all_steps or full or clipped:
            records.append(rec)

    return records

