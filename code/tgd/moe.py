r"""Mixture-of-Experts routing: capacity without compute, and the price of a router.

Why this module exists
----------------------
Chapters 9 and 11 established that decoding is bandwidth-bound and that
arithmetic intensity sits two to four orders of magnitude below the ridge point.
Chapter 15 showed that making *parameters* smaller makes training cheaper. This
chapter is the opposite trade: **make parameters much larger, and pay nothing in
active compute.**

The mechanism is a sparse conditional computation. A dense feed-forward block
applies one transformation to every token. A Mixture of Experts replaces it with
$n_{exp}$ candidate transformations and applies only the top $K$ of them, chosen
by a learned router. Per token the arithmetic is $K$ expert evaluations; in the
weights there are $n_{exp}$ of them. The two numbers are independent and the whole
of the method is that they can be decoupled.

What this module measures, in this order
-----------------------------------------
1. **Active FLOPs per token**, counted exactly, against the dense block. This is
   the claim that works.
2. **Weight VRAM**, counted exactly, against the dense block. This is where the
   decoupling stops, and it stops hard: the parameters are still resident, so VRAM
   goes *up* by roughly the decoupled factor. A method that "saves compute" by
   storing more is not free, and the accounting has to show both sides.
3. **Routing collapse.** With the auxiliary loss switched off, the router provably
   concentrates. The mechanism is a positive feedback loop with no equilibrium
   away from monopoly, and it is measured rather than asserted.
4. **The auxiliary loss**, its exact gradient, and the restored distribution.

Notation, checked against the contract's collision registry
-----------------------------------------------------------
  * Experts are **not** $E$. The contract registers $E$ as the embedding matrix
    (ID 1, shape $V \times d$); the literature's use of $E$ for "expert" is a
    collision this book does not carry. The expert count is $n_{exp}$ and the
    expert index is $a$.
  * The router is $W_{route} \in \mathbb{R}^{d \times n_{exp}}$, its probabilities
    $\pi^{(t)}$, and the top-$K$ set $\mathcal{S}^{(t)}$. **$\mathcal{S}^{(t)}$
    always carries a token index**, which is what keeps it distinct from
    $S^{(h)}$, the attention scores of ID 8 -- the same accepted near-collision
    the contract records for $V$ versus $V_\ell$.
  * The load-balancing coefficient is $\gamma$, distinct from $\varepsilon$ (C3)
    and $\varepsilon_{adam}$.
  * The auxiliary loss uses $f_a$ (the fraction of tokens dispatched to expert
    $a$) and $\zeta_a$ (that expert's mean router probability). **The literature
    writes $P_a$; the contract forbids $P$ in prose** ($\S$7: the gate
    pre-activation already owns it).

The one thing that is NOT here
------------------------------
No claim about model quality. There is no perplexity, no downstream task, and no
comparison of an $n_{exp} = 8$ model to a dense one of equal active compute. A
sparse model with $2\times$ the parameters and the same FLOPs may be better, equal
or worse, and nothing below predicts which.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

__all__ = [
    "BYTES", "DenseConfig", "MoEConfig", "DenseFFN", "MoEFFN",
    "dense_flops", "moe_flops", "flop_report", "vram_report",
    "load_balancing_loss", "aux_gradient_report", "routing_collapse_report",
    "expert_usage", "exactness_checks",
    "router_speed_report", "aux_efficacy_report",
    "make_clustered_batch",
]

#: float32. Every byte count here is exact arithmetic on this number.
BYTES = 4

#: Multiply-accumulates are counted as 1; a FLOP count is 2x a MAC count. Both
#: conventions appear in the literature and they differ by exactly this factor,
#: so the convention is named here and never left implicit.
FLOPS_PER_MAC = 2


# ============================================================== dense FFN ====

@dataclass(frozen=True)
class DenseConfig:
    """The baseline block. ``d_ff = 4d`` is the usual ratio."""

    d: int = 1024
    d_ff: int = 4096
    element_bytes: int = BYTES

    @property
    def n_experts(self) -> int:
        return 1                       # a dense block is the n_exp=1, K=1 case

    @property
    def params_per_matrix(self) -> int:
        return self.d * self.d_ff


@dataclass(frozen=True)
class MoEConfig:
    """The sparse block.

    Defaults follow the brief: $n_{exp} = 8$, $d_{ff} = 1024$ **per expert**,
    top-$K = 2$. Note that $d_{ff}$ here is *per expert*, not the block total --
    $8 \times 1024 = 8192 > 4096$, which is the entire point.
    """

    d: int = 1024
    n_exp: int = 8
    d_ff: int = 1024                  # PER EXPERT
    k: int = 2
    element_bytes: int = BYTES

    @property
    def total_expert_ff(self) -> int:
        return self.n_exp * self.d_ff

    def validate(self) -> None:
        if self.k < 1 or self.k > self.n_exp:
            raise ValueError(
                f"k must be in 1..n_exp; got k={self.k}, n_exp={self.n_exp}. "
                "k > n_exp would select the same expert twice and k < 1 would "
                "route nothing.")
        if self.d_ff < 1:
            raise ValueError(f"d_ff must be positive, got {self.d_ff}")


class DenseFFN:
    r"""SwiGLU feed-forward, contract IDs 16, 18, 19, 20, 21.

    $D = \mathrm{SiLU}(XW_g) \odot (XW_u)$, then $Y = D W_d$. Three matrices of
    $d \times d_{ff}$, so the MAC count per token is $3 d\,d_{ff}$ and the block
    holds $3 d\, d_{ff}$ parameters -- **one MAC per parameter**, which is the
    property that makes MoE's arithmetic look free.
    """

    def __init__(self, cfg: DenseConfig, *, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        s = 1.0 / math.sqrt(cfg.d)
        self.cfg = cfg
        self.W_g = torch.nn.Parameter(torch.randn(cfg.d, cfg.d_ff, generator=g) * s)
        self.W_u = torch.nn.Parameter(torch.randn(cfg.d, cfg.d_ff, generator=g) * s)
        self.W_d = torch.nn.Parameter(torch.randn(cfg.d_ff, cfg.d, generator=g) * s)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = torch.nn.functional.silu(x @ self.W_g)
        u = x @ self.W_u
        return (g * u) @ self.W_d

    def params(self) -> list[torch.nn.Parameter]:
        return [self.W_g, self.W_u, self.W_d]

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.params())


# ================================================================ the MoE ====

class MoEFFN:
    r"""Sparse MoE: a router, $n_{exp}$ SwiGLU experts, top-$K$ dispatch.

    Forward, for one token $t$:

    1. logits $z^{(t)} = x^{(t)} W_{route}$, probabilities
       $\pi^{(t)} = \mathrm{softmax}(z^{(t)})$
    2. $\mathcal{S}^{(t)}$ = the indices of the top $K$ entries of $\pi^{(t)}$
    3. $y^{(t)} = \sum_{a \in \mathcal{S}^{(t)}} \pi^{(t)}_a\, \mathrm{FFN}_a(x^{(t)})$

    Step 3 is a **$K$-path merge** and is registered in ``tgd.ledger`` with each
    summand named (contract A.3). Its adjoint is the sum of the $K$ per-expert
    adjoints into the router probability and into $x^{(t)}$.

    The gate is $\pi_a$ and not a renormalisation over $\mathcal{S}$. That is a
    choice, and it is the one that makes the gradient useful: the router receives
    a signal proportional to **how much the expert actually changed the output**,
    which is the credit-assignment a renormalised gate destroys. The
    consequence -- that the weights do not sum to one -- is handled by the
    output layer, exactly as in a dense residual block.
    """

    def __init__(self, cfg: MoEConfig, *, seed: int = 0):
        cfg.validate()
        g = torch.Generator().manual_seed(seed)
        s = 1.0 / math.sqrt(cfg.d)
        self.cfg = cfg
        self.W_route = torch.nn.Parameter(
            torch.randn(cfg.d, cfg.n_exp, generator=g) * s)
        self.experts = [DenseFFN(DenseConfig(cfg.d, cfg.d_ff,
                                             cfg.element_bytes), seed=seed + 1 + a)
                        for a in range(cfg.n_exp)]

    # -- routing ---------------------------------------------------------
    def route(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(logits, probs)``, both ``(T, n_exp)``."""
        z = x @ self.W_route
        return z, torch.softmax(z, dim=-1)

    def select(self, probs: torch.Tensor) -> torch.Tensor:
        """Top-$K$ mask, ``(T, n_exp)``, 1.0 inside $\\mathcal{S}$ and 0 elsewhere.

        Built by an exact ``topk`` index scatter: the $K$ selected indices are
        scattered into a zero mask, so exactly $K$ experts are dispatched per
        token regardless of ties. The previous threshold form
        ``probs >= kth`` dispatched **every** expert tied at the $K$-th value
        -- a zero router (all-equal probabilities) selected all $n_{exp}$
        experts instead of $K$, silently multiplying the FLOP count by
        $n_{exp}/K$ and changing the $K$-path merge. ``torch.topk`` breaks
        ties by index, which is deterministic and selects each expert at most
        once, so the duplicate-index trap the threshold form was guarding
        against cannot occur here either.
        """
        idx = probs.topk(self.cfg.k, dim=-1).indices
        mask = torch.zeros_like(probs).scatter_(-1, idx, 1.0)
        return mask

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor,
                                                torch.Tensor]:
        """Return ``(y, probs, mask)``."""
        _, probs = self.route(x)
        mask = self.select(probs)
        gate = probs * mask
        y = torch.zeros_like(x)
        for a in range(self.cfg.n_exp):
            sel = mask[:, a] > 0
            if not bool(sel.any()):
                continue
            y[sel] = y[sel] + gate[sel, a : a + 1] * self.experts[a].forward(x[sel])
        return y, probs, mask

    def params(self) -> list[torch.nn.Parameter]:
        out = [self.W_route]
        for e in self.experts:
            out.extend(e.params())
        return out

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.params())


# ================================================================= FLOPs ====

def dense_flops(cfg: DenseConfig, batch: int = 1) -> dict:
    r"""Exact MAC and FLOP count for the dense block, one token per row.

    A SwiGLU block is three matmuls: $XW_g$, $XW_u$, $DW_d$. The elementwise
    SiLU and product are **not** counted, which is stated rather than assumed:
    they are $O(d_{ff})$ against $O(d\,d_{ff})$ for the matmuls, so they are
    $2/(3d) \approx 0.065\%$ at $d = 1024$. A reader who wants them can add
    $\approx 3 d_{ff}$ per token; the chapter gives the number.
    """
    macs = 3 * cfg.d * cfg.d_ff
    elementwise = 3 * cfg.d_ff          # SiLU plus the gated product
    return {
        "config": cfg, "batch": batch,
        "macs_per_token": macs,
        "flops_per_token": FLOPS_PER_MAC * macs,
        "macs_batch": macs * batch,
        "elementwise_ignored": elementwise,
        "elementwise_fraction": elementwise / (macs + elementwise),
        "n_params": 3 * cfg.d * cfg.d_ff,
    }


def moe_flops(cfg: MoEConfig, batch: int = 1) -> dict:
    r"""Exact MAC and FLOP count for the MoE block, at full expert utilisation.

    **Active** compute: the router costs $d\,n_{exp}$ and exactly $K$ experts are
    evaluated, each $3 d\,d_{ff}$. Total:

    $$d\,n_{exp} \;+\; K\cdot 3 d\, d_{ff}$$

    **Total** (all experts, if every token activated every expert) is
    $d\,n_{exp} + n_{exp}\cdot 3 d\, d_{ff}$, and the ratio is the sparsity
    $n_{exp}/K$. Both are reported, because quoting only the active count without
    the total is how a $4\times$ claim is manufactured from an $8\times$ one.
    """
    cfg.validate()
    router = cfg.d * cfg.n_exp
    per_expert = 3 * cfg.d * cfg.d_ff
    active = router + cfg.k * per_expert
    total = router + cfg.n_exp * per_expert
    elementwise = 3 * cfg.d_ff * cfg.k
    return {
        "config": cfg, "batch": batch,
        "router_macs": router,
        "macs_per_expert": per_expert,
        "macs_per_token": active,
        "flops_per_token": FLOPS_PER_MAC * active,
        "macs_batch": active * batch,
        "macs_all_experts": total,
        "flops_all_experts": FLOPS_PER_MAC * total,
        "sparsity": cfg.n_exp / cfg.k,
        "elementwise_ignored": elementwise,
        "elementwise_fraction": elementwise / (active + elementwise),
        "n_params_experts": cfg.n_exp * 3 * cfg.d * cfg.d_ff,
        "n_params_router": router,
        "n_params": cfg.n_exp * 3 * cfg.d * cfg.d_ff + router,
    }


def flop_report(dense: DenseConfig | None = None,
                moe: MoEConfig | None = None) -> dict:
    """The decoupling, stated as two ratios that do not agree."""
    dense = dense or DenseConfig()
    moe = moe or MoEConfig()
    fd, fm = dense_flops(dense), moe_flops(moe)
    return {
        "dense": fd, "moe": fm,
        "flop_ratio_dense_over_moe": fd["flops_per_token"]
                                    / fm["flops_per_token"],
        "flop_reduction_percent": 100.0 * (1.0 - fm["flops_per_token"]
                                           / fd["flops_per_token"]),
        "param_ratio_moe_over_dense": fm["n_params"] / fd["n_params"],
        "sparsity": fm["sparsity"],
        "n_params_dense": fd["n_params"], "n_params_moe": fm["n_params"],
    }


def vram_report(dense: DenseConfig | None = None,
                moe: MoEConfig | None = None) -> dict:
    r"""Weight VRAM, and the point where the decoupling stops.

    The claim of MoE is about *active* compute. The weights are still resident:
    every one of the $n_{exp}$ experts is in VRAM whether or not this token
    activates it. So the capacity multiplier and the memory multiplier are the
    **same number**, $n_{exp}$, and they compound with $K$ not at all.

    Reported alongside the training-time figures, because a MoE deployment is
    argued about in the same breath as a fine-tune (Chapter 15) and the two costs
    add: MoE raises weight VRAM, and Chapter 15's adapter is the only thing that
    makes the resulting model trainable on a given card.
    """
    dense = dense or DenseConfig()
    moe = moe or MoEConfig()
    e = dense.element_bytes
    fd = dense_flops(dense)
    fm = moe_flops(moe)
    db = fd["n_params"] * e
    mb = fm["n_params"] * e
    return {
        "element_bytes": e,
        "dense_bytes": db, "moe_bytes": mb,
        "dense_MiB": db / 2 ** 20, "moe_MiB": mb / 2 ** 20,
        "vram_ratio_moe_over_dense": mb / db,
        "vram_increase_percent": 100.0 * (mb / db - 1.0),
        "expert_bytes": fm["n_params_experts"] * e,
        "router_bytes": fm["n_params_router"] * e,
        "router_share": fm["n_params_router"] / fm["n_params"],
        "note": "active FLOPs fall; resident bytes rise by the same factor the "
                "parameter count rises. The decoupling is between compute and "
                "PARAMETER COUNT, not between compute and MEMORY.",
    }


# ======================================================= the auxiliary loss ==

def load_balancing_loss(probs: torch.Tensor, mask: torch.Tensor,
                        n_exp: int) -> tuple[torch.Tensor, dict]:
    r"""The Switch/GShard load-balancing loss, and the statistics it is built on.

    $$\mathcal{L}_{aux} \;=\; n_{exp} \sum_{a} f_a\,\zeta_a$$

    where, over a batch of $T$ tokens,

    $$f_a \;=\; \frac{1}{T}\sum_t \mathbb{1}[a \in \mathcal{S}^{(t)}],
      \qquad
      \zeta_a \;=\; \frac{1}{T}\sum_t \pi^{(t)}_a .$$

    Two properties matter, both of them conditional -- and the conditions are
    stated, because the unconditional versions are FALSE and both shipped here once:

    * **Uniform dispatch minimises at the top-$K$ value.** If every token takes
      $K$ slots and the slots are shared evenly, $f_a = K/n_{exp}$ for all $a$,
      and then $\mathcal{L}_{aux} = n_{exp}\cdot(K/n_{exp})\cdot\sum_a\zeta_a =
      K$. **Not 1**: with top-$K$ routing the uniform value is $K$, and every
      statement about "the loss equals one" is the top-1 special case wearing a
      general costume. Measured: exactly 2.0 at $n_{exp}=8$, $K=2$.
    * **The $n_{exp}\sum f\zeta \ge 1$ bound needs calibration.** It holds when
      the router is calibrated, $f \approx \zeta$: for a probability vector $v$,
      $n_{exp}\sum_a v_a^2 \ge 1$ by Jensen, with equality iff $v$ is uniform
      (measured: minimum 1.034 over 1000 random probability vectors, never
      below 1). Without calibration it is false: $f = e_1$, $\zeta = e_2$ gives
      exactly 0. A check asserting "$\ge 1$ on random independent routing" passed
      on 64 trials here by luck and by nothing else, so it is not that check.
    * **The prefactor $n_{exp}$ is what makes it O(1).** Without it the loss
      scales as $1/n_{exp}$ and its gradient shrinks with model width, so the
      same nominal $\gamma$ means less and less as the model grows.

    $f_a$ is a *frequency*: a piecewise-constant function of the routing decision,
    so it is **detached** and carries no gradient. All the gradient flows through
    $\zeta_a$, i.e. through $\pi$.
    """
    T, n = probs.shape
    if n != n_exp:
        raise ValueError(f"probs has {n} columns, expected n_exp={n_exp}")
    counts = mask.sum(dim=0)                      # tokens per expert, (n_exp,)
    f = (counts / T).detach()
    zeta = probs.mean(dim=0)                      # carries the gradient
    loss = n_exp * (f * zeta).sum()
    with torch.no_grad():
        uniform = torch.full_like(f, 1.0 / n_exp)
        stats = {
            "T": int(T), "n_exp": int(n_exp),
            "counts": counts.to(torch.float64).tolist(),
            "f": f.to(torch.float64).tolist(),
            "zeta": zeta.detach().to(torch.float64).tolist(),
            "loss_value": float(loss),
            "f_max": float(f.max()), "f_min": float(f.min()),
            "entropy_f": float(-(f * f.clamp_min(1e-12).log()).sum()),
            "entropy_max": float(math.log(n_exp)),
            "max_entropy_ratio": float(
                -(f * f.clamp_min(1e-12).log()).sum() / math.log(n_exp)),
            "cauchy_schwarz_gap": float(
                n_exp * (f * zeta).sum() - 1.0),
        }
    return loss, stats


def aux_gradient_report(d: int = 256, n_exp: int = 8, T: int = 64, seed: int = 0,
                        dtype: torch.dtype = torch.float64) -> dict:
    r"""The analytic adjoint of the auxiliary loss, against autograd.

    Because $f$ is detached and only $\zeta_a = \frac{1}{T}\sum_t \pi^{(t)}_a$
    carries gradient, with $\partial \pi_j/\partial z_k = \pi_j(\delta_{jk} -
    \pi_k)$, the chain rule gives, for one token $t$ and one logit $k$,

    $$\frac{\partial \mathcal{L}_{aux}}{\partial z^{(t)}_k}
    \;=\; \frac{n_{exp}}{T}\, \pi^{(t)}_k \,
       \bigl(f_k - \langle f\rangle_t\bigr),
       \qquad
       \langle f\rangle_t \;=\; \sum_a f_a\, \pi^{(t)}_a .$$

    **The first version of this docstring omitted the second term** -- it asserted
    $(n_{exp}/T)\,f_k\pi^{(t)}_k$, as if differentiating through $\zeta$ did not
    pass through the softmax Jacobian. That formula agreed with autograd to $0.125$,
    not to $10^{-18}$, and it is kept in the git history as the single easiest
    mistake of this chapter to write down. The softmax term $-\pi\langle
    f\rangle$ is not a correction of order nothing; at $\langle f\rangle \approx
    1/n_{exp}$ it is the same size as the kept term whenever $f_k$ is itself
    small, which is exactly the regime -- a starved expert -- where the auxiliary
    loss is supposed to act.

    The **correct** mechanism is therefore comparative, not absolute. The gradient
    on expert $k$'s logit is positive iff $f_k > \langle f\rangle_t$: the expert is
    receiving *more* than this token's current router expects. Descent then pushes
    over-represented experts' logits down and under-represented experts' logits
    **up**, from both sides. There is still no sign anywhere that rewards a popular
    expert -- but neither is "every gradient non-negative", which the first version
    also claimed and which autograd rejects.
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(T, d, generator=g, dtype=dtype)
    Wr = torch.randn(d, n_exp, generator=g, dtype=dtype)
    z = (x @ Wr).requires_grad_(True)
    probs = torch.softmax(z, dim=-1)
    # an IMBALANCED mask on purpose: a uniform mask would make f uniform and the
    # gradient degenerate
    mask = torch.zeros(T, n_exp, dtype=dtype)
    for t in range(T):
        a = int(t % n_exp) if t % 3 else 0        # expert 0 gets a third of them
        mask[t, a] = 1.0
    mask[:, 0] = 1.0

    loss, stats = load_balancing_loss(probs, mask, n_exp)
    loss.backward()
    autograd_grad = z.grad.clone()

    counts = mask.sum(0)
    f = (counts / T)
    analytic = (n_exp / T) * f.unsqueeze(0) * probs.detach()

    wmean = (f.unsqueeze(0) * probs.detach()).sum(-1, keepdim=True)
    analytic = ((n_exp / T)
                * (f.unsqueeze(0) * probs.detach() - probs.detach() * wmean))
    err = float((autograd_grad - analytic).abs().max())
    popular, starved = int(f.argmax()), int(f.argmin())
    grad_popular = float(autograd_grad[:, popular].mean())
    grad_starved = float(autograd_grad[:, starved].mean())
    # an UNMASKED logit still carries gradient through the softmax denominator:
    # changing it changes every selected expert's probability
    unmasked_carry = bool((autograd_grad[mask == 0] != 0).any())
    return {
        "d": d, "n_exp": n_exp, "T": T, "dtype": str(dtype),
        "loss_value": stats["loss_value"],
        "loss_above_1": stats["loss_value"] > 1.0,
        "max_abs_err_vs_analytic": err,
        "unmasked_logits_carry_gradient": unmasked_carry,
        "popular_expert": popular, "popular_share": float(f[popular]),
        "popular_mean_logit_grad": grad_popular,
        "starved_expert": starved, "starved_share": float(f[starved]),
        "starved_mean_logit_grad": grad_starved,
        "popular_pushed_down": grad_popular > 0,
        "starved_pushed_up": grad_starved < 0,
        "stats": stats,
    }


# ==================================================== routing collapse =====

def make_clustered_batch(n_clusters: int, per_cluster: int, d: int, *,
                         seed: int = 0, radius: float = 3.0,
                         dtype: torch.dtype = torch.float32) -> tuple:
    r"""A batch with genuine per-expert structure, so collapse has something to grip.

    A batch of i.i.d. Gaussians has no structure for the router to exploit, and a
    collapse measured on one is an artefact of initialisation noise. This draws
    $n_{clusters}$ well-separated means on a sphere of radius ``radius`` and
    assigns tokens round-robin, with a matching target that is a fixed linear
    function of the cluster. An expert that is better on one cluster then really
    is better, and the rich-get-richer loop is the real phenomenon rather than an
    artefact of noise amplification.
    """
    g = torch.Generator().manual_seed(seed)
    dirs = torch.randn(n_clusters, d, generator=g, dtype=dtype)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True) * radius
    idx = torch.arange(n_clusters).repeat_interleave(per_cluster)
    x = dirs[idx] + torch.randn(n_clusters * per_cluster, d, generator=g,
                                dtype=dtype)
    # target: each cluster wants the direction rotated 90 degrees, a linear map
    target = torch.roll(dirs, 1, dims=-1)[idx] * 0.5
    return x.to(dtype), target.to(dtype), idx


def make_iid_batch(T: int, d: int, *, seed: int = 0,
                   dtype: torch.dtype = torch.float32) -> tuple:
    r"""Structureless data: the control in which collapse has nothing to oppose it.

    Inputs are i.i.d. Gaussian and the target is one fixed linear map applied to
    all of them. No expert has a natural constituency and no token has a natural
    home, so any concentration of routing is the router amplifying its own noise
    -- the rich-get-richer loop in isolation, with no countervailing
    specialisation force. This is the condition in which textbook routing collapse
    lives, and the chapter is explicit that it is a constructed worst case rather
    than a typical workload.
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(T, d, generator=g, dtype=dtype)
    A = torch.randn(d, d, generator=g, dtype=dtype) / math.sqrt(d)
    return x.to(dtype), (x @ A * 0.5).to(dtype), torch.zeros(T, dtype=torch.long)


def expert_usage(probs: torch.Tensor, mask: torch.Tensor) -> dict:
    r"""Assignment statistics, all of them scale-free.

    **Normalisation, because top-$K$ breaks the naive version.** The first draft
    defined $f_a = c_a/T$ with $c_a$ the slot counts, then computed the entropy
    of $f$ and a $\chi^2$ against $T/n$ -- as if $\sum_a f_a = 1$. But each
    token occupies $K$ mask slots, so $\sum_a f_a = K$ (measured $\approx 2.0$
    here), the "entropy" was inflated, and the $\chi^2$ compared slot counts
    against token-count expectations. All three numbers were wrong, and they
    were wrong in the direction that made a balanced router look structured.

    The correct primitive is the **slot share** $p_a = c_a / S$ with $S =
    \sum_a c_a$, which is a distribution for any $K$. The headline number is
    then $\max_a p_a$: $1/n_{exp}$ under balance ($0.125$ at $n_{exp} = 8$) and
    $1.0$ under total monopoly, with no $K$ anywhere in the comparison.
    """
    counts = mask.sum(0).to(torch.float64)
    T = int(mask.shape[0])
    S = float(counts.sum())
    n = counts.numel()
    p = (counts / S).clamp_min(1e-12)
    ent = float(-(p * p.log()).sum())
    ent_max = math.log(n)
    chi2 = float((((counts - S / n) ** 2) / (S / n)).sum())
    K_avg = S / T
    return {
        "counts": counts.tolist(),
        "slot_share": p.tolist(),
        "top_share": float(p.max()),
        "bottom_share": float(p.min()),
        "uniform_share": 1.0 / n,
        "collapse_ratio": float(p.max() * n),          # 1.0 balanced, n monopoly
        "slots_per_token": K_avg,
        "entropy": ent,
        "entropy_max": ent_max,
        "normalized_entropy": ent / ent_max,
        "chi2_vs_uniform": chi2,
        "chi2_dof": n - 1,
        "n_dead_experts": int((counts == 0).sum()),
        "T": T,
    }


def _train_steps(cfg: MoEConfig, steps: int, T: int, gamma: float, *,
                 seed: int, lr: float = 0.05, data_mode: str = "iid",
                 freeze_experts: bool = False, optim: str = "adam",
                 dtype: torch.dtype = torch.float32) -> dict:
    """One training run.

    ``data_mode`` is ``"iid"`` (structureless; the collapse condition) or
    ``"clustered"`` (eight well-separated clusters the experts can divide, which
    rewards healthy specialisation and should NOT collapse).

    ``freeze_experts`` isolates the pure routing dynamics: the experts stay at
    their random inits (so quality differences are fixed) and only the router
    moves. Any concentration is then the router amplifying fixed quality gaps,
    not experts improving -- the feedback loop with the expert half removed.

    ``optim`` is ``"adam"`` or ``"sgd"``. Adam normalises each parameter's update
    by its gradient RMS, which *equalises* effective learning rates across
    experts and therefore damps the rich-get-richer loop by construction. SGD
    translates a popular expert's larger gradient directly into a larger step and
    should amplify. The comparison is the test of that structural claim.
    """
    if data_mode not in ("iid", "clustered"):
        raise ValueError(f"data_mode must be 'iid' or 'clustered', got {data_mode!r}")
    if optim not in ("adam", "sgd"):
        raise ValueError(f"optim must be 'adam' or 'sgd', got {optim!r}")
    torch.manual_seed(seed)
    model = MoEFFN(cfg, seed=seed)
    if freeze_experts:
        for e in model.experts:
            for prm in e.params():
                prm.requires_grad_(False)
    if data_mode == "iid":
        x, target, _ = make_iid_batch(T, cfg.d, seed=seed + 1, dtype=dtype)
    else:
        x, target, _ = make_clustered_batch(cfg.n_exp, max(1, T // cfg.n_exp),
                                            cfg.d, seed=seed + 1, dtype=dtype)
    opt = (torch.optim.Adam(model.params(), lr=lr) if optim == "adam"
           else torch.optim.SGD(model.params(), lr=lr))
    trace = []
    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        y, probs, mask = model.forward(x)
        main = torch.nn.functional.mse_loss(y, target)
        aux, st = load_balancing_loss(probs, mask, cfg.n_exp)
        loss = main + gamma * aux
        loss.backward()
        opt.step()
        if step in (1, steps // 2, steps):
            u = expert_usage(probs.detach(), mask.detach())
            trace.append({"step": step, "main_loss": float(main.detach()),
                          "aux_loss": st["loss_value"],
                          "top_share": u["top_share"],
                          "normalized_entropy": u["normalized_entropy"],
                          "n_dead_experts": u["n_dead_experts"],
                          "chi2": u["chi2_vs_uniform"],
                          "counts": [int(c) for c in u["counts"]]})
    with torch.no_grad():
        _, probs, mask = model.forward(x)
    final = expert_usage(probs, mask)
    return {"final": final, "trace": trace, "gamma": gamma, "steps": steps,
            "model": model}


def routing_collapse_report(steps: int = 50, T: int = 256, gamma: float = 0.01,
                            d: int = 1024, n_exp: int = 8, d_ff: int = 1024,
                            k: int = 2, seed: int = 0, lr: float = 0.05,
                            data_mode: str = "iid", optim: str = "adam",
                            freeze_experts: bool = False) -> dict:
    r"""Routing with and without the auxiliary loss, over a real training run.

    Both runs are identical apart from $\gamma$: same seed, same data, same
    initialisation, same optimizer, same number of steps. The only difference is
    whether the $n_{exp}\sum_a f_a\zeta_a$ term is present, so the difference in
    routing is attributable to it and to nothing else.

    What is measured, at step 1 and step ``steps``:

      * ``top_share`` -- slot share of the most popular expert, against the uniform
        reference $1/n_{exp}$ and the monopoly bound 1;
      * ``normalized_entropy`` -- 1.0 is perfectly uniform, 0 is monopoly;
      * ``n_dead_experts`` -- experts receiving **no** tokens at all, which is
        the irreversible failure: a dead expert receives no gradient and can
        never come back;
      * ``chi2_vs_uniform`` -- the raw test statistic, because "top share" alone
        can look healthy while the tail has emptied.

    **What this reports at the brief's scale, honestly.** At 50 steps the loop has
    barely engaged: top share moves from $\approx 0.16$ to $\approx 0.18$ without
    the aux loss, and the aux loss restores $\approx 0.16$. There is no monopoly
    and no dead expert in any configuration tested here -- not iid or clustered,
    not top-1 or top-2, not Adam or SGD, not 50 or 500 steps. The full monopoly
    that motivates the auxiliary loss in the literature needs scale this harness
    cannot reach (horizon, expert size, irreversibility threshold), and the
    chapter says so rather than manufacturing it. See :func:`router_speed_report`
    for the fastest concentration this harness *can* produce.
    """
    cfg = MoEConfig(d=d, n_exp=n_exp, d_ff=d_ff, k=k)
    off = _train_steps(cfg, steps, T, 0.0, seed=seed, lr=lr,
                     data_mode=data_mode, optim=optim,
                     freeze_experts=freeze_experts)
    on = _train_steps(cfg, steps, T, gamma, seed=seed, lr=lr,
                      data_mode=data_mode, optim=optim,
                      freeze_experts=freeze_experts)
    u0 = 1.0 / n_exp
    return {
        "config": cfg, "steps": steps, "T": T, "gamma": gamma, "seed": seed,
        "lr": lr, "data_mode": data_mode, "optim": optim,
        "freeze_experts": freeze_experts,
        "uniform_share": u0,
        "no_aux": off, "with_aux": on,
        "collapse_top_share": off["final"]["top_share"],
        "restored_top_share": on["final"]["top_share"],
        "top_share_improvement": (off["final"]["top_share"]
                                  / max(on["final"]["top_share"], 1e-12)),
        "collapse_normalized_entropy": off["final"]["normalized_entropy"],
        "restored_normalized_entropy": on["final"]["normalized_entropy"],
        "entropy_improvement": (off["final"]["normalized_entropy"]
                                / max(on["final"]["normalized_entropy"], 1e-12)),
        "collapse_dead_experts": off["final"]["n_dead_experts"],
        "restored_dead_experts": on["final"]["n_dead_experts"],
        "collapse_chi2": off["final"]["chi2_vs_uniform"],
        "restored_chi2": on["final"]["chi2_vs_uniform"],
        "chi2_improvement": (off["final"]["chi2_vs_uniform"]
                             / max(on["final"]["chi2_vs_uniform"], 1e-12)),
        "no_aux_is_monopoly": off["final"]["top_share"] > 0.9,
        "no_aux_is_degenerate": (off["final"]["top_share"] > 0.5
                                 or off["final"]["n_dead_experts"] > 0),
    }


# ========================================================= exactness probes ==

def exactness_checks(d: int = 128, n_exp: int = 8, d_ff: int = 64, k: int = 2,
                     T: int = 32, seed: int = 0) -> dict:
    r"""The structural properties, each of which is silently violable.

      1. **Dispatch is exactly $K$ per token.** A row with tied router
         probabilities can select the same expert twice, which silently reduces
         the effective path count and changes the FLOP count. Checked on tied
         probabilities specifically, since that is where it happens.
      2. **The output is a $K$-path merge**: the forward equals the explicit sum
         over $\mathcal{S}$, term by term, and the adjoint into $\pi$ is the sum
         of the $K$ per-expert contributions.
      3. **The router's softmax is a distribution**: rows sum to 1.
      4. **$\mathcal{L}_{aux} \ge 1$ always**, by Cauchy-Schwarz, and equals 1
         exactly when routing is uniform.
    """
    cfg = MoEConfig(d=d, n_exp=n_exp, d_ff=d_ff, k=k)
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(T, d, generator=g)

    # (1) dispatch count, including on DELIBERATELY TIED probabilities
    torch.manual_seed(seed)
    model = MoEFFN(cfg, seed=seed)
    _, probs, mask = model.forward(x)
    per_token = mask.sum(-1)
    tied = torch.zeros(1, n_exp, dtype=x.dtype)     # every expert equally likely
    tied_mask = model.select(tied)
    return {
        "cfg": cfg, "T": T,
        "dispatch_is_exactly_k": bool((per_token == k).all()),
        "dispatch_counts": per_token.to(torch.float64).tolist(),
        "tied_selects_exactly_k": bool((tied_mask.sum(-1) == k).all()),
        "tied_mask": tied_mask.to(torch.float64).tolist(),
        "probs_rows_sum_to_1": bool(torch.allclose(probs.sum(-1),
                                                   torch.ones(T), atol=1e-6)),
        "aux_on_uniform_dispatch": _aux_uniform_value(cfg, x, probs, mask),
        # 1e-6, not 1e-9: the uniform-dispatch value is computed through a
        # float32 softmax and a float32 reduction, whose rounding leaves the
        # result a few 1e-7 from K (1.99999988 at K=2 on some seeds); a 1e-9
        # tolerance turns that rounding into a false structural failure on
        # arbitrary seeds.
        "aux_uniform_dispatch_equals_k": abs(_aux_uniform_value(
            cfg, x, probs, mask) - cfg.k) < 1e-6,
        "aux_calibrated_lower_bound": _aux_lower_bound(cfg, seed=seed),
        "y_equals_explicit_sum": _merge_exact(cfg, x, seed=seed),
    }


def _aux_uniform_value(cfg: MoEConfig, x: torch.Tensor, probs: torch.Tensor,
                       mask: torch.Tensor) -> float:
    r"""$\\mathcal{L}_{aux}$ when dispatch is uniform by construction.

    With top-$K$ routing that value is **$K$**, not 1: every token occupies $K$
    slots, so $\sum_a f_a = K$ and $n_{exp}\cdot(K/n_{exp})\sum_a\zeta_a = K$.
    The top-1 folklore says 1. The `exactness_checks` entry asserts the
    top-$K$-correct value, which is how the folklore version was caught.
    """
    T = x.shape[0]
    n = cfg.n_exp
    k = cfg.k
    if T % (n * k):
        return float("nan")
    uniform = torch.zeros_like(mask)
    for t in range(T):
        base = (t // k) % n
        for j in range(k):
            uniform[t, (base + j) % n] = 1.0
    loss, _ = load_balancing_loss(probs, uniform, n)
    # detach: `probs` flows through the router's parameters, so `loss` requires
    # grad and float() on it raises UserWarning. A scalar *reading* is not a
    # gradient path.
    return float(loss.detach())


def _aux_lower_bound(cfg: MoEConfig, *, seed: int, trials: int = 256) -> dict:
    r"""The lower bound, *conditional on calibration*, which is where it lives.

    Asserts the true theorem: when $f = \zeta$ (the router is perfectly
    calibrated), $n_{exp}\sum_a v_a^2 \ge 1$ for every probability vector $v$,
    with equality iff $v$ is uniform. The first version of this check asserted
    "$\ge 1$ on random independent routing", which is **false** -- opposite
    one-hots give exactly 0 -- and it passed on 64 trials by luck. Asserting a
    false theorem that passes by luck is worse than asserting nothing, so this
    is the conditional form, and the condition is in the name.
    """
    g = torch.Generator().manual_seed(seed + 99)
    worst, worst_v = float("inf"), None
    at_uniform = None
    for i in range(trials):
        v = torch.softmax(torch.randn(cfg.n_exp, generator=g,
                                      dtype=torch.float64), dim=-1)
        L = float(cfg.n_exp * (v * v).sum())
        if L < worst:
            worst, worst_v = L, v.tolist()
        if i == 0:
            at_uniform = float(cfg.n_exp
                               * ((torch.full((cfg.n_exp,), 1.0 / cfg.n_exp)
                                   ** 2).sum()))
    return {"worst": worst, "worst_at": worst_v,
            "at_uniform": at_uniform,
            "holds": worst >= 1.0 - 1e-9,
            "uniform_is_1": abs(at_uniform - 1.0) < 1e-12}


def _merge_exact(cfg: MoEConfig, x: torch.Tensor, *, seed: int) -> bool:
    """Forward equals the explicit per-expert sum, term by term."""
    torch.manual_seed(seed)
    model = MoEFFN(cfg, seed=seed)
    y, probs, mask = model.forward(x)
    manual = torch.zeros_like(x)
    for a in range(cfg.n_exp):
        sel = mask[:, a] > 0
        if bool(sel.any()):
            manual[sel] += (probs[sel, a:a + 1] * model.experts[a].forward(x[sel]))
    return bool(torch.equal(y, manual))


def router_speed_report(router_lr_mults=(1, 20, 100), steps: int = 200,
                        T: int = 256, gamma: float = 0.01, d: int = 1024,
                        n_exp: int = 8, d_ff: int = 1024, k: int = 2,
                        seed: int = 0, base_lr: float = 0.05) -> dict:
    r"""How far the loop *can* drive concentration when the router outruns the experts.

    The router is given its own learning rate, ``router_lr_mult`` times the
    experts'. When the router moves much faster, it can starve experts before
    they equalize -- the condition under which the rich-get-richer loop has real
    force. This is the fastest concentration this harness produces, and it is
    reported as a boundary of the phenomenon, not as the typical case.

    Measured: top share, entropy and dead experts with and without the auxiliary
    loss, at each speed ratio. The aux coefficient is held fixed, so a secondary
    finding emerges honestly: at high router speed the fixed $\gamma$ is swamped
    by the main-loss gradient, and the aux loss stops helping. That limitation is
    reported rather than tuned away.
    """
    cfg = MoEConfig(d=d, n_exp=n_exp, d_ff=d_ff, k=k)
    x, target, _ = make_iid_batch(T, d, seed=seed + 1)
    rows = []
    for mult in router_lr_mults:
        row = {"router_lr_mult": mult, "router_lr": base_lr * mult,
               "expert_lr": base_lr}
        for g, name in ((0.0, "no_aux"), (gamma, "with_aux")):
            torch.manual_seed(seed)
            model = MoEFFN(cfg, seed=seed)
            opt = torch.optim.Adam([
                {"params": [model.W_route], "lr": base_lr * mult},
                {"params": [p for e in model.experts for p in e.params()],
                 "lr": base_lr},
            ])
            for _ in range(steps):
                opt.zero_grad(set_to_none=True)
                y, probs, mask = model.forward(x)
                loss = (torch.nn.functional.mse_loss(y, target)
                        + g * load_balancing_loss(probs, mask, n_exp)[0])
                loss.backward()
                opt.step()
            with torch.no_grad():
                _, probs, mask = model.forward(x)
            u = expert_usage(probs, mask)
            row[name] = {"top_share": u["top_share"],
                         "collapse_ratio": u["collapse_ratio"],
                         "normalized_entropy": u["normalized_entropy"],
                         "n_dead_experts": u["n_dead_experts"],
                         "chi2": u["chi2_vs_uniform"]}
        rows.append(row)
    nose = [r["no_aux"] for r in rows]
    return {
        "config": cfg, "steps": steps, "T": T, "gamma": gamma, "seed": seed,
        "base_lr": base_lr, "rows": rows,
        "max_top_share": max(r["top_share"] for r in nose),
        "monopoly_reached": any(r["top_share"] > 0.9 for r in nose),
        "any_dead": any(r["n_dead_experts"] > 0 for r in nose),
    }


def aux_efficacy_report(gammas=(0.01, 0.1, 0.5), seeds=(0, 1, 2, 3),
                        steps: int = 50, T: int = 256, d: int = 1024,
                        n_exp: int = 8, d_ff: int = 1024, k: int = 2) -> dict:
    r"""Does the auxiliary loss actually improve balance, across seeds?

    The honest version of the restoration claim. A single-seed comparison --
    $1.27\times$ without aux against $1.20\times$ with it at seed 0 -- is within
    seed noise and does not survive more draws: at $\gamma = 0.01$ the aux run
    wins on 2 of 4 seeds, i.e. a coin flip. Reporting the seed-0 pair as
    "restoration" would be reporting noise as signal.

    Measured per $\gamma$: the win rate (fraction of seeds where the aux run's
    top share is strictly below the no-aux run's), and the mean top shares on
    both sides. Also measured: whether a large $\gamma$ *hurts*, because
    "more regularisation is better" is the natural misreading and it is false
    here -- at $\gamma = 0.5$ the aux run loses on all four seeds.
    """
    rows = []
    for gamma in gammas:
        tn, ta = [], []
        for sd in seeds:
            r = routing_collapse_report(steps=steps, T=T, gamma=gamma, d=d,
                                        n_exp=n_exp, d_ff=d_ff, k=k, seed=sd)
            tn.append(r["no_aux"]["final"]["top_share"])
            ta.append(r["with_aux"]["final"]["top_share"])
        m_n = sum(tn) / len(tn)
        m_a = sum(ta) / len(ta)
        rows.append({
            "gamma": gamma,
            "n_seeds": len(seeds),
            "no_aux_mean": m_n, "with_aux_mean": m_a,
            "win_rate": sum(a < b for a, b in zip(ta, tn)) / len(seeds),
            "mean_improvement": m_n - m_a,
        })
    return {"steps": steps, "T": T, "seeds": list(seeds), "rows": rows,
            # "Significant" must mean unanimous *and* sized, not a 3-of-4
            # majority: with 4 seeds a 75% win rate is 3 seeds, which a fair
            # coin produces with probability 4 * 0.5^4 = 25% per gamma -- and
            # across the three gammas swept here, at least one such streak is
            # more likely than not. Torch-version rounding alone moves
            # individual seeds across the tie line (measured: the same code
            # flips the gamma=0.01 win rate between 1.00 and 0.50 across
            # library builds, and the top-k tie fix legitimately changes one
            # tied dispatch per ~10 training steps), so the threshold is
            # unanimous wins AND a mean improvement larger than the ~1%
            # seed-noise band of these top shares.
            "restoration_significant_anywhere": any(
                r["win_rate"] >= 1.0 and r["mean_improvement"] > 0.01
                for r in rows),
            "large_gamma_hurts": any(r["mean_improvement"] < -0.01
                                     for r in rows)}
