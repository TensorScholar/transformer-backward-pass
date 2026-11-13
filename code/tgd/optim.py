"""AdamW, written out explicitly, with per-step instrumentation.

Why this module exists
----------------------
The Chapter 7 training harness calls ``torch.optim.AdamW``. That is the right
call for producing results, and the wrong call for a book that claims the
optimizer's behaviour can be derived. Two reasons:

1. **The update rules in Chapter 8 must be *measured*, not asserted.** Every
   claim in that chapter -- the step-0 scale invariance, the way accumulated
   state penalizes a spike, the role of epsilon as a step-size limiter -- is a
   statement about a specific arithmetic sequence. Reading it off a library's
   documentation is not a measurement.

2. **The harness's defaults are the evidence.** ``betas=(0.9, 0.95)`` and
   ``weight_decay=0.01`` are what produced every number in Chapter 7. Those
   values belong in the book's notation contract and in a place a reader can
   check, not buried in a training script.

So this module is a reference implementation whose *only* job is to be obviously
correct and to expose the intermediate quantities. It is verified elementwise
against ``torch.optim.AdamW`` by ``code/ch08/ch08_optimizer_verify.py``.

The update, in the book's notation (contract section 5)
--------------------------------------------------------
    m_k = beta_1 * m_{k-1} + (1 - beta_1) * g_k              first moment
    v_k = beta_2 * v_{k-1} + (1 - beta_2) * g_k ** 2         second moment
    m_hat_k = m_k / (1 - beta_1 ** k)                        bias correction
    v_hat_k = v_k / (1 - beta_2 ** k)
    delta = -eta * m_hat_k / (sqrt(v_hat_k) + eps) - lambda * w * theta
    theta += delta

with ``m_0 = v_0 = 0`` and ``k`` starting at 1. ``k`` is the optimizer step and
is never a tensor index.
"""
from __future__ import annotations

from typing import Iterable, NamedTuple

import torch

#: The hyperparameters the Chapter 7 harness actually used. These are the
#: defaults of every measurement in Chapter 7 and are recorded here so the
#: manuscript and the code cannot drift apart.
BETA_1 = 0.9
BETA_2 = 0.95
WEIGHT_DECAY = 0.01
EPS_ADAM = 1e-8


class AdamWTrace(NamedTuple):
    """One step's intermediates. Everything the chapter quotes is here."""

    step: int
    grad_norm_pre: torch.Tensor        # ||g_k|| before any clipping
    grad_norm_post: torch.Tensor       # ||g_k|| after clipping
    m_hat_norm: torch.Tensor           # ||m_hat_k||
    v_hat_norm: torch.Tensor           # ||v_hat_k||
    ratio_norm: torch.Tensor           # || m_hat / (sqrt(v_hat) + eps) ||
    update_norm: torch.Tensor          # ||delta||, the actual move
    weight_norm: torch.Tensor          # ||theta|| before the step
    rho: torch.Tensor                  # update-to-weight ratio
    eps_term_max: torch.Tensor         # max_i eps / (sqrt(v_hat_i) + eps)


class AdamW:
    """Explicit AdamW with decoupled weight decay.

    Parameters follow ``torch.optim.AdamW`` so the two are interchangeable in a
    test. ``capturable``/``foreach``/``fused`` are deliberately absent: this
    implementation is for reading, not for throughput.
    """

    def __init__(self, params: Iterable[torch.nn.Parameter], *,
                 lr: float = 1e-3, betas: tuple[float, float] = (BETA_1, BETA_2),
                 eps: float = EPS_ADAM, weight_decay: float = WEIGHT_DECAY):
        self.params = [p for p in params if p.requires_grad]
        self.lr = lr
        self.beta_1, self.beta_2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.step_count = 0
        self.trace: list[AdamWTrace] = []

        # m_0 = v_0 = 0 : the zero initial state is what makes the first step
        # special, and it is the entire content of Chapter 8's section 8.2.
        self.state: dict[int, tuple[torch.Tensor, torch.Tensor]] = {
            i: (torch.zeros_like(p, dtype=torch.float32),
                torch.zeros_like(p, dtype=torch.float32))
            for i, p in enumerate(self.params)
        }

    @torch.no_grad()
    def step(self, clip_fn=None) -> AdamWTrace:
        """Take one step. ``clip_fn`` may rescale gradients in place; it returns
        nothing, and the trace records the norms either side of it."""
        self.step_count += 1
        k = self.step_count
        b1, b2 = self.beta_1, self.beta_2
        bc1 = 1.0 - b1 ** k
        bc2 = 1.0 - b2 ** k

        pre_sq = post_sq = 0.0
        m_sq = v_sq = r_sq = upd_sq = w_sq = 0.0
        eps_term_max = 0.0

        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            pre_sq += float(g.detach().pow(2).sum())
            if clip_fn is not None:
                clip_fn()
                g = p.grad
            post_sq += float(g.detach().pow(2).sum())

            m, v = self.state[i]
            # ---- the two moments -------------------------------------------
            m.mul_(b1).add_(g, alpha=1.0 - b1)
            v.mul_(b2).addcmul_(g, g, value=1.0 - b2)
            # ---- bias correction -------------------------------------------
            m_hat = m / bc1
            v_hat = v / bc2
            # ---- the ratio, which is what the chapter is about -------------
            denom = v_hat.sqrt().add_(self.eps)
            ratio = m_hat / denom
            # ---- decoupled decay: applied to the parameter, NOT to g -------
            upd = ratio.mul(-self.lr)
            if self.weight_decay != 0.0:
                upd.add_(p, alpha=-self.lr * self.weight_decay)
            p.add_(upd)

            m_sq += float(m_hat.detach().pow(2).sum())
            v_sq += float(v_hat.detach().pow(2).sum())
            r_sq += float(ratio.detach().pow(2).sum())
            upd_sq += float(upd.detach().pow(2).sum())
            w_sq += float(p.detach().pow(2).sum())
            eps_term_max = max(eps_term_max,
                               float((self.eps / denom).max()))

        tr = AdamWTrace(
            step=k,
            grad_norm_pre=torch.tensor(pre_sq ** 0.5),
            grad_norm_post=torch.tensor(post_sq ** 0.5),
            m_hat_norm=torch.tensor(m_sq ** 0.5),
            v_hat_norm=torch.tensor(v_sq ** 0.5),
            ratio_norm=torch.tensor(r_sq ** 0.5),
            update_norm=torch.tensor(upd_sq ** 0.5),
            weight_norm=torch.tensor(w_sq ** 0.5),
            rho=torch.tensor(upd_sq ** 0.5 / (w_sq ** 0.5 + 1e-12)),
            eps_term_max=torch.tensor(eps_term_max),
        )
        self.trace.append(tr)
        return tr

    @torch.no_grad()
    def zero_grad(self, set_to_none: bool = True) -> None:
        for p in self.params:
            p.grad = None if set_to_none else torch.zeros_like(p)


def scale_clip(c: float):
    """A clip callback matching global-norm clipping at threshold ``c``.

    Global rather than per-parameter on purpose: global is the case where the
    first-step invariance holds exactly, and per-group is one of the conditions
    that breaks it.
    """
    def _fn() -> None:
        total = torch.norm(torch.stack([
            p.grad.norm() for p in _CURRENT if p.grad is not None]))
        if float(total) > c:
            for p in _CURRENT:
                if p.grad is not None:
                    p.grad.mul_(c / (float(total) + 1e-12))
    return _fn


_CURRENT: list = []


def _bind(params) -> None:
    global _CURRENT
    _CURRENT = list(params)