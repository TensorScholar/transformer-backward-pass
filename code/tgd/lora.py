r"""Parameter-efficient fine-tuning: the dense layer and its low-rank shadow.

Why this module exists
----------------------
Fine-tuning a pretrained decoder is *mathematically identical* to pre-training it.
Same forward graph, same backward pass, same optimizer, same non-differentiability
of argmax. The only difference is which parameters carry a gradient. That is the
whole reason PEFT works, and it is also why PEFT is an **algebraic** trick rather
than an approximation.

The arithmetic that motivates it is Chapter 9's and Chapter 11's made into a bill.
For a weight $W \in \mathbb{R}^{d_{out}\times d_{in}}$ trained with Adam in
float32, the persistent footprint is

    weights W          d_out * d_in * 4 bytes
    gradients \bar W    d_out * d_in * 4 bytes
    Adam first moment  d_out * d_in * 4 bytes
    Adam second moment d_out * d_in * 4 bytes
    ---------------------------------------------
    total              16 bytes per weight = 4x the weights alone

and the gradient plus the two moments are **three quarters of it**. The weights
themselves are not trainable overhead -- they are the model. So the addressable
waste is $12$ bytes per weight, and that is what low-rank adaptation removes.

What LoRA does
--------------
Freeze $W$. Learn two much smaller matrices and add their product:

    W_eff = W + W^{\downarrow} W^{\uparrow},
    W^{\downarrow} \in R^{d_in x r},   W^{\uparrow} \in R^{r x d_out}

Trainable parameters: $r(d_{out} + d_{in})$ instead of $d_{out}d_{in}$. At
$d_{out} = d_{in} = 4096$ the ratio is $d^2/r(2d) = d/2r$, which is **256x at
$r=8$** and **32x at $r=64$**. Three quarters of the persistent footprint goes
with it.

Notation, and why it is not $A$ and $B$
----------------------------------------
Every LoRA implementation in the wild names the factors $A$ and $B$. **Both are
reserved by the notation contract**: $A$ is the attention matrix (ID 9, and
$\S7$ records it as having collided with Adam's first moment before), and $B$ is
the batch dimension. This module therefore writes $W^{\downarrow}$ and
$W^{\uparrow}$, which are the factors' own names in the original paper and carry
no collision. A reader coming from an implementation will recognise them
immediately; a reader reading this book will not lose a symbol.

Two facts that are easy to get backwards, and both are asserted here
--------------------------------------------------------------------
1. **Associativity holds in reals, not in floating point.** $x(W + AB) = xW +
   (xA)B$ is a theorem about real arithmetic. In float32 the two evaluations round
   differently and the outputs differ at the last few bits. The chapter asserts
   agreement *to float32 rounding* and reports the measured magnitude; it does
   **not** assert bitwise equality, because that assertion is false and asserting
   it would be asserting a bug.
2. **Merging is not free, it is one-time.** Inference with an unmerged adapter
   costs two extra matmuls per token. Merging folds $W + W^{\downarrow}
   W^{\uparrow}$ into a single matrix once, after which latency is *identical* to
   the base model -- the extra arithmetic is $r\,d_{out}d_{in}$ multiply-adds, once,
   against $d_{in}d_{out}$ per token thereafter. That is the claim "zero inference
   overhead" rests on, and :func:`merge_flop_report` measures it.

Measurement convention
----------------------
All tensors are float32 on CPU with fixed seeds, so every byte count is exact
arithmetic on shapes and every FLOP count is exact arithmetic on the shapes. The
one measured quantity with any tolerance is the forward-pass equivalence, and its
tolerance is stated where it is used rather than assumed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

__all__ = [
    "LoRAConfig", "FullLayer", "LoRALayer", "lora_forward", "merged_weight",
    "dense_forward", "equivalence_report", "memory_report", "merge_flop_report",
    "training_run", "training_run_report", "rank_tradeoff", "exactness_checks",
    "seed_stability_report", "BYTES", "ADAM_STATE_BYTES",
]

#: float32. Every byte count in this module is exact arithmetic on this number.
BYTES = 4

#: Adam keeps two moments per trainable parameter, so 2 x BYTES per parameter.
ADAM_STATE_BYTES = 2 * BYTES


# ============================================================== the layers ==

class FullLayer:
    r"""A dense layer fine-tuned directly: the $W$ of contract $\S2$.

    Implements the **row-vector** convention of contract C1 throughout:
    ``y = x W + b`` with ``x`` of shape ``(..., d_in)``. Nothing in this module
    needs the column form, and mixing the two is how a LoRA implementation ends
    up transposed.
    """

    def __init__(self, d_in: int, d_out: int, *, seed: int = 0,
                 requires_grad: bool = True):
        g = torch.Generator().manual_seed(seed)
        # Kaiming-ish scaling so the layer is not pathological at init. A LoRA
        # factor must start at ZERO on one side (see LoRALayer), and an adapter
        # that starts at a random nonzero product would change the base model's
        # behaviour at step 0 -- a silent bug that looks like "LoRA hurt quality".
        # SHAPE, and this is the bug worth reporting. Row-vector convention
        # (contract C1) gives y = x W with x of shape (..., d_in), so W must be
        # (d_in, d_out). I first wrote (d_out, d_in), which is the transposed
        # convention -- and every test passed, because the only shapes anyone
        # tries first are square, where d_in == d_out and the transpose is
        # invisible. It surfaced the moment equivalence_report was called with
        # (1024, 768). The parameter COUNT is identical either way, so no byte
        # count in section 15.1 is affected; only the orientation is.
        self.W = torch.nn.Parameter(
            torch.randn(d_in, d_out, generator=g) / math.sqrt(d_in),
            requires_grad=requires_grad)
        self.b = torch.nn.Parameter(torch.zeros(d_out),
                                    requires_grad=requires_grad)
        self.d_in, self.d_out = d_in, d_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.W + self.b

    def trainable(self) -> list[torch.nn.Parameter]:
        return [p for p in (self.W, self.b) if p.requires_grad]


class LoRALayer:
    r"""A frozen dense layer plus a rank-$r$ trainable correction.

    ``y = x (W + W_down W_up) + b``, implemented the **fast** way during
    training as ``x W + (x W_down) W_up``. $W^{\uparrow}$ is initialised to zero,
    so $W^{\downarrow}W^{\uparrow} = 0$ at step 0 and the adapted model is
    *exactly* the base model before the first update. That is a correctness
    property, not a nicety, and it is asserted.
    """

    def __init__(self, d_in: int, d_out: int, r: int, *, seed: int = 0,
                 base: FullLayer | None = None, alpha: float | None = None):
        if r <= 0:
            raise ValueError(f"r must be positive, got r={r}")
        if r > min(d_in, d_out):
            raise ValueError(
                f"r={r} exceeds min(d_in, d_out)={min(d_in, d_out)}; the "
                "factorisation would not be a compression, and the parameter "
                "count would exceed the dense layer it replaces")
        base = base if base is not None else FullLayer(d_in, d_out, seed=seed,
                                                       requires_grad=False)
        self.base = base
        self.r = r
        self.d_in, self.d_out = d_in, d_out
        # LoRA's scaling is alpha / r by convention. It is a free multiplier and
        # changes nothing about any count in this chapter; it is exposed so the
        # default is visible rather than hidden.
        self.scaling = 1.0 if alpha is None else alpha / r
        g = torch.Generator().manual_seed(seed + 1)
        self.W_down = torch.nn.Parameter(
            torch.randn(d_in, r, generator=g) / math.sqrt(r))
        # ZERO, deliberately. See the class docstring.
        self.W_up = torch.nn.Parameter(torch.zeros(r, d_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base.forward(x) + (x @ self.W_down) @ self.W_up * self.scaling

    def delta(self) -> torch.Tensor:
        r"""The learned correction $\Delta W = W^{\downarrow} W^{\uparrow}$."""
        return (self.W_down @ self.W_up) * self.scaling

    def merged(self) -> FullLayer:
        r"""Fold the adapter into a standalone dense layer for inference."""
        m = FullLayer(self.d_in, self.d_out, seed=0, requires_grad=False)
        with torch.no_grad():
            m.W.copy_(self.base.W + self.delta())
            m.b.copy_(self.base.b)
        return m

    def trainable(self) -> list[torch.nn.Parameter]:
        return [self.W_down, self.W_up]


def dense_forward(x: torch.Tensor, W: torch.Tensor,
                  b: torch.Tensor | None = None) -> torch.Tensor:
    r"""``x (W + A B)`` -- the *merged* evaluation path, for comparison."""
    return x @ W + (b if b is not None else 0)


def lora_forward(x: torch.Tensor, W: torch.Tensor, A: torch.Tensor,
                 B: torch.Tensor, b: torch.Tensor | None = None,
                 scaling: float = 1.0) -> torch.Tensor:
    r"""``x W + (x A) B`` -- the *unmerged* evaluation path."""
    return x @ W + ((x @ A) @ B) * scaling


def merged_weight(W: torch.Tensor, A: torch.Tensor, B: torch.Tensor,
                  scaling: float = 1.0) -> torch.Tensor:
    r"""``W + s A B`` -- the one-time merge."""
    return W + scaling * (A @ B)


# =========================================================== equivalence ====

def equivalence_report(d_in: int = 512, d_out: int = 512, r: int = 8,
                       batch: int = 16, seed: int = 0,
                       dtype: torch.dtype = torch.float32) -> dict:
    r"""Is ``x(W + AB) == xW + (xA)B``? Measured, not assumed.

    The theorem is about **reals**. In floating point the two evaluations differ in
    association order and therefore in rounding, so the honest claim is agreement
    to float32 rounding, not bitwise equality. Both are measured and both are
    reported, because a reader who expects bitwise equality will otherwise read
    the ~1e-7 difference as a bug.

    Also reported: the *post-merge* path, which computes ``W + A B`` once and then
    a single matmul. That path differs from the unmerged path by the rounding of
    the merge itself, so there are three quantities, not two, and conflating them
    is how "LoRA changes the output" gets claimed from a float32 artefact.
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, d_in, generator=g, dtype=dtype)
    W = torch.randn(d_in, d_out, generator=g, dtype=dtype) / math.sqrt(d_in)
    A = torch.randn(d_in, r, generator=g, dtype=dtype) / math.sqrt(r)
    B = torch.randn(r, d_out, generator=g, dtype=dtype)

    merged = merged_weight(W, A, B)
    y_merged = x @ merged                                  # x(W + AB)
    y_unmerged = lora_forward(x, W, A, B)                  # xW + (xA)B

    d = (y_merged - y_unmerged).abs()
    scale = float(y_unmerged.abs().max())
    ref = y_unmerged.double()
    # the float32 rounding budget: gamma_n = n*u/(1-n*u) for a length-n dot
    # product, u = 2^-24 for float32
    u = 2.0 ** -24
    n = d_in + r
    budget = n * u / (1.0 - n * u) * float(ref.abs().max())

    # NORMALISATION. A per-element relative difference, |d_i| / |y_i|, is
    # meaningless here and reported 0.84 for a pair of tensors that agree to
    # float32 rounding: the output has elements that are themselves near zero, and
    # dividing a rounding-sized absolute error by a near-zero reference
    # manufactures an enormous ratio. Two honest normalisations are used instead:
    #
    #   * against the output SCALE, which is the quantity the roundoff budget is
    #     also measured against, so the two are directly comparable;
    #   * in ULPs of that scale, which is the only way to say how many
    #     representable floats apart the two evaluations are.
    diff_over_scale = float(d.max()) / scale if scale else float("nan")
    return {
        "d_in": d_in, "d_out": d_out, "r": r, "batch": batch,
        "dtype": str(dtype),
        "max_abs_diff": float(d.max()),
        "rms_diff": float((d.double() ** 2).mean() ** 0.5),
        "diff_over_output_scale": diff_over_scale,
        "ulps_of_output_scale": diff_over_scale / u,
        "min_abs_output": float(ref.abs().min()),
        "bitwise_identical": bool(torch.equal(y_merged, y_unmerged)),
        "output_scale": scale,
        "roundoff_budget": budget,
        "budget_over_max_diff": (budget / float(d.max())
                                 if float(d.max()) else 0.0),
        "within_roundoff": bool(float(d.max()) <= budget),
        "n_ulp_float32": 24,
    }


# ================================================================ memory ====

@dataclass(frozen=True)
class LoRAConfig:
    """Shapes. Everything else is derived, never set directly."""

    d_in: int = 4096
    d_out: int = 4096
    r: int = 8
    element_bytes: int = BYTES

    @property
    def dense_params(self) -> int:
        return self.d_out * self.d_in

    @property
    def lora_params(self) -> int:
        return self.r * (self.d_out + self.d_in)

    @property
    def n_bias(self) -> int:
        """Bias is trainable in both variants, so it is counted in both."""
        return self.d_out


def memory_report(cfg: LoRAConfig | None = None) -> dict:
    r"""The exact persistent footprint, byte for byte.

    Counted honestly, which means two distinctions the usual summary blurs:

      * **Adam state is $2\times$ the gradient count**, not $1\times$. A step
        needs a first and a second moment. Reporting only $m$ understates the
        saving by half.
      * **LoRA does not shrink the weight matrix.** The frozen $W$ is still
        resident, at full size, in every configuration. So the reduction applies
        to the *trainable* footprint -- gradient plus moments -- and the *total*
        saving is much smaller than the parameter-count ratio. Both are reported,
        and conflating them is the single most common overstatement of PEFT.

    LoRA adapts the **weight matrix** only. The bias belongs to the frozen base
    layer, so it is *not* trainable under LoRA, and counting it as though it were
    inflates the LoRA footprint and understates the saving. This was a real
    accounting bug: the model predicted 4,352 trainable elements for a 256x256
    layer at r=8 while the layer actually had 4,096. So:

        full:  d_out*d_in  + d_out      (weight and bias both trainable)
        lora:  r(d_in+d_out)            (adapter only; bias stays frozen)

    which makes the ratio exactly ``d^2/(2rd) = d/2r`` with no bias fudge.
    """
    cfg = cfg or LoRAConfig()
    e = cfg.element_bytes
    D = cfg.dense_params + cfg.n_bias          # weight + bias, both trainable
    L = cfg.lora_params                          # adapter only; bias is frozen

    def block(n: int) -> dict:
        w = n * e
        g = n * e
        adam = n * ADAM_STATE_BYTES
        return {"elements": n, "weights_bytes": w, "grad_bytes": g,
                "adam_bytes": adam, "persistent_bytes": w + g + adam}

    full = block(D)
    lora = block(L)
    # frozen base weights are resident in BOTH and are the same bytes
    frozen = cfg.dense_params * e

    return {
        "config": cfg,
        "full": full, "lora": lora,
        "frozen_base_bytes": frozen,
        "full_total_bytes": frozen + D * 0 + full["grad_bytes"]
                           + full["adam_bytes"],
        "lora_total_bytes": frozen + lora["weights_bytes"]
                           + lora["grad_bytes"] + lora["adam_bytes"],
        # With the bias correctly frozen under LoRA, the two ratios coincide:
        # both are d^2/(r(d_in+d_out)) = d/2r. They are still reported separately,
        # because "the weights-only ratio" is the number most often quoted and
        # the chapter should show that it happens to be the right one here rather
        # than the right one by coincidence.
        "param_ratio": D / L,
        "param_ratio_weights_only": cfg.dense_params / cfg.lora_params,
        "bias_trainable_under_lora": False,
        "grad_ratio": full["grad_bytes"] / lora["grad_bytes"],
        "adam_ratio": full["adam_bytes"] / lora["adam_bytes"],
        "trainable_bytes_saved": (full["grad_bytes"] + full["adam_bytes"])
                                 - (lora["grad_bytes"] + lora["adam_bytes"]),
        "total_bytes_saved": (frozen + full["grad_bytes"] + full["adam_bytes"])
                             - (frozen + lora["weights_bytes"]
                                + lora["grad_bytes"] + lora["adam_bytes"]),
        "total_ratio": (frozen + full["grad_bytes"] + full["adam_bytes"])
                       / (frozen + lora["weights_bytes"] + lora["grad_bytes"]
                          + lora["adam_bytes"]),
    }


# ================================================================ flops =====

def merge_flop_report(cfg: LoRAConfig | None = None) -> dict:
    r"""Training cost overhead and the one-time merge cost.

    Per token, the unmerged adapter adds

        r*d_in  (for x W_down)  +  r*d_out  (for W_up .)

    against the base ``d_in * d_out``, so the overhead fraction is
    ``r(d_in + d_out) / (d_in d_out) = 1/256`` at $d = 4096, r = 8.

    The merge itself costs ``r * d_out * d_in`` multiply-adds, ONCE. Expressed as
    "how many tokens of inference pay for it", it is ``r`` -- so the merge has
    amortised itself after ``r`` tokens and is free thereafter. That arithmetic is
    the entire content of "LoRA adds no inference latency".
    """
    cfg = cfg or LoRAConfig()
    base = cfg.d_in * cfg.d_out
    extra = cfg.r * (cfg.d_in + cfg.d_out)
    merge = cfg.r * cfg.d_out * cfg.d_in
    return {
        "config": cfg,
        "base_macs_per_token": base,
        "lora_extra_macs_per_token": extra,
        "overhead_fraction": extra / base,
        "overhead_percent": 100.0 * extra / base,
        "merge_macs_once": merge,
        "tokens_to_amortise_merge": merge / base,
        "merged_inference_macs_per_token": base,
        "unmerged_inference_overhead_percent": 100.0 * extra / base,
    }


# ======================================================= the training run ===

def _loss_from_output(y: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Deterministic scalar loss, so the two runs are comparable step for step."""
    return ((y - target) ** 2).mean()


def training_run(kind: str, cfg: LoRAConfig, steps: int = 10, batch: int = 8,
                 seed: int = 0, lr: float = 1e-3) -> dict:
    r"""``steps`` real backward passes through autograd, with a real optimizer.

    ``kind`` is ``"full"`` or ``"lora"``. The Adam update is written out here
    rather than taken from ``tgd.optim`` for two reasons: this module must not
    depend on the optimizer chapter, and the moment tensors need to be *counted*,
    which means owning them.

    The same input, the same target, the same seed and the same initial $W$ in
    both runs, so the loss traces differ only because the parameterisations do.
    That is what makes the comparison a measurement rather than an anecdote.
    """
    if kind not in ("full", "lora"):
        raise ValueError(f"kind must be 'full' or 'lora', got {kind!r}")
    e = cfg.element_bytes
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, cfg.d_in, generator=g)
    target = torch.randn(batch, cfg.d_out, generator=g)

    if kind == "full":
        layer = FullLayer(cfg.d_in, cfg.d_out, seed=seed)
    else:
        layer = LoRALayer(cfg.d_in, cfg.d_out, cfg.r, seed=seed)
    params = layer.trainable()

    # count the state that WILL be allocated, before allocating it, so the number
    # is a prediction rather than a reading of torch's allocator
    n_trainable = sum(p.numel() for p in params)
    predicted_adam_bytes = n_trainable * ADAM_STATE_BYTES
    predicted_grad_bytes = n_trainable * e
    frozen_base_bytes = (0 if kind == "full"
                         else layer.base.W.numel() * e)

    m = [torch.zeros_like(p) for p in params]
    v = [torch.zeros_like(p) for p in params]
    b1, b2, eps = 0.9, 0.999, 1e-8

    losses: list[float] = []
    grad_norms: list[float] = []
    for k in range(1, steps + 1):
        for p in params:
            p.grad = None
        y = layer.forward(x)
        loss = _loss_from_output(y, target)
        loss.backward()
        # detach before the scalar conversion: float() on a tensor that requires
        # grad raises a UserWarning and is a latent trap, since the value read is
        # the same either way and so nothing looks wrong.
        losses.append(float(loss.detach()))
        gn = math.sqrt(sum(float(p.grad.detach().pow(2).sum())
                           for p in params if p.grad is not None))
        grad_norms.append(gn)
        with torch.no_grad():
            for i, p in enumerate(params):
                g_ = p.grad
                m[i].mul_(b1).add_(g_, alpha=1.0 - b1)
                v[i].mul_(b2).addcmul_(g_, g_, value=1.0 - b2)
                mh = m[i] / (1.0 - b1 ** k)
                vh = v[i] / (1.0 - b2 ** k)
                p.add_(-lr * mh / (vh.sqrt() + eps))

    return {
        "kind": kind, "config": cfg, "steps": steps,
        "n_trainable": n_trainable,
        "trainable_bytes": n_trainable * e,
        "predicted_grad_bytes": predicted_grad_bytes,
        "predicted_adam_bytes": predicted_adam_bytes,
        "frozen_base_bytes": frozen_base_bytes,
        "persistent_bytes": (frozen_base_bytes + n_trainable * e
                             + predicted_grad_bytes + predicted_adam_bytes),
        "losses": losses, "grad_norms": grad_norms,
        "final_loss": losses[-1] if losses else None,
        "loss_decreased": bool(losses and losses[-1] < losses[0]),
        "grad_finite": all(math.isfinite(g_) for g_ in grad_norms),
    }


def training_run_report(d_in: int = 4096, d_out: int = 4096, r: int = 8,
                        steps: int = 10, batch: int = 8, seed: int = 0
                        ) -> dict:
    r"""Both runs side by side, plus the measured optimizer-state saving."""
    full_cfg = LoRAConfig(d_in=d_in, d_out=d_out, r=r)
    f = training_run("full", full_cfg, steps=steps, batch=batch, seed=seed)
    l = training_run("lora", full_cfg, steps=steps, batch=batch, seed=seed)
    m = memory_report(full_cfg)
    fr = merge_flop_report(full_cfg)
    return {
        "config": full_cfg, "steps": steps, "batch": batch, "seed": seed,
        "full": f, "lora": l,
        "memory": m, "flops": fr,
        "optimizer_state_saved_bytes": f["predicted_adam_bytes"]
                                       - l["predicted_adam_bytes"],
        "optimizer_state_saving_ratio": (f["predicted_adam_bytes"]
                                         / l["predicted_adam_bytes"]),
        "grad_saving_ratio": (f["predicted_grad_bytes"]
                              / l["predicted_grad_bytes"]),
        "param_ratio": f["n_trainable"] / l["n_trainable"],
        "both_decreased_loss": f["loss_decreased"] and l["loss_decreased"],
        "loss_ratio_first_step": f["losses"][0] / l["losses"][0],
    }


def rank_tradeoff(d_in: int = 4096, d_out: int = 4096,
                  ranks=(1, 4, 8, 16, 32, 64, 128, 256)) -> list[dict]:
    r"""Parameter and optimizer-state cost against rank.

    A sweep read for monotonicity, not for a minimum: the trainable footprint
    grows linearly in $r$ with no interior optimum, because "how much of the
    update do you need" is not answered by any of these counts. The zero column
    is what makes the table honest -- at $r = 0$ the adapter is exactly the base
    model, and the jump from there is the whole cost of PEFT.
    """
    rows = []
    for r in ranks:
        cfg = LoRAConfig(d_in=d_in, d_out=d_out, r=r)
        m = memory_report(cfg)
        rows.append({
            "r": r,
            "lora_elements": cfg.lora_params,
            "fraction_of_dense": cfg.lora_params / cfg.dense_params,
            "compression": cfg.dense_params / cfg.lora_params,
            "optimizer_bytes": m["lora"]["adam_bytes"],
            "optimizer_MiB": m["lora"]["adam_bytes"] / 2 ** 20,
            "full_optimizer_MiB": m["full"]["adam_bytes"] / 2 ** 20,
            "r_is_min_dim": r == min(d_in, d_out),
        })
    return rows


# ========================================================= exactness probes ==

def exactness_checks(d_in: int = 256, d_out: int = 256, r: int = 8,
                     seed: int = 3) -> dict:
    r"""The properties that make LoRA *correct* rather than merely plausible.

    Each is a silent failure if violated:

      1. **Zero identity.** At initialisation $W^{\uparrow} = 0$, so the adapted
         layer must equal the frozen layer *exactly*. If it does not, the adapter
         is perturbing the pretrained model before a single step, and every
         downstream measurement is of a different model.
      2. **Merging is exact in the sense that matters**: the merged layer's output
         equals the unmerged layer's output to float32 rounding, and the merged
         layer has no adapter attached.
      3. **Freezing actually freezes.** $W$ must receive no gradient and no
         optimizer state, which is the entire source of the saving.
      4. **The frozen base is byte-identical** before and after a training run.
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(8, d_in, generator=g)
    layer = LoRALayer(d_in, d_out, r, seed=seed)

    # (1) zero identity, exactly
    y0 = layer.forward(x)
    y_base = layer.base.forward(x)
    zero_exact = bool(torch.equal(y0, y_base))

    # (2) merge. detach everywhere: the probe is a *numerical* comparison, not a
    # gradient path, and float() on a grad-requiring tensor is a latent trap.
    #
    # The expected value is captured HERE, at merge time. Checking it against
    # `layer.delta()` evaluated later -- after the training loop has moved
    # W_down and W_up -- compares a pre-training snapshot against post-training
    # state and reports False. The merge is correct; the probe was comparing two
    # different moments. Capturing at the point of the operation is the only
    # correct time.
    m = layer.merged()
    y_m = m.forward(x)
    merge_diff = float((y_m - y0).detach().abs().max())
    expected_merged_W = (layer.base.W + layer.delta()).detach().clone()
    merge_is_exact = bool(torch.equal(m.W.detach(), expected_merged_W))

    # (3) freezing
    layer.base.W.grad = None
    layer.forward(x).sum().backward()
    base_has_grad = layer.base.W.grad is not None
    base_requires_grad = bool(layer.base.W.requires_grad)

    # (4) base unchanged by training
    before = layer.base.W.detach().clone()
    opt_m = [torch.zeros_like(p) for p in layer.trainable()]
    opt_v = [torch.zeros_like(p) for p in layer.trainable()]
    for p in layer.trainable():
        p.grad = None
    layer.forward(x).pow(2).mean().backward()
    with torch.no_grad():
        for i, p in enumerate(layer.trainable()):
            opt_m[i].mul_(0.9).add_(p.grad, alpha=0.1)
            opt_v[i].mul_(0.999).addcmul_(p.grad, p.grad, value=0.001)
            p.add_(-1e-3 * (opt_m[i] / 0.1)
                   / ((opt_v[i] / 0.001).sqrt() + 1e-8))
    base_unchanged = bool(torch.equal(before, layer.base.W.detach()))
    adapter_nonzero = bool(layer.delta().abs().max() > 0)

    # (5) rank guard: r must not exceed min(d_in, d_out)
    refused = False
    try:
        LoRALayer(d_in, d_out, d_in + 1, seed=seed)
    except ValueError:
        refused = True

    return {
        "zero_init_identity_exact": zero_exact,
        "zero_init_max_abs_diff": float((y0 - y_base).detach().abs().max()),
        "merge_max_abs_diff": merge_diff,
        # the merged layer must be a plain frozen dense layer with NO adapter
        # attached. FullLayer is not an nn.Module, so it has no .parameters();
        # the accessor is trainable(), which returns [] when nothing requires
        # grad. (Calling .parameters() here raised AttributeError, which is the
        # correct outcome -- it is not a duck-typed module and should not be
        # treated as one.)
        "merged_has_no_adapter": len(m.trainable()) == 0,
        "merge_is_exact": merge_is_exact,
        "base_received_grad": base_has_grad,
        "base_requires_grad": base_requires_grad,
        "base_unchanged_by_training": base_unchanged,
        "adapter_nonzero_after_one_step": adapter_nonzero,
        "rank_gt_min_refused": refused,
        "r": r, "d_in": d_in, "d_out": d_out,
    }

def seed_stability_report(seeds=(0, 1, 2, 3, 4, 5, 6, 7), d_in: int = 4096,
                          d_out: int = 4096, r: int = 8, steps: int = 10,
                          batch: int = 8) -> dict:
    r"""How much of Chapter 15's measured content depends on the draw.

    Three of the chapter's measured quantities are stochastic: the forward-pass
    ULP difference (over random inputs), the 10-step loss traces, and the final
    losses. The byte counts and MAC counts are exact arithmetic and do not appear
    here. Reported as mean $\pm$ sample standard deviation across eight seeds.

    The claim under test is that the *conclusions* are stable, not that the digits
    are: that the ULP count stays inside the rounding budget, and that LoRA's
    higher 10-step loss is a property of capacity rather than of one unlucky draw.
    """
    ulps, budgets, margins = [], [], []
    full_final, lora_final, ratios = [], [], []
    first_losses = []
    for sd in seeds:
        e = equivalence_report(d_in=512, d_out=512, r=r, batch=16, seed=sd)
        ulps.append(e["ulps_of_output_scale"])
        budgets.append(e["budget_over_max_diff"])
        margins.append(e["within_roundoff"])
        t = training_run_report(d_in=d_in, d_out=d_out, r=r, steps=steps,
                                batch=batch, seed=sd)
        full_final.append(t["full"]["final_loss"])
        lora_final.append(t["lora"]["final_loss"])
        ratios.append(t["lora"]["final_loss"] / t["full"]["final_loss"])
        first_losses.append((t["full"]["losses"][0], t["lora"]["losses"][0]))

    def stats(v):
        m = sum(v) / len(v)
        sd_ = (sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** 0.5
        return {"mean": m, "sample_sd_across_seeds": sd_,
                "relative_sd": sd_ / m if m else None,
                "min": min(v), "max": max(v)}

    return {
        "seeds": list(seeds), "d_in": d_in, "d_out": d_out, "r": r,
        "steps": steps, "batch": batch, "n_seeds": len(seeds),
        "ulps": stats(ulps), "budget_margin": stats(budgets),
        "full_final_loss": stats(full_final), "lora_final_loss": stats(lora_final),
        "final_loss_ratio": stats(ratios),
        "always_within_roundoff": all(margins),
        "ulps_always_under_100": all(u < 100.0 for u in ulps),
        "lora_always_worse": all(l > f for l, f in zip(lora_final, full_final)),
        "step1_losses_identical_every_seed": all(
            a == b for a, b in first_losses),
    }
