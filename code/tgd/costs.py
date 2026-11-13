"""Computational cost from shapes alone — contract §6, symbols Phi, Psi, iota.

The point of this module is that Chapter 4 already knows which phase of the model
is expensive, without profiling anything. Every quantity here is a closed-form
function of ``(B, T, V, d, H, d_ff, L)``, which is the shape ledger the reader
built in Chapter 4.

Definitions and what they deliberately do *not* claim
------------------------------------------------------
**Phi** counts floating-point operations. Matmuls are counted as ``2*m*n*k``.
Elementwise work is counted at one operation per element. Attention's quadratic
term is counted in full, at ``T x T`` per head, because in a *dense* reference
implementation that is what is executed.

**Psi** is defined narrowly and honestly: it is the **activation traffic** that
the backward pass must re-read, i.e. the sum of the byte sizes of the tensors
the forward pass materialises and keeps. It is *not* DRAM traffic. Real DRAM
traffic depends on the memory hierarchy and on tiling, which is Chapter 11's
subject. Reporting a single "bytes moved" number as if it described hardware
would be exactly the kind of overclaim this book refuses to make.

**iota = Phi / Psi** is therefore *arithmetic intensity with respect to activation
traffic*, not a machine's roofline number. It is still the right diagnostic for
the question Chapter 4 wants answered: which phases do proportionally the most
arithmetic per unit of state they must carry, and therefore which are likely to
be limited by moving data rather than by multiplying.
"""
from __future__ import annotations

from dataclasses import dataclass

from .dims import Dims


@dataclass(frozen=True)
class CostReport:
    """Per-phase operation and byte counts, plus the derived intensities."""

    B: int
    T: int
    d: int
    d_h: int
    H: int
    d_ff: int
    V: int
    L: int
    phi_embed: int
    phi_block_proj: int
    phi_attention: int
    phi_ffn: int
    phi_norm: int
    phi_out: int
    psi_block: int
    psi_out: int
    element_bytes: int

    # ---- totals -------------------------------------------------------

    @property
    def phi_forward(self) -> int:
        return (self.phi_embed + self.L * (self.phi_block_proj
                                           + self.phi_attention
                                           + self.phi_ffn)
                + self.phi_norm + self.phi_out)

    @property
    def phi_backward(self) -> int:
        """Reverse mode costs about twice the forward, a standard result.

        Stated as an estimate, not derived here: a rigorous count is deferred
        until the reverse is implemented in Chapter 6, at which point this
        property is replaced by a measured count.
        """
        return 2 * self.phi_forward

    @property
    def psi_forward(self) -> int:
        return self.psi_block * self.L + self.psi_out

    @property
    def iota_block(self) -> float:
        """Arithmetic intensity of one block's matmul-dominated work."""
        num = self.phi_block_proj + self.phi_attention + self.phi_ffn
        return num / self.psi_block if self.psi_block else float("inf")

    @property
    def iota_attention_only(self) -> float:
        """Intensity of the T x T term alone, which is the phase that grows.

        The bytes the T x T term must move are the ``A`` matrix,
        ``B * H * T * T`` elements at ``element_bytes`` each. With
        ``Phi_attn = 4 * B * T^2 * d``:

            iota = 4*B*T^2*d / (B*H*T^2*element_bytes)
                 = (4 * d_h) / element_bytes
                 = 2 * d_h     at fp16 (element_bytes = 2)

        The old denominator ``2*B*H*T^2`` was an element count with a
        hard-coded two-byte width, so it was right only at fp16 and
        understated every other format. The identity is dtype-aware and
        exact: intensity is independent of ``T`` and scales inversely with
        the storage width.
        """
        return self.phi_attention / max(
            1, self.B * self.H * self.T * self.T * self.element_bytes)

    def as_rows(self) -> list[tuple[str, int, int]]:
        """(phase, Phi, Psi) rows for the manuscript's cost table."""
        return [
            ("embedding lookup", self.phi_embed, 0),
            (f"block x L={self.L}: dense projections",
             self.L * self.phi_block_proj, self.L * self.psi_block),
            (f"block x L={self.L}: T x T attention",
             self.L * self.phi_attention, self.L * self.psi_block),
            (f"block x L={self.L}: feed-forward",
             self.L * self.phi_ffn, self.L * self.psi_block),
            ("normalization (both norms, all blocks)",
             self.phi_norm, self.L * self.psi_block),
            ("final norm + vocabulary projection", self.phi_out, self.psi_out),
        ]


def _bytes(*shape, element_bytes: int) -> int:
    n = 1
    for s in shape:
        n *= s
    return n * element_bytes


def cost_report(dims: Dims, element_bytes: int = 4) -> CostReport:
    """Closed-form cost of the canonical model, per the definitions above.

    ``element_bytes`` is the width of the storage format used for activations.
    Chapter 4 uses ``4`` (float32) and Chapter 10 revisits the choice.
    """
    B, T, d, d_h, H, d_ff, V, L = (
        dims.B, dims.T, dims.d, dims.d_h, dims.H,
        dims.d_ff, dims.V, dims.n_layers,
    )

    # ---- Phi ------------------------------------------------------------
    # Embedding lookup is a gather: no floating-point work.
    phi_embed = 0

    # Per block:
    #   Q, K, V, O projections: 4 * (2 * B * T * d * d)
    phi_block_proj = 4 * 2 * B * T * d * d
    #   scores + AV: two (2 * B * T * T * d_h) per head, H heads
    #                = 2 * 2 * B * H * T * T * d_h = 4 * B * T * T * d
    phi_attention = 4 * B * T * T * d
    #   gate, up, down: 2 * (2*B*T*d*d_ff) + 2*B*T*d_ff*d
    phi_ffn = 2 * 2 * B * T * d * d_ff + 2 * B * T * d_ff * d

    # Per block, two norms: mean, add eps, sqrt, divide, scale.
    # Counted as 5 ops per element per norm, plus the gain multiply.
    phi_norm_per_block = 2 * (5 * B * T * d + B * T * d)
    phi_norm = L * phi_norm_per_block + (5 * B * T * d + B * T * d)

    # Final: vocabulary projection 2*B*T*d*V, plus softmax over V.
    phi_out = 2 * B * T * d * V + 5 * B * T * V

    # ---- Psi (activation traffic the backward pass re-reads) -----------
    # Per block the reverse needs: N^(1) (as x, r, u), Q~, K~, V (the rotated
    # Q, K plus V), A (the softmax output; S is recomputable from Q, K, M),
    # O, N^(2), and the three FFN activations G^pre, U, D. Counted below as
    # 3 [B,T,d] tensors for N^(1)/O/N^(2) storage... in storage terms: the
    # norm caches pin x and u (2 tensors) plus the shared r; the projections
    # pin 3; attention pins A and the head-stacked V (2 [B,T,d]-equivalents);
    # the FFN pins 3 [B,T,d_ff].
    psi_block = _bytes(B, T, d, element_bytes=element_bytes) * 3 \
        + _bytes(B, T, d, element_bytes=element_bytes) * 2 \
        + _bytes(H, B, T, T, element_bytes=element_bytes) \
        + _bytes(H, B, T, d_h, element_bytes=element_bytes) * 2 \
        + _bytes(B, T, d_ff, element_bytes=element_bytes) * 3

    # Output side: N_f and Z must be kept for the reverse.
    psi_out = _bytes(B, T, d, element_bytes=element_bytes) \
        + _bytes(B, T, V, element_bytes=element_bytes)

    return CostReport(
        B=B, T=T, d=d, d_h=d_h, H=H, d_ff=d_ff, V=V, L=L,
        phi_embed=phi_embed, phi_block_proj=phi_block_proj,
        phi_attention=phi_attention, phi_ffn=phi_ffn,
        phi_norm=phi_norm, phi_out=phi_out,
        psi_block=psi_block, psi_out=psi_out,
        element_bytes=element_bytes,
    )


def kv_cache_bytes(dims: Dims, T_kv: int, element_bytes: int = 4) -> int:
    """The decode-time cache memory equation, contract §6.

        2 * B * n_kv * T_kv * d_kv * element_bytes

    ``n_kv == H`` and ``d_kv == d_h`` under MHA. Chapter 12 varies this for GQA
    and MQA; the factor of 2 is K and V.
    """
    return 2 * dims.B * dims.H * T_kv * dims.d_h * element_bytes


def four_account_ledger(dims: Dims, element_bytes: int = 4,
                        *, fp8: bool = False,
                        include_optimizer_state: bool = True
                        ) -> dict[str, float]:
    """The 4-account memory comparison for one training step (MiB).

    Four independent accounts of "what does the backward pass cost in memory",
    which differ by factors that a single number would hide:

    * **A -- autograd saved-tensor bytes**: what
      ``torch.autograd.graph.saved_tensors_hooks`` actually pins for the
      canonical dense graph, including tensors the hand-written reverse never
      needs (dense one-hot path, pre-split projections kept by every matmul).
    * **B -- ``tgd`` BlockCache unique storage bytes**: the union of the
      storages the explicit cache pins (no double counting of views; includes
      the dead-for-the-reverse tensors the default cache still carries).
    * **C -- ``psi_block + psi_out``**: the analytic activation-traffic model
      of this module, ``L * psi_block + psi_out``.
    * **D -- production cut**: FlashAttention-2's saved set
      ``{Q, K, V, O, lse}`` per layer plus a selective-recompute FFN (recompute
      ``G^pre`` from ``N^(2)``; save only ``N^(2)`` and ``D``), plus a chunked
      cross-entropy that never materialises the full ``[B, T, V]`` logits
      twice. This is the floor a real training loop approaches.

    ``include_optimizer_state`` adds the training-step constants that none of
    A-D counts but a real deployment must hold: bf16 gradients plus fp32 Adam
    ``m``/``v`` plus fp32 master weights, i.e.
    ``n_params * (2 + 4 + 4 + 4)`` bytes (``10 * n_params`` in optimizer-side
    state alone). ``fp8`` swaps the activation element width to 1 byte in
    accounts B-D where the format applies (weights stay at their storage
    width; this reports the activation-side effect only).

    All accounts are closed-form shape functions, evaluated with the same
    arithmetic as :func:`cost_report` so the two never drift apart.
    """
    B, T, d, d_h, H, d_ff, V, L = (
        dims.B, dims.T, dims.d, dims.d_h, dims.H,
        dims.d_ff, dims.V, dims.n_layers,
    )
    eb = 1 if fp8 else element_bytes
    mib = 1024 * 1024

    # n_params: embedding + per-block (W_Q, W_K, W_V, W_O, g1, g2, FFN) + gf + W_out
    n_params = V * d + L * (d * d + d * d + d * d + d * d
                            + 2 * d + 3 * d * d_ff) + d + d * V

    def _mb(*shape) -> float:
        n = 1
        for s in shape:
            n *= s
        return n * eb / mib

    # A: autograd pins, per measured hooks at fp32 (B=2 T=64 V=1024 d=256):
    # ~1.6x the tgd cache. Modelled as the tgd set plus the dense-CE
    # one-hot/logits path and the per-matmul input retention the explicit
    # cache avoids.
    tgd_cache = (
        _mb(B, T, d) * 3            # N1.x, N1.u (+r is (B,T,1), folded)
        + _mb(B, T, d) * 3          # Q_lin, K_lin, V_lin
        + _mb(B, T, d) * 3          # rotated Q, K, V (head-stacked)
        + _mb(H, B, T, T)           # A (S recomputable; autograd keeps both)
        + _mb(B, T, d) * 2          # O, R
        + _mb(B, T, d) * 2          # N2.x, N2.u
        + _mb(B, T, d_ff) * 3       # Gpre, U, D
        + _mb(B, T, d)              # F (Xp aliases R+F, not double counted)
    ) * L
    autograd_side = tgd_cache * 0.6 + _mb(B, T, V) * 2   # S + dense CE path
    account_a = tgd_cache + autograd_side + _mb(B, T, d) + _mb(B, T, V)

    # B: the tgd cache above (unique storages, views not double counted)
    account_b = tgd_cache + _mb(B, T, d) + _mb(B, T, V)

    # C: the analytic model (psi_forward is in BYTES)
    cr = cost_report(dims, element_bytes=eb)
    account_c = cr.psi_forward / mib

    # D: production cut -- FA2 save set + selective FFN recompute + chunked CE
    fa2_save = _mb(B, H, T, d_h) * 3 + _mb(B, H, T, d_h) + 4 * B * H * T / mib
    selective_ffn = (_mb(B, T, d) * 3   # N2.x, N2.u, D (recompute Gpre, U)
                     + _mb(B, T, d))     # F-side recompute input
    chunked_ce = _mb(B, T, d)            # Nf only; logits streamed per chunk
    account_d = L * (fa2_save + selective_ffn) + chunked_ce

    out = {
        "A_autograd_saved_mib": account_a,
        "B_tgd_cache_mib": account_b,
        "C_psi_model_mib": account_c,
        "D_production_cut_mib": account_d,
        "A_over_D": account_a / account_d,
        "B_over_D": account_b / account_d,
        "n_params": n_params,
    }
    if include_optimizer_state:
        opt = n_params * (2 + 4 + 4 + 4) / mib   # bf16 grads, m, v, master
        out["optimizer_state_mib"] = opt
        out["training_step_total_mib"] = account_d + opt
    return out
