"""The closed dimension set — notation contract §1.

Contract §1 exempts these symbols from the "scalars are lowercase" law. This
exemption is deliberately finite and enumerated here, so that adding a new
uppercase scalar is a visible act rather than an accident.

Collision reminders enforced by the assertions below (from contract §7):
  * ``H`` is the head count and only the head count; the SwiGLU gated product is
    ``D``;
  * ``L`` is the block count; the loss is calligraphic and lives in ``loss.py``;
  * ``d_h = d / H`` is the per-head width, and the attention scale factor uses
    it and never ``d`` (contract C2).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Dims:
    """Dimensions of the canonical decoder-only Transformer.

    Attributes mirror contract §1 exactly. ``L`` is the number of blocks; the
    loss is not represented here and is never spelled ``L``.
    """

    B: int = 1          # batch size
    T: int = 4          # sequence length
    V: int = 7          # vocabulary size
    d: int = 4          # model (residual-stream) width
    H: int = 1          # number of attention heads  (reserved: this is all H means)
    d_ff: int = 8       # feed-forward width
    n_layers: int = 1   # contract calls this L; exposed as n_layers to keep the
                        # loss symbol L free (see module docstring)
    eps: float = 1e-6   # RMSNorm epsilon, contract C3
    rope_base: float = 10000.0

    def __post_init__(self) -> None:
        if self.d % self.H != 0:
            raise ValueError(
                f"d={self.d} must be divisible by H={self.H} "
                "(contract: d_h = d / H)"
            )
        if self.d_ff <= 0:
            raise ValueError(f"d_ff must be positive, got {self.d_ff}")
        if self.n_layers < 1:
            raise ValueError(f"need at least one block, got {self.n_layers}")
        if self.T < 1 or self.V < 1:
            raise ValueError("T and V must be positive")

    # ---- derived dimensions (never stored, never overridable) -----------

    @property
    def d_h(self) -> int:
        """Per-head width. The attention scale is 1/sqrt(d_h), not 1/sqrt(d)."""
        return self.d // self.H

    @property
    def n_sup(self) -> int:
        """Supervised positions for an all-supervised batch (contract C5)."""
        return self.B * self.T

    @property
    def L(self) -> int:
        """Number of blocks, for display. Never use this for the loss."""
        return self.n_layers

    # ---- parameter accounting (contract §2, parameter registry) --------

    @property
    def per_block_params(self) -> int:
        """Q, K, V, O projections; gate, up, down; two norm gains."""
        return 4 * self.d * self.d + 2 * self.d * self.d_ff + self.d_ff * self.d \
            + 2 * self.d

    @property
    def n_params(self) -> int:
        """Total trainable parameters: embedding, blocks, final gain, unembedding."""
        return (
            self.V * self.d                      # E
            + self.n_layers * self.per_block_params
            + self.d                             # g_f
            + self.d * self.V                    # W_out
        )

    def __str__(self) -> str:  # pragma: no cover - display only
        return (
            f"Dims(B={self.B}, T={self.T}, V={self.V}, d={self.d}, H={self.H}, "
            f"d_h={self.d_h}, d_ff={self.d_ff}, L={self.n_layers})"
        )


#: Chapter 6's declared one-head, one-block specialization (contract C7).
CH6_TINY = Dims(B=1, T=4, V=7, d=4, H=1, d_ff=8, n_layers=1)

#: Chapter 8's two-head transfer case: same idea, wider, so H > 1 is exercised.
CH8_TWO_HEAD = Dims(B=1, T=4, V=7, d=8, H=2, d_ff=8, n_layers=1)

#: Chapter 7's training family: four heads, four blocks.
CH7_TRAIN = Dims(B=16, T=24, V=32, d=48, H=4, d_ff=96, n_layers=4)
