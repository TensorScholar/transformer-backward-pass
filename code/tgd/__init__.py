"""From Gradients to Generation — shared canonical implementation.

This package is the **single** canonical implementation of the decoder forward
and reverse graph. Verification scripts import from here rather than restating
the forward graph.

Rationale, which is a hard requirement rather than a preference: earlier
revisions of this project carried three independent copies of the forward graph
(manuscript spec, Chapter 6 script, Chapter 7 harness) and they disagreed about
whether a final RMSNorm existed. That single disagreement was the root of the
entire P0 defect class. One importable implementation makes it unrepresentable.

Modules
-------
dims       the closed dimension set from contract §1
rope       row-vector rotary position embedding, forward and reverse
rmsnorm    RMSNorm forward and reverse, epsilon per contract C3
attention  head split, concat, causal masking, masked-softmax VJP
swiglu     gate, up, SiLU, gated product
residual   both residual additions and the identity adjoint (invariant I3)
loss       log-softmax, masked-mean cross-entropy, tied-weight transpose
graph      the canonical forward graph, shape-asserted
costs      Phi, Psi, arithmetic intensity from shapes alone
ledger     the merge register and adjoint-completeness checker
"""
from __future__ import annotations

__all__ = [
    "dims", "rope", "rmsnorm", "attention", "swiglu", "residual", "loss",
    "graph", "costs", "ledger",
]

__version__ = "1.0"
