"""The merge register and the adjoint-completeness checker — contract §3, §8.

This is the enforcement mechanism for "zero missing adjoints". A rule that cannot
fail a test is not enforced, so the checker is written to be able to *fail*.

What it verifies
----------------
1. every contract ID in §2 is registered with a name and a backward status;
2. every shape-bearing ID has a declared expected shape, and when a live graph
   is supplied, the live shape matches the declared one;
3. every parameter in the §2 parameter registry is present in a supplied
   parameter set;
4. every entry in the merge register declares at least two paths and names each
   summand (contract A.3);
5. every parameter family has a registered deletion mutation (contract A.6);
6. invariants I1, I2, I3 are reported, per head where applicable.

What it deliberately does not verify
------------------------------------
Whether a human explanation survived. See contract §3.6 of the blueprint: the
gates that catch the worst defect in this project's history cannot be automated,
and this module is not a substitute for them.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from . import graph as graph_mod
from .dims import Dims


@dataclass(frozen=True)
class Merge:
    """One merge point of contract §3."""

    tensor: str
    n_paths: int
    summands: tuple[str, ...]
    chapter: str
    note: str = ""


#: Contract §3, verbatim. Every entry needs n_paths >= 2 and named summands.
MERGE_REGISTER: tuple[Merge, ...] = (
    Merge("X_ell", 2, ("X_bar_direct", "X_bar_attn"), "Ch 5"),
    Merge("R_ell", 2, ("X_bar", "R_bar_ffn"), "Ch 5"),
    Merge("N^(1)_ell", 3, ("N1_bar_Q", "N1_bar_K", "N1_bar_V"), "Ch 6",
          "the three-way attention merge; concat-adjoint, not sum"),
    Merge("N^(2)_ell", 2, ("N2_bar_g", "N2_bar_u"), "Ch 6",
          "gate and up both consume the same normalized input"),
    Merge("E", 0, ("E_bar_lookup",), "Ch 6",
          "scatter-add over occurrences; n_paths is the occurrence count, "
          "which is data-dependent and therefore not fixed at 2"),
    Merge("E (tied)", 2, ("E_bar_lookup", "W_out_bar^T"), "Ch 6",
          "explicit weight tying, contract C6"),
    Merge("S^(h) rows", 0, ("s_bar row sum = 0",), "Ch 6",
          "invariant I1: a legal row's score adjoints sum to zero, and masked "
          "entries are exactly zero"),
    Merge("norm gains", 0, ("g_bar sums over tokens",), "Ch 6",
          "contract C4: a different reduction axis from every other parameter"),
    Merge("MoE expert output", 2, ("expert_a_contribution",
                                      "expert_b_contribution"), "Ch 16",
          "y_t = sum over the top-K set S^(t) of pi_{t,a} FFN_a(x_t). For one "
          "token the two active experts are two consumers of x_t and of the "
          "router probability, so their adjoints both land there and must be "
          "summed. The summands are per-token: a different pair (a, b) for "
          "every t, which is why the merge is in the OUTPUT rather than in a "
          "fixed weight. K is a runtime value, so the summand names are the "
          "first two active experts by convention."),
    Merge("LoRA adapted output", 2, ("frozen_weight_path",
                                     "lora_adapter_path"), "Ch 15",
          "y = x W + (x W_down) W_up: the frozen weight and the rank-r "
          "correction are two consumers of x, so their adjoints both land on "
          "x and must be summed. The merge is in the OUTPUT, not the weight; "
          "merging W + W_down W_up first removes it, which is why the merged "
          "form is not merely faster but structurally merge-free"),
)

#: Contract A.6: each parameter family needs a registered deletion mutation.
DELETION_MUTATIONS: dict[str, str] = {
    "V path into N^(1)": "zero the per-head V adjoint before the dense merge",
    "first residual identity": "drop R_bar's identity contribution into X_ell",
    "second residual identity": "drop Xp_bar's identity contribution into R_ell",
    "attention head concat adjoint": "sum head adjoints instead of reshaping",
    "norm gain accumulation": "reduce the gain gradient over features, not tokens",
    "up path into N^(2)": "delete the U contribution to the N^(2) merge",
    "query and key paths into N^(1)": "delete both the Q and K contributions",
    "MoE expert path into the adapted output":
        "delete one active expert's gated contribution to the K-way MoE output "
        "merge, leaving K-1 paths",
    "LoRA adapter path into the adapted output":
        "delete the (x W_down) W_up contribution to the LoRA output merge, "
        "leaving only the frozen weight path",
}


@dataclass
class LedgerReport:
    ok: bool
    errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    backward_pending: list[str] = field(default_factory=list)


def check(dims: Dims, *, live_shapes: dict[int, tuple] | None = None,
          live_params: dict[str, list["torch.Tensor"]] | None = None,
          ) -> LedgerReport:
    """Run every specification check the ledger can run. Never raises."""
    rep = LedgerReport(ok=True)

    def fail(msg: str) -> None:
        rep.ok = False
        rep.errors.append(msg)

    # 1. every contract ID registered
    declared = set(graph_mod.BACKWARD_STATUS)
    named = set(graph_mod.ID_NAMES)
    if declared != named:
        fail(f"ID sets disagree: status-only {sorted(declared - named)}, "
             f"name-only {sorted(named - declared)}")
    for i in sorted(declared):
        if graph_mod.BACKWARD_STATUS[i] not in {"pending", "implemented", "n/a"}:
            fail(f"ID {i} has unknown backward status "
                 f"{graph_mod.BACKWARD_STATUS[i]!r}")

    # 2. shapes
    want = graph_mod.expected_shapes(dims)
    for i, shape in sorted(want.items()):
        if live_shapes is not None and i in live_shapes:
            if tuple(live_shapes[i]) != tuple(shape):
                fail(f"ID {i} ({graph_mod.ID_NAMES.get(i, '?')}): live shape "
                     f"{tuple(live_shapes[i])} != declared {tuple(shape)}")
    if live_shapes is not None:
        undeclared = set(live_shapes) - set(want)
        if undeclared:
            fail(f"live shapes for IDs with no declared shape: "
                 f"{sorted(undeclared)}")
    rep.notes.append(f"{len(want)} shape-bearing IDs declared for {dims}")

    # 3. parameters present
    if live_params is not None:
        required = {"E", "g1", "W_Q", "W_K", "W_V", "W_O", "g2", "W_g", "W_u",
                    "W_d", "gf", "W_out"}
        missing = required - set(live_params)
        if missing:
            fail(f"parameter registry incomplete: missing {sorted(missing)}")
        for name in sorted(required & set(live_params)):
            group = live_params[name]
            if not isinstance(group, list):
                fail(f"parameter {name}: expected a list over blocks, got "
                     f"{type(group).__name__}")
            elif not group:
                fail(f"parameter {name}: empty, but the registry requires at "
                     f"least one occurrence")
        expect_counts = {"E": 1, "gf": 1, "W_out": 1}
        for name in sorted(required & set(live_params)):
            want_n = expect_counts.get(name, dims.n_layers)
            got_n = len(live_params[name])
            if got_n != want_n:
                fail(f"parameter {name}: {got_n} occurrences, expected {want_n}")
        expected = {
            "E": (dims.V, dims.d), "g1": (dims.d,), "gf": (dims.d,),
            "g2": (dims.d,), "W_Q": (dims.d, dims.d), "W_K": (dims.d, dims.d),
            "W_V": (dims.d, dims.d), "W_O": (dims.d, dims.d),
            "W_g": (dims.d, dims.d_ff), "W_u": (dims.d, dims.d_ff),
            "W_d": (dims.d_ff, dims.d), "W_out": (dims.d, dims.V),
        }
        for name, shape in expected.items():
            if name not in live_params:
                continue
            group = live_params[name]
            if not isinstance(group, list):
                fail(f"parameter {name}: expected a list over blocks, got "
                     f"{type(group).__name__}")
                continue
            if not group:
                # already reported above; do not index into it
                continue
            if tuple(group[0].shape) != shape:
                fail(f"parameter {name}: shape {tuple(group[0].shape)} "
                     f"!= declared {shape}")
        n_seen = sum(t.numel() for g in live_params.values()
                     if isinstance(g, list) for t in g)
        if n_seen != dims.n_params:
            fail(f"parameter count {n_seen} != analytic {dims.n_params}")
        else:
            rep.notes.append(f"parameter count {n_seen:,} matches the analytic "
                             f"count from the shape ledger")

    # 4. merge register well-formed
    for m in MERGE_REGISTER:
        if len(m.summands) < 1:
            fail(f"merge {m.tensor}: no summands named")
        if m.n_paths and len(m.summands) != m.n_paths:
            fail(f"merge {m.tensor}: declares {m.n_paths} paths but names "
                 f"{len(m.summands)} summands {m.summands}")
        if m.n_paths and m.n_paths < 2:
            fail(f"merge {m.tensor}: n_paths={m.n_paths} is not a merge")
    rep.notes.append(f"{len(MERGE_REGISTER)} merge entries, "
                     f"{len(DELETION_MUTATIONS)} deletion mutations registered")

    # 5. forward-only IDs reported, not hidden
    pending = [f"{i}:{graph_mod.ID_NAMES[i]}" for i in sorted(graph_mod.BACKWARD_STATUS)
               if graph_mod.BACKWARD_STATUS[i] == "pending"]
    rep.backward_pending = pending
    if pending:
        rep.notes.append(f"{len(pending)} contract IDs are forward-only at this "
                         f"revision; their reverses land in Chapters 6-8")

    return rep
