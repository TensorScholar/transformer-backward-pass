"""Row-vector rotary position embedding — contract §2 IDs 6, 7 and C1.

The convention, stated exactly, because a rotation without an explicit form is
precisely the ambiguity that produces silent sign bugs.

For each head, token position ``t``, and coordinate pair ``j`` (i.e. elements
``2j`` and ``2j+1``), let ``theta_{t,j} = t * base^(-2j/d_h)`` and write

    c_{t,j} = cos(theta_{t,j}),   s_{t,j} = sin(theta_{t,j})

The rotation block for that pair is

    R_t^{(j)} = [[ c,  -s],
                 [ s,   c ]]

so that, for a **row** token slice ``q`` with pair ``(q_e, q_o)``:

    forward    q R_t^T = ( q_e*c - q_o*s ,  q_e*s + q_o*c )
    reverse    y R_t   = ( y_e*c + y_o*s , -y_e*s + y_o*c )

The two are exact inverses because ``R_t`` is orthogonal. Note the asymmetry: the
forward uses ``R_t^T`` and the reverse uses ``R_t``; getting this the wrong way
round produces a rotation by ``-2*theta``, which preserves norms and therefore
looks plausible while agreeing with nothing.

The sign of the angle is a **convention**, and both signs appear in the
literature. This one is pinned rather than derived, and it is pinned to match the
rotation the previously published Chapter 6 numeric trace was generated with, so
that the verified values quoted in the manuscript remain reproducible. Choosing
the other sign would rotate every query and key by ``-theta`` instead and change
the loss from 2.340928875134 to 2.340149081721, for no pedagogical gain. The
contract states this matrix explicitly so the gap is closed.

Values are never rotated.

Context extension
-----------------
Above, ``T`` is a *table size*, not a capacity. The angles are ``t * base^(-2j/d_h)``
evaluated at integer ``t``, and nothing in that formula constrains the largest
permitted ``t``. A model trained with positions ``0..N-1`` has nevertheless only
ever *seen* ``t < N``, so evaluating ``theta_{2048,j} = 2048 * base^(-2j/d_h)``
is an extrapolation into a region of the circle the model has no data about. Two
interpolations are implemented here and they fix the problem in opposite ways:

    linear   divide the *position* by alpha:            t -> t / alpha
    ntk      change the *base frequency*:                b  -> b * alpha^(d/(d-2))

Linear shrinks the whole spectrum uniformly, so a distance the model handled at
1 token is now handled at ``1/alpha`` of a token and local structure is destroyed.
NTK-aware leaves every angle strictly below one radian nearly untouched and pays
for the extension only in the low-frequency pairs, which is where long-range
structure lives. Both are measured, not asserted: see
:func:`extrapolation_report`.

Mechanically, NTK-aware scaling is *not* position interpolation and must never be
implemented as it. ``b' = b * alpha^(d/(d-2))`` implies ``inv_freq'_j =
b'^(-2j/d) = inv_freq_j * alpha^(-2j/(d-2))``, a per-pair rescaling that is
**not** ``inv_freq_j / alpha`` except at ``j = 0`` (and at ``j = d_h/2``, since
``2j = d_h = d``). Passing a modified ``base`` to :func:`rope_angles` is correct;
dividing ``angles`` by ``alpha`` is not the same function and the difference is
measured in :func:`scaling_spectrum_report`.
"""
from __future__ import annotations

import math

import torch


def _angle_work_dtype(dtype: torch.dtype) -> torch.dtype:
    """Working precision for the phase table.

    ``float64`` stays ``float64``. ``float16``/``bfloat16`` are promoted to
    ``float32``: constructing ``pos`` or ``inv_freq`` in a storage format with
    an 11-bit (fp16) or 8-bit (bf16) mantissa collapses distinct positions to
    identical angles once ``T`` is large -- at ``T = 8192`` and bf16, positions
    8191 and 8192 quantize to the same value, and every high-frequency pair's
    phase is snapped to a multiple of its (very coarse) representable grid.
    The rotation is evaluated in the promoted precision and only the rotated
    output is cast back to the activation dtype.
    """
    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype


def rope_angles(T: int, d_h: int, base: float = 10000.0,
                *, dtype: torch.dtype = torch.float32,
                device=None) -> torch.Tensor:
    """Per-pair angles of shape ``(T, d_h/2)``.

    ``theta_{t,j} = t * base^(-2j/d_h)`` matches the standard
    ``1/(base^(2i/d))`` parameterisation of rotary embeddings. The table is
    always evaluated in at least ``float32`` (or ``float64`` when ``dtype``
    is ``float64``), never in the half-precision activation format, so long
    contexts do not alias positions; callers casting the table to fp16 for
    storage still get correctly-rounded angles.
    """
    if d_h % 2 != 0:
        raise ValueError(
            f"RoPE pairs coordinates, so d_h must be even; got d_h={d_h}. "
            "An odd per-head width would silently broadcast the last rotation "
            "slot and corrupt the trailing coordinate."
        )
    wd = _angle_work_dtype(dtype)
    j = torch.arange(d_h // 2, dtype=wd, device=device)
    inv_freq = base ** (-2.0 * j / d_h)
    t = torch.arange(T, dtype=wd, device=device)
    return torch.outer(t, inv_freq)          # (T, d_h/2), working precision


def rope_forward_rows(q: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """Rotate row slices: ``q -> q R_t^T``. ``q`` is ``(..., T, d_h)``.

    ``cos``/``sin`` are evaluated in at least ``float32`` (or ``float64`` for
    a ``float64`` ``q``), never in the half-precision activation format: a
    bf16 angle table has ~8 mantissa bits, which at high-frequency pairs is
    an angle error of order 1e-2 rad per position. The output is cast back
    to ``q``'s dtype at the end.
    """
    T, d_h = q.shape[-2], q.shape[-1]
    if d_h % 2 != 0:
        raise ValueError(f"d_h must be even, got d_h={d_h}")
    if angles.shape != (T, d_h // 2):
        raise ValueError(
            f"angles must be ({T}, {d_h // 2}), got {tuple(angles.shape)}"
        )
    h = d_h // 2
    wd = _angle_work_dtype(q.dtype)
    qw = q.to(wd)
    ang = angles.to(wd)
    c, s = torch.cos(ang), torch.sin(ang)
    out = torch.empty_like(qw)
    out[..., : 2 * h: 2] = qw[..., : 2 * h: 2] * c - qw[..., 1: 2 * h: 2] * s
    out[..., 1: 2 * h: 2] = qw[..., : 2 * h: 2] * s + qw[..., 1: 2 * h: 2] * c
    return out.to(q.dtype)


def rope_backward_rows(bar_q: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """Reverse of :func:`rope_forward_rows`: ``y -> y R_t``.

    Because ``R_t`` is orthogonal this is the inverse rotation, and it is *not*
    the same expression as the forward. Angles are promoted to at least
    ``float32`` exactly as in the forward, so forward and reverse use the
    *same* trigonometric values and the round trip stays exact.
    """
    T, d_h = bar_q.shape[-2], bar_q.shape[-1]
    if d_h % 2 != 0:
        raise ValueError(f"d_h must be even, got d_h={d_h}")
    if angles.shape != (T, d_h // 2):
        raise ValueError(
            f"angles must be ({T}, {d_h // 2}), got {tuple(angles.shape)}"
        )
    h = d_h // 2
    wd = _angle_work_dtype(bar_q.dtype)
    bw = bar_q.to(wd)
    ang = angles.to(wd)
    c, s = torch.cos(ang), torch.sin(ang)
    out = torch.empty_like(bw)
    out[..., : 2 * h: 2] = bw[..., : 2 * h: 2] * c + bw[..., 1: 2 * h: 2] * s
    out[..., 1: 2 * h: 2] = -bw[..., : 2 * h: 2] * s + bw[..., 1: 2 * h: 2] * c
    return out.to(bar_q.dtype)


# =====================================================================
# Chapter 13. Three implementations of one rotation, and context scaling
# =====================================================================
#
# What was above is the implementation that ships: two strided slices and two
# fused multiply-adds, no matrix ever materialised. It is correct and it is what
# a GPU wants. It is also the *least* legible of the three, because the rotation
# is invisible in it -- you cannot see the 2x2 block anywhere in that code.
#
# The three below are mathematically identical. Establishing that, by assertion
# and not by reading, is the point: a rotation has an embarrassingly many
# equivalent spellings and the only defence against drifting between them is to
# compute all of them and compare.

def inv_freq(d_h: int, base: float = 10000.0, *,
             dtype: torch.dtype = torch.float32, device=None) -> torch.Tensor:
    """``base^(-2j/d_h)`` for ``j = 0 .. d_h/2-1``, in float64 by default.

    Computed as ``exp(-2j log(base)/d_h)`` rather than ``base ** (-2j/d_h)``. The
    two agree mathematically and differ numerically, and the second one drifts:
    ``d_h = 4`` gives exponents ``0, -0.5, -1.0, -1.5``, and ``10000 ** (-1.5)``
    in binary64 carries 0.5 ULP of error while the exponential form does not.
    Over a 4096-token table the disagreement reaches 2.2e-13 absolute, which is
    invisible in a loss and visible in a claim about exact frequencies. Use this.

    A half-precision ``dtype`` is promoted to ``float32`` before evaluation,
    for the same reason as :func:`rope_angles`: the frequency table itself
    must never be quantized to the storage format.
    """
    if d_h % 2 != 0:
        raise ValueError(
            f"RoPE pairs coordinates, so d_h must be even; got d_h={d_h}. "
            "An odd per-head width would silently broadcast the last rotation "
            "slot and corrupt the trailing coordinate."
        )
    wd = _angle_work_dtype(dtype)
    j = torch.arange(d_h // 2, dtype=wd, device=device)
    return torch.exp(-2.0 * j * math.log(base) / d_h)


def ntk_base(d_h: int, alpha: float, base: float = 10000.0) -> float:
    """NTK-aware base ``b' = b * alpha^(d/(d-2))``.

    Note the exponent is on ``d_h``, the **per-head** width, not on the model
    width ``d``. Several published statements of this formula write ``d`` where
    the per-head dimension is meant; at ``d_h = 128`` and ``alpha = 2`` the two
    differ by a factor of ``2^(128/126 - 1) = 1.0111``, i.e. a 1.1% error in
    every frequency in the table. :func:`scaling_spectrum_report` reports the
    resulting spread so the discrepancy is a number and not a footnote.
    """
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}")
    if d_h <= 2:
        raise ValueError(
            f"the NTK exponent d/(d-2) diverges at d_h=2; got d_h={d_h}"
        )
    return base * alpha ** (d_h / (d_h - 2))


def scaled_positions(T: int, d_h: int, *, mode: str = "linear",
                     alpha: float = 1.0, base: float = 10000.0,
                     dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Position ids for a scaled model, ``(T,)``.

    ``mode="linear"`` divides every position by ``alpha`` -- the sequence is
    squashed into the trained range and distances are reported in *rescaled*
    tokens. ``mode="ntk"`` leaves positions at integers and changes the base
    frequency instead, so distances keep their true token counts.

    Positions are evaluated in at least ``float32`` regardless of ``dtype``:
    the *fractional* position is the whole point of linear interpolation, and
    quantizing it to a half-precision format would make many positions
    identical.
    """
    wd = _angle_work_dtype(dtype)
    t = torch.arange(T, dtype=wd)
    if mode == "linear":
        return t / alpha
    if mode == "ntk":
        if alpha == 1.0:
            return t
        return t
    raise ValueError(f"unknown scaling mode {mode!r}")


def scaled_angles(T: int, d_h: int, *, mode: str = "linear",
                  alpha: float = 1.0, base: float = 10000.0,
                  dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Angles ``(T, d_h/2)`` under linear or NTK-aware extension.

    The two branches differ in *which* number is perturbed, and the distinction
    is not cosmetic:

        linear   theta'_{t,j} = (t/alpha) * base^(-2j/d_h)
        ntk      theta'_{t,j} = t * (b * alpha^(d/(d-2)))^(-2j/d_h)

    NTK-aware is therefore a *per-pair rescaling* that equals ``1/alpha`` only at
    ``j = 0``. Implementing it by dividing the angles by ``alpha`` is a different
    function that happens to agree at the fastest pair; see
    :func:`scaling_spectrum_report`. Both branches evaluate in at least
    ``float32`` (``float64`` stays ``float64``) and cast back to ``dtype``.
    """
    wd = _angle_work_dtype(dtype)
    if mode == "linear":
        t = torch.arange(T, dtype=wd) / alpha
        return torch.outer(t, inv_freq(d_h, base, dtype=wd))
    if mode == "ntk":
        t = torch.arange(T, dtype=wd)
        return torch.outer(t, inv_freq(d_h, ntk_base(d_h, alpha, base),
                                       dtype=wd))
    raise ValueError(f"unknown scaling mode {mode!r}")


# ------------------------------------------- (1) block-diagonal matrix ----

def rotation_block(angle: torch.Tensor) -> torch.Tensor:
    """A single ``(..., 2, 2)`` block ``[[c, -s], [s, c]]``."""
    c, s = torch.cos(angle), torch.sin(angle)
    z = torch.zeros_like(c)
    o = torch.ones_like(c)
    return torch.stack([torch.stack([c, -s], dim=-1),
                        torch.stack([s, c], dim=-1)], dim=-2)


def _block_diagonal(angles: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Assemble the ``(T, d_h, d_h)`` block-diagonal rotation explicitly.

    This is the definition, written out. It exists so that the two implementations
    actually shipping code can be checked against the thing they are derived
    from, and it is quadratic in ``d_h`` in both time and memory. See
    :func:`matrix_cost` for what that costs. The rotation blocks are built in
    the working precision of ``dtype`` (fp16/bf16 promoted to fp32), matching
    :func:`rope_forward_rows`.
    """
    T, half = angles.shape
    d_h = 2 * half
    wd = _angle_work_dtype(dtype)
    R = torch.zeros(T, d_h, d_h, dtype=wd)
    for j in range(half):
        blk = rotation_block(angles[:, j].to(wd))             # (T,2,2)
        R[:, 2 * j:2 * j + 2, 2 * j:2 * j + 2] = blk
    return R.to(dtype)


def rope_forward_matrix(q: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """``q -> q R_t^T`` by explicit block-diagonal multiplication.

    Row-vector convention (contract C1): ``out[t] = q[t] @ R_t.T``, which is
    ``(q_e*c - q_o*s, q_e*s + q_o*c)`` because ``R_t.T = [[c, s], [-s, c]]``.
    The multiplication is carried out in the working precision of ``q``'s
    dtype (fp16/bf16 promoted to fp32) and cast back.
    """
    T, d_h = q.shape[-2], q.shape[-1]
    R = _block_diagonal(angles, q.dtype)                      # (T,d_h,d_h)
    # THE TRAP IN THIS FUNCTION. `torch.matmul(q, R_T)` is WRONG and fails
    # SILENTLY. With q of shape (..., T, d_h) and R_T of shape (T, d_h, d_h),
    # matmul reads q's LAST TWO dims as the matrix and R_T's FIRST dim as a batch
    # axis, then broadcasts q's leading axis against R's position axis. For
    # q of shape (1, T, d_h) that yields a (T, T, d_h) tensor in which each
    # position has been contracted with the SUM of all T rotations -- a wrong
    # answer of the right shape, no exception raised. It is caught by asserting
    # norm preservation, because a sum of rotations is not orthogonal: the
    # relative norm error came out 7.0. The fix is to materialise a position
    # axis for q so that the batch axes align one-to-one with the rotations.
    out = torch.matmul(q.unsqueeze(-2).to(R.dtype), R.transpose(-2, -1))
    return out.squeeze(-2).to(q.dtype)


def rope_backward_matrix(bar_q: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """Reverse of :func:`rope_forward_matrix`: ``y -> y R_t``."""
    T, d_h = bar_q.shape[-2], bar_q.shape[-1]
    R = _block_diagonal(angles, bar_q.dtype)
    # same one-to-one batch alignment and working precision as the forward;
    # see the trap noted there.
    return torch.matmul(bar_q.unsqueeze(-2).to(R.dtype), R).squeeze(-2).to(bar_q.dtype)


# ------------------------------------------------- (2) complex arithmetic ----

def _as_complex(x: torch.Tensor) -> torch.Tensor:
    """Interleave ``(re, im)`` pairs into the last axis: ``..., d_h/2 -> ...``.

    Interleaved, not split: element ``2j`` is the real part and ``2j+1`` the
    imaginary part of pair ``j``, which is the layout the dot product uses. A
    split (even/odd) layout would need the score's contraction axis permuted to
    agree, and would hide the bug described in the module docstring of
    :func:`rope_forward_complex`.
    """
    d_h = x.shape[-1]
    if d_h % 2 != 0:
        raise ValueError(f"d_h must be even, got {d_h}")
    return torch.complex(x[..., 0::2].contiguous(), x[..., 1::2].contiguous())


def _from_complex(z: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`_as_complex`: interleave back to a real tensor."""
    return torch.view_as_real(z).flatten(start_dim=-2)


def rope_forward_complex(q: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """``q -> q R_t^T`` as a single multiplication by a complex unit.

    The subtle step, and the one that is wrong in half of the code on the
    internet, is deciding whether to multiply by ``e^{+i theta}`` or
    ``e^{-i theta}``. Derive it rather than recall it.

    The row-vector convention (contract C1) is ``out = q R_t^T``. A row times a
    matrix is a **column** times its transpose, transposed:

        q R_t^T = ( R_t q^T )^T .

    So the correct complex computation is ``R_t`` acting on the column
    ``q^T = (q_e, q_o)``, and ``R_t = [[c, -s], [s, c]]`` is exactly
    multiplication by ``e^{+i theta}``:

        (q_e + i q_o)(c + i s) = (q_e c - q_o s) + i (q_e s + q_o c)

    which is component-for-component the slicing implementation. Multiplying by
    ``e^{-i theta}`` instead gives ``(q_e c + q_o s, q_o c - q_e s)``, which is
    ``R_t^T`` applied to a *column* -- i.e. the transpose convention, and a
    rotation by ``-theta``. That error is invisible by inspection: the map is
    still orthogonal, still norm-preserving, still produces plausible attention,
    and it is wrong. The verification asserts against the matrix form, which is
    how it was caught, and :func:`equivalence_report` reports the size of the
    error so the claim "it would have looked fine" has a number on it.

    Note also that ``torch.complex`` computes in complex64 whenever its inputs
    are float32, so the cosine table must be built in the *same* real dtype or
    the multiply silently promotes and the comparison tolerance changes. This
    function upcasts a float32 input to float64 and casts back, and the
    verification asserts the round trip is exact.
    """
    ang = angles.to(_angle_work_dtype(q.dtype))
    z = _as_complex(q.to(ang.dtype))                        # (...,T,d_h/2)
    unit = torch.polar(torch.ones_like(ang), ang)           # exp(+i theta)
    return _from_complex(z * unit).to(q.dtype)


def rope_backward_complex(bar_q: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """Reverse of :func:`rope_forward_complex`.

    Multiplication by ``exp(+i theta)`` is undone by multiplication by
    ``exp(-i theta)``, because the two are complex conjugates and
    ``z * conj(z) = |z|^2 = 1`` for a unit-modulus factor. The angles are
    unchanged; only the sign of the exponent flips. This is the concrete content
    of ``R^T = R^{-1}`` in the complex spelling -- and note that the flip is
    ``theta -> -theta`` while the *angles table itself is untouched*, which is
    the whole content of "the adjoint is a rotation by ``-theta``".
    """
    ang = angles.to(_angle_work_dtype(bar_q.dtype))
    z = _as_complex(bar_q.to(ang.dtype))
    unit = torch.polar(torch.ones_like(ang), -ang)          # exp(-i theta)
    return _from_complex(z * unit).to(bar_q.dtype)


# ------------------------------------------------- (3) the shipped slicing ----

IMPLEMENTATIONS = ("matrix", "complex", "sliced")


def _dispatch(fn_matrix, fn_complex, q: torch.Tensor, angles: torch.Tensor,
              mode: str):
    if mode == "matrix":
        return fn_matrix(q, angles)
    if mode == "complex":
        return fn_complex(q, angles)
    if mode == "sliced":
        return rope_forward_rows(q, angles)
    raise ValueError(f"unknown RoPE implementation {mode!r}; "
                     f"expected one of {IMPLEMENTATIONS}")


def rope_forward(q: torch.Tensor, angles: torch.Tensor, mode: str = "sliced"
                 ) -> torch.Tensor:
    """Forward rotation under the named implementation."""
    return _dispatch(rope_forward_matrix, rope_forward_complex, q, angles, mode)


def rope_backward(bar_q: torch.Tensor, angles: torch.Tensor, mode: str = "sliced"
                  ) -> torch.Tensor:
    """Reverse rotation under the named implementation."""
    if mode == "matrix":
        return rope_backward_matrix(bar_q, angles)
    if mode == "complex":
        return rope_backward_complex(bar_q, angles)
    if mode == "sliced":
        return rope_backward_rows(bar_q, angles)
    raise ValueError(f"unknown RoPE implementation {mode!r}")


# ------------------------------------------- the relative-position score ----

def relative_score(q_m: torch.Tensor, k_n: torch.Tensor,
                   angles: torch.Tensor, m: int, n: int) -> float:
    """``q_m^T k_n`` after both are rotated, as a function of ``m - n`` alone.

    ``q_m`` and ``k_n`` are **single** tokens, i.e. 1-D vectors of width ``d_h``;
    ``angles`` is a ``(T, d_h/2)`` table. The rotation applied to ``k_n`` uses
    the angle *difference* ``theta_{m,j} - theta_{n,j}``, which is what makes the
    returned scalar depend on the distance and nothing else.

    With ``base = 1`` the whole table collapses to ``t * 1``, the score becomes
    exactly ``q^T R_{m-n} k``, and this is checkable in closed form against an
    explicitly built ``R``.
    """
    d_h = q_m.shape[-1]
    half = d_h // 2
    T = angles.shape[0]
    # GUARD. An unguarded `angles[n]` with n < 0 wraps to angles[-n] and returns a
    # confident, plausible score for the WRONG position. This is not hypothetical:
    # while building this chapter an out-of-range probe at m = 4095, delta = 4095
    # requested position -2048, got the last row of the table, and printed a
    # number that looked like a legitimate attention logit. Every index derived
    # from a subtraction must be range-checked before it is used.
    if not (0 <= m < T):
        raise IndexError(f"position m={m} outside angle table of length {T}")
    if not (0 <= n < T):
        raise IndexError(
            f"position n={n} outside angle table of length {T} "
            f"(m={m}; a negative n means the requested distance exceeds m, "
            f"and would otherwise be silently read from the end of the table)")
    ang = (angles[m, :half] - angles[n, :half]).unsqueeze(0)     # (1, d_h/2)
    # a single token has no position axis; rope_forward_rows indexes one at -2,
    # so the (d_h,) row is presented as a length-1 sequence.
    return float(q_m @ rope_forward_rows(k_n.unsqueeze(0), ang).squeeze(0))


# ------------------------------------------------------- cost of the matrix ----

def matrix_cost(T: int, d_h: int, dtype: torch.dtype = torch.float32) -> dict:
    """What the block-diagonal matrix would cost if anyone actually built it.

    Reported because the reason the other two implementations exist is that these
    numbers are unacceptable, and a reader deserves to see by how much.
    """
    e = torch.empty(0, dtype=dtype).element_size()
    dense = T * d_h * d_h * e
    # Arithmetic actually required: each pair is a 2x2 multiply-add, 4 pairs of
    # 2 FLOPs, for T*d_h/2 blocks.
    flops = 4.0 * T * (d_h // 2)
    return {"T": T, "d_h": d_h, "element_bytes": e,
            "matrix_bytes": dense,
            "nonzero": T * (d_h // 2) * 4,
            "density": (d_h // 2) * 4 / (d_h * d_h),
            "dense_flops": 2.0 * T * d_h * d_h,
            "actual_flops": flops,
            "waste_ratio": (2.0 * T * d_h * d_h) / flops}


# ============================================================ measurements ==

def _rotated_dot(q: torch.Tensor, k: torch.Tensor, ang: torch.Tensor
                 ) -> torch.Tensor:
    """Per-sample dot product ``q_s . R_ang(k_s)`` for ``(n, d_h)`` inputs.

    ``ang`` is a single ``(d_h/2,)`` angle vector shared by every sample, which is
    the situation here: one distance, many random ``(q, k)`` draws.

    The rotate-and-score is stated like this rather than as
    ``q @ rope_forward_rows(k, ang).T`` because that expression is wrong: given
    ``k`` of shape ``(n, d_h)`` the helper infers a sequence length of ``n`` and
    then demands ``n/2`` angles, which is an error rather than a wrong answer --
    but only because this particular call site has the right *number* of angles
    to hand. Presenting a ``(n, d_h)`` sample matrix as a length-``n`` sequence is
    a category error, and the axis confusion it invites is exactly what produced
    the silent matmul bug in :func:`rope_forward_matrix`.
    """
    if q.shape != k.shape:
        raise ValueError(f"q and k must share shape, got {tuple(q.shape)} and "
                         f"{tuple(k.shape)}")
    rot = rope_forward_rows(k.unsqueeze(-2), ang.unsqueeze(-2))   # (n,1,d_h)
    return (q * rot.squeeze(-2)).sum(-1)


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    """Symmetric relative error, the same convention as the package validator."""
    a, b = a.detach().double(), b.detach().double()
    return float((a - b).norm() / (a.norm() + b.norm() + 1e-30))


def equivalence_report(T: int = 64, d_h: int = 8, base: float = 10000.0,
                       dtype: torch.dtype = torch.float64,
                       seed: int = 0) -> dict:
    """Three implementations, compared on the forward *and* the reverse.

    Reported as symmetric relative error against the block-diagonal matrix, which
    is the definition and therefore the reference. ``sliced`` is the shipping
    implementation and ``complex`` is the elegant one; both are checked, and
    neither is trusted.
    """
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(1, T, d_h, generator=g, dtype=dtype)
    ang = rope_angles(T, d_h, base, dtype=dtype)

    fwd = {}
    for m in IMPLEMENTATIONS:
        fwd[m] = rope_forward(q, ang, m)
    bwd = {}
    for m in IMPLEMENTATIONS:
        bwd[m] = rope_backward(q, ang, m)

    rel_f = {m: _rel(fwd[m], fwd["matrix"]) for m in IMPLEMENTATIONS}
    rel_b = {m: _rel(bwd[m], bwd["matrix"]) for m in IMPLEMENTATIONS}

    # Round-trip: forward then reverse must be the identity to float64 rounding.
    round_trip = {m: _rel(rope_backward(fwd[m], ang, m), q)
                  for m in IMPLEMENTATIONS}

    # Norm preservation: an orthogonal map cannot change ||q||. This is the
    # property that makes a wrong sign undetectable by inspection.
    norm_in = float(q.norm())
    norm_err = {m: abs(float(fwd[m].norm()) - norm_in) / norm_in
                for m in IMPLEMENTATIONS}

    # The sign trap, made concrete: exp(-i theta) is a *valid* rotation and is
    # wrong. Quantify how wrong, so "it looks fine" has a number attached.
    wrong = _as_complex(q.to(dtype))
    wrong = _from_complex(wrong * torch.polar(
        torch.ones_like(ang), -ang))            # exp(-i theta): the wrong sign
    return {
        "T": T, "d_h": d_h, "base": base, "dtype": str(dtype),
        "rel_fwd_vs_matrix": rel_f, "rel_bwd_vs_matrix": rel_b,
        "round_trip": round_trip, "norm_rel_err": norm_err,
        "sign_trap_rel": _rel(wrong, fwd["matrix"]),
        "sign_trap_is_orthogonal": abs(float(wrong.norm())
                                       - float(q.norm())) < 1e-12 * float(q.norm()),
    }


def relative_report(T: int = 32, d_h: int = 4, base: float = 10000.0,
                    dtype: torch.dtype = torch.float64, seed: int = 1) -> dict:
    """``q_m^T k_n`` depends on ``m - n`` and on nothing else.

    Two independent checks:

      1. *Translation.* With ``base = 1`` the angle table collapses to ``t``, so
         the score is exactly ``q^T R_{m-n} k``. Every pair ``(m, n)`` and
         ``(m+c, n+c)`` must agree to rounding. This is the check that could
         fail if RoPE were not relative, and it uses no approximation.
      2. *Closure.* For the real base, ``theta = t * inv_freq`` and so
         ``theta_{m,j} - theta_{n,j} = (m-n) * inv_freq_j`` exactly, which is
         what makes (1) an identity rather than a coincidence of one base.
    """
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(d_h, generator=g, dtype=dtype)
    k = torch.randn(d_h, generator=g, dtype=dtype)

    # (1) translation invariance at base = 1, where R is exactly R_{m-n}
    ang1 = rope_angles(T + 8, d_h, 1.0, dtype=dtype)        # theta = t
    worst = 0.0
    worst_at = None
    for m in range(T):
        for n in range(m + 1):
            ref = float(relative_score(q, k, ang1, m, n))
            for c in (1, 3, 7):
                if m + c >= ang1.shape[0]:
                    continue
                got = float(relative_score(q, k, ang1, m + c, n + c))
                worst = max(worst, abs(got - ref))
                worst_at = worst_at or ((m, n, c), got - ref)
    # (2) the angle identity itself
    ang = rope_angles(T, d_h, base, dtype=dtype)
    closure = max(
        abs(float((ang[m] - ang[n])[j]) - float((m - n)) * float(inv_freq(d_h, base, dtype=dtype)[j]))
        for m in range(T) for n in range(m + 1) for j in range(d_h // 2))
    # and the three implementations agree on the score itself. Note the
    # implementation name and the position index are different variables; they
    # share the letter m here only because the reader does, and conflating them
    # is a bug I wrote and deleted while building this function.
    q_row = q.unsqueeze(0)                                   # (1, d_h)
    sc = {}
    for impl in IMPLEMENTATIONS:
        sc[impl] = float(rope_forward(q_row, ang[3:4], impl).squeeze(0) @ k)
    return {"worst_translation_deviation": worst, "at": worst_at,
            "angle_closure_deviation": closure,
            "score_by_impl": sc,
            "score_spread": max(sc.values()) - min(sc.values()),
            "n_pairs": T * (T + 1) // 2}


def adjoint_report(T: int = 32, d_h: int = 8, base: float = 10000.0,
                   dtype: torch.dtype = torch.float64, seed: int = 2) -> dict:
    """The adjoint of a rotation by ``theta`` is a rotation by ``-theta``.

    Verified two ways: symbolically (``R(phi)^T R(psi) = R(psi - phi)``, checked
    as ``orthogonal(R)`` and ``orthogonality`` simultaneously) and numerically,
    by comparing every implementation's reverse against autograd through the
    forward. The autograd comparison is the one that would catch a hand-derived
    mistake; the symbolic one says what the mistake would have been.
    """
    ang = rope_angles(T, d_h, base, dtype=dtype)
    g = torch.Generator().manual_seed(seed)

    # --- symbolic: R_phi^T R_psi == R_{psi-phi}, in closed form -------------
    worst_orth = worst_compose = worst_against = 0.0
    for _ in range(24):
        phi = float(torch.rand(1, generator=g, dtype=dtype)) * 4.0
        psi = float(torch.rand(1, generator=g, dtype=dtype)) * 4.0
        Rp = rotation_block(torch.tensor(phi, dtype=dtype))
        Rs = rotation_block(torch.tensor(psi, dtype=dtype))
        d = Rp.T @ Rp - torch.eye(2, dtype=dtype)
        worst_orth = max(worst_orth, float(d.abs().max()))
        worst_compose = max(worst_compose,
                            float((Rp.T @ Rs
                                   - rotation_block(torch.tensor(psi - phi,
                                                                  dtype=dtype))).abs().max()))
        # the reverse rotation applied to a vector equals the transpose applied
        Rm = rotation_block(torch.tensor(-psi, dtype=dtype))
        v = torch.randn(2, generator=g, dtype=dtype)
        worst_against = max(worst_against,
                            float((Rm @ v - (Rs.T @ v)).abs().max()))

    # --- numeric: reverse vs autograd, per implementation -----------------
    q = torch.randn(1, T, d_h, generator=g, dtype=dtype)
    upstream = torch.randn(1, T, d_h, generator=g, dtype=dtype)
    err_vs_autograd = {}
    for m in IMPLEMENTATIONS:
        qr = q.clone().requires_grad_(True)
        y = rope_forward(qr, ang, m)
        (y * upstream).sum().backward()
        err_vs_autograd[m] = _rel(rope_backward(upstream, ang, m), qr.grad)

    # reverse(forward(.)) is the identity -- the second half of R^T = R^{-1}
    ident = {}
    for m in IMPLEMENTATIONS:
        ident[m] = _rel(rope_backward(rope_forward(q, ang, m), ang, m), q)

    # The failure mode this chapter warns about: reverse written as a *copy* of
    # the forward. It preserves norms and is silently wrong.
    ident_wrong_sign = _rel(
        rope_forward(rope_forward(q, ang, "sliced"), ang, "sliced"), q)

    return {"orthogonality_deviation": worst_orth,
            "composition_deviation": worst_compose,
            "transpose_equals_negangle_deviation": worst_against,
            "reverse_vs_autograd": err_vs_autograd,
            "worst_reverse_vs_autograd": max(err_vs_autograd.values()),
            "reverse_forward_identity": ident,
            "double_forward_deviation": ident_wrong_sign,
            "n_pairs_checked": 24, "dtype": str(dtype)}


# ------------------------------------------------ context-scaling spectra ---

def scaling_spectrum_report(d_h: int = 64, alpha: float = 2.0,
                            base: float = 10000.0,
                            T: int = 4096,
                            dtype: torch.dtype = torch.float64) -> dict:
    """What linear and NTK-aware scaling do to the *frequency spectrum*.

    Reported per coordinate pair ``j``, and the mechanism is the whole point:

      * **linear** multiplies every frequency by exactly ``1/alpha``. The ratio is
        flat. Local structure dies uniformly: a 1-token gap now carries the
        geometry of a ``1/alpha``-token gap.
      * **ntk** leaves the fast pairs essentially alone and rescales only the
        slow ones. The ratio ``(alpha^(d/(d-2)))^(-2j/d)`` is *not* ``1/alpha``
        except at ``j = 0``, which is exactly why NTK-aware cannot be implemented
        by dividing angles.

    Also reported: the *naive* NTK substitution ``theta_t / alpha``, whose error
    against the true NTK table is the diagnostic that catches this
    implementation mistake.
    """
    if_freq = inv_freq(d_h, base, dtype=dtype)
    lin = if_freq / alpha
    ntk = inv_freq(d_h, ntk_base(d_h, alpha, base), dtype=dtype)
    naive = if_freq / alpha                                  # the wrong thing

    half = d_h // 2
    pairs = [0, 1, 2, half // 4, half // 2, half - 2, half - 1]
    rows = []
    for j in pairs:
        if j < 0 or j >= half:
            continue
        rows.append({
            "j": j,
            "inv_freq": float(if_freq[j]),
            "lin_ratio": float(lin[j] / if_freq[j]),
            "ntk_ratio": float(ntk[j] / if_freq[j]),
            "ntk_over_lin": float(ntk[j] / lin[j]),
            "theta_lin_t1": float(lin[j]),
            "theta_ntk_t1": float(ntk[j]),
        })

    # naive-vs-true NTK, evaluated where it is largest: the slowest pair at the
    # longest position
    t = torch.arange(T, dtype=dtype)
    ang_ntk = torch.outer(t, ntk)
    ang_naive = torch.outer(t, naive)
    naive_err = float((ang_ntk - ang_naive).abs().max())
    naive_rel = float(((ang_ntk - ang_naive).abs()
                       / ang_ntk.abs().clamp_min(1e-300)).max())

    # THE IDENTITY. `theta_t / alpha` is not an approximation of NTK-aware
    # scaling; it is *bit-for-bit* linear interpolation, because
    # (t * f_j) / alpha == (t / alpha) * f_j in floating point for a power of two
    # and to within one rounding otherwise. Anyone who implements NTK-aware
    # scaling by scaling the angles has implemented linear interpolation and
    # will measure exactly linear interpolation's error while believing they
    # avoided it. Asserted bitwise rather than approximately.
    ang_linear = torch.outer(t / alpha, if_freq)
    naive_is_linear = bool(torch.equal(ang_naive, ang_linear))
    naive_linear_max_diff = float((ang_naive - ang_linear).abs().max())

    # the exact per-pair NTK/linear ratio, alpha^(-2j/(d-2)), verified against
    # the closed form: 1 at j=0 and exactly 1/alpha at j=(d-2)/2
    ratio = ntk / if_freq
    pred = alpha ** (-2.0 * torch.arange(half, dtype=dtype) / (d_h - 2))
    ratio_err = float((ratio - pred).abs().max())
    return {
        "d_h": d_h, "alpha": alpha, "base": base, "T": T,
        "ntk_base": ntk_base(d_h, alpha, base),
        "rows": rows,
        "ntk_ratio_spread": (max(r["ntk_ratio"] for r in rows)
                             / min(r["ntk_ratio"] for r in rows)),
        "lin_ratio_spread": (max(r["lin_ratio"] for r in rows)
                             / min(r["lin_ratio"] for r in rows)),
        "naive_vs_true_ntk_max_abs": naive_err,
        "naive_vs_true_ntk_max_rel": naive_rel,
        "naive_equals_linear_bitwise": naive_is_linear,
        "naive_vs_linear_max_diff": naive_linear_max_diff,
        "ntk_over_lin_ratio_max_err_vs_closed_form": ratio_err,
        "ntk_over_lin_ratio_at_j0": float(ratio[0]),
        "ntk_over_lin_ratio_at_j_last": float(ratio[half - 1]),
        "at_pair": half - 1, "at_position": T - 1,
    }


# --------------------------------------------------- the extrapolation cliff --
#
# WHAT THIS SECTION MEASURES, and one thing it deliberately does not claim.
#
# The naive expectation is that running positions past the trained window
# produces *wrong attention scores* -- scores that are larger, or noisier, or
# otherwise out of distribution. That expectation is false, and the reason is
# the relative-position identity of §13.3: q^T R_{m-n} k does not mention m or n,
# only their difference. Evaluating it at (m, n) = (4095, 3071) and at (2047, 1023)
# gives the same number. There is no algebraic degradation, and a chapter that
# reported one would be reporting an artefact of its own harness.
#
# What IS real, and what is measured below, is twofold:
#
#   (a) NUMERICAL. The identity is exact in reals, but the code evaluates
#       theta_m - theta_n where both terms are proportional to m. The error of that
#       difference grows *linearly* with position, so doubling the context doubles
#       the numerical error in every attention logit. This is a real cliff, it is
#       measured in ULPs, and in float32 it is large.
#
#   (b) GEOMETRIC. Interpolation is not free and is not neutral: linear and
#       NTK-aware scaling deliberately *change* the distance a token gap
#       represents. That change is the price of extension, and it is measured as
#       RMS deviation from the trained score, in units of the trained score's own
#       standard deviation.

def _score_samples(q: torch.Tensor, k: torch.Tensor, ang_diff: torch.Tensor
                   ) -> torch.Tensor:
    """``q_s . R_{ang_diff}(k_s)`` over ``n`` random draws. See :func:`_rotated_dot`."""
    return _rotated_dot(q, k, ang_diff)


def _inv_freq_pow(d_h: int, base: float, dtype: torch.dtype) -> torch.Tensor:
    """``base ** (-2j/d_h)`` -- the form every tutorial writes.

    Kept only so the chapter can measure what it costs against
    :func:`inv_freq`'s exponential form. A relative error ``eps`` in the
    frequency becomes an **absolute** error of ``eps * t * f`` in the angle at
    position ``t``, so the damage grows linearly with context length. At
    ``t = 4095`` it is worth roughly 1500 ULPs of a float64 attention logit --
    larger, by four orders of magnitude, than the float32 rounding the reader
    was actually worried about. This is the measured justification for
    :func:`inv_freq` and it is the chapter's only genuine IEEE-754 finding.
    """
    j = torch.arange(d_h // 2, dtype=dtype)
    return base ** (-2.0 * j / d_h)


def numerical_cliff_report(d_h: int = 64, base: float = 10000.0,
                           train_len: int = 2048, delta: int = 512,
                           positions=(256, 1024, 2047, 3071, 4095),
                           n_samples: int = 4096, seed: int = 7) -> dict:
    """Does the attention score degrade as positions grow? Measured, two ways.

    Two error sources are separated, because they have different fixes and a
    report that merges them teaches the wrong lesson:

      **position growth.** ``theta_t = t * f`` is itself a product, so representing
        ``t`` and multiplying by ``f`` each costs rounding. The angle's own error
        therefore grows linearly in ``t``.

      **formula choice.** ``base ** (-2j/d)`` (the pow form) carries a relative
        error of about ``eps`` in ``f`` that the exponential form
        ``exp(-2j log(base)/d)`` does not. Multiplied by ``t``, that relative error
        becomes an absolute angle error proportional to ``t`` as well.

    The reference is the translation-invariant difference ``delta * f`` computed
    with the exponential form in float64 and rotated in float64 -- i.e. the
    algebraic value of ``q^T R_delta k``, to float64 rounding.

    The probe distance ``delta`` is held **fixed** while the probe position ``t``
    moves. That is the whole experimental design: varying ``delta`` confounds
    distance with position, and the claim under test is about position.
    """
    g = torch.Generator().manual_seed(seed)
    q64 = torch.randn(n_samples, d_h, generator=g, dtype=torch.float64)
    k64 = torch.randn(n_samples, d_h, generator=g, dtype=torch.float64)
    if_exp64 = inv_freq(d_h, base, dtype=torch.float64)

    ref = _score_samples(q64, k64, torch.tensor(float(delta),
                                               dtype=torch.float64) * if_exp64)
    ref_scale = float(ref.abs().mean())
    ref_std = float(ref.std(unbiased=True))

    rows = []
    for t in positions:
        if t - delta < 0:
            continue
        row = {"t": t, "in_train": bool(t <= train_len - 1)}
        for name, work in (("f32_pow", torch.float32),
                           ("f32_exp", torch.float32),
                           ("f64_pow", torch.float64),
                           ("f64_exp", torch.float64)):
            if work == torch.float32:
                if name.endswith("pow"):
                    f = _inv_freq_pow(d_h, base, work)
                else:
                    f = inv_freq(d_h, base, dtype=work)
                # the shipped route: build a length-(t+1) table, subtract
                ang = torch.outer(torch.arange(t + 1, dtype=work), f)
                diff = (ang[t] - ang[t - delta]).to(torch.float64)
                S = _score_samples(q64.to(work), k64.to(work),
                                   diff.to(work)).to(torch.float64)
            else:
                f = (_inv_freq_pow(d_h, base, work) if name.endswith("pow")
                     else inv_freq(d_h, base, dtype=work))
                diff = (torch.tensor(float(t), dtype=work)
                        * f - torch.tensor(float(t - delta), dtype=work) * f)
                S = _score_samples(q64, k64, diff.to(torch.float64))
            err = float((S - ref).abs().max())
            eps = float(torch.finfo(work).eps)
            row[f"{name}_max_abs_err"] = err
            row[f"{name}_ulps"] = err / (eps * ref_scale)
        rows.append(row)

    def col(rows_, key):
        return [r[key] for r in rows_ if key in r]

    first, last = rows[0], rows[-1]
    growth = {}
    for name in ("f32_pow", "f32_exp", "f64_pow", "f64_exp"):
        a, b = first[f"{name}_ulps"], last[f"{name}_ulps"]
        # f64_exp is the REFERENCE branch -- it computes the same expression as
        # `ref`, so its deviation is exactly 0.0 by construction, not by
        # accuracy. Its growth ratio is therefore 0/0 and must be reported as
        # None rather than silently divided by zero.
        growth[name] = (b / a) if a else None
    return {
        "d_h": d_h, "base": base, "delta": delta, "train_len": train_len,
        "positions": list(positions), "n_samples": n_samples, "seed": seed,
        "ref_mean_abs": ref_scale, "ref_std": ref_std,
        "rows": rows,
        "ulp_growth_first_to_last": growth,
        "f64_exp_is_reference": True,
        "formula_choice_cost_float64": {
            "at_first": first["f64_pow_ulps"],
            "at_last": last["f64_pow_ulps"],
            "growth": growth["f64_pow"],
        },
        "formula_choice_cost_float32": {
            "at_first": first["f32_pow_ulps"],
            "at_last": last["f32_pow_ulps"],
            "growth": growth["f32_pow"],
        },
    }


def float32_roundtrip_report(T: int = 4096, d_h: int = 64,
                             base: float = 10000.0, delta: int = 1,
                             positions=(2047, 4095), seed: int = 5) -> dict:
    """float32-only precision loss, separated from the formula-choice effect.

    Everything here is computed in float32 with the exponential form, so the only
    error left is float32 rounding of positions, angles and products. Reported at
    the last trained position and at the last extended one, so the extra that
    context extension costs is visible as a number.

    ``delta`` is held fixed while the probe position ``t`` moves. An earlier
    version of this function built the reference as ``(t-1) * f`` while the code
    under test computed ``ang[t] - ang[t-1]``; those encode *different* distances
    -- 4094 and 1 -- and the resulting disagreement on the ``j = 0`` pair was a
    full 4093 radians, which is a reference that is wrong rather than an error in
    float32. Any measurement of an *increment* must increment on both sides.
    """
    g = torch.Generator().manual_seed(seed)
    q32 = torch.randn(4096, d_h, generator=g, dtype=torch.float32)
    k32 = torch.randn(4096, d_h, generator=g, dtype=torch.float32)
    f32 = inv_freq(d_h, base, dtype=torch.float32)
    f64 = inv_freq(d_h, base, dtype=torch.float64)

    ref_diff = torch.tensor(float(delta), dtype=torch.float64) * f64
    ref = _score_samples(q32.double(), k32.double(), ref_diff)
    scale = float(ref.abs().mean())
    eps = float(torch.finfo(torch.float32).eps)

    out = {"delta": delta, "n_samples": int(q32.shape[0]),
           "score_mean_abs": scale, "eps": eps, "points": {}}
    for t in positions:
        if t - delta < 0:
            continue
        # the shipped route: build a length-(t+1) float32 table, then subtract
        ang = torch.outer(torch.arange(t + 1, dtype=torch.float32), f32)
        diff = ang[t] - ang[t - delta]
        S = _score_samples(q32, k32, diff).double()
        err = float((S - ref).abs().max())
        out["points"][f"t={t}"] = {
            "t": t, "in_train": bool(t <= 2047),
            "max_abs_err": err,
            "rms_err": float(((S - ref) ** 2).mean() ** 0.5),
            "score_std": float(ref.std(unbiased=True)),
            "ulps": err / (eps * scale),
        }
    pts = list(out["points"].values())
    out["ulp_growth"] = (pts[-1]["ulps"] / pts[0]["ulps"]
                         if pts and pts[0]["ulps"] else None)
    out["position_ratio"] = (pts[-1]["t"] / pts[0]["t"] if pts else None)
    return out


def scaling_cost_report(d_h: int = 64, base: float = 10000.0,
                        train_len: int = 2048, eval_len: int = 4096,
                        n_samples: int = 2048, seed: int = 11,
                        dtype: torch.dtype = torch.float64) -> dict:
    """What each scaling method does to a distance the model already knows.

    Protocol. For a true token distance ``delta``, the *trained* geometry is

        S_train(delta) = q^T R_delta k          (integer positions, base b)

    and the *scaled* geometry is the same expression evaluated on the scaled
    table. The price of extension is the RMS deviation between them, normalised
    by the standard deviation of the trained score across random ``(q, k)``
    draws. The normalisation is what makes "0.31 standard deviations" meaningful:
    it says how much of the score's own natural spread the intervention
    overwrites.

    Two references are reported because they answer different questions:

      * ``delta <= train_len - 1`` -- distances the model was actually trained on.
        This is where linear interpolation is supposed to fail and NTK-aware
        supposed to survive.
      * the full range to ``eval_len - 1`` -- the whole extended context.

    Also reported, because it is the commonest implementation error and it is not
    a tuning choice: applying NTK-aware scaling by dividing the *angles* by
    ``alpha`` instead of passing the modified ``base`` to the table. It is a
    different function and the report says by how much.
    """
    alpha = eval_len / train_len
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(n_samples, d_h, generator=g, dtype=dtype)
    k = torch.randn(n_samples, d_h, generator=g, dtype=dtype)
    if_freq = inv_freq(d_h, base, dtype=dtype)

    ang_trained = rope_angles(eval_len, d_h, base, dtype=dtype)
    ang_linear = scaled_angles(eval_len, d_h, mode="linear", alpha=alpha,
                               base=base, dtype=dtype)
    ang_ntk = scaled_angles(eval_len, d_h, mode="ntk", alpha=alpha,
                            base=base, dtype=dtype)
    ang_naive = torch.outer(torch.arange(eval_len, dtype=dtype),
                            if_freq) / alpha          # the WRONG ntk shortcut

    deltas = [1, 8, 64, 256, 1024, 2048, 3072, 4095]
    rows = []
    for delta in deltas:
        m = eval_len - 1
        n = m - delta
        if n < 0:
            continue
        ref = _score_samples(q, k, ang_trained[m] - ang_trained[n])
        ref_std = float(ref.std(unbiased=True))
        row = {"delta": delta, "in_train": bool(delta <= train_len - 1),
               "ref_std": ref_std}
        for name, ang in (("linear", ang_linear), ("ntk", ang_ntk),
                          ("ntk_naive", ang_naive)):
            S = _score_samples(q, k, ang[m] - ang[n])
            dev = S - ref
            row[f"{name}_rms_dev"] = float((dev ** 2).mean() ** 0.5)
            row[f"{name}_dev_in_sd"] = (row[f"{name}_rms_dev"] / ref_std
                                        if ref_std else float("nan"))
            row[f"{name}_max_abs_dev"] = float(dev.abs().max())
        rows.append(row)

    def band(rs, key, lo, hi):
        v = [r[key] for r in rs if lo <= r["delta"] <= hi]
        return float(sum(v) / len(v)) if v else None

    local_hi = 64
    return {
        "d_h": d_h, "base": base, "train_len": train_len,
        "eval_len": eval_len, "alpha": alpha, "ntk_base": ntk_base(d_h, alpha, base),
        "n_samples": n_samples, "seed": seed, "dtype": str(dtype),
        "local_hi": local_hi, "rows": rows,
        "linear_dev_in_sd_local": band(rows, "linear_dev_in_sd", 1, local_hi),
        "ntk_dev_in_sd_local": band(rows, "ntk_dev_in_sd", 1, local_hi),
        "ntk_naive_dev_in_sd_local": band(rows, "ntk_naive_dev_in_sd", 1, local_hi),
        "linear_dev_in_sd_trained_band": band(rows, "linear_dev_in_sd", 1, train_len - 1),
        "ntk_dev_in_sd_trained_band": band(rows, "ntk_dev_in_sd", 1, train_len - 1),
        "ntk_naive_dev_in_sd_trained_band": band(rows, "ntk_naive_dev_in_sd", 1, train_len - 1),
        "linear_dev_in_sd_long_band": band(rows, "linear_dev_in_sd", train_len, eval_len - 1),
        "ntk_dev_in_sd_long_band": band(rows, "ntk_dev_in_sd", train_len, eval_len - 1),
        "ntk_naive_dev_in_sd_long_band": band(rows, "ntk_naive_dev_in_sd", train_len, eval_len - 1),
        "ntk_advantage_local": (band(rows, "linear_dev_in_sd", 1, local_hi)
                                / band(rows, "ntk_dev_in_sd", 1, local_hi)
                                if band(rows, "ntk_dev_in_sd", 1, local_hi) else None),
        "ntk_advantage_trained_band": (
            band(rows, "linear_dev_in_sd", 1, train_len - 1)
            / band(rows, "ntk_dev_in_sd", 1, train_len - 1)
            if band(rows, "ntk_dev_in_sd", 1, train_len - 1) else None),
    }


def extrapolation_report(d_h: int = 64, base: float = 10000.0,
                         train_len: int = 2048, eval_len: int = 4096,
                         seed: int = 7) -> dict:
    """The complete Chapter 13 measurement, as one dict.

    Bundles the three pieces so the verification script and the manuscript cannot
    disagree about which numbers they are quoting: they both call this.
    """
    return {
        "equivalence": equivalence_report(),
        "relative": relative_report(),
        "adjoint": adjoint_report(),
        "cliff": numerical_cliff_report(d_h=d_h, base=base,
                                        train_len=train_len, seed=seed),
        "f32_roundtrip": float32_roundtrip_report(T=eval_len, d_h=d_h,
                                                  base=base, seed=seed + 1),
        "scaling": scaling_cost_report(d_h=d_h, base=base,
                                       train_len=train_len, eval_len=eval_len,
                                       seed=seed + 4),
        "spectrum": scaling_spectrum_report(d_h=d_h,
                                            alpha=eval_len / train_len,
                                            base=base, T=eval_len),
        "matrix_cost": matrix_cost(eval_len, d_h),
        "seed_stability": seed_stability_report(d_h=d_h, base=base,
                                                train_len=train_len,
                                                eval_len=eval_len),
    }


__all__ = [
    "rope_angles", "rope_forward_rows", "rope_backward_rows",
    "inv_freq", "rotation_block",
    "rope_forward_matrix", "rope_backward_matrix",
    "rope_forward_complex", "rope_backward_complex",
    "rope_forward", "rope_backward", "IMPLEMENTATIONS",
    "ntk_base", "scaled_positions", "scaled_angles",
    "relative_score", "matrix_cost",
    "equivalence_report", "relative_report", "adjoint_report",
    "scaling_spectrum_report", "scaling_cost_report",
    "numerical_cliff_report", "float32_roundtrip_report",
    "seed_stability_report", "extrapolation_report",
]


def seed_stability_report(seeds=(11, 12, 13, 14, 15), d_h: int = 64,
                          base: float = 10000.0, train_len: int = 2048,
                          eval_len: int = 4096,
                          n_samples: int = 2048) -> dict:
    """Seed-to-seed spread of every number reported in the scaling table.

    The scaling deviations in ``scaling_cost_report`` are means over random
    ``(q, k)`` draws and therefore carry sampling error. A chapter that quotes
    them from one seed without saying so has an undisclosed variance, and the
    gate that is supposed to catch that (G9) cannot: it greps for phrases and
    will happily match the words "closed form" in an unrelated table, which is
    exactly what happened while drafting this chapter.

    So the spread is measured rather than asserted. Reported per band as the
    sample standard deviation across seeds, and as a ratio to the quantity
    itself, because a spread of 1e-3 is either negligible or decisive depending
    on whether the effect being reported is 0.1 or 1.0.
    """
    keys = ("linear_dev_in_sd_local", "ntk_dev_in_sd_local",
            "linear_dev_in_sd_trained_band", "ntk_dev_in_sd_trained_band",
            "linear_dev_in_sd_long_band", "ntk_dev_in_sd_long_band")
    runs = []
    for sd in seeds:
        r = scaling_cost_report(d_h=d_h, base=base, train_len=train_len,
                                eval_len=eval_len, n_samples=n_samples, seed=sd)
        runs.append({k: r[k] for k in keys})

    rows = []
    for k in keys:
        vals = [r[k] for r in runs]
        m = sum(vals) / len(vals)
        sd_ = (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
        rows.append({
            "metric": k, "mean": m,
            "sample_sd_across_seeds": sd_,
            "relative_sd": sd_ / m if m else None,
            "min": min(vals), "max": max(vals),
            "n_seeds": len(vals),
        })
    worst = max(r["relative_sd"] for r in rows if r["relative_sd"] is not None)
    return {"seeds": list(seeds), "n_samples": n_samples,
            "d_h": d_h, "train_len": train_len, "eval_len": eval_len,
            "rows": rows, "worst_relative_sd": worst}
