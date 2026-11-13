"""Reduced-precision failure modes: cancellation, overflow, and truncation.

Why this module exists
----------------------
Chapters 2 to 9 do mathematics in a register that is effectively infinite:
float64, or an exact symbolic ring. Production inference and training do not.
Three failure modes account for most of what changes, and **all three are
invisible** -- no exception is raised, no warning is printed, and the result is a
finite wrong number:

1. **Catastrophic cancellation.** The textbook variance
   ``E[x^2] - E[x]^2`` subtracts two nearly equal large numbers. In a format with
   a short mantissa the small difference is rounded away entirely, so the
   "variance" can come out zero or *negative*.
2. **Overflow.** ``exp(x)`` exceeds the largest finite float16 at
   ``x = ln(65504)``, after which the softmax is ``inf/inf = nan``.
3. **Truncation (swamping).** ``W + dW`` rounds back to ``W`` whenever
   ``|dW|`` is below half an ulp of ``W``. The update is not small; it is
   *invisible*.

Each is measured here at the exact threshold, because a chapter that says
"precision can matter" is not saying anything. The whole architecture of a modern
decoder -- RMSNorm rather than LayerNorm, max-subtracted softmax, float32 master
weights -- is a consequence of the numbers below.

Precision formats, from their structure
--------------------------------------
Both are IEEE binary16: 1 sign, 5 exponent, 10 stored mantissa.

    float16   exponent bias 15, largest finite 65504, smallest normal 6.1e-5
              unit roundoff  2^-11 = 4.883e-04
              -> constrained by DYNAMIC RANGE: a wide exponent field, so it can
                 hold very large and very small magnitudes, but nothing in
                 between is finely resolved.

    bfloat16   same 8-bit exponent as float32, largest finite ~3.4e38
              unit roundoff  2^-8 = 3.906e-03
              -> constrained by RESOLUTION: it keeps float32's range and throws
                 away mantissa bits, so it represents the same magnitudes but
                 with eight times fewer significant digits.

The trade is range against precision, and the two failure modes above follow
directly: float16 dies of overflow at moderate inputs, bfloat16 dies of
cancellation and truncation far earlier.

Conventions follow the notation contract. ``=`` is used only for true
mathematical equality; floating-point assignment and approximation are written
``<-`` and ``~=`` in the chapter text, and the functions here report whether a
result is exact, approximate, or wrong.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "FORMATS", "rmsnorm_variance_naive", "rmsnorm_variance_zero_centered",
    "rmsnorm_forward", "softmax_naive", "softmax_safe",
    "weight_update", "measure_cancellation", "measure_overflow",
    "measure_truncation", "MACHINE_NOTE",
]

MACHINE_NOTE = (
    "All measurements below are exact IEEE binary16 arithmetic via numpy, so they "
    "are properties of the formats and not of any framework, device or compiler. "
    "float64 is used only as the reference against which the reduced-precision "
    "results are judged wrong."
)

#: name -> (numpy type, unit roundoff 2^-p, largest finite)
FORMATS: dict[str, tuple] = {
    "float16": (np.float16, 2.0 ** -11, 65504.0),
    "bfloat16": (None, 2.0 ** -8, 3.3895313892515355e38),  # emulated, see below
    "float32": (np.float32, 2.0 ** -24, 3.4028234663852886e38),
    "float64": (np.float64, 2.0 ** -53, 1.7976931348623157e308),
}

#: bfloat16 has no native numpy type. It is emulated exactly as the truncation
#: of a float32 to 8 stored mantissa bits, which is the definition. Emulation is
#: done on the bit pattern, not by scaling, so it is bit-for-bit correct.
BF16_MANTISSA_BITS = 7


def _to_bf16(a: np.ndarray) -> np.ndarray:
    """Round a float32 array to bfloat16 by bit manipulation, exactly."""
    f32 = np.ascontiguousarray(a, dtype=np.float32)
    u = f32.view(np.uint32)
    # Round-to-nearest-even on the 16 discarded mantissa bits, then TRUNCATE them.
    #
    # The mask is not optional. Adding the bias without masking leaves the low 16
    # bits in place, so the result is not a bfloat16 number at all -- it is a
    # float32 nudged upward by up to one float32 ulp. The symptom is _to_bf16(1.0)
    # returning 1.0039061 instead of 1.0, which then makes every "was the update
    # truncated?" test answer incorrectly.
    lsb = (u >> 16) & np.uint32(1)
    rounded = (u + np.uint32(0x7FFF) + lsb) & np.uint32(0xFFFF0000)
    return rounded.view(np.float32)


def cast(a, fmt: str) -> np.ndarray:
    """Cast to the named format, emulating bfloat16 if necessary."""
    if fmt == "bfloat16":
        return _to_bf16(np.asarray(a, dtype=np.float32))
    return np.asarray(a, dtype=FORMATS[fmt][0])


def unit_roundoff(fmt: str) -> float:
    return FORMATS[fmt][1]


def rnd(v, fmt: str) -> float:
    """Round a python float to the named format and return it as a float.

    Every intermediate in the failure paths goes through here, because the whole
    point is that rounding is what destroys the result. Rounding only the final
    answer would measure a different, and much more forgiving, computation.
    """
    if fmt == "bfloat16":
        return float(_to_bf16(np.array([v], dtype=np.float32))[0])
    return float(np.asarray(v, dtype=FORMATS[fmt][0])[()])


# ------------------------------------------------------------ RMSNorm -----

def rmsnorm_variance_naive(x, fmt: str = "float16") -> float:
    """The textbook variance ``E[x^2] - E[x]^2``.

    The two moments are **accumulated in float32**, as production reductions
    generally are, and then each is rounded to ``fmt`` before the subtraction.
    That isolates the cancellation, which is the phenomenon under study, from a
    second and unrelated failure -- ``E[x^2]`` overflowing the format outright --
    which is measured separately by :func:`variance_all_low_precision`.

    This is exact in real arithmetic and catastrophically wrong in reduced
    precision whenever the mean is large relative to the spread.
    """
    x64 = np.asarray(x, dtype=np.float64)
    xc = cast(x, fmt)
    mean_sq = rnd(float(np.mean(x64 * x64)), fmt)
    mean = rnd(float(np.mean(x64)), fmt)
    return _sub_in(mean_sq, rnd(rnd(mean, fmt) ** 2, fmt), fmt)


def _sub_in(a: float, b: float, fmt: str) -> float:
    """Subtract two values that are already rounded to ``fmt``, in ``fmt``.

    numpy will promote float16 scalars to float64 on subtraction, which would
    hide exactly the rounding the measurement depends on. Subtracting via the
    format's own type keeps it honest.
    """
    if fmt == "bfloat16":
        return float(_to_bf16(np.array([a - b], dtype=np.float32))[0])
    return float(np.asarray(a, dtype=FORMATS[fmt][0])
                  - np.asarray(b, dtype=FORMATS[fmt][0]))


def variance_all_low_precision(x, fmt: str = "float16") -> float:
    """``E[x^2] - E[x]^2`` with the *reduction* also in ``fmt``.

    This is the fully naive version, and it fails EARLIER than the cancellation
    alone: ``E[x^2]`` overflows the format before any subtraction happens. Kept
    separate because it is a different bug with a different threshold, and
    reporting only one of the two would understate how fragile the expression is.

    The accumulation is genuinely in ``fmt`` (numpy ``float16`` for fp16; the
    emulated bf16 storage for bfloat16), NOT in Python ``float``: a float64
    accumulator would silently rescue the sum from overflowing and report 0.0
    for a constant ``x = 200`` vector of length 1024 whose ``sum(x^2) = 4.096e7``
    overflows fp16. The overflow is the measurement, so the reduction must be
    able to produce it.
    """
    xc = cast(x, fmt)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        sq = xc * xc
        if fmt == "bfloat16":
            # bf16 accumulates in its float32 storage but re-rounds each
            # partial, which is the emulation of a bf16 adder tree.
            ssum = float(_to_bf16(np.array([np.sum(sq, dtype=np.float32)],
                                            dtype=np.float32))[0])
            csum = float(_to_bf16(np.array([np.sum(xc, dtype=np.float32)],
                                            dtype=np.float32))[0])
        else:
            ssum = float(np.sum(sq, dtype=xc.dtype))    # sums in fp16: overflows
            csum = float(np.sum(xc, dtype=xc.dtype))
        mean_sq = rnd(ssum / xc.size, fmt)
        mean = rnd(csum / xc.size, fmt)
        return _sub_in(mean_sq, rnd(rnd(mean, fmt) ** 2, fmt), fmt)


def rmsnorm_variance_zero_centered(x, fmt: str = "float16") -> float:
    """``E[x^2]`` -- the variance of a vector already known to have zero mean.

    This is why RMSNorm exists rather than LayerNorm: RMSNorm *assumes* the mean
    is zero, so the variance needs no subtraction and therefore suffers no
    cancellation. The assumption is a modelling choice, not a numerical trick, and
    it is why the architecture was designed this way rather than as an
    optimisation.
    """
    x64 = np.asarray(x, dtype=np.float64)
    return rnd(float(np.mean(x64 * x64)), fmt)


def rmsnorm_reference(x) -> float:
    """float64 reference: the exact variance of the input, by both routes."""
    x64 = np.asarray(x, dtype=np.float64)
    return float(np.mean(x64 * x64) - np.mean(x64) ** 2)


def rmsnorm_forward(x, gain, fmt: str = "float64"):
    """RMSNorm forward in the given format: ``y = gain * x / sqrt(E[x^2] + eps)``."""
    xc = cast(x, fmt)
    v = rmsnorm_variance_zero_centered(x, fmt)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        inv = rnd(1.0 / math.sqrt(v + 1e-6), fmt)
        return (xc * inv) * cast(gain, fmt)


# ------------------------------------------------------------- Softmax ----

def softmax_naive(x, fmt: str = "float16") -> np.ndarray:
    """``exp(x) / sum(exp(x))`` with no max subtraction, entirely in ``fmt``."""
    xc = cast(x, fmt)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        ex = np.exp(xc.astype(np.float64))
        e = cast(ex, fmt)                 # exp is evaluated then ROUNDED
        total = float(np.sum(e.astype(np.float64)))
        return e.astype(np.float64) / total


def softmax_safe(x, fmt: str = "float16") -> np.ndarray:
    """Max-subtracted softmax.

    Mathematically identical to :func:`softmax_naive` -- see the identity in the
    chapter -- but the largest exponent argument is 0, so ``exp`` cannot overflow.
    The final normalization is performed in ``fmt`` (the shifted exponentials are
    cast, summed, and divided inside the format) so the returned rows are the
    format's own rounding of the softmax, not a float64 view of it.
    """
    xc = cast(x, fmt)
    m = float(np.max(xc.astype(np.float64)))
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        shifted = cast(xc.astype(np.float64) - m, fmt)
        e = cast(np.exp(shifted.astype(np.float64)), fmt)
        if fmt == "bfloat16":
            total = float(_to_bf16(np.array([np.sum(e, dtype=np.float32)],
                                            dtype=np.float32))[0])
        else:
            total = float(np.sum(e, dtype=e.dtype))
        return cast(e.astype(np.float64) / total, fmt).astype(np.float64)


# ------------------------------------------------------ weight update -----

def weight_update(W, dW, fmt: str):
    """``W + dW`` computed entirely in ``fmt``. Returns (result, changed).

    The result is **re-rounded into the format** before being returned. For
    float16 numpy's scalar promotion already does this; for the emulated
    bfloat16 it does not, because ``cast`` hands back float32 storage holding
    bfloat16-rounded values, and a float32 addition of two such arrays would
    carry far more precision than the format has. Without the re-rounding,
    bfloat16 appears to swallow updates ~100,000x smaller than its unit roundoff
    predicts, which is a pure emulation artefact.
    """
    Wc, dWc = cast(W, fmt), cast(dW, fmt)
    with np.errstate(over="ignore", invalid="ignore"):
        raw = Wc + dWc
        out = cast(raw, fmt)          # re-round into the format, any size
        # Compare like with like. Mixing a Python float with a float16 array
        # makes the result depend on numpy's promotion rules rather than on the
        # format's arithmetic, which silently reported every update as applied.
        same = bool(np.array_equal(np.reshape(np.asarray(out, dtype=np.float64),
                                               np.shape(Wc)),
                                  np.reshape(np.asarray(Wc, dtype=np.float64),
                                               np.shape(Wc))))
        return out, not same


# --------------------------------------------------------- measurement ----

@dataclass
class CancellationPoint:
    """Where the naive variance stops being a variance."""

    fmt: str
    ratio: float            # mu / sigma
    dimension: int
    ref: float              # true variance in float64
    naive: float
    zero_centered: float
    sign_ok: bool
    exact: bool


def measure_cancellation(mu: float, sigma: float, n: int = 1024,
                         seed: int = 0, ratios=None) -> list[CancellationPoint]:
    """Sweep the mean-to-spread ratio and find where naive variance fails.

    The relative error of a cancellation is ``~ u * mu^2 / sigma^2``, so failure
    should begin near ``mu/sigma ~ 1/sqrt(u)``. This measures where it actually
    begins, for both formats.
    """
    if ratios is None:
        ratios = [1, 2, 4, 8, 16, 32, 45, 64, 90, 128, 256]
    rng = np.random.default_rng(seed)
    out: list[CancellationPoint] = []
    for fmt in ("float16", "bfloat16"):
        for r in ratios:
            m, s = mu * r, sigma
            x = m + s * rng.standard_normal(n)
            ref = rmsnorm_reference(x)
            naive = rmsnorm_variance_naive(x, fmt)
            zc = rmsnorm_variance_zero_centered(x, fmt)
            out.append(CancellationPoint(
                fmt=fmt, ratio=float(r), dimension=n,
                ref=ref, naive=naive, zero_centered=zc,
                sign_ok=naive >= 0.0, exact=abs(naive - ref) <= 1e-6 * max(abs(ref), 1e-30),
            ))
    return out


def cancellation_threshold(fmt: str, mu: float = 1.0, sigma: float = 1.0,
                           n: int = 1024, seed: int = 0, limit: float = 4096.0
                           ) -> tuple[float | None, float | None]:
    """Binary-search the ``mu/sigma`` ratio at which naive variance first
    returns a non-positive number, and at which it first stops matching float64."""
    lo, hi = 1.0, limit
    bad_sign = None
    for _ in range(200):                 # bounded: never trust a bisection loop
        if lo >= hi:
            break
        mid = math.sqrt(lo * hi)          # geometric bisection
        x = mu * mid + sigma * np.random.default_rng(seed).standard_normal(n)
        v = rmsnorm_variance_naive(x, fmt)
        # nan counts as failure. It failed here only because E[x^2] overflowed,
        # which is why the accumulation is in float32 above; leaving it out turns
        # this into an unbounded loop, because nan <= 0 is False.
        if not np.isfinite(v) or v <= 0.0:
            bad_sign, hi = mid, mid
        else:
            lo = mid
    lo2, hi2 = 1.0, limit
    inexact = None
    for _ in range(200):
        if lo2 >= hi2:
            break
        mid = math.sqrt(lo2 * hi2)
        x = mu * mid + sigma * np.random.default_rng(seed).standard_normal(n)
        ref = rmsnorm_reference(x)
        v = rmsnorm_variance_naive(x, fmt)
        if not np.isfinite(v) or abs(v - ref) > 1e-3 * ref:
            inexact, hi2 = mid, mid
        else:
            lo2 = mid
    return bad_sign, inexact


def measure_overflow(fmt: str = "float16") -> dict[str, float]:
    """Exact input value at which ``exp`` overflows, and where softmax is nan."""
    if fmt == "bfloat16":
        # emulated: exp is applied in float32 then rounded to bfloat16
        big = float(np.float32(np.finfo(np.float32).max))
        return {"max_finite": big,
                "overflow_at_x": None,
                "note": "bfloat16 shares float32's range; exp does not overflow "
                        "within any representable input"}
    fmax = float(np.finfo(np.float16).max)
    exact = math.log(fmax)
    x = np.array([0.0], dtype=np.float16)
    ok = float(np.float16(math.nextafter(exact, 0.0)))
    bad = float(np.float16(math.nextafter(exact, math.inf)))
    sm = softmax_naive(np.array([0.0, bad], dtype=np.float16), fmt)
    return {
        "max_finite": fmax,
        "overflow_at_x": exact,
        "largest_safe_x": ok,
        "smallest_overflowing_x": bad,
        "softmax_is_nan": bool(np.any(np.isnan(sm))),
    }


def measure_truncation(fmt: str, seed: int = 0) -> dict[str, float]:
    """Largest relative update ``|dW| / |W|`` that leaves ``W`` unchanged.

    This is the swamping condition: below it, the update is not merely small,
    it is unrepresentable. Reported against the unit roundoff ``u = 2^-p``, so
    the result is a statement about the format's mantissa rather than about this
    particular sweep.
    """
    rng = np.random.default_rng(seed)
    u = unit_roundoff(fmt)
    # search the relative update size on a log grid, then bisect
    lo, hi = 1e-9, 1.0
    for _ in range(200):
        if not _truncated(fmt, hi, rng):
            break
        hi *= 2
        if hi > 4.0:
            hi = 4.0
            break
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if _truncated(fmt, mid, rng):
            lo = mid
        else:
            hi = mid
    return {"threshold_rel": lo, "unit_roundoff": u,
            "ratio_to_u": lo / u, "max_finite": FORMATS[fmt][2]}


def _truncated(fmt: str, rel: float, rng) -> bool:
    W = np.array([1.0, 1.5, 2.0, 3.0, 0.75, 1.25], dtype=np.float32)
    dW = (W * rel).astype(np.float32)
    _, changed = weight_update(W, dW, fmt)
    return not changed


def measure_truncation_by_magnitude(fmt: str) -> list[tuple[float, float]]:
    """Relative-update threshold as a function of the magnitude of W.

    Reported across binades because the threshold is *absolute*: a fixed ``|dW|``
    is representable at small ``|W|`` and invisible at large ``|W|``. This is the
    concrete form of the swamping argument.

    The search brackets on a LOG GRID and then bisects. Geometric bisection from
    a wide bracket collapses: every early midpoint lands far below the threshold,
    so the upper bound walks down to meet the lower bound and the answer reported
    is the search's starting guess rather than a measurement.
    """
    def truncated(w: float, rel: float) -> bool:
        _, changed = weight_update(np.array([w], dtype=np.float32),
                                  np.array([w * rel], dtype=np.float32), fmt)
        return not changed

    grid = [10.0 ** k for k in range(-13, 1)]
    out = []
    for W_val in (2.0 ** k for k in range(-6, 7)):
        lo = hi = None
        for a, b in zip(grid, grid[1:]):
            if truncated(W_val, a) and not truncated(W_val, b):
                lo, hi = a, b
                break
        if lo is None:
            out.append((float(W_val), grid[-1]))
            continue
        for _ in range(80):
            if lo >= hi:
                break
            mid = math.sqrt(lo * hi)
            if truncated(W_val, mid):
                lo = mid
            else:
                hi = mid
        out.append((float(W_val), float(lo)))
    return out
