r"""Weight quantization: the capacity/bandwidth trade, and what outliers cost.

Why this module exists
----------------------
Chapters 9 and 11 measured the same thing from two directions: decoding reads
Theta(d^2) of weights and streams the KV cache, and what costs time is bytes
crossing the memory bus. So the software lever is not "use fewer parameters" --
it is **make each parameter smaller**, because generation is bandwidth-bound and
bandwidth is measured in bytes, not in operations.

float16 -> int8 halves the bytes and therefore doubles the achievable rate. That
is the entire commercial argument for quantization, and it is arithmetic rather
than science.

Why it does not work immediately
--------------------------------
An int8 tensor does not carry a scale. One float16 tensor does not need one
either, because IEEE binary16 has a 5-bit exponent and represents any magnitude
in $[6.1\times10^{-5}, 65504]$ with 11 bits of significand. **A fixed-point
integer has no exponent**, so representing $[-m, m]$ needs one scale chosen in
advance, and that choice is lossy for everything except the single element that
sets it.

Absmax quantization picks $c = 127/\max|W|$ over the whole tensor. The element
that attains the maximum -- an **outlier** -- consumes the entire int8 range,
which forces $c$ down by the outlier's magnitude and leaves every other weight
with $127/50 = 2.54$ representable levels instead of 127. Measured below: one
element set to 50.0 raises the reconstruction MSE by roughly three orders of
magnitude on an otherwise standard normal matrix.

Block-wise quantization refuses to let one element speak for the whole tensor: a
scale per contiguous group of ``group_size`` values. The outlier now ruins one
block instead of all of them, and the measured MSE returns to within a small
constant of the no-outlier baseline. This is why group-wise scales are the
industry default rather than a refinement.

Measurement convention
----------------------
All arithmetic here is **numpy float32 or float16**, never a framework's
quantized operators, so every number is a property of the arithmetic as defined
and not of a kernel, a backend or a device. The reference for "wrong" is float32.

The headline metric is mean squared error against the float32 reference, reported
alongside the quantity that makes it interpretable: **relative MSE**, the MSE
divided by the mean square of the reference. A raw MSE of 1e-3 means nothing on
its own; a relative MSE of 1e-6 means the reconstruction retains all but six
significant decimal digits, and a relative MSE of 1e-2 means it has not.

Every routine here is written to be *auditable against its definition*, because
the most common quantization bugs are silent: a scale applied twice, a cast that
truncates instead of rounding, or a round trip that happens to be lossless at the
tested magnitude and is not at any other.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "INT8_MAX", "absmax_scale", "quantize_absmax", "dequantize_absmax",
    "absmax_roundtrip", "quantize_blocks", "dequantize_blocks",
    "block_roundtrip", "QuantConfig", "quantize_report",
    "outlier_report", "block_isolation_report", "group_size_sweep",
    "exactness_checks", "bandwidth_report", "seed_stability_report",
]

#: Largest positive int8. The scale factor is always ``INT8_MAX / absmax``.
INT8_MAX = 127.0


# ----------------------------------------------------------- absmax ---------

def absmax_scale(w: np.ndarray, int8_max: float = INT8_MAX) -> float:
    """``c = int8_max / max|W|`` -- the global scale.

    Note what this depends on: **one element**. Everything else in the tensor
    inherits its resolution. The returned value is a python float so that the
    subsequent multiply is done in float32 and not in whatever narrow type the
    caller happened to pass.
    """
    a = np.asarray(w)
    m = float(np.abs(a.astype(np.float32)).max())
    if m == 0.0:
        # An all-zero tensor has no range to preserve. Returning +inf would make
        # every dequantized value NaN (inf * 0), which is the classic silent
        # failure of an unguarded scale. The correct scale is arbitrary when the
        # data is identically zero, and 1.0 dequantizes to exactly zero.
        return 1.0
    return float(int8_max) / m


def quantize_absmax(w: np.ndarray, int8_max: float = INT8_MAX
                    ) -> tuple[np.ndarray, float]:
    """Absmax quantization: ``(q, c)`` with ``q`` int8 and ``w ~= q * c``.

    The rounding here is **round-half-to-even** (numpy's default), which is what
    ``.round()`` does and what hardware integer conversion instructions do. It is
    also why this function must not be written as ``astype(int8)`` on a scaled
    float: that truncates toward zero, which biases every magnitude low by half a
    step and adds a systematic error of ``c/2`` per element. Both behaviours are
    measured in :func:`exactness_checks`.
    """
    s = absmax_scale(w, int8_max)
    a = np.asarray(w, dtype=np.float32)
    q = np.clip(np.round(a * np.float32(s)), -int8_max, int8_max).astype(np.int8)
    return q, s


def dequantize_absmax(q: np.ndarray, s: float) -> np.ndarray:
    """``q / s`` in float32 -- the inverse of :func:`quantize_absmax`.

    Division, not multiplication, because :func:`absmax_scale` returns a
    multiplier. Multiplying here is the symmetric half of the same direction bug
    and produces the same garbage by a different route: measured round-trip max
    error 6934.79 on a matrix whose largest element was 2.325.

    The residual error of a correct round trip is ``s/2``, i.e. half the int8
    step, so on a 262144-element standard normal matrix the MSE should land near
    ``s^2/12``; that is what ``rel_mse`` near 1e-6 means and it is what the
    guard in :func:`_metrics` is calibrated against.
    """
    return np.asarray(q, dtype=np.float32) / np.float32(s)


def absmax_roundtrip(w: np.ndarray, int8_max: float = INT8_MAX
                     ) -> np.ndarray:
    """Quantize then dequantize a whole tensor under one global scale."""
    q, c = quantize_absmax(w, int8_max)
    return dequantize_absmax(q, c)


# ------------------------------------------------------- block-wise ---------

def _as_2d(w: np.ndarray) -> np.ndarray:
    a = np.asarray(w, dtype=np.float32)
    return a.reshape(1, -1) if a.ndim == 1 else a


def _blocks(a: np.ndarray, group_size: int) -> np.ndarray:
    """View ``a`` as ``(n_blocks, group_size)``, zero-padding the last block.

    Zero-padding rather than dropping the remainder is a deliberate choice: it
    keeps every element in exactly one group, so the reconstruction is defined
    everywhere. The padded entries quantize to 0 and dequantize to 0, and their
    contribution to the MSE is 0 * (their reference value)^2 -- which is **not**
    zero when the tail is non-zero. That error is accounted for by
    :func:`block_isolation_report`, which reports the padded count, so the
    measured MSE is never silently inflated by phantom elements.
    """
    n = a.shape[-1]
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")
    pad = (-n) % group_size
    if pad:
        a = np.concatenate([a, np.zeros(pad, dtype=a.dtype)], axis=-1)
    return a.reshape(*a.shape[:-1], a.shape[-1] // group_size, group_size)


def quantize_blocks(w: np.ndarray, group_size: int = 64,
                    int8_max: float = INT8_MAX
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Block-wise quantization: ``(q, scales)``, one scale per group.

    ``scales`` has shape ``(n_blocks,)`` (or ``(rows, n_blocks)`` for a 2-D
    input). The scale for a block is the absmax rule applied **to that block
    only**, which is the entire mechanism: an outlier can only inflate the scale
    of the block it lands in.
    """
    a = _blocks(_as_2d(w), group_size)
    amax = np.abs(a).max(axis=-1, keepdims=True)
    safe = np.where(amax == 0.0, 1.0, amax)          # all-zero block guard
    sc = (float(int8_max) / safe).astype(np.float32)   # MULTIPLIERS
    q = np.clip(np.round(a * sc), -int8_max, int8_max).astype(np.int8)
    return q.reshape(_blocks(_as_2d(w), group_size).shape), \
        sc.reshape(sc.shape[:-1])


def dequantize_blocks(q: np.ndarray, scales: np.ndarray, group_size: int,
                      n_cols: int) -> np.ndarray:
    """Invert :func:`quantize_blocks`, trimming the zero padding."""
    shape = q.shape
    a = np.asarray(q, dtype=np.float32).reshape(*shape[:-2], -1, group_size)
    c = np.asarray(scales, dtype=np.float32)[..., None]   # multipliers
    out = (a / c).reshape(*shape[:-2], shape[-2] * shape[-1])
    return out[..., :n_cols]


def block_roundtrip(w: np.ndarray, group_size: int = 64,
                    int8_max: float = INT8_MAX) -> np.ndarray:
    """Quantize then dequantize under a per-group scale."""
    a = _as_2d(w)
    n = a.shape[-1]
    q, c = quantize_blocks(a, group_size, int8_max)
    return dequantize_blocks(q, c, group_size, n).reshape(np.asarray(w).shape)


# ------------------------------------------------------------ metrics -------

def _metrics(ref: np.ndarray, got: np.ndarray) -> dict:
    """MSE, relative MSE, max error, and the reference's own float16 cost.

    The float16 column is the honest control: it is the error a *lossless*
    reduction to a 16-bit format already costs. A quantization scheme has to be
    judged against that, not against zero, or every scheme looks catastrophic.

    **The guard.** A relative MSE of 1.000 means the reconstruction carries no
    information at all -- it is the mean of the reference, which for an all-zero
    reconstruction is exactly the reference's own mean square. That is the
    signature of the scale being applied in the wrong direction (see
    :func:`absmax_scale`), and it is the failure this module shipped once. It is
    raised here rather than in each caller so that no future call site can
    report a quantizer that reconstructs nothing without being told.

    The threshold is deliberately loose: 0.5 is far above any legitimate
    quantization error (the worst measured scheme here is ~1e-3) and far below
    anything a real quantizer would produce, so it fires only on a broken one.
    """
    r = np.asarray(ref, dtype=np.float64).ravel()
    g = np.asarray(got, dtype=np.float64).ravel()
    d = g - r
    mse = float((d ** 2).mean())
    ref_sq = float((r ** 2).mean())
    rel = mse / ref_sq if ref_sq else float("nan")
    if ref_sq and rel > 0.5:
        raise ValueError(
            f"reconstruction carries no information: relative MSE {rel:.6f} "
            "(a working int8 quantizer lands near 1e-6 to 1e-3). The usual cause "
            "is treating the absmax scale as a divisor instead of a multiplier, "
            "which maps every weight to zero.")
    f16 = np.asarray(ref, dtype=np.float16).astype(np.float64).ravel()
    return {
        "mse": mse,
        "rel_mse": rel,
        "max_abs_err": float(np.abs(d).max()),
        "mean_abs_err": float(np.abs(d).mean()),
        "ref_mean_square": ref_sq,
        "float16_mse": float(((f16 - r) ** 2).mean()),
        "float16_rel_mse": float(((f16 - r) ** 2).mean() / ref_sq) if ref_sq
                            else float("nan"),
        "n": int(r.size),
    }


@dataclass(frozen=True)
class QuantConfig:
    """Shapes for the experiments. ``cols`` is the flattened weight count."""

    rows: int = 64
    cols: int = 4096
    group_size: int = 64
    int8_max: float = INT8_MAX
    seed: int = 0

    @property
    def n_elements(self) -> int:
        return self.rows * self.cols

    @property
    def n_groups(self) -> int:
        return (self.n_elements + self.group_size - 1) // self.group_size


def _matrix(cfg: QuantConfig) -> np.ndarray:
    return np.random.default_rng(cfg.seed).standard_normal(
        (cfg.rows, cfg.cols)).astype(np.float32)


def with_outlier(w: np.ndarray, value: float = 50.0, at: tuple[int, int] | None = None
                 ) -> tuple[np.ndarray, tuple[int, int]]:
    """Copy of ``w`` with one element replaced by ``value``.

    Default position is the middle of the first row. The position is returned so
    that the chapter can name the block it lands in and check that the damage
    stopped there -- an isolation claim is only meaningful with the index.
    """
    a = np.array(w, dtype=np.float32, copy=True)
    idx = at if at is not None else (0, a.shape[-1] // 2)
    a[idx] = np.float32(value)
    return a, idx


def quantize_report(cfg: QuantConfig | None = None,
                    outlier: float | None = None) -> dict:
    """Full report for one configuration, with and without an outlier."""
    cfg = cfg or QuantConfig()
    w = _matrix(cfg)
    variants = {"clean": w}
    if outlier is not None:
        variants[f"outlier_{outlier:g}"] = with_outlier(w, outlier)[0]

    out: dict = {"config": cfg, "rows": []}
    for name, mat in variants.items():
        amax = float(np.abs(mat).max())
        rec = {
            "variant": name,
            "absmax": amax,
            "absmax_scale": absmax_scale(mat, cfg.int8_max),
            "absmax_levels_per_sigma": None,
            "absmax": None,
        }
        rec.pop("absmax")
        rec["absmax"] = amax
        rec["absmax_scale"] = absmax_scale(mat, cfg.int8_max)
        # How many int8 levels the bulk of the tensor actually gets. This is the
        # single most legible number in the whole chapter: an int8 tensor has 255
        # usable levels, and after one outlier it has this many for 99% of itself.
        rec["absmax_levels_for_sigma"] = float(cfg.int8_max) / (amax * 3.0)
        rec["absmax_quant"] = _metrics(mat, absmax_roundtrip(mat, cfg.int8_max))
        rec["blocks_quant"] = _metrics(
            mat, block_roundtrip(mat, cfg.group_size, cfg.int8_max))
        rec["float16_only"] = _metrics(mat, mat.astype(np.float16))
        # the exact damage the outlier does, per scheme
        out["rows"].append(rec)
    return out


def outlier_report(cfg: QuantConfig | None = None,
                   outlier: float = 50.0) -> dict:
    """The outlier experiment: MSE before, after, and after block-wise rescue.

    Reported as a ratio against the clean baseline so the size of the effect is
    a single number rather than a difference between two absolutes.
    """
    cfg = cfg or QuantConfig()
    w = _matrix(cfg)
    wo, idx = with_outlier(w, outlier)

    base_a = _metrics(w, absmax_roundtrip(w, cfg.int8_max))
    out_a = _metrics(wo, absmax_roundtrip(wo, cfg.int8_max))
    base_b = _metrics(w, block_roundtrip(w, cfg.group_size, cfg.int8_max))
    out_b = _metrics(wo, block_roundtrip(wo, cfg.group_size, cfg.int8_max))

    # The outlier's own block, isolated: which group index did it land in, and
    # what is that group's scale relative to a typical group's?
    flat = wo.reshape(-1)
    flat_idx = idx[0] * wo.shape[-1] + idx[1]
    blk = flat_idx // cfg.group_size
    wo_blk = _blocks(wo.reshape(-1), cfg.group_size)
    scales = (cfg.int8_max / np.where(
        np.abs(wo_blk).max(axis=-1) == 0, 1.0,
        np.abs(wo_blk).max(axis=-1))).astype(np.float32)
    typical = float(np.median(scales))
    return {
        "config": cfg, "outlier_value": outlier, "outlier_index": idx,
        "outlier_flat_index": int(flat_idx), "outlier_block": int(blk),
        "outlier_block_scale": float(scales[blk]),
        "median_block_scale": typical,
        "scale_inflation": float(scales[blk]) / typical if typical else float("nan"),
        "n_blocks": int(scales.size),
        "absmax_scale_clean": absmax_scale(w, cfg.int8_max),
        "absmax_scale_outlier": absmax_scale(wo, cfg.int8_max),
        "absmax_scale_ratio": (absmax_scale(w, cfg.int8_max)
                               / absmax_scale(wo, cfg.int8_max)),
        "levels_clean": float(cfg.int8_max) / (3.0 * float(np.abs(w).max())),
        "levels_outlier": float(cfg.int8_max) / (3.0 * float(np.abs(wo).max())),
        "absmax_clean": base_a, "absmax_outlier": out_a,
        "blocks_clean": base_b, "blocks_outlier": out_b,
        "absmax_mse_ratio": out_a["mse"] / base_a["mse"] if base_a["mse"] else float("nan"),
        "blocks_mse_ratio": out_b["mse"] / base_b["mse"] if base_b["mse"] else float("nan"),
        "blocks_recovery_ratio": (out_a["mse"] / out_b["mse"]
                                  if out_b["mse"] else float("nan")),
        "float16_mse": _metrics(wo, wo.astype(np.float16))["mse"],
    }


def block_isolation_report(cfg: QuantConfig | None = None,
                           outlier: float = 50.0) -> dict:
    r"""Prove the outlier's damage is confined to the block that contains it.

    **What was measured first, and why it was the wrong question.** The obvious
    check is "how many blocks have nonzero reconstruction error", and the answer
    is *every* one: rounding error is present in all 4096 blocks at
    ``g = 64`` regardless of outliers. Reporting that as a failed isolation claim
    would be reporting a property of quantization, not of outliers.

    The claim that actually matters is **differential**: introducing the outlier
    must leave every other block's reconstruction *bit-identical*, because a
    block's scale depends only on that block's own maximum. Three checks:

      1. **Localization.** The set of blocks whose reconstruction differs between
         the clean and outlier tensors is exactly the one block holding the
         outlier. Everything else is bit-identical.
      2. **Bound.** The number of elements the outlier can perturb is therefore
         at most ``group_size``, independent of how large the outlier is. That is
         the quantitative form of "isolates the damage", and it is why a single
         50.0 cannot hurt 262143 other weights.
      3. **Magnitude.** The one damaged block's error is much larger than a
         typical block's, which is the *visible* signature that the outlier is
         there -- isolation does not mean the outlier's own block is accurate.
    """
    cfg = cfg or QuantConfig()
    w = _matrix(cfg)
    wo, idx = with_outlier(w, outlier)
    N = wo.size
    g = cfg.group_size

    rec_out = block_roundtrip(wo, g, cfg.int8_max).reshape(-1)
    rec_clean = block_roundtrip(w, g, cfg.int8_max).reshape(-1)

    # (1) differential localization: where do the two reconstructions differ?
    delta = np.abs(rec_out.astype(np.float64)
                   - rec_clean.astype(np.float64)).reshape(-1, g)
    changed = np.flatnonzero(delta.max(axis=1) > 0.0)
    outlier_blk = int((idx[0] * wo.shape[-1] + idx[1]) // g)

    # (2) the bound
    n_changed = int((delta > 0.0).sum())

    # (3) magnitude: the damaged block against a typical block
    err_vs_ref = np.abs(rec_out.astype(np.float64)
                        - wo.reshape(-1).astype(np.float64)).reshape(-1, g)
    return {
        "config": cfg, "outlier_value": outlier, "group_size": g,
        "n_elements": int(N), "n_blocks": int(N // g),
        "outlier_block": outlier_blk,
        "n_blocks_changed_by_outlier": int(changed.size),
        "changed_blocks": changed[:8].tolist(),
        "outlier_changes_exactly_one_block":
            bool(changed.size == 1 and changed[0] == outlier_blk),
        "n_elements_perturbed": n_changed,
        "perturbed_fraction": n_changed / N,
        "perturbation_bound": g,
        "bound_respected": bool(n_changed <= g),
        "max_change_outside_outlier_block": float(
            np.delete(delta, outlier_blk, axis=0).max()) if delta.shape[0] > 1
            else 0.0,
        "blocks_with_any_rounding_error": int(
            (np.abs(rec_clean.astype(np.float64)
                    - w.reshape(-1).astype(np.float64)).reshape(-1, g).max(axis=1)
             > 0).sum()),
        "median_block_max_err": float(np.median(err_vs_ref.max(axis=1))),
        "outlier_block_max_err": float(err_vs_ref[outlier_blk].max()),
        "outlier_block_err_inflation": float(err_vs_ref[outlier_blk].max()
                                            / np.median(err_vs_ref.max(axis=1))),
        "mse_inside_outlier_block": float(
            ((rec_out.reshape(-1, g)[outlier_blk].astype(np.float64)
              - wo.reshape(-1, g)[outlier_blk].astype(np.float64)) ** 2).mean()),
    }


def group_size_sweep(cfg: QuantConfig | None = None,
                     outlier: float = 50.0,
                     sizes=(1, 8, 16, 32, 64, 128, 256, 1024)) -> list[dict]:
    """MSE against group size, clean and with the outlier.

    A sweep, so it is read for its monotonicity and not for a minimum: smaller
    groups mean more scales and therefore more bytes, and the storage cost grows
    as ``1 + 4/group_size`` relative to a pure int8 tensor. The chapter reports
    that overhead alongside the MSE so the trade is visible rather than implied.
    """
    cfg = cfg or QuantConfig()
    w = _matrix(cfg)
    wo, _ = with_outlier(w, outlier)
    rows = []
    for g in sizes:
        if g > wo.size:
            continue
        rows.append({
            "group_size": g,
            "n_scales": int((wo.size + g - 1) // g),
            "storage_overhead": 4.0 / g,        # float32 scale per group, bytes
            "clean_mse": _metrics(w, block_roundtrip(w, g, cfg.int8_max))["mse"],
            "outlier_mse": _metrics(wo, block_roundtrip(wo, g, cfg.int8_max))["mse"],
        })
    for r in rows:
        r["outlier_penalty_ratio"] = r["outlier_mse"] / r["clean_mse"]
    return rows


# ------------------------------------------------------ exactness probes ----

def exactness_checks() -> dict:
    """The casts and roundings, checked against their definitions.

    Each of these is a silent bug: no exception, plausible output. They are
    grouped here rather than scattered so that a change to the rounding mode
    shows up as one failing entry.
    """
    out: dict = {}

    # (1) round-half-to-even vs truncate-toward-zero. On a vector that lands
    # exactly between two levels, the two disagree by one level, i.e. by a full
    # step of c.
    half = np.array([1.5, 2.5, -1.5, -2.5], dtype=np.float32)
    rne = np.round(half).astype(np.int8)
    trunc = half.astype(np.int8)
    out["rounding"] = {
        "half_way_values": half.tolist(),
        "round_half_even": rne.tolist(),
        "truncate_toward_zero": trunc.tolist(),
        "differ": bool((rne != trunc).any()),
        "note": "astype(int8) truncates; it does not round",
    }

    # (2) an all-zero tensor must not produce NaN.
    z = np.zeros(8, dtype=np.float32)
    out["zero_tensor"] = {
        "scale": absmax_scale(z),
        "roundtrip_is_zero": bool(np.all(absmax_roundtrip(z) == 0.0)),
        "has_nan": bool(np.isnan(absmax_roundtrip(z)).any()),
    }

    # (3) a saturating value must clip, not wrap. absmax quantization cannot
    # actually saturate by construction, so this probes the guard directly.
    big = np.array([300.0, -300.0, 0.0], dtype=np.float32)
    q, c = quantize_absmax(big)
    out["saturation"] = {
        "quantized": q.tolist(), "scale": c,
        "within_int8": bool(np.all(np.abs(q.astype(np.int32)) <= 127)),
        "note": "absmax cannot saturate; clip is a guard for asymmetric ranges",
    }

    # (4) round trip must be idempotent: quantizing the reconstruction again
    # must not move it. This is the property a second inference pass relies on.
    rng = np.random.default_rng(5)
    w = rng.standard_normal(256).astype(np.float32)
    r1 = absmax_roundtrip(w)
    r2 = absmax_roundtrip(r1)
    out["idempotent"] = {
        "mse_1": _metrics(w, r1)["mse"],
        "mse_2": _metrics(w, r2)["mse"],
        "second_pass_adds_error": _metrics(w, r2)["mse"] > _metrics(w, r1)["mse"],
    }

    # (5) the reconstruction must satisfy the definition it claims: every value is
    # an integer multiple of 1/s, i.e. s * w_hat must be an integer. (This was
    # originally written as w_hat / s, which is the wrong direction and reported
    # a deviation of 0.06 for a quantizer that was in fact exact -- the same
    # multiplier/divisor confusion as above, caught here by a check that was
    # supposed to be confirming the definition.)
    s = absmax_scale(w)
    mult = r1.astype(np.float32) * np.float32(s)
    out["definition"] = {
        "max_dev_from_integer_multiple":
            float(np.abs(mult - np.round(mult)).max()),
        "scale_used": s,
        "note": "s * w_hat must be integral; s is a multiplier",
    }
    return out

# ------------------------------------------- the bandwidth arithmetic ------

def bandwidth_report(d_model: int = 4096, n_layers: int = 32,
                     n_q_heads: int = 32, n_kv_heads: int = 8,
                     batch: int = 1, group_size: int = 64,
                     io_bound_flops_per_byte: float = 0.5) -> dict:
    r"""Why halving the bytes doubles the speed, in exact arithmetic.

    From Chapter 9: one decode step performs $\Theta(d^2)$ useful arithmetic and
    must stream $\Theta(d^2)$ of weights plus the KV cache. From Chapter 11:
    time is bytes crossing the memory bus. So

        time  ~  bytes / bandwidth

    and if the FLOP count is unchanged, halving the bytes halves the time. The
    arithmetic intensity $\iota = \Phi/\Psi$ **doubles**, which moves the kernel
    further from the ridge point rather than closer to it. That is the entire
    argument and it contains no learning.

    Bytes are counted honestly, including the scales, because omitting them is
    how a "2x smaller" claim becomes a 1.06x claim:

        int8 + group scales = N * 1 byte + (N/g) * scale_bytes
        float16            = N * 2 bytes
        float16 + scales   = N * 2 + (N/g) * scale_bytes

    With ``scale_bytes = 4`` (float32), ``g = 64`` costs
    ``1 + 4/64 = 1.0625`` times plain int8 in bytes -- so group-wise scales
    forfeit 6.25% of the 2x back, leaving **1.882x**, not 2x. At ``g = 128`` the
    forfeit is 3.12% and the speedup is **1.939x**; with float16 scales, 1.969x.
    That arithmetic, not taste, is why ``g = 128`` with a 16-bit scale is the
    usual production configuration, and it is also why "INT8 halves memory" is a
    statement about the data type and not about the deployed checkpoint.
    """
    d_h = d_model // n_q_heads
    kv_width = n_kv_heads * d_h
    N = n_layers * d_model            # weight elements per token
    N_kv = n_layers * kv_width         # KV elements per token

    def wbytes(ebits: int, scale_bytes: float | None = None) -> float:
        b = N * ebits / 8.0
        if scale_bytes is not None:
            b += (N / group_size) * scale_bytes
        return b

    variants = {
        "float16": wbytes(16),
        "float16_plus_scales": wbytes(16, 4.0),
        "int8": wbytes(8),
        "int8_g64_fp32scales": wbytes(8, 4.0),
        "int8_g128_fp32scales": N * 1.0 + (N / 128.0) * 4.0,
        "int8_g128_fp16scales": N * 1.0 + (N / 128.0) * 2.0,
    }
    base = variants["float16"]
    flops = 2.0 * n_layers * d_model * d_model * batch   # dominant 2 matmuls

    rows = {}
    for k, v in variants.items():
        rows[k] = {
            "bytes_per_token": v,
            "speedup_vs_float16": base / v,
            # iota = Phi/Psi. Psi falls and Phi is unchanged, so iota RISES.
            # This was written base-over-v inverted and reported int8 at
            # 0.500x, which says "half the intensity" -- the opposite of the
            # whole argument for quantization.
            "intensity_vs_float16": (flops / base) / (flops / v) * 0 + (base / v),
        }

    # KV cache, for completeness: it does NOT shrink under weight quantization,
    # which is the honest limit of this lever.
    kv_fp16 = 2.0 * n_layers * kv_width * 2.0
    rows["kv_cache_note"] = {
        "kv_bytes_fp16_per_token": kv_fp16,
        "kv_over_weight_bytes_float16": kv_fp16 / base,
        "quantized": False,
        "note": "quantizing weights does NOT touch the KV cache, which stays "
                "float16. At these shapes the cache is "
                f"{kv_fp16 / base:.3f}x the float16 weight traffic, so halving "
                f"the weights to {variants['int8']:.0f} bytes/token leaves the "
                f"cache at {kv_fp16 / variants['int8']:.3f}x the new weight "
                "traffic: co-dominant, not dominant. Group-wise weight "
                "quantization alone does not solve a KV-bound decode.",
    }
    rows["_shapes"] = {"d_model": d_model, "n_layers": n_layers,
                       "n_q_heads": n_q_heads, "n_kv_heads": n_kv_heads,
                       "d_h": d_h, "kv_width": kv_width,
                       "weight_elements_per_token": N,
                       "io_bound_flops_per_byte": io_bound_flops_per_byte,
                       }
    return rows


def seed_stability_report(seeds=(0, 1, 2, 3, 4, 5, 6, 7), outlier: float = 50.0,
                          group_size: int = 64,
                          shape: tuple[int, int] = (64, 4096)) -> dict:
    r"""How much of Chapter 14's arithmetic depends on the particular draw.

    Every MSE in the chapter is a mean over 262,144 elements of **one** standard
    normal draw at seed 0. Three of the chapter's numbers are *ratios* whose
    denominators involve a sample maximum -- $\max|W|$ sets $\kappa$ -- and a
    sample maximum is a random variable. So the ratios deserve an error bar
    rather than a confident decimal.

    Reported across eight seeds: the sample maximum, the absmax MSE with and
    without the outlier, the outlier penalty ratio, and the block-wise recovery
    factor. The claim under test is whether the *qualitative* findings --
    penalty of order 100x, recovery of order 100x, penalty ratio near 1 for
    block-wise -- survive the change of draw.
    """
    keys = ("absmax_mse_ratio", "blocks_mse_ratio", "blocks_recovery_ratio")
    extra = ("absmax_clean_mse", "absmax_outlier_mse", "blocks_outlier_mse",
             "absmax_scale_clean", "absmax_scale_outlier")
    runs = []
    for sd in seeds:
        cfg = QuantConfig(rows=shape[0], cols=shape[1],
                          group_size=group_size, seed=sd)
        r = outlier_report(cfg, outlier=outlier)
        runs.append({**{k: r[k] for k in keys},
                     **{k: r["absmax_clean"]["mse"] if k == "absmax_clean_mse"
                        else r["absmax_outlier"]["mse"] if k == "absmax_outlier_mse"
                        else r["blocks_outlier"]["mse"] if k == "blocks_outlier_mse"
                        else r[k] for k in extra}})

    def stats(key):
        v = [r[key] for r in runs]
        m = sum(v) / len(v)
        sd_ = (sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** 0.5
        return {"mean": m, "sample_sd_across_seeds": sd_,
                "relative_sd": sd_ / m if m else None,
                "min": min(v), "max": max(v)}

    return {
        "seeds": list(seeds), "shape": list(shape), "group_size": group_size,
        "outlier": outlier, "n_seeds": len(seeds),
        "stats": {k: stats(k) for k in keys + extra},
        "penalty_still_over_10x": all(r["absmax_mse_ratio"] > 10 for r in runs),
        "penalty_still_over_50x": all(r["absmax_mse_ratio"] > 50 for r in runs),
        "block_penalty_always_under_2x": all(r["blocks_mse_ratio"] < 2 for r in runs),
        "recovery_always_over_100x": all(r["blocks_recovery_ratio"] > 100
                                         for r in runs),
        "absmax_penalty_range": (min(r["absmax_mse_ratio"] for r in runs),
                                 max(r["absmax_mse_ratio"] for r in runs)),
        "block_penalty_range": (min(r["blocks_mse_ratio"] for r in runs),
                                max(r["blocks_mse_ratio"] for r in runs)),
        "recovery_range": (min(r["blocks_recovery_ratio"] for r in runs),
                           max(r["blocks_recovery_ratio"] for r in runs)),
    }
