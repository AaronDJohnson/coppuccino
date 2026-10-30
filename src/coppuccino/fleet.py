"""Fleet fitting: train one bounded-support copula flow per source, simultaneously.

Designed for the LISA galactic-binary use case: ~10^4 sources, each with its
own bounded posterior samples, whose proposal flows are refit periodically.
A single flow fit is a tiny network that would leave a GPU idle; the win is
that all sources' fits are the *same program with different data and
parameters*, which ``jax.vmap`` fuses into single kernels: one optimizer step
advances every source's flow at once. The same code runs on CPU (where vmap
still vectorizes) and becomes fast on a GPU simply by installing ``jax[cuda]``.

Architecture is FIXED across the fleet (vmap requires identical parameter
pytrees): the default ``knots=4, flow_layers=6`` is the measured middle
ground for galactic-binary posteriors -- near-optimal on simple (few-parameter,
unimodal) fits and mid-rung on hard (8-parameter, bimodal) ones. Both are
exposed as kwargs.

Pipeline per source (boundary-safe by construction):

1. *Marginals*: empirical-quantile monotone RQS CDFs built on CPU (cheap,
   O(n log n) per source-dimension) and stored as stacked knot tensors
   ``(n_sources, dim, K)`` -- no per-source Python closures, so evaluation is
   vmappable end to end. Endpoints map to exactly 0/1 at the prior bounds
   (Hazen plotting positions; strictly positive wall density).
2. *Copula*: an RQS coupling flow on the unit cube over a uniform base, one
   parameter stack for the whole fleet.

Example
-------
>>> import numpy as np
>>> from coppuccino import normalizing_flows_fit_fleet
>>> chains = np.random.rand(64, 2000, 8)          # (n_sources, n_samples, dim)
>>> bounds = np.tile(np.array([0.0, 1.0]), (8, 1))
>>> fleet = normalizing_flows_fit_fleet(chains, prior_bounds=bounds,
...                                     max_epochs=100)      # doctest: +SKIP
>>> x = fleet.sample(512, rng_seed=0)             # (64, 512, 8)  # doctest: +SKIP
>>> lp = fleet.log_prob(x)                        # (64, 512)     # doctest: +SKIP
>>> x7 = fleet.sample(512, rng_seed=0, source_index=7)  # doctest: +SKIP
"""
import io
from typing import Any, Callable, Optional

import cloudpickle
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from flowjax.bijections import RationalQuadraticSpline
from flowjax.distributions import Uniform
from flowjax.flows import coupling_flow

from coppuccino.bijections import (
    _POS_FLOOR,
    _rqs_derivative,
    _rqs_forward,
    _rqs_inverse,
)
from coppuccino.model_io import (
    FLEET_KIND,
    FORMAT_VERSION,
    _atomic_write_bytes,
    _check_payload,
)

__all__ = ["normalizing_flows_fit_fleet", "FleetFlow"]

_MIN_SAMPLES = 64
# Source-block size for the knot sort, in float64 elements (~64 MB).
_SORT_BLOCK_ELEMENTS = 8_000_000
# Number of live (block, dim, K) temporaries the vectorized PCHIP slope kernel
# holds at its peak; the block loop divides the sort budget by this so that the
# two phases have comparable high-water marks.
_SLOPE_TEMPORARIES = 8
# Fleet size at which the traced `_stack_params` build overtakes the Python
# loop. See that function's Notes for the measurement.
_VMAP_STACK_MIN_SOURCES = 300


# ----------------------------------------------------------------------
# input validation (parity with copula_flows._validate_prior_bounds)
# ----------------------------------------------------------------------

def _validate_fleet_bounds(prior_bounds, n_src: int, dim: int
                           ) -> tuple[np.ndarray, np.ndarray]:
    """Coerce ``prior_bounds`` to per-source ``lo`` / ``hi`` arrays and check them.

    The fleet's marginals anchor their outer knots at the bounds, so a
    misordered, degenerate or non-finite box is not a slightly-wrong fit but a
    knot vector that is not increasing. Without this the failure modes all
    surfaced somewhere else: reversed and equal bounds as "chains contain
    samples outside prior_bounds", which points at the data rather than at the
    bounds, and non-finite bounds as a bare scipy complaint about non-finite
    ``x`` that names no coppuccino symbol at all.

    Parameters
    ----------
    prior_bounds : array_like
        ``(dim, 2)`` shared bounds or ``(n_sources, dim, 2)`` per source.
    n_src, dim : int
        Fleet shape, taken from ``chains``.

    Returns
    -------
    tuple of np.ndarray
        ``(lo, hi)``, both ``(n_sources, dim)`` float arrays.

    Raises
    ------
    ValueError
        If the shape is wrong, or any row is non-finite or not strictly
        increasing.
    """
    pb = np.asarray(prior_bounds, dtype=float)
    shared = pb.shape == (dim, 2)
    if shared:
        pb = np.broadcast_to(pb, (n_src, dim, 2))
    if pb.shape != (n_src, dim, 2):
        raise ValueError(f"prior_bounds must be (dim, 2) = {(dim, 2)} or "
                         f"(n_sources, dim, 2) = {(n_src, dim, 2)}; "
                         f"got {pb.shape}.")

    def where(s: int, j: int) -> str:
        return (f"prior_bounds row {j}" if shared
                else f"prior_bounds[{s}] row {j}")

    nonfinite = ~np.isfinite(pb).all(axis=-1)
    if nonfinite.any():
        s, j = (int(v) for v in np.argwhere(nonfinite)[0])
        raise ValueError(
            f"{where(s, j)} must be finite; got "
            f"[{pb[s, j, 0]:g}, {pb[s, j, 1]:g}]. The fleet models a bounded "
            f"prior box exactly and puts its outermost marginal knots on the "
            f"bounds, so an infinite or NaN side has no meaning here -- pass "
            f"a finite box, or use normalizing_flows_fit for an unbounded "
            f"parameter.")
    # ``~(low < high)`` rather than ``low >= high`` to stay parallel with
    # copula_flows._validate_prior_bounds; NaN is already gone by here.
    misordered = ~(pb[..., 0] < pb[..., 1])
    if misordered.any():
        s, j = (int(v) for v in np.argwhere(misordered)[0])
        raise ValueError(
            f"{where(s, j)} must be [low, high] with low < high; got "
            f"[{pb[s, j, 0]:g}, {pb[s, j, 1]:g}].")
    return pb[..., 0].copy(), pb[..., 1].copy()


def _validate_fleet_chains(chains: np.ndarray, lo: np.ndarray, hi: np.ndarray
                           ) -> tuple[np.ndarray, np.ndarray]:
    """Check a fleet's chains against its bounds; return per-source min/max.

    Covers the two states the fitter used to accept or misdiagnose: any
    non-finite entry (``+inf`` was reported as a containment failure) and a
    column that never varies. A constant column has no empirical CDF -- its
    knots collapse onto the tie bumps and the fitted marginal then samples
    that parameter across the whole prior box -- and the single-flow path
    already rejects it, so the fleet does too.
    """
    bad = ~np.isfinite(chains)
    if bad.any():
        s, _, j = (int(v) for v in np.argwhere(bad)[0])
        n_nan = int(np.isnan(chains).sum())
        raise ValueError(
            f"chains hold non-finite values: {int(bad.sum())} in total, of "
            f"which {n_nan} are NaN; the first is at source {s}, parameter "
            f"(column) {j}. Clean or subsample upstream.")
    cmin, cmax = chains.min(axis=1), chains.max(axis=1)         # (N, d) each
    outside = (cmin < lo) | (cmax > hi)
    if outside.any():
        s, j = (int(v) for v in np.argwhere(outside)[0])
        raise ValueError(
            f"chains contain samples outside prior_bounds: source {s}, "
            f"parameter (column) {j} spans [{cmin[s, j]:g}, {cmax[s, j]:g}] "
            f"but its bounds are [{lo[s, j]:g}, {hi[s, j]:g}].")
    degenerate = cmax <= cmin
    if degenerate.any():
        s, j = (int(v) for v in np.argwhere(degenerate)[0])
        raise ValueError(
            f"source {s}, parameter (column) {j}: every one of the "
            f"{chains.shape[1]} samples equals {cmin[s, j]:g}. A marginal CDF "
            f"needs at least two distinct sample values; drop the column or "
            f"supply a chain that explores it.")
    return cmin, cmax


# ----------------------------------------------------------------------
# marginal knots (CPU construction; stacked arrays, no closures)
# ----------------------------------------------------------------------

def _bump_ties(a: np.ndarray, eps) -> np.ndarray:
    """Make ``a`` strictly increasing along its last axis, in place.

    Ties are pushed ``eps`` past their predecessor, left to right. The scan is
    sequential, but the *test* is not: rows that are already increasing (the
    overwhelming majority for continuous posteriors) are found with one
    vectorized pass and left bit-for-bit alone. ``eps`` broadcasts against
    ``a[..., 0]``.
    """
    eps = np.broadcast_to(eps, a.shape[:-1])
    for idx in np.argwhere(~np.all(np.diff(a, axis=-1) > 0.0, axis=-1)):
        row, step = a[tuple(idx)], float(eps[tuple(idx)])
        for i in range(1, row.shape[0]):
            if row[i] <= row[i - 1]:
                row[i] = row[i - 1] + step
    return a


def _pchip_knot_derivatives(xk: np.ndarray, yk: np.ndarray) -> np.ndarray:
    """Monotone PCHIP knot slopes over a stack of knot rows, floored positive.

    Bit-for-bit equal to :func:`coppuccino.bijections._rqs_knot_derivatives` at
    its defaults applied to each ``(source, dimension)`` row -- which is what
    it replaces. That loop is one ``scipy.interpolate.PchipInterpolator``
    construction per source-dimension (80,000 of them at the ``10^4``-source
    scale in the module docstring) and dominated the marginal build; the array
    form runs the same arithmetic once over the whole stack.

    Two details are load-bearing for the bit-identity. SciPy's Fritsch-Carlson
    branch writes exactly 0 wherever an adjacent secant vanishes or changes
    sign and the weighted harmonic mean elsewhere, and its endpoints use the
    one-sided three-point estimate with Moler's two shape-preserving
    corrections. And ``PchipInterpolator(...).derivative()(xk)`` does not read
    the slopes back: the *last* knot is evaluated as the last cubic segment's
    derivative at its right end, so that quadratic has to be reproduced term
    by term rather than short-circuited to ``dk[-1]``.

    Parameters
    ----------
    xk : np.ndarray
        Strictly increasing knot locations, shape ``(..., K)``, ``K >= 3``.
    yk : np.ndarray
        Strictly increasing knot values, same shape as ``xk``.

    Returns
    -------
    np.ndarray
        Knot derivatives, shape ``(..., K)``, clamped at ``_POS_FLOOR``.

    Raises
    ------
    ValueError
        If ``xk`` has fewer than 3 knots, or any row is not strictly
        increasing.
    """
    if xk.shape[-1] < 3:
        raise ValueError(f"_pchip_knot_derivatives needs at least 3 knots; "
                         f"got {xk.shape[-1]}.")
    hk = xk[..., 1:] - xk[..., :-1]
    # scipy rejected a non-increasing `x` and was right to. Every branch below
    # stays finite on one -- a negative `hk` only flips the sign of a secant,
    # which reads as an ordinary sign change -- so without this the caller gets
    # a plausible-looking slope vector for knots that are not a function, and
    # the fit is silently wrong rather than absent. One pass over an array that
    # is already materialized.
    if not np.all(hk > 0.0):
        bad = np.argwhere(hk <= 0.0)[0]
        row, k = tuple(int(v) for v in bad[:-1]), int(bad[-1])
        n_bad = int(np.count_nonzero(np.any(hk <= 0.0, axis=-1)))
        raise ValueError(
            f"_pchip_knot_derivatives needs strictly increasing knots; "
            f"{n_bad} of {hk[..., 0].size} rows in this batch are not. Row "
            f"{row} has x[{k}] = {xk[row][k]!r} >= x[{k + 1}] = "
            f"{xk[row][k + 1]!r}. From a fleet build this is the marginal "
            f"knot grid failing to fit inside the prior box: when the box "
            f"spans only a few hundred float64 ULP at its own magnitude, the "
            f"tie bumps walk the quantile grid past the upper bound and the "
            f"endpoint guard pulls only the last interior knot back. Rescale "
            f"or recentre the parameter so its box is wider than the rounding "
            f"error of its own endpoints.")
    mk = (yk[..., 1:] - yk[..., :-1]) / hk

    def edge(h0, h1, m0, m1):
        # one-sided three-point estimate, then Moler's corrections: kill it if
        # it disagrees in sign with the adjacent secant, cap it at 3 x that
        # secant when the two secants themselves disagree. That second sign
        # test is implied by surviving the first -- |d| > 3|m0| with sign(d) ==
        # sign(m0) forces sign(m1) != sign(m0) -- so no input can distinguish
        # it from the bare magnitude test; it is spelled out to stay a
        # line-by-line transcription of scipy's `_edge_case`.
        d = ((2 * h0 + h1) * m0 - h0 * m1) / (h0 + h1)
        flip = np.sign(d) != np.sign(m0)
        steep = ((np.sign(m0) != np.sign(m1))
                 & (np.abs(d) > 3.0 * np.abs(m0)))
        return np.where(flip, 0.0, np.where(steep, 3.0 * m0, d))

    dk = np.empty_like(xk)
    flat = ((np.sign(mk[..., 1:]) != np.sign(mk[..., :-1]))
            | (mk[..., 1:] == 0) | (mk[..., :-1] == 0))
    w1 = 2 * hk[..., 1:] + hk[..., :-1]
    w2 = hk[..., 1:] + 2 * hk[..., :-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        # the divisions that blow up are exactly the ones `flat` discards
        whmean = (w1 / mk[..., :-1] + w2 / mk[..., 1:]) / (w1 + w2)
        dk[..., 1:-1] = np.where(flat, 0.0, 1.0 / whmean)
    dk[..., 0] = edge(hk[..., 0], hk[..., 1], mk[..., 0], mk[..., 1])
    dk[..., -1] = edge(hk[..., -1], hk[..., -2], mk[..., -1], mk[..., -2])

    # replay the last Hermite segment's derivative at its right end, the way
    # PPoly evaluates it: c = [t/h, (m - d0)/h - t, d0, y0] and the derivative
    # polynomial accumulates c2 + 2*c1*h + 3*c0*h*h in that order
    h, m, d0 = hk[..., -1], mk[..., -1], dk[..., -2]
    t = (d0 + dk[..., -1] - 2 * m) / h
    dk[..., -1] = d0 + (2 * ((m - d0) / h - t)) * h + (3 * (t / h)) * (h * h)
    return np.maximum(dk, _POS_FLOOR)


def _build_fleet_knots(chains: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                       num_points: int):
    """Stacked marginal knots for the whole fleet: three (N, d, K) arrays.

    Per source-dimension the knots have fixed length ``num_points + 2``:
    interior knots at the data quantiles with Hazen ``(counts - 1/2)/n`` CDF
    values, plus the exact endpoints ``(lo, 0)`` and ``(hi, 1)``. ``dk`` are
    monotone PCHIP slopes (strictly positive), so each rational-quadratic
    spline is an exact bijection ``[lo, hi] <-> [0, 1]`` with positive wall
    density.

    Sorting, the quantile grid, the tie bumps and the PCHIP slopes all run over
    ``(source, dimension)`` pairs at once; ``np.searchsorted`` needs a 1-D
    haystack, so that one stays in the Python loop below.
    """
    n_src, n, dim = chains.shape
    span = np.maximum(hi - lo, _POS_FLOOR)                     # (N, d)
    # keep the quantile grid strictly inside the bounds: samples sitting
    # exactly on a bound (e.g. clipped data) would otherwise collide with the
    # endpoint knots after tie-bumping.
    #
    # The inset and the tie bump are scaled by whichever span is *smaller*,
    # the prior box's or the data's. A prior-relative scale alone is wrong
    # whenever the box is much wider than the posterior: under
    # prior_bounds=[[0, 1e14]] on a chain spanning a few hundred, 1e-9 * span
    # is 1e5, so the clip below moves the entire chain onto one point and the
    # fit is silently meaningless. What counts as "sitting on the bound" is
    # set by the resolution of the samples, not by how uninformative the prior
    # is. This mirrors bijections.make_smooth_marginal, whose eps0 had the
    # same defect. The ULP floor keeps ``bound + margin`` representable when
    # the data span underflows the bounds' own magnitude.
    data_span = np.maximum(chains.max(axis=1) - chains.min(axis=1), _POS_FLOOR)
    scale = np.minimum(span, data_span)                        # (N, d)
    ulp = 4.0 * np.spacing(np.maximum(np.abs(lo), np.abs(hi)))
    margin = np.maximum(1e-9 * scale, ulp)
    probs = np.linspace(0.0, 1.0, num_points)
    # strictly increasing interior knots (bump ties; preserves count)
    eps_x = np.maximum(1e-12 * scale, ulp)                     # (N, d)

    xq = np.empty((n_src, dim, num_points))
    yq = np.empty((n_src, dim, num_points))
    # The sorted copy is the high-water mark here -- the chain length dwarfs
    # num_points -- so it is built a block of sources at a time. Peak stays
    # around 2 x block (the clip and the sort) instead of 2 x fleet, which at
    # the 10^4-source scale in the module docstring is ~128 MB rather than
    # ~2.6 GB.
    step = max(1, _SORT_BLOCK_ELEMENTS // max(dim * n, 1))
    for s0 in range(0, n_src, step):
        s1 = min(s0 + step, n_src)
        xs = np.sort(np.clip(np.moveaxis(chains[s0:s1], 1, 2),
                             (lo[s0:s1] + margin[s0:s1])[..., None],
                             (hi[s0:s1] - margin[s0:s1])[..., None]),
                     axis=-1)                                  # (B, d, n)
        block = _bump_ties(
            np.moveaxis(np.quantile(xs, probs, axis=-1), 0, -1), eps_x[s0:s1])
        xq[s0:s1] = block
        # np.searchsorted needs a 1-D haystack, so this pair stays a loop
        for s in range(s1 - s0):
            for j in range(dim):
                yq[s0 + s, j] = np.searchsorted(xs[s, j], block[s, j],
                                                side="right")
    yq = _bump_ties(np.clip((yq - 0.5) / n, 1e-7, 1.0 - 1e-7), 1e-9)

    xk = np.concatenate([lo[..., None], xq, hi[..., None]], axis=-1)
    yk = np.concatenate([np.zeros((n_src, dim, 1)), yq,
                         np.ones((n_src, dim, 1))], axis=-1)
    # guard: quantile grid must sit strictly inside the bounds
    xk[..., 1] = np.where(xk[..., 1] <= xk[..., 0], xk[..., 0] + eps_x,
                          xk[..., 1])
    xk[..., -2] = np.where(xk[..., -2] >= xk[..., -1], xk[..., -1] - eps_x,
                           xk[..., -2])

    # same block treatment as the sort: the slope kernel is array-wide, so its
    # working set is a fixed number of (block, dim, K) temporaries
    dk = np.empty_like(xk)
    dstep = max(1, _SORT_BLOCK_ELEMENTS
                // max(_SLOPE_TEMPORARIES * dim * xk.shape[-1], 1))
    for s0 in range(0, n_src, dstep):
        s1 = min(s0 + dstep, n_src)
        dk[s0:s1] = _pchip_knot_derivatives(xk[s0:s1], yk[s0:s1])
    return jnp.asarray(xk), jnp.asarray(yk), jnp.asarray(dk)


# ----------------------------------------------------------------------
# per-source marginal maps, built on the shared rational-quadratic spline
# kernel in :mod:`coppuccino.bijections`: x (b, d) with knots (d, K)
# ----------------------------------------------------------------------

def _u_and_logdet(x, xk, yk, dk):
    per_dim = jax.vmap(
        lambda col, a, b_, c: (
            _rqs_forward(col, a, b_, c),
            jnp.log(_rqs_derivative(col, a, b_, c, floor=_POS_FLOOR)),
        ),
        in_axes=(1, 0, 0, 0))
    u_t, ld_t = per_dim(x, xk, yk, dk)          # (d, b) each
    # the spline kernel evaluates an out-of-range query at the nearest end
    # knot, so a row outside the prior box would otherwise be scored as if it
    # sat on the nearest wall. The marginal density there is zero: report -inf
    # so that finiteness is a support test. A NaN coordinate compares False
    # against both bounds, so it needs naming separately -- otherwise the NaN
    # propagates into the returned density and poisons a downstream MH ratio
    # instead of being reported as unsupported.
    oob = jnp.any((x < xk[..., 0]) | (x > xk[..., -1]) | jnp.isnan(x), axis=-1)
    ld = jnp.where(oob, -jnp.inf, jnp.sum(ld_t, axis=0))
    return u_t.T, ld                             # (b, d), (b,)


def _x_from_u(u, xk, yk, dk):
    per_dim = jax.vmap(lambda col, a, b_, c: _rqs_inverse(col, a, b_, c),
                       in_axes=(1, 0, 0, 0))
    return per_dim(u, xk, yk, dk).T              # (b, d)


# ----------------------------------------------------------------------
# stacked copula flows
# ----------------------------------------------------------------------

def _build_cube_inner(dim: int, knots: int, flow_layers: int, rng_seed: int = 0):
    """The (untrained) unit-cube coupling flow every fleet member shares.

    RQS coupling layers map ``[0, 1]^d`` to itself with strictly positive
    derivative everywhere (flowjax enforces ``min_derivative > 0``), so support
    is exactly the cube and no clipping is involved. ``FleetFlow.load``
    deserialises trained arrays into this skeleton, so its *structure* must
    stay deterministic; ``rng_seed`` only sets initial values (overwritten on
    load) and the layer permutations (which is why the template seed is stored
    with the fleet).
    """
    return coupling_flow(
        jr.key(rng_seed),
        base_dist=Uniform(jnp.zeros(dim), jnp.ones(dim)),
        transformer=RationalQuadraticSpline(knots=knots, interval=(0, 1)),
        flow_layers=flow_layers,
        invert=True,
    )


def _template(dim: int, knots: int, flow_layers: int, rng_seed: int = 0):
    return _build_cube_inner(dim, knots, flow_layers, rng_seed=rng_seed)


def _stack_params(dim, knots, flow_layers, n_src, rng_seed):
    """Stacked array leaves of N template flows (leading axis = source).

    ``eqx.filter_vmap`` builds the whole fleet's initial parameters in one
    traced pass, which is flat in ``n_src`` where the equivalent Python loop
    grows linearly. Only the inexact-array partition leaves the traced
    function: the coupling layers' integer permutations stay inside it, so
    they cannot acquire a spurious leading source axis. ``static`` therefore
    comes from a separate scalar build at the *base* seed -- exactly the
    template the loop used to take it from.

    Notes
    -----
    The traced build is flat but not free: it pays one trace and one compile
    whatever the fleet size -- seconds of it on a CPU backend -- which makes
    it a *regression* below a few hundred sources, and the fleet in this
    module's own Example block is 64. So small fleets keep the loop. Both
    paths are bitwise identical (each source is built at seed ``rng_seed + i``
    either way), so the threshold only trades wall time.

    Where the two cross is a property of the machine, not of this package.
    The loop grows linearly in ``n_src`` -- it constructs one flow per source
    in Python -- while the traced build is flat, so they cross wherever the
    trace-and-compile cost equals that many per-source constructions. No table
    of seconds is quoted here on purpose: this package's development machine
    could not be measured idle, and repeated timings of the *identical* build
    moved by more than the gap the threshold decides (the same 256-source loop
    build came out anywhere from 2.8 s to 6.6 s across 10 cold runs).

    What did survive that noise is the *paired* ordering within a run, which
    is what the threshold actually needs. At ``dim=2, knots=4, flow_layers=6,
    rng_seed=0``, one cold ``_stack_params`` call per freshly started
    interpreter, jax 0.7.2 on a CPU backend: over 10 runs spread across two
    sessions with different competing load, the loop was the faster branch at
    256 sources in 9 of the 10 and the slower one at 300 in 10 of the 10.
    Hence ``_VMAP_STACK_MIN_SOURCES = 300``. Near the crossover the choice is
    worth a fraction of a second either way; the threshold earns its keep on
    fleets of thousands, where the loop's linear term dominates. Re-measure
    the ordering rather than moving the constant on a hunch.
    """
    static = eqx.partition(_template(dim, knots, flow_layers, rng_seed=rng_seed),
                           eqx.is_inexact_array)[1]
    if n_src < _VMAP_STACK_MIN_SOURCES:
        per_source = [
            eqx.partition(_template(dim, knots, flow_layers, rng_seed=rng_seed + i),
                          eqx.is_inexact_array)[0] for i in range(n_src)]
        return (jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *per_source),
                static)

    @eqx.filter_vmap
    def build(seed):
        model = _template(dim, knots, flow_layers, rng_seed=seed)
        return eqx.partition(model, eqx.is_inexact_array)[0]

    return build(jnp.arange(rng_seed, rng_seed + n_src)), static


# ----------------------------------------------------------------------
# FleetFlow
# ----------------------------------------------------------------------

class FleetFlow:
    """A fleet of independently fitted boundary-safe copula flows.

    Holds one stacked parameter pytree (leading axis = source) plus stacked
    marginal knot tensors. ``sample`` / ``log_prob`` / ``sample_and_log_prob``
    operate on the whole fleet at once (vmapped) or on a single source via
    ``source_index``. Returns NumPy arrays, matching the coppuccino API
    convention; the package-level verbs of the same names take single flows and
    reject a fleet.
    """

    def __init__(self, params, static, arch, xk, yk, dk, lo, hi):
        self.params = params            # stacked pytree, leading axis n_sources
        self.static = static            # shared template statics
        # ("fleet-uniform", dim, knots, layers, n_sources, template_seed).
        # The leading tag is an opaque on-disk marker kept stable so existing
        # fleet files still load and warm-start.
        self.arch = arch
        self.xk, self.yk, self.dk = xk, yk, dk   # (N, d, K)
        self.lo, self.hi = lo, hi                # (N, d)
        self.n_sources = int(xk.shape[0])
        self.dim = int(xk.shape[1])
        self._jit_cache = {}

    # ---- internals -----------------------------------------------------
    def _model(self, p):
        return eqx.combine(p, self.static)

    def _source_index(self, i) -> int:
        """Validate one ``source_index``; negatives count from the end.

        JAX clamps out-of-bounds indices instead of raising, so without this
        an off-by-one (or a refit with fewer sources) would silently return a
        neighbouring source's flow.
        """
        idx = int(i)
        if idx < 0:
            idx += self.n_sources
        if not 0 <= idx < self.n_sources:
            raise IndexError(f"source_index {int(i)} out of range for a fleet "
                             f"of {self.n_sources} sources.")
        return idx

    def _check_x(self, x, source_index) -> None:
        """Reject a mis-shaped query batch before it reaches ``jax.vmap``.

        A ``(b, dim)`` array passed without ``source_index`` is the common
        fleet mistake. Unchecked it reaches the vmapped evaluator, which
        reports it as inconsistent mapped axis sizes and names an internal
        parameter leaf (``p.base_dist.bijection.loc.tree``) rather than the
        argument the caller actually got wrong.
        """
        got = tuple(x.shape)
        if source_index is None:
            if x.ndim == 3 and x.shape[0] == self.n_sources \
                    and x.shape[2] == self.dim:
                return
            hint = ""
            if x.ndim == 2 and x.shape[1] == self.dim:
                hint = (" That looks like one source's batch: pass "
                        "source_index=i to score it against source i, or "
                        f"np.broadcast_to(x, ({self.n_sources},) + x.shape) "
                        "to score it against every source.")
            raise ValueError(
                f"x must be (n_sources, b, dim) = ({self.n_sources}, b, "
                f"{self.dim}) when source_index is None; got {got}.{hint}")
        if x.ndim != 2 or x.shape[1] != self.dim:
            raise ValueError(
                f"x must be (b, dim) = (b, {self.dim}) when source_index is "
                f"given; got {got}.")

    def _take(self, i):
        i = self._source_index(i)
        p = jax.tree_util.tree_map(lambda a: a[i], self.params)
        return p, self.xk[i], self.yk[i], self.dk[i]

    def _fn(self, name):
        """Build (and cache) the jitted batched evaluation functions."""
        if name in self._jit_cache:
            return self._jit_cache[name]
        static = self.static

        def logp_one(p, x, xk, yk, dk):
            u, ld = _u_and_logdet(x, xk, yk, dk)
            return eqx.combine(p, static).log_prob(u) + ld

        def sample_one(p, key, n, xk, yk, dk):
            u = eqx.combine(p, static).sample(key, (n,))
            return _x_from_u(u, xk, yk, dk)

        f: Callable[..., Any]
        if name == "logp_all":
            f = eqx.filter_jit(jax.vmap(logp_one, in_axes=(0, 0, 0, 0, 0)))
        elif name == "logp_one":
            f = eqx.filter_jit(logp_one)
        elif name == "sample_all":
            f = eqx.filter_jit(jax.vmap(sample_one,
                                        in_axes=(0, 0, None, 0, 0, 0)))
        else:  # sample_one
            f = eqx.filter_jit(sample_one)
        self._jit_cache[name] = f
        return f

    # ---- public API ------------------------------------------------------
    def log_prob(self, x, source_index: Optional[int] = None) -> np.ndarray:
        """Log density. ``x``: (n_sources, b, dim) batched, or (b, dim) with
        ``source_index`` (negatives count from the end). Returns
        (n_sources, b) or (b,). Rows outside the prior box, and rows holding a
        NaN, score ``-inf``, so finiteness of the result is a support test.

        Raises
        ------
        ValueError
            If ``x`` does not have the shape implied by ``source_index``.
        IndexError
            If ``source_index`` is out of range for this fleet.
        """
        x = jnp.asarray(x)
        self._check_x(x, source_index)
        if source_index is None:
            return np.asarray(self._fn("logp_all")(self.params, x,
                                                   self.xk, self.yk, self.dk))
        p, xk, yk, dk = self._take(source_index)
        return np.asarray(self._fn("logp_one")(p, x, xk, yk, dk))

    def sample(self, n_samples: int, rng_seed: Optional[int] = None,
               source_index: Optional[int] = None) -> np.ndarray:
        """Draw samples: (n_sources, n_samples, dim), or (n_samples, dim) for
        one source (negatives count from the end). Fresh entropy when
        ``rng_seed`` is None; for a fixed seed the single-source draws are the
        corresponding slice of the whole-fleet draws."""
        from coppuccino.copula_flows import _resolve_seed
        key = jr.key(_resolve_seed(rng_seed))
        keys = jr.split(key, self.n_sources)
        if source_index is None:
            return np.asarray(self._fn("sample_all")(self.params, keys,
                                                     n_samples, self.xk,
                                                     self.yk, self.dk))
        idx = self._source_index(source_index)
        p, xk, yk, dk = self._take(idx)
        return np.asarray(self._fn("sample_one")(p, keys[idx], n_samples,
                                                 xk, yk, dk))

    def sample_and_log_prob(self, n_samples: int, rng_seed: Optional[int] = None,
                            source_index: Optional[int] = None
                            ) -> tuple[np.ndarray, np.ndarray]:
        """Draw samples and their log densities, the fleet analogue of the
        package-level ``sample_and_log_prob``. Shapes and ``source_index``
        follow :meth:`sample` and :meth:`log_prob`.

        This is a convenience wrapper, not a fused kernel: it scores its own
        draws with :meth:`log_prob`, so the returned densities are bitwise
        identical to a later ``log_prob`` call on the same points. Fusing the
        two would score the *pre*-inverse u instead, which near a wall differs
        from the reconstructed value by nats.
        """
        x = self.sample(n_samples, rng_seed=rng_seed, source_index=source_index)
        return x, self.log_prob(x, source_index=source_index)

    # ---- persistence -----------------------------------------------------
    def save(self, path) -> None:
        """Write the whole fleet to ``path``; read it back with
        :meth:`FleetFlow.load`. The package-level ``save_flow`` handles single
        flows only and rejects a fleet.

        The payload is published atomically (temporary sibling plus
        :func:`os.replace`), so an interrupted save leaves any previously saved
        fleet at ``path`` intact instead of truncating it -- the difference
        matters at the documented ~10^4-source scale, where the write window is
        long. Note that the payload is buffered in memory before it is written:
        the serialised parameters and the pickle of them are both resident
        while :func:`cloudpickle.dumps` runs, so peak usage is around twice the
        file size. Releasing the serialisation buffer first (below) keeps it
        from being three times.
        """
        buf = io.BytesIO()
        eqx.tree_serialise_leaves(buf, self.params)
        params_bytes = buf.getvalue()
        buf.close()          # the copy above is the only one worth keeping
        _atomic_write_bytes(path, cloudpickle.dumps({
            "format_version": FORMAT_VERSION,
            "kind": FLEET_KIND,
            "arch": self.arch,
            "params_bytes": params_bytes,
            "xk": np.asarray(self.xk), "yk": np.asarray(self.yk),
            "dk": np.asarray(self.dk),
            "lo": np.asarray(self.lo), "hi": np.asarray(self.hi),
        }))

    @classmethod
    def load(cls, path) -> "FleetFlow":
        """Load a fleet written by :meth:`FleetFlow.save`.

        .. warning::
            The saved format is a Python pickle. Loading it executes arbitrary
            code contained in the file, so only load fleets from a source you
            trust. Never load a fleet received over a network or from an
            untrusted third party. This is the same caveat that applies to
            `pickle`, `joblib` and `torch.load`.

        Parameters
        ----------
        path : str or Path
            Path to the saved fleet file.

        Returns
        -------
        FleetFlow
            The fleet, ready for sampling and density evaluation.

        Notes
        -----
        Reading a fleet costs more memory than the file it comes from: the
        unpickled payload and the deserialised parameters are both resident.
        The two avoidable copies are avoided here -- the pickled parameter
        bytes are dropped from the payload as they are handed to the reader,
        and the ``like`` skeleton is made of
        :class:`jax.ShapeDtypeStruct` rather than real arrays, since
        deserialisation reads shape and dtype from it and then overwrites
        every value.

        Raises
        ------
        ValueError
            If the file holds a single copula flow (saved by ``save_flow``)
            rather than a fleet, or was written in a newer save format than
            this install understands.
        """
        with open(path, "rb") as f:
            d = cloudpickle.load(f)
        # Fleet and single-flow payloads share no keys, so without this the
        # wrong file dies on `KeyError: 'arch'`. Files written before the
        # stamps existed are still recognised, by their key set.
        _check_payload(d, path, FLEET_KIND)
        # rebuild with the SAME template seed: the coupling layers' integer
        # permutation arrays live in the static partition and are
        # seed-dependent, so an arbitrary-seed skeleton would evaluate the
        # deserialized parameters under different permutations
        _, dim, knots, layers, n_src, tmpl_seed = d["arch"]
        params0, static = eqx.partition(
            _template(dim, knots, layers, rng_seed=tmpl_seed),
            eqx.is_inexact_array)
        # deserialisation reads only shape and dtype from the skeleton and then
        # overwrites every value, so the skeleton needs no storage at all --
        # equinox accepts ShapeDtypeStruct leaves in `like` and still checks
        # both against the file. Materialising it (n_src of them, or one
        # broadcast) would allocate a second full copy of the fleet's
        # parameters purely to be thrown away.
        skeleton = jax.tree_util.tree_map(
            lambda a: jax.ShapeDtypeStruct((n_src,) + a.shape, a.dtype),
            params0)
        # pop, so the payload's copy of the parameter bytes is released as the
        # reader takes it rather than staying alive for the whole read
        params = eqx.tree_deserialise_leaves(
            io.BytesIO(d.pop("params_bytes")), skeleton)
        return cls(params, static, d["arch"],
                   jnp.asarray(d["xk"]), jnp.asarray(d["yk"]),
                   jnp.asarray(d["dk"]),
                   np.asarray(d["lo"]), np.asarray(d["hi"]))


# ----------------------------------------------------------------------
# fleet fitting
# ----------------------------------------------------------------------

def normalizing_flows_fit_fleet(chains: np.ndarray,
                                prior_bounds: np.ndarray,
                                knots: int = 4,
                                flow_layers: int = 6,
                                learning_rate: float = 1e-3,
                                max_epochs: int = 200,
                                patience: int = 30,
                                batch_size: int = 128,
                                val_prop: float = 0.1,
                                num_points: int = 200,
                                rng_seed: int = 999,
                                warm_start: Optional[FleetFlow] = None,
                                verbose: bool = False) -> FleetFlow:
    """Fit one boundary-safe copula flow per source, all sources at once.

    Every source shares a fixed architecture (``knots``, ``flow_layers`` --
    default 4/6, the measured middle ground for galactic-binary posteriors),
    which is what lets ``vmap`` fuse the whole fleet's training into single
    kernels: one Adam step advances every source simultaneously. On CPU this
    vectorizes; on a GPU (``jax[cuda]``) the same code trains ~10^4 sources
    in minutes.

    Parameters
    ----------
    chains : np.ndarray
        Posterior samples, shape ``(n_sources, n_samples, dim)``. Use a few
        thousand *decorrelated* (thinned) samples per source; all sources
        must supply the same ``n_samples`` (subsample upstream). Every entry
        must be finite, every column must take at least two distinct values,
        and every sample must lie inside its source's ``prior_bounds``
        (clean per source upstream).
    prior_bounds : np.ndarray
        ``(dim, 2)`` shared bounds or ``(n_sources, dim, 2)`` per source;
        each row ``[low, high]`` with ``low < high`` and both sides finite.
        Required: the fleet models bounded (prior-box) support exactly, and
        anchors its outermost marginal knots on the bounds, so -- unlike
        ``normalizing_flows_fit`` -- it has no representation for an
        unbounded side.
    knots, flow_layers : int, optional
        Fleet-wide copula architecture. Defaults 4 and 6.
    learning_rate, max_epochs, patience, batch_size, val_prop : optional
        Training controls. ``patience`` acts on the fleet-mean validation
        loss; per-source best parameters are tracked and returned regardless.
    num_points : int, optional
        Marginal CDF quantile-grid size (clamped to ``n_samples // 3``).
    rng_seed : int, optional
        Seed for initialization, batching, and the validation split.
    warm_start : FleetFlow, optional
        Reuse a previous fleet's parameters as initialization (architecture
        and ``n_sources`` must match). For periodic refreshes this typically
        cuts the required epochs several-fold.
    verbose : bool, optional
        Print per-epoch fleet-mean validation loss.

    Returns
    -------
    FleetFlow

    Raises
    ------
    ValueError
        If ``chains`` is not 3-D, is too short, holds a non-finite value, has
        a constant column, or leaves its box; or if ``prior_bounds`` has the
        wrong shape, a non-finite side, or a row that is not strictly
        increasing.

    Notes
    -----
    Memory. Per-source parameters mean the optimizer holds several
    simultaneous copies of the whole fleet's parameter stack -- the current
    parameters, the step's output, Adam's two moment estimates, the tracked
    per-source best, and one transient inside the jitted step. At
    ``n_sources=256, dim=8`` and the default architecture one stack is
    38.2 MB (36.4 MiB), so those copies alone are a few hundred MB, and the
    term grows linearly with the fleet.

    The other term is the training batch, which is vmapped across sources.
    Peak process RSS (``getrusage``) for a whole default fit of that fleet --
    256 sources, 512 samples each, ``dim=8``, everything else at its default
    -- spanned 1704-1755 MiB over three runs at the default
    ``batch_size=128``, against 1451-1487 MiB over three at ``batch_size=32``.
    Lowering ``batch_size`` is the knob if a large fleet does not fit; note
    that it changes the optimization, not just the memory, and that it buys
    much less than proportionally, since the parameter copies above do not
    shrink with it.
    """
    chains = np.asarray(chains, dtype=float)
    if chains.ndim != 3:
        raise ValueError(f"chains must be (n_sources, n_samples, dim); "
                         f"got shape {chains.shape}.")
    n_src, n_samp, dim = chains.shape
    if n_src < 1 or dim < 1:
        raise ValueError(f"chains must hold at least one source and one "
                         f"parameter; got shape {chains.shape}.")
    if n_samp < _MIN_SAMPLES:
        raise ValueError(f"need at least {_MIN_SAMPLES} samples per source; "
                         f"got {n_samp}.")

    lo, hi = _validate_fleet_bounds(prior_bounds, n_src, dim)
    _validate_fleet_chains(chains, lo, hi)

    # ---- marginals (CPU) + u-space training tensors ----
    num_points = max(20, min(num_points, n_samp // 3))
    xk, yk, dk = _build_fleet_knots(chains, lo, hi, num_points)

    # per-source val split (shared indices; sizes identical by construction)
    rng = np.random.default_rng(rng_seed)
    perm = rng.permutation(n_samp)
    n_val = max(1, int(round(n_samp * val_prop)))
    # map the two halves separately. Mapping the whole chain and then slicing
    # it would pin a second (n_sources, n_samples, dim) buffer -- 1.3 GB at the
    # documented 10^4-source scale -- for the whole fit, alongside a marginal
    # log-det that training never reads.
    to_u = eqx.filter_jit(jax.vmap(
        lambda c, a, b_, e: _u_and_logdet(c, a, b_, e)[0]))
    u_val = to_u(jnp.asarray(chains[:, perm[:n_val]]), xk, yk, dk)   # (N, ., d)
    u_tr = to_u(jnp.asarray(chains[:, perm[n_val:]]), xk, yk, dk)
    n_tr = u_tr.shape[1]
    batch_size = min(batch_size, n_tr)
    steps = n_tr // batch_size

    # ---- stacked flows ----
    arch = ("fleet-uniform", dim, knots, flow_layers, n_src, rng_seed)
    if warm_start is not None:
        if warm_start.arch[:5] != arch[:5]:
            raise ValueError(f"warm_start arch {warm_start.arch[:5]} does not "
                             f"match requested {arch[:5]}.")
        # adopt the previous fleet's parameters AND static (its params were
        # trained under its own template's permutations)
        params, static, arch = warm_start.params, warm_start.static, warm_start.arch
    else:
        params, static = _stack_params(dim, knots, flow_layers, n_src, rng_seed)

    def nll(p, u):
        return -jnp.mean(eqx.combine(p, static).log_prob(u))

    opt = optax.adam(learning_rate)
    opt_state = opt.init(eqx.filter(params, eqx.is_inexact_array))

    @eqx.filter_jit
    def train_step(params, opt_state, u_batch):
        losses, grads = jax.vmap(jax.value_and_grad(nll))(params, u_batch)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = eqx.apply_updates(params, updates)
        return params, opt_state, losses

    @eqx.filter_jit
    def val_loss(params):
        return jax.vmap(nll)(params, u_val)                    # (N,)

    key = jr.key(rng_seed)
    best_val = np.asarray(val_loss(params))
    best_params = params
    fleet_best = float(np.mean(best_val))
    stall = 0
    for epoch in range(max_epochs):
        key, sub = jr.split(key)
        order = jr.permutation(sub, n_tr)
        for s in range(steps):
            idx = order[s * batch_size:(s + 1) * batch_size]
            params, opt_state, _ = train_step(params, opt_state,
                                              u_tr[:, idx])
        v = np.asarray(val_loss(params))                        # (N,)
        improved = v < best_val
        if improved.any():
            imp = jnp.asarray(improved)
            best_params = jax.tree_util.tree_map(
                # `imp` is bound as a default so the closure carries this
                # epoch's mask rather than reading the loop variable later.
                lambda b, n, imp=imp: jnp.where(
                    imp.reshape((-1,) + (1,) * (n.ndim - 1)), n, b),
                best_params, params)
            best_val = np.where(improved, v, best_val)
        if verbose:
            print(f"[fleet] epoch {epoch + 1}/{max_epochs}: "
                  f"mean val NLL {np.mean(v):.4f} "
                  f"(best {np.mean(best_val):.4f})", flush=True)
        if float(np.mean(best_val)) < fleet_best - 1e-12:
            fleet_best = float(np.mean(best_val))
            stall = 0
        else:
            stall += 1
            if stall >= patience:
                break

    return FleetFlow(best_params, static, arch, xk, yk, dk, lo, hi)
