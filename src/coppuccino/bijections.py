from typing import Any, Callable, ClassVar, NamedTuple, Optional

import jax.numpy as jnp
import numpy as np
from flowjax.bijections import AbstractBijection
from interpax._ppoly import PchipInterpolator
from jax import Array
from jax.scipy.special import log_ndtr, ndtr, ndtri
from jax.typing import ArrayLike

__all__ = [
    "make_empirical_cdf_spline",
    "make_smooth_marginal",
    "EmpiricalMarginalToGaussian",
    "SmoothMarginalToGaussian",
]

# Numerical floors, named so the intent is explicit at each use site.
_POS_FLOOR = 1e-30    # keep quantities that sit in denominators strictly positive
_SCALE_FLOOR = 1e-10  # keep fitted scale (sigma) parameters away from zero
_NDTRI_FLOOR = 1e-300  # smallest ndtri argument that stays inside float64 range
_LOG_SQRT_2PI = float(0.5 * np.log(2.0 * np.pi))


def _ndtri_floor(x: ArrayLike) -> float:
    """Smallest usable ``ndtri`` argument in ``x``'s floating dtype.

    ``_NDTRI_FLOOR`` underflows to zero below float64, and ``ndtri(0)`` is
    ``-inf``, so the tail inversion needs the per-dtype limit rather than the
    float64 constant. The package enables x64 on import, but that is a
    process-wide setting a caller may turn back off (see the package
    docstring), and the marginals must stay finite when they do.
    """
    return max(_NDTRI_FLOOR, float(jnp.finfo(jnp.asarray(x).dtype).tiny))


def _dtype_max(x: ArrayLike) -> float:
    """Largest finite value in ``x``'s floating dtype.

    The mirror of :func:`_ndtri_floor` at the top of the range. The Gaussian
    tail scale factors are float64 build-time constants; on heavy-tailed data
    they routinely exceed 1e100, and casting one to a coarser runtime dtype
    (x64 turned back off) overflows to ``inf`` with a numpy "overflow
    encountered in cast" warning. Clamping to the representable maximum keeps
    the same limit for ``u / scale`` -- underflow to zero, then floored by
    :func:`_ndtri_floor` -- without the overflow. A no-op in float64, where
    the constants are finite by construction.
    """
    return float(jnp.finfo(jnp.asarray(x).dtype).max)


def _process_array(arr: ArrayLike) -> np.ndarray:
    """
    Check for duplicate values in array and perturb them to ensure uniqueness.

    If duplicates exist, perturbs the values slightly to make them unique
    and resorts the array if necessary after perturbation. This is useful
    for creating monotonic interpolators that require strictly increasing inputs.

    Parameters
    ----------
    arr : numpy.ndarray
        Input array (will be sorted internally).

    Returns
    -------
    numpy.ndarray
        Processed array with no duplicate values.

    Examples
    --------
    >>> import numpy as np
    >>> from coppuccino.bijections import _process_array
    >>> arr = np.array([1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0])
    >>> processed = _process_array(arr)
    >>> len(processed) == len(np.unique(processed))
    True
    """
    # Convert to float for perturbation
    arr = np.sort(arr)
    arr = np.asarray(arr, dtype=float)

    # Check for duplicates
    unique, counts = np.unique(arr, return_counts=True)
    if np.all(counts <= 1):
        return arr

    # Determine epsilon for perturbation
    max_count = np.max(counts)
    diffs = np.diff(arr)
    pos_diffs = diffs[diffs > 0]
    if len(pos_diffs) > 0:
        min_pos_diff = np.min(pos_diffs)
        epsilon = min_pos_diff / (max_count * 2)
    else:
        epsilon = 1e-10

    # Copy array for modification
    new_arr = arr.copy()

    # Perturb duplicates
    i = 0
    while i < len(new_arr):
        val = new_arr[i]
        j = i
        while j < len(new_arr) and new_arr[j] == val:
            j += 1
        group_size = j - i
        if group_size > 1:
            for k in range(group_size):
                new_arr[i + k] += k * epsilon
        i = j

    # Check if still sorted
    is_sorted = np.all(np.diff(new_arr) >= 0)
    if not is_sorted:
        new_arr.sort()

    return new_arr


def _fit_gpd(excesses: ArrayLike) -> tuple[float, float]:
    """Fit a Generalized Pareto Distribution to non-negative excesses.

    Returns (xi, sigma). Falls back to a moment-based estimate when
    scipy's MLE fails (rare, but happens on tiny or degenerate samples).
    """
    from scipy.stats import genpareto
    excesses = np.asarray(excesses, dtype=float)
    excesses = excesses[excesses > 0]
    if len(excesses) < 5:
        # Too few excesses for any sensible fit — exponential fallback.
        sigma = float(np.mean(excesses)) if len(excesses) else 1.0
        return 0.0, max(sigma, _SCALE_FLOOR)
    try:
        xi, _, sigma = genpareto.fit(excesses, floc=0)
    except Exception:
        # Method-of-moments fallback (Hosking & Wallis 1987).
        m = float(np.mean(excesses))
        v = float(np.var(excesses, ddof=1))
        if v <= 0:
            return 0.0, max(m, _SCALE_FLOOR)
        xi = 0.5 * (1.0 - m * m / v)
        sigma = 0.5 * m * (1.0 + m * m / v)
    return float(xi), float(max(sigma, _SCALE_FLOOR))


def _gpd_cdf(y: ArrayLike, xi: float, sigma: float,
             xi_eps: float = 1e-6) -> Array:
    """GPD CDF on excesses y ≥ 0. Handles ξ≈0 (exponential) limit."""
    y = jnp.maximum(y, 0.0)
    z = y / sigma
    # For ξ<0 the support is bounded at y = -σ/ξ; clip to keep argument > 0.
    base = 1.0 + xi * z
    safe_arg = jnp.maximum(base, _POS_FLOOR)
    xi_safe = jnp.where(jnp.abs(xi) < xi_eps, 1.0, xi)  # avoid /0 in unused branch
    cdf = jnp.where(
        jnp.abs(xi) < xi_eps,
        1.0 - jnp.exp(-z),
        1.0 - jnp.power(safe_arg, -1.0 / xi_safe),
    )
    # Past the bounded support edge (ξ<0, base≤0) all mass lies below: CDF = 1.
    return jnp.where(base <= 0.0, 1.0, cdf)


def _gpd_pdf(y: ArrayLike, xi: float, sigma: float,
             xi_eps: float = 1e-6) -> Array:
    """GPD PDF on excesses y ≥ 0. Zero beyond the bounded support when ξ<0."""
    y = jnp.maximum(y, 0.0)
    z = y / sigma
    base = 1.0 + xi * z
    safe_arg = jnp.maximum(base, _POS_FLOOR)
    xi_safe = jnp.where(jnp.abs(xi) < xi_eps, 1.0, xi)
    pdf = jnp.where(
        jnp.abs(xi) < xi_eps,
        jnp.exp(-z) / sigma,
        jnp.power(safe_arg, -1.0 / xi_safe - 1.0) / sigma,
    )
    # For ξ<-1 the exponent -1/ξ-1 is negative, so the clipped safe_arg would
    # blow up to ~1e10 past the support edge (y > -σ/ξ, base≤0). The density is
    # actually zero there — enforce it explicitly.
    return jnp.where(base <= 0.0, 0.0, pdf)


def _gpd_quantile(p: ArrayLike, xi: float, sigma: float,
                  xi_eps: float = 1e-6) -> Array:
    """GPD inverse CDF: y(p) for p ∈ [0, 1)."""
    p = jnp.clip(p, 0.0, 1.0 - 1e-15)
    xi_safe = jnp.where(jnp.abs(xi) < xi_eps, 1.0, xi)
    return jnp.where(
        jnp.abs(xi) < xi_eps,
        -sigma * jnp.log1p(-p),
        sigma * (jnp.power(1.0 - p, -xi_safe) - 1.0) / xi_safe,
    )


# ----------------------------------------------------------------------
# Monotone rational-quadratic spline kernel (Gregory & Delbourgo 1982;
# Durkan et al. 2019), shared by every RQS in the package: the bulk
# marginal splines below, the fleet's stacked marginals
# (:mod:`coppuccino.fleet`) and the post-fit edge-calibration windows
# (:mod:`coppuccino.calibration`). This module is the dependency leaf, so
# both of those import the kernel from here rather than restating it.
# ----------------------------------------------------------------------


def _rqs_bin(index_knots: Array, query: Array, xk: Array, yk: Array,
             dk: Array) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
    """Locate ``query`` in the RQS bins and gather that bin's parameters.

    ``index_knots`` is ``xk`` for the forward and derivative maps and ``yk``
    for the inverse. The bin index is clamped to the interior bins, so a
    query outside the knot range is evaluated on the nearest end bin.

    Parameters
    ----------
    index_knots : jax.Array
        1-D strictly increasing knots to search in.
    query : jax.Array
        Points to locate; any shape.
    xk, yk, dk : jax.Array
        1-D knot locations, knot values and knot derivatives.

    Returns
    -------
    x0, w, y0, dy, s, d0, d1 : jax.Array
        Left knot, bin width, left value, bin height, secant slope and the
        two knot derivatives of the containing bin, shaped like ``query``.
    """
    n_bins = index_knots.shape[0] - 1
    k = jnp.clip(jnp.searchsorted(index_knots, query, side="right") - 1, 0, n_bins - 1)
    x0, x1 = xk[k], xk[k + 1]
    y0, y1 = yk[k], yk[k + 1]
    w = x1 - x0
    dy = y1 - y0
    return x0, w, y0, dy, dy / w, dk[k], dk[k + 1]


def _rqs_forward(x: ArrayLike, xk: Array, yk: Array, dk: Array) -> Array:
    """Evaluate the monotone RQS: knot domain -> knot range.

    ``x`` outside the knot range is evaluated at the nearest end knot, so
    the map saturates at ``yk[0]`` / ``yk[-1]``. The rational-quadratic form
    is monotone only for ``xi`` in ``[0, 1]``: extrapolated it turns over,
    returning values that leave ``[yk[0], yk[-1]]`` and rise again as ``x``
    moves further out — and, in :func:`_rqs_derivative`, a negative density.
    Saturating instead is what the fleet's stacked marginals have always
    done (:mod:`coppuccino.fleet` layers its own out-of-box support test on
    top); the marginal and calibration call sites clip their own inputs
    first, so for them the clamp is a no-op that closes the mode off for
    good.
    """
    x = jnp.clip(jnp.asarray(x), xk[0], xk[-1])
    x0, w, y0, dy, s, d0, d1 = _rqs_bin(xk, x, xk, yk, dk)
    xi = (x - x0) / w
    num = dy * (s * xi**2 + d0 * xi * (1 - xi))
    den = s + (d1 + d0 - 2 * s) * xi * (1 - xi)
    return y0 + num / den


def _rqs_derivative(x: ArrayLike, xk: Array, yk: Array, dk: Array,
                    floor: Optional[float] = None) -> Array:
    """Closed-form derivative of :func:`_rqs_forward` (the density).

    Strictly positive for strictly positive knot derivatives, and constant
    beyond either end knot, where the query saturates exactly as in
    :func:`_rqs_forward`. ``floor``, when given, is a lower clamp for
    callers that feed the result straight to ``log``; the default leaves
    the raw value alone.
    """
    x = jnp.clip(jnp.asarray(x), xk[0], xk[-1])
    x0, w, y0, dy, s, d0, d1 = _rqs_bin(xk, x, xk, yk, dk)
    xi = (x - x0) / w
    den = s + (d1 + d0 - 2 * s) * xi * (1 - xi)
    num = s**2 * (d1 * xi**2 + 2 * s * xi * (1 - xi) + d0 * (1 - xi) ** 2)
    derivative = num / den**2
    return derivative if floor is None else jnp.maximum(derivative, floor)


def _rqs_inverse(u: ArrayLike, xk: Array, yk: Array, dk: Array) -> Array:
    """Invert the monotone RQS: knot range -> knot domain.

    The exact inverse of :func:`_rqs_forward` on the knot range, via the
    numerically stable root of the bin's quadratic (the form that stays in
    ``[0, 1]`` as the bin approaches linear). Clamping that root is what
    saturates ``u`` outside the knot range, so no separate guard on ``u``
    is needed here.
    """
    u = jnp.asarray(u)
    x0, w, y0, dy, s, d0, d1 = _rqs_bin(yk, u, xk, yk, dk)
    alpha = u - y0
    coef = d1 + d0 - 2 * s
    a = dy * (s - d0) + alpha * coef
    b = dy * d0 - alpha * coef
    c = -s * alpha
    disc = jnp.maximum(b**2 - 4 * a * c, 0.0)
    xi = jnp.clip(2 * c / (-b - jnp.sqrt(disc)), 0.0, 1.0)
    return x0 + xi * w


def _rqs_knot_derivatives(
    xk: ArrayLike,
    yk: ArrayLike,
    floor: float = _POS_FLOOR,
    edge_slope_floor: float = 0.0,
    edge_slopes: Optional[tuple[Optional[float], Optional[float]]] = None,
) -> np.ndarray:
    """Monotone PCHIP knot slopes for an RQS, floored strictly positive.

    The RQS reproduces the shape-preserving PCHIP slopes at its knots while
    admitting a closed-form inverse, so every RQS in the package takes its
    knot derivatives from here.

    Parameters
    ----------
    xk : array_like
        Strictly increasing knot locations.
    yk : array_like
        Strictly increasing knot values.
    floor : float, optional
        Lower clamp on every knot derivative. Slopes sit in RQS
        denominators, so this must stay strictly positive; the default
        ``_POS_FLOOR`` is the generic "not exactly zero" guard, while a
        caller whose map must stay resolvably invertible (the calibration
        windows) wants something far larger.
    edge_slope_floor : float, optional
        If positive, floor the two *endpoint* derivatives at this fraction
        of their end-bin secant slope. PCHIP's endpoint rule can return
        exactly 0 there (a one-sided estimate clamped on sign change),
        which is fatal for callers that extend tails from the boundary
        slope. The default 0.0 disables it.
    edge_slopes : tuple of (float or None, float or None), optional
        Explicit endpoint derivatives, overriding both the PCHIP estimate
        and ``edge_slope_floor``. Used when the caller knows the true slope
        at an endpoint that is not a distribution endpoint.

    Returns
    -------
    numpy.ndarray
        Knot derivatives, one per knot.
    """
    from scipy.interpolate import PchipInterpolator as _ScipyPchip
    xk = np.asarray(xk, dtype=float)
    yk = np.asarray(yk, dtype=float)
    dk = np.maximum(
        np.asarray(_ScipyPchip(xk, yk).derivative()(xk), dtype=float), floor)
    if edge_slope_floor > 0.0:
        sec = np.diff(yk) / np.diff(xk)
        dk[0] = max(dk[0], edge_slope_floor * sec[0])
        dk[-1] = max(dk[-1], edge_slope_floor * sec[-1])
    if edge_slopes is not None:
        # caller-supplied slopes win over both the PCHIP estimate and the
        # floor; the RQS is monotone for any strictly positive knot slopes
        if edge_slopes[0] is not None:
            dk[0] = max(float(edge_slopes[0]), floor)
        if edge_slopes[1] is not None:
            dk[-1] = max(float(edge_slopes[1]), floor)
    return dk


def _rqs_bulk(
    xg: ArrayLike,
    cdf_vals: ArrayLike,
    edge_slope_floor: float = 0.0,
    edge_slopes: Optional[tuple[Optional[float], Optional[float]]] = None,
) -> tuple[Callable[[ArrayLike], Array], Callable[[ArrayLike], Array],
           Callable[[ArrayLike], Array]]:
    """Build closed-form monotone rational-quadratic spline (RQS) callables.

    Returns ``(cdf, inverse, deriv)`` callables over the bulk knots, matching the
    call interface of the interpax ``PchipInterpolator`` objects they replace in
    :func:`make_empirical_cdf_spline`. Unlike the PCHIP pair (two independently
    constructed splines that are only approximate inverses), the RQS forward,
    inverse and derivative all come from a single parameterization, so the
    transform is self-consistent to machine precision.

    Knot derivatives come from :func:`_rqs_knot_derivatives` and the three
    callables from the shared kernel (:func:`_rqs_forward`,
    :func:`_rqs_inverse`, :func:`_rqs_derivative`); this function only
    prepares the knots.

    Parameters
    ----------
    xg : array_like
        Strictly sorted bulk knot locations (CDF domain).
    cdf_vals : array_like
        Strictly increasing CDF values at ``xg``.
    edge_slope_floor : float, optional
        Endpoint knot-derivative floor, as a fraction of the end-bin secant
        slope; see :func:`_rqs_knot_derivatives`. Rescuing PCHIP's zeroed
        endpoint derivative matters for callers that extend tails from the
        boundary slope (:func:`make_smooth_marginal`): the C^1-matched tail
        rate would otherwise collapse to the generic ``_POS_FLOOR`` and the
        tail map degenerates. The default 0.0 preserves the legacy behavior
        exactly.
    edge_slopes : tuple of (float or None, float or None), optional
        Explicit endpoint knot derivatives, replacing PCHIP's one-sided
        estimate (and any ``edge_slope_floor``) at the first / last knot.
        Used when the caller knows the true density at a knot that is not a
        distribution endpoint — under ``tail_model="gpd"`` the outer knots
        are the POT thresholds, where the GPD supplies a far better slope
        than a one-sided secant and matching it makes the join C^1.

    Returns
    -------
    cdf, inverse, deriv : callables
        ``cdf(x)`` maps x -> CDF, ``inverse(u)`` maps CDF -> x, ``deriv(x)``
        is the closed-form CDF derivative (density). All are jnp-vectorized.
    """
    xg = np.asarray(xg, dtype=float)
    yg = np.asarray(cdf_vals, dtype=float)
    # RQS needs strictly increasing x knots; drop any ties (cdf_vals is already
    # strictly increasing thanks to the upstream monotonicity bump).
    keep = np.concatenate([[True], np.diff(xg) > 0])
    xk, yk = xg[keep], yg[keep].copy()
    # the tie-drop keeps the FIRST knot of a tied block; the top endpoint's
    # CDF value must survive on the kept boundary knot or the spline's range
    # silently tops out below it and the discarded mass reappears as an
    # exact-at-bound sampling atom
    yk[-1] = yg[-1]
    dk = _rqs_knot_derivatives(xk, yk, edge_slope_floor=edge_slope_floor,
                               edge_slopes=edge_slopes)

    xk_j, yk_j, dk_j = jnp.asarray(xk), jnp.asarray(yk), jnp.asarray(dk)

    def cdf(x: ArrayLike) -> Array:
        return _rqs_forward(x, xk_j, yk_j, dk_j)

    def deriv(x: ArrayLike) -> Array:
        return _rqs_derivative(x, xk_j, yk_j, dk_j)

    def inverse(u: ArrayLike) -> Array:
        return _rqs_inverse(u, xk_j, yk_j, dk_j)

    return cdf, inverse, deriv


# noqa reason: complexity 21 against the project ceiling of 15. The branching is
# the cross product of two build-time switches (tail_model in {none, gaussian,
# gpd} x marginal in {rqs, pchip}) that the three returned closures must all
# read consistently. Splitting it is tracked as an open refactor; the exemption
# is per-function so new code stays under the ceiling.
def make_empirical_cdf_spline(  # noqa: C901
    samples: np.ndarray,
    num_points: int = 200,
    min_eps: float = 1e-7,
    tail_extension: bool = False,
    prior_low: Optional[float] = None,
    prior_high: Optional[float] = None,
    tail_model: str = "gaussian",
    tail_quantile: float = 0.05,
    marginal: str = "rqs",
) -> tuple[np.ndarray, Callable, Callable, Callable]:
    """
    Create empirical CDF, inverse CDF (quantile), and PDF functions from samples.

    This function constructs smooth monotonic spline interpolators for the empirical
    CDF and its inverse based on input samples. The PDF is computed via automatic
    differentiation of the CDF spline.

    When ``prior_low`` / ``prior_high`` are provided, the CDF grid is extended to
    cover the full prior support. This allows the flow to generate samples all the
    way to the prior boundaries rather than being limited to the most extreme
    training sample. This is the recommended mode for MCMC chains.

    When ``tail_extension=True``, the CDF and quantile functions are extended
    beyond the observed data range using a Gaussian tail model fitted to match
    the empirical CDF at the boundaries. This allows the flow to generate
    samples beyond the training range. Default is False (clip to data bounds),
    which is appropriate for MCMC chains bounded by prior support.

    For heavy-tailed marginals, set ``tail_model="gpd"`` to use a Generalized
    Pareto Distribution fitted via peaks-over-threshold to the top/bottom
    ``tail_quantile`` of samples (default 5%). Unlike the Gaussian tail model,
    the GPD threshold lives *inside* the observed data range — the empirical
    spline covers only the bulk, and the GPD takes over wherever it is more
    statistically efficient. This is the recommended choice when accuracy in
    the tails matters more than absolute simplicity.

    Parameters
    ----------
    samples : array_like
        Training data samples from which to construct the empirical distribution.
    num_points : int, optional
        Number of grid points to use for spline construction. Default is 200.
        Automatically clamped to ``max(20, min(num_points, len(samples) // 3))``
        to avoid degenerate grids with small sample sizes.
    min_eps : float, optional
        Minimum epsilon to avoid CDF values exactly 0 or 1. Default is 1e-7.
    tail_extension : bool, optional
        If True, extend the CDF/quantile functions beyond the observed data
        range using a Gaussian tail model. Default is False (clip to data
        bounds), which is appropriate for MCMC chains bounded by prior support.
    prior_low : float, optional
        Lower bound of the prior support for this parameter. If provided, the
        CDF grid is extended down to this value (with CDF = min_eps), allowing
        the flow to sample all the way to the prior edge.
    prior_high : float, optional
        Upper bound of the prior support for this parameter. If provided, the
        CDF grid is extended up to this value (with CDF = 1 - min_eps).
    tail_model : str, optional
        ``"gaussian"`` (default, current behavior) or ``"gpd"`` (peaks-over-
        threshold Generalized Pareto). ``"gpd"`` activates GPD tails inside
        the data range and is appropriate for heavy-tailed marginals; it
        ignores ``tail_extension`` and (for now) ``prior_low``/``prior_high``.
        With ``marginal="rqs"`` the density is continuous across both GPD
        joins (the bulk spline takes its endpoint slopes from the fitted tail).
        With ``marginal="pchip"`` the join is continuous in CDF only: PCHIP's
        one-sided endpoint slope cannot be overridden, leaving a density step
        at each threshold of median 0.24 and up to 1.33 nats over 40 seeds of
        Student-t(3) data — a kink at the 5th/95th percentile of the marginal,
        which costs proposal efficiency but not correctness.
    tail_quantile : float, optional
        Fraction of samples in each GPD tail (only used when
        ``tail_model="gpd"``). Default 0.05 (5%).
    marginal : str, optional
        Bulk interpolant family (default ``"rqs"``): ``"rqs"`` builds a single
        monotone rational-quadratic spline whose forward map, inverse and
        derivative are all closed-form and mutually exact (self-consistent to
        machine precision). ``"pchip"`` instead builds two independent PCHIP
        splines for the CDF and its inverse, which are only approximate inverses
        of each other; it is retained for reproducing older fits. Both use the
        same empirical-quantile knots and the same tail / prior-bound handling,
        so they produce nearly identical transforms, but ``"rqs"`` guarantees an
        exact, cheaper inverse, admits explicit endpoint knot slopes (used to
        close the GPD join, see ``tail_model``) and is preferred.
        (``normalizing_flows_fit``
        defaults to the clip-free ``"smooth"`` marginal; this family is used
        when a legacy tail model is requested.)

        Neither family can carry the smooth family's wall construction, and
        with ``prior_low`` / ``prior_high`` both still report a wall density
        that is wrong by nats. Both are *clipped*: a prior bound is one more
        spline knot, its CDF is pinned at ``min_eps`` / ``1 - min_eps``, and
        so a query at a bound is handed
        ``z = ndtri(min_eps) = -5.2`` whatever the fit. There is no analytic
        tail object to bound that against the copula's training range the way
        :func:`make_smooth_marginal` does -- the depth is set by a caller's
        ``min_eps``, not by anything measured from the data. On
        ``U(0, 1)^2``, prior [[0, 1], [0, 1]], n = 1500, 80 epochs, seeds
        0-3, ``log_prob`` at both walls comes out +1.2 to +5.7 above the flat
        truth, essentially all of it the copula extrapolating at that depth
        (its excess over its own normal base there is +0.2 to +5.0). Use
        ``marginal="smooth"`` when the density at a bound matters.

    Returns
    -------
    cdf_vals : numpy.ndarray
        Empirical CDF values at the grid points.
    cdf_fn : callable
        Function mapping x → CDF(x), with extrapolation handling.
    quantile_fn : callable
        Function mapping u ∈ [0,1] → x (inverse CDF).
    pdf_fn : callable
        Function mapping x → PDF(x) via automatic differentiation.

    Examples
    --------
    >>> import numpy as np
    >>> from coppuccino.bijections import make_empirical_cdf_spline
    >>> samples = np.random.normal(0, 1, 1000)
    >>> cdf_vals, cdf_fn, quantile_fn, pdf_fn = make_empirical_cdf_spline(samples)
    >>> import jax.numpy as jnp
    >>> median = float(cdf_fn(jnp.array(0.0)))   # ~0.5 for standard normal
    >>> centre = float(quantile_fn(jnp.array(0.5)))  # ~0.0
    >>> # With prior bounds (recommended for MCMC chains)
    >>> cdf_vals, cdf_fn, quantile_fn, pdf_fn = make_empirical_cdf_spline(
    ...     samples, prior_low=-5.0, prior_high=5.0)
    """
    use_gpd = (tail_model == "gpd")

    # Tail-model parameters below are each bound only inside their guarding
    # branch (GPD or Gaussian tail_extension). Pre-declare them so the
    # cdf/quantile/pdf closures — which reference them under the same guards —
    # can never raise NameError if a future edit shifts a branch boundary. The
    # declared type is what they carry wherever they are *read*; the None
    # sentinel is deliberately not a float, so a guard that slips still fails
    # loudly (TypeError) instead of propagating a plausible number. mypy cannot
    # tie "use_gpd is True" to "F_lo is not None", hence the ignores here rather
    # than on each of the 13 reads downstream.
    F_lo: float
    F_hi: float
    u_lo: float
    u_hi: float
    xi_lo: float
    sigma_lo: float
    xi_hi: float
    sigma_hi: float
    mu: float
    sigma: float
    lo_scale: float
    hi_scale: float
    F_lo = F_hi = u_lo = u_hi = None  # type: ignore[assignment]
    xi_lo = sigma_lo = xi_hi = sigma_hi = None  # type: ignore[assignment]
    mu = sigma = lo_scale = hi_scale = None  # type: ignore[assignment]
    bulk_edge_slopes = None

    # ------------------------------------------------------------------
    # GPD branch: peaks-over-threshold model with thresholds *inside* the
    # data range. The spline covers only the bulk [u_lo, u_hi]; the GPD
    # takes over for x outside that interval.
    # ------------------------------------------------------------------
    if use_gpd:
        F_lo = float(tail_quantile)
        F_hi = 1.0 - F_lo
        u_lo = float(np.quantile(samples, F_lo))
        u_hi = float(np.quantile(samples, F_hi))
        if u_hi <= u_lo:
            # Degenerate: data is essentially constant. Fall back to Gaussian.
            use_gpd = False
        else:
            excesses_lo = u_lo - samples[samples < u_lo]
            excesses_hi = samples[samples > u_hi] - u_hi
            xi_lo, sigma_lo = _fit_gpd(excesses_lo)
            xi_hi, sigma_hi = _fit_gpd(excesses_hi)
            # Density each tail contributes *at* its threshold: the GPD pdf at
            # zero excess is 1/sigma, weighted by the tail's mass. Handing these
            # to the bulk spline as its endpoint knot slopes makes the density
            # continuous across the join. The threshold is an interior point of
            # the distribution, so PCHIP's one-sided endpoint rule (built for a
            # true support boundary) is the weaker of the two estimates: it sees
            # only the last bulk secant, while the GPD scale is an MLE over the
            # whole excess sample. Constraining the tail to the bulk instead
            # would also close the join, but it degrades held-out tail NLL.
            bulk_edge_slopes = (F_lo / sigma_lo, (1.0 - F_hi) / sigma_hi)

            bulk = samples[(samples >= u_lo) & (samples <= u_hi)]
            num_points_bulk = max(20, min(num_points, len(bulk) // 3))
            xg = np.asarray(np.quantile(bulk, np.linspace(0, 1, num_points_bulk)))
            # Anchor endpoints exactly at the thresholds so the spline
            # joins the GPD at known CDF values.
            xg[0], xg[-1] = u_lo, u_hi

            sorted_chain = _process_array(samples)
            counts = np.searchsorted(sorted_chain, xg, side='right')
            cdf_vals = counts / len(sorted_chain)
            cdf_vals[0], cdf_vals[-1] = F_lo, F_hi
            # data_min / data_max here demark the *spline* domain, not the
            # full sample range — outside this, the GPD branch is used.
            data_min, data_max = u_lo, u_hi

    if not use_gpd:
        # Adaptive grid size: avoid degenerate grids with small N
        num_points = max(20, min(num_points, len(samples) // 3))

        xg = np.asarray(np.quantile(samples, np.linspace(0, 1, num_points)))

        # Empirical CDF
        sorted_chain = _process_array(samples)
        counts = np.searchsorted(sorted_chain, xg, side='right')
        cdf_vals = counts / len(sorted_chain)

        # Data bounds (may be extended by prior bounds)
        data_min, data_max = np.min(samples), np.max(samples)

    # Extend grid to prior bounds if provided. Skipped under the GPD tail model,
    # whose thresholds live inside the data range and supply the out-of-bulk
    # tails directly (prior bounds would only inject spurious 0/1-clipped knots).
    if not use_gpd:
        # The empirical CDF is ``counts / n``, so the largest sample sits at
        # exactly 1.0 and the interval between it and an appended prior_high
        # would carry no mass at all: both knots clip to ``1 - min_eps``, the
        # tie bump below separates them by ``min_eps / (10 K)``, and the
        # boundary bin ends up with ~5e-11 of CDF where the plotting position
        # says ``0.5 / n``. That collapses the spline slope there, and with
        # it the reported density just inside the upper bound -- measured on
        # U(0, 1)^2 with prior [[0, 1], [0, 1]], n = 1500, 80 epochs, seeds
        # 0-3, log_prob at (1, 0.5) came out -14.6 to -11.6 against a truth
        # of 0. The lower wall has no such problem -- the smallest sample
        # sits at ``1 / n``, which already leaves the bottom bin its mass --
        # so only the top needs holding back, to its Hazen position. With it
        # the same four fits report +1.2 to +5.2 at (1, 0.5), matching the
        # +1.2 to +5.7 at (0, 0.5), and no training row is mapped to the
        # spurious z = ndtri(1 - min_eps) = 5.2 outlier any more.
        if prior_low is not None and prior_low < xg[0]:
            xg = np.concatenate([[prior_low], xg])
            cdf_vals = np.concatenate([[0.0], cdf_vals])
            data_min = prior_low
        if prior_high is not None and prior_high > xg[-1]:
            cdf_vals = np.minimum(cdf_vals, 1.0 - 0.5 / len(sorted_chain))
            xg = np.concatenate([xg, [prior_high]])
            cdf_vals = np.concatenate([cdf_vals, [1.0]])
            data_max = prior_high
    # Clip CDF values to avoid 0 and 1, which cause issues with inverse normal CDF
    cdf_vals = np.clip(cdf_vals, min_eps, 1.0 - min_eps)
    # Ensure strict monotonicity for the inverse CDF spline: prior bound
    # extension + clipping can create duplicate CDF values (e.g. the last
    # quantile grid point and the prior_high point both clip to 1 - min_eps).
    # PCHIP requires strictly increasing x-values, so bump any ties.
    eps_bump = min_eps / (len(cdf_vals) * 10)
    for i in range(1, len(cdf_vals)):
        if cdf_vals[i] <= cdf_vals[i - 1]:
            cdf_vals[i] = cdf_vals[i - 1] + eps_bump
    # Store actual CDF bounds after clipping (may differ from min_eps due to empirical CDF)
    cdf_min, cdf_max = float(np.min(cdf_vals)), float(np.max(cdf_vals))

    # Bulk interpolant. Both families share the empirical-quantile knots above
    # and the tail / prior-bound closures below; they differ only in how the
    # bulk CDF is interpolated, inverted and differentiated.
    if marginal == "rqs":
        # Single closed-form rational-quadratic spline: forward, inverse and
        # derivative come from one parameterization (machine-precision inverse).
        # The edge slope floor rescues PCHIP's zeroed endpoint derivative, which
        # otherwise collapses the density at the outermost knot — with no prior
        # bounds that knot *is* each dimension's most extreme training sample.
        bulk_cdf, bulk_inverse, bulk_deriv = _rqs_bulk(
            xg, cdf_vals, edge_slope_floor=0.05, edge_slopes=bulk_edge_slopes)
    elif marginal == "pchip":
        # Two independent PCHIP splines for CDF and inverse-CDF; the derivative
        # is the analytic (piecewise-cubic) derivative of the CDF spline,
        # strictly cheaper than running jax.grad through the spline per call.
        cdf_spline = PchipInterpolator(xg, cdf_vals, extrapolate=False, check=False)
        bulk_inverse = PchipInterpolator(cdf_vals, xg, extrapolate=False, check=False)
        bulk_cdf = cdf_spline
        bulk_deriv = cdf_spline.derivative()
    else:
        raise ValueError(f"marginal must be 'pchip' or 'rqs', got {marginal!r}")

    # ------------------------------------------------------------------
    # Gaussian tail model: fit a Gaussian to the marginal and use it for
    # extrapolation beyond the observed data range.
    # ------------------------------------------------------------------
    if tail_extension:
        mu = float(np.mean(samples))
        sigma = float(np.std(samples, ddof=1))
        sigma = max(sigma, _SCALE_FLOOR)  # guard against zero std

        # CDF values of the Gaussian at the data boundaries. The upper one is
        # taken as a survival function: ``1 - norm.cdf`` cancels to exactly 0.0
        # once data_max exceeds ~8.3 fitted sigmas, which sends hi_scale to
        # ~1e26 and destroys the right tail.
        from scipy.stats import norm as _norm
        gauss_cdf_lo = float(_norm.cdf(data_min, loc=mu, scale=sigma))
        gauss_sf_hi = float(_norm.sf(data_max, loc=mu, scale=sigma))

        # Empirical CDF at boundaries (from spline)
        emp_cdf_lo = float(cdf_min)
        emp_cdf_hi = float(cdf_max)

        # Scale factors so the Gaussian tail joins the empirical CDF at
        # the boundary.  We want:
        #   tail_cdf(data_min) = emp_cdf_lo
        #   tail_cdf(data_max) = emp_cdf_hi
        # Using:  tail_cdf(x) = emp_cdf_lo * Phi_gauss(x) / Phi_gauss(data_min)   [left]
        #         tail_cdf(x) = 1 - (1 - emp_cdf_hi) * Phi_gauss_sf(x) / Phi_gauss_sf(data_max)  [right]
        # Floored at the float64 subnormal edge rather than at _POS_FLOOR:
        # 1e-30 is reached by an entirely ordinary 11.5-sigma data boundary,
        # and truncating there rescales the whole tail (both the quantile
        # inversion below and the tail density in pdf_fn) by many decades.
        lo_scale = emp_cdf_lo / max(gauss_cdf_lo, _NDTRI_FLOOR)
        hi_scale = (1.0 - emp_cdf_hi) / max(gauss_sf_hi, _NDTRI_FLOOR)
    # ------------------------------------------------------------------

    def cdf_fn(x: ArrayLike) -> Array:
        x = jnp.asarray(x)
        x_clipped = jnp.clip(x, data_min, data_max)
        spline_val = bulk_cdf(x_clipped)
        if use_gpd:
            left_tail = F_lo * (1.0 - _gpd_cdf(u_lo - x, xi_lo, sigma_lo))
            right_tail = F_hi + (1.0 - F_hi) * _gpd_cdf(x - u_hi, xi_hi, sigma_hi)
            y = jnp.where(
                x <= u_lo,
                left_tail,
                jnp.where(x >= u_hi, right_tail, spline_val),
            )
        elif tail_extension:
            gauss_cdf = ndtr((x - mu) / sigma)
            # Clamped so the cast to x's dtype cannot overflow to inf, which
            # turned inf * 0 into NaN below x64 (see :func:`_dtype_max`).
            left_tail = min(lo_scale, _dtype_max(x)) * gauss_cdf
            right_tail = 1.0 - min(hi_scale, _dtype_max(x)) * (1.0 - gauss_cdf)
            y = jnp.where(
                x <= data_min,
                left_tail,
                jnp.where(x >= data_max, right_tail, spline_val),
            )
        else:
            # No tail model: clip flat to the *boundary* CDF value (cdf_min /
            # cdf_max), not to min_eps. Using min_eps here created a jump of
            # ~1/N at data_min/data_max — mapping each dimension's extreme
            # sample to a spurious z≈±5.2 outlier. cdf_min == bulk_cdf(data_min)
            # (and cdf_max == bulk_cdf(data_max)), so this join is continuous.
            # (With prior_bounds, cdf_min/cdf_max already equal min_eps/1-min_eps,
            # so that recommended path is unchanged.)
            y = jnp.where(
                x <= data_min,
                cdf_min,
                jnp.where(x >= data_max, cdf_max, spline_val),
            )
        # Final bounds check to ensure we never return exactly 0 or 1
        return jnp.clip(y, min_eps, 1.0 - min_eps)

    def quantile_fn(u: ArrayLike) -> Array:
        u = jnp.asarray(u)
        if use_gpd:
            # Avoid 1-0 inside GPD quantile when u is at the endpoints.
            safe_left = jnp.clip(1.0 - u / max(F_lo, min_eps), min_eps, 1.0 - min_eps)
            safe_right = jnp.clip((u - F_hi) / max(1.0 - F_hi, min_eps),
                                   min_eps, 1.0 - min_eps)
            left_x = u_lo - _gpd_quantile(safe_left, xi_lo, sigma_lo)
            right_x = u_hi + _gpd_quantile(safe_right, xi_hi, sigma_hi)
            u_bulk = jnp.clip(u, cdf_min, cdf_max)
            mid_x = bulk_inverse(u_bulk)
            return jnp.where(
                u <= F_lo,
                left_x,
                jnp.where(u >= F_hi, right_x, mid_x),
            )
        if tail_extension:
            # For values below or above the empirical CDF range, invert the
            # Gaussian tail model. Both arguments are floored at the ndtri
            # domain limit, not at min_eps: ndtri(1e-7) is only -5.199, so a
            # min_eps floor pins both tails at mu -/+ 5.199 sigma. Whenever the
            # data reach past that (routine for t, lognormal, even exponential
            # at n = 5000) the tail branch then returns a value *inside* the
            # data range and quantile_fn goes non-monotone at the join.
            # Left tail:  u = lo_scale * Phi((x - mu) / sigma)
            #   => x = mu + sigma * Phi^{-1}(u / lo_scale)
            ndtri_lo = _ndtri_floor(u)
            # Clamped so the divisor cannot overflow the cast to u's dtype;
            # see :func:`_dtype_max`. Both are no-ops in float64.
            scale_lo = min(lo_scale, _dtype_max(u))
            scale_hi = min(hi_scale, _dtype_max(u))
            left_x = mu + sigma * ndtri(
                jnp.clip(u / scale_lo, ndtri_lo, 1.0 - min_eps))
            # Right tail: 1 - u = hi_scale * (1 - Phi((x - mu) / sigma))
            #   => x = mu - sigma * Phi^{-1}((1 - u) / hi_scale), via
            # Phi^{-1}(1 - p) = -Phi^{-1}(p) so that no 1 - x cancellation
            # (which loses every digit of a small survival probability) enters
            # the ndtri argument.
            right_x = mu - sigma * ndtri(
                jnp.clip((1.0 - u) / scale_hi, ndtri_lo, 1.0 - min_eps))
            # A no-op wherever the join is continuous (the tail values are then
            # already outside the data range). It only binds past ~37 sigma,
            # where the ndtri floor above saturates, and turns what would be a
            # fold — quantile_fn decreasing, i.e. a non-invertible marginal —
            # into a flat extension.
            left_x = jnp.minimum(left_x, data_min)
            right_x = jnp.maximum(right_x, data_max)

            u_clipped = jnp.clip(u, cdf_min, cdf_max)
            mid_x = bulk_inverse(u_clipped)

            y = jnp.where(
                u < cdf_min,
                left_x,
                jnp.where(u > cdf_max, right_x, mid_x)
            )
        else:
            u_clipped = jnp.clip(u, cdf_min, cdf_max)
            y = bulk_inverse(u_clipped)
            # no tail model: the map saturates at the support ends, one notch
            # inside them (see _q_limits) so no draw lands on a wall
            y = jnp.clip(y, *_q_limits)
        return y

    # Strictly-inside saturation points for the quantile map. Returning
    # data_min / data_max themselves puts sample atoms exactly on the support
    # ends, which under prior_bounds are hard walls: measured 4 draws of
    # 20000 landing on exactly 0.0 for U(0, 1) data with prior_bounds [0, 1],
    # marginal="rqs" and "pchip" alike. One notch inside is invisible to every
    # density and CDF value here and removes the atom.
    _x_inset = max(1e-15 * float(data_max - data_min),
                   4.0 * float(np.spacing(max(abs(float(data_min)),
                                              abs(float(data_max))))))
    _q_limits = (float(data_min) + _x_inset, float(data_max) - _x_inset)

    _inv_sqrt_2pi = 1.0 / jnp.sqrt(2.0 * jnp.pi)

    def pdf_fn(x: ArrayLike) -> Array:
        """PDF via the analytic derivative of the bulk CDF interpolant.

        Mirrors the branch structure of ``cdf_fn``:
          * middle (within data range): closed-form derivative of the bulk
            interpolant (PCHIP or RQS), no jax.grad.
          * left/right tail with ``tail_extension``: derivative of the
            Gaussian-tail CDF, i.e. ``scale * φ((x-μ)/σ) / σ``.
          * no tail extension: PDF is zero outside the data range.

        Every branch is floored to ``min_eps`` on return. This is a deliberate
        numerical guard so ``log(pdf)`` (used by the Jacobian log-det and by
        ``log_prob``) stays finite everywhere. The consequence, by design: with
        no tail model the density does not integrate to exactly 1 and a query
        point outside the observed data range reports a small floor density
        (``min_eps``) rather than a true 0. Supply ``prior_bounds`` or
        ``tail_extension``/``tail_model`` to model beyond the data range.

        Works on scalar or vector input (jax.grad previously forced
        scalar-only).
        """
        x = jnp.asarray(x)
        x_in = jnp.clip(x, data_min, data_max)
        mid_pdf = bulk_deriv(x_in)

        if use_gpd:
            left_pdf = F_lo * _gpd_pdf(u_lo - x, xi_lo, sigma_lo)
            right_pdf = (1.0 - F_hi) * _gpd_pdf(x - u_hi, xi_hi, sigma_hi)
            y = jnp.where(
                x <= u_lo,
                left_pdf,
                jnp.where(x >= u_hi, right_pdf, mid_pdf),
            )
        elif tail_extension:
            z = (x - mu) / sigma
            tail_normal_pdf = _inv_sqrt_2pi * jnp.exp(-0.5 * z * z) / sigma
            left_pdf = min(lo_scale, _dtype_max(x)) * tail_normal_pdf
            right_pdf = min(hi_scale, _dtype_max(x)) * tail_normal_pdf
            y = jnp.where(
                x <= data_min,
                left_pdf,
                jnp.where(x >= data_max, right_pdf, mid_pdf),
            )
        else:
            # strict comparisons: the endpoints themselves are training rows
            # (or, under prior bounds, the walls), and the interpolant is
            # defined there. Testing ``<=`` / ``>=`` handed them the
            # out-of-range 0.0 instead, so the extreme sample of each
            # dimension reported the ``min_eps`` floor -- 1.0e-7 against a
            # true 4.7e-4 on N(0, 1) with n = 3000, a 7.6-nat hole one ULP
            # wide, while ``data_min + 1e-12`` was already correct.
            y = jnp.where(
                (x < data_min) | (x > data_max),
                0.0,
                mid_pdf,
            )
        return jnp.maximum(y, min_eps)

    return cdf_vals, cdf_fn, quantile_fn, pdf_fn


class EmpiricalMarginalToGaussian(AbstractBijection):
    """
    Bijection transforming empirical marginal distribution to standard Gaussian.

    This bijection uses the probability integral transform (PIT) combined with
    the inverse normal CDF to map samples from an empirical distribution to
    a standard Gaussian distribution. It's useful for copula modeling where
    marginals need to be transformed to Gaussian.

    Attributes
    ----------
    samples : np.ndarray
        Original data samples used to construct the empirical distribution.
    cdf_fn : Callable
        Empirical CDF function.
    quantile_fn : Callable
        Inverse CDF (quantile) function.
    pdf_fn : Callable
        Empirical PDF function (derivative of CDF).
    min_eps : float, default=1e-7
        Minimum epsilon to avoid CDF values of exactly 0 or 1.

    The remaining attributes are reconstruction metadata: they record the
    arguments passed to :func:`make_empirical_cdf_spline` so that
    :func:`coppuccino.model_io.load_flow` can rebuild ``cdf_fn`` /
    ``quantile_fn`` / ``pdf_fn`` identically from ``samples``.

    num_points : int, default=200
        Number of grid points used for the CDF spline.
    tail_extension : bool, default=False
        Whether a Gaussian tail model was used beyond the data range.
    prior_low, prior_high : float or None, default=None
        Prior support bounds the CDF grid was extended to, if any.
    tail_model : str, default="gaussian"
        Tail model used (``"gaussian"`` or the experimental ``"gpd"``).
    tail_quantile : float, default=0.05
        GPD tail fraction (only meaningful when ``tail_model="gpd"``).
    marginal : str, default="rqs"
        Which bulk interpolant family built ``cdf_fn``/``quantile_fn``/``pdf_fn``
        (``"rqs"`` or ``"pchip"``). Must match the family
        :func:`make_empirical_cdf_spline` actually used, since
        :mod:`coppuccino.model_io` reconstructs the closures from this field;
        the default tracks that function's own default.

    Examples
    --------
    >>> import numpy as np
    >>> import jax.numpy as jnp
    >>> from coppuccino.bijections import EmpiricalMarginalToGaussian, make_empirical_cdf_spline
    >>> # Generate samples from a non-Gaussian distribution
    >>> samples = np.random.exponential(2.0, 1000)
    >>> cdf_vals, cdf_fn, quantile_fn, pdf_fn = make_empirical_cdf_spline(
    ...     samples, marginal="rqs")
    >>> # Create bijection — pass the same ``marginal`` used to build the
    >>> # closures, so the recorded metadata can rebuild them faithfully.
    >>> bij = EmpiricalMarginalToGaussian(
    ...     samples=samples, cdf_fn=cdf_fn, quantile_fn=quantile_fn,
    ...     pdf_fn=pdf_fn, marginal="rqs"
    ... )
    >>> # Transform sample to Gaussian space
    >>> x = jnp.array(2.0)
    >>> z, log_det = bij.inverse_and_log_det(x)
    >>> # Transform back
    >>> x_recovered, log_det_fwd = bij.transform_and_log_det(z)
    """
    samples: np.ndarray
    cdf_fn: Callable
    quantile_fn: Callable
    pdf_fn: Callable
    min_eps: float = 1e-7
    num_points: int = 200
    tail_extension: bool = False
    prior_low: Optional[float] = None
    prior_high: Optional[float] = None
    tail_model: str = "gaussian"
    tail_quantile: float = 0.05
    marginal: str = "rqs"
    # flowjax declares shape/cond_shape as equinox ``AbstractVar``, which mypy
    # reads as instance variables of the base; a concrete ``ClassVar`` is the
    # intended implementation here (both are per-class constants).
    cond_shape: ClassVar[None] = None  # type: ignore[misc]
    shape: ClassVar[tuple[int, ...]] = ()  # type: ignore[misc]

    def inverse_and_log_det(self, x: ArrayLike,
                            condition: Optional[ArrayLike] = None) -> tuple[Array, Array]:
        """
        Transform from original space to standard Gaussian.

        Applies the probability integral transform using the empirical CDF,
        then maps the uniform variable to Gaussian via the inverse normal CDF.

        Parameters
        ----------
        x : array_like
            Values in the original (empirical) space.
        condition : None
            Not used (for API compatibility).

        Returns
        -------
        z : jax.Array
            Transformed values in standard Gaussian space.
        log_det : float
            Log absolute determinant of the Jacobian.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> # bij is an EmpiricalMarginalToGaussian (see the class docstring)
        >>> z, log_det = bij.inverse_and_log_det(jnp.array([1.0, 2.0, 3.0]))  # doctest: +SKIP
        """
        # Get CDF value using smooth interpolation
        u = self.cdf_fn(x)

        # Transform to standard Gaussian
        z = ndtri(u)  # JAX equivalent of norm.ppf

        # Jacobian: |dz/dx| = |dz/du| * |du/dx| = (1/φ(z)) * pdf(x)
        pdf_x = self.pdf_fn(x)
        # JAX normal PDF
        pdf_z = jnp.exp(-0.5 * z**2) / jnp.sqrt(2 * jnp.pi)
        log_det = jnp.log(pdf_x) - jnp.log(pdf_z)

        return z, jnp.sum(log_det)

    def transform_and_log_det(self, z: ArrayLike,
                              condition: Optional[ArrayLike] = None) -> tuple[Array, Array]:
        """
        Transform from standard Gaussian to original space.

        Applies the normal CDF to get a uniform variable, then maps to the
        original space using the empirical quantile function.

        Parameters
        ----------
        z : array_like
            Values in standard Gaussian space.
        condition : None
            Not used (for API compatibility).

        Returns
        -------
        x : jax.Array
            Transformed values in the original (empirical) space.
        log_det : float
            Log absolute determinant of the Jacobian.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> # bij is an EmpiricalMarginalToGaussian (see the class docstring)
        >>> z = jnp.array([0.0, 1.0, -1.0])
        >>> x, log_det = bij.transform_and_log_det(z)  # doctest: +SKIP
        """
        # Transform to uniform using JAX normal CDF
        u = ndtr(z)  # JAX equivalent of norm.cdf
        # Clip u to ensure it's in valid range for quantile function
        u = jnp.clip(u, self.min_eps, 1.0 - self.min_eps)

        x = self.quantile_fn(u)
        # Jacobian: |dx/dz| = |dx/du| * |du/dz| = (1/pdf(x)) * φ(z)
        pdf_x = self.pdf_fn(x)

        # JAX normal PDF
        pdf_z = jnp.exp(-0.5 * z**2) / jnp.sqrt(2 * jnp.pi)
        log_det = jnp.log(1.0 / pdf_x) + jnp.log(pdf_z)

        return x, jnp.sum(log_det)


# ----------------------------------------------------------------------
# Smooth-tail marginals: a true diffeomorphism z in R  <->  theta in (lo, hi)
# ----------------------------------------------------------------------

# Both tails of a smooth marginal are the same map seen from opposite ends,
# so they are written once in each side's own *outward* coordinates: on the
# left ``u = z`` and ``y = theta``, on the right ``u = -z`` and
# ``y = -theta``. In those coordinates every tail is increasing in ``u`` and
# runs outward as ``u`` decreases, so one implementation serves both sides
# and the mirroring costs only exact sign flips.

_EXP_MAX = 700.0  # exp overflow guard for values that get where()-selected away
# Conditioning guard on the wall shape's join-side derivative. It is a ratio
# of densities, so a collapsed endpoint-slope estimate can send it to 1e30 or
# beyond, where the spline kernel's quadratic squares it. Past this value the
# shape is numerically its own limit anyway.
_WALL_BETA_MAX = 1e12
# How far past its own outermost knot, in z, a bounded tail may run before it
# saturates. It is a *margin*, not an absolute |z| cap, because the quantity
# that has to be bounded is extrapolation distance: the outermost bulk knot
# sits on the most extreme sample, so a tail's ``u_edge`` is exactly the
# deepest z the copula flow downstream was ever trained on in this dimension,
# and everything below it is extrapolation. Measured on U(0, 1)^2, n = 1500,
# 80 epochs, seeds 0-3, the fitted copula's excess over its own N(0, 1) base
# at (z, 0) ran -0.0 to +0.2 nats at |z| = 3 (inside training, |u_edge| =
# 3.40), +0.0 to +0.8 at 3.4, -0.3 to +1.9 at 4, +0.5 to +4.5 at 5 and +1.6 to
# +14.9 at 8 -- roughly 2 nats per unit z beyond the edge. An absolute cap
# cannot express that: |u_edge| is ndtri(0.5 / n), so a fixed 4.0 is 0.6 past
# the edge at n = 1500, 0.24 past it at n = 6000 and *inside* it from
# n = 15790 on. Well before that the half-gap guard below takes over and the
# constant stops meaning anything: with a fixed 4.0 the z actually reported at
# the bound of a U(0, 1) fit was 4.09 at n = 6000, 4.29 at 30000 and 4.51 at
# 100000 -- the guard's number, not the cap's.
# The margin is the whole trade. Everything below the cap collapses onto one
# theta, so a smaller margin buys a smaller copula error at a wall and pays in
# wall resolution: the saturation point moves out to about
# Phi(u_edge - margin) / Phi(u_edge) of the knot-to-bound distance, which at
# n = 1500 is 11% of it at margin 0.6 (and hits the half-gap guard on 2 of 4
# U(0, 1) seeds), 1.7% at 1.0 and 0.14% at 1.5. 1.5 is the smallest margin
# measured at which the marginal's own wall-density error is indistinguishable
# from the uncapped tail's (median 1.09 / worst 2.03 nats against the analytic
# truth on truncated N(0, 1), 8 seeds x both walls x {0, 1e-9, 1e-6} of the
# span, versus 1.09 / 1.97 uncapped and 1.13 / 2.92 at margin 1.0).
#
# It also brings the at-bound density back down near the bulk of the fitted
# distribution, where the uncapped tail put it clearly above: comparing a box
# mid-edge against the maximum log-density over 20000 draws, the uncapped tail
# ran 2.9 to 9.9 nats *above* on U(0, 1), truncated N(0, 1) and Beta(0.5, 3),
# and margin 1.5 brings it to roughly the same order as that maximum. Treat
# that comparison as indicative only -- the maximum over a finite sample is
# itself random, and holding one fit and varying only the draw seed moves it
# by a couple of nats in either direction, so it is not a bound. Box *corners*,
# where both dimensions sit at the cap at once, stay worse than the mid-edges
# (+0.5 to +11.9 against +0.1 to +4.6 on U(0, 1)^2 at the same settings)
# -- see SmoothMarginalToGaussian.
_WALL_Z_MARGIN = 1.5
# Largest |knot value| the spline kernel can carry in float64 (its quadratic
# squares theta-scale products, so the headroom is the fourth root of the
# float64 range) and largest usable unbounded-tail growth rate.
_MAX_ABS_KNOT = 1e75
_TAIL_ALPHA_MAX = 10.0


def _log_phi(u: Array) -> Array:
    """Standard normal log-density, finite for every representable ``u``."""
    return -0.5 * u * u - _LOG_SQRT_2PI


class _MarginalTail(NamedTuple):
    """One analytic tail of :func:`make_smooth_marginal`, in ``(u, y)`` coords.

    Two shapes share this record. ``wall`` selects between them, and it is a
    build-time Python bool, so each closure below traces only one branch.

    * ``wall=True`` -- *bounded side*. The CDF has to reach 0 at the bound,
      which in Gaussian space happens only as ``u -> -inf``, so the tail is
      written as the inverse of a CDF shape::

          Phi(u) = p_edge * S((y - y_bound) / gap)

      with ``S`` a monotone rational-quadratic spline on the unit interval
      pinned at ``S(0) = 0``, ``S(1) = 1``, ``S'(1) = beta`` (the bulk
      spline's own density at the join, so the density is continuous there)
      and ``S'(0) = 1``. That last pin is what the family exists for: the
      marginal density at the bound is ``p_edge / gap``, the mean density
      over the knot-to-bound interval -- finite, strictly positive, and
      reached smoothly. An exponential or power-law approach instead makes
      ``y - y_bound`` log-normal or a power law, whose density at the bound
      can only be 0 or infinity. It is also the estimator the fleet's
      endpoint knots (:mod:`coppuccino.fleet`) use, whose wall accuracy this
      reproduces.
    * ``wall=False`` -- *unbounded side*. ``dy/du = slope *
      exp(alpha |u - u_edge|)``: linear for ``alpha = 0``
      (Gaussian tails in theta), heavier above it. There is no saturation
      point on such a side, and ``tail_alpha`` is consumed only here.

    ``y_min`` is where a bounded forward map saturates: the bound plus the
    larger of the smallest offset that survives ``bound + offset`` in
    floating point and the offset whose ``u`` is ``_WALL_Z_MARGIN`` below
    ``u_edge``, so no finite ``u`` maps onto a bound exactly (``-inf`` on an
    unbounded side). ``r_floor`` is the same limit as a fractional wall
    offset; the inverse and the log-derivative clamp on it too, so all three
    directions agree there. It is also capped at half the knot-to-bound
    distance, which binds only when ``beta`` is large enough that the wall
    shape puts the margin's quantile past that point.
    """
    u_edge: float
    y_edge: float
    slope: float
    alpha: float
    y_min: float
    wall: bool
    y_bound: float
    gap: float
    beta: float
    log_p_edge: float
    r_floor: float
    shape_x: jnp.ndarray
    shape_y: jnp.ndarray
    shape_d: jnp.ndarray


def _inset_wall_atoms(y: np.ndarray, bound: float, eps0: float,
                      max_inset: float) -> np.ndarray:
    """Move samples sitting on ``bound`` (approached from above) just inside it.

    Values within ``eps0`` of the bound -- the reflection / in-bounds
    artifacts of a bounded MCMC kernel, which arrive as an exact atom -- are
    placed at half the gap to the first strictly interior value rather than
    at the raw ``eps0`` margin. The atom then becomes a *resolvable* shoulder
    whose knot offset is comparable to the first bin width, instead of a
    sub-ULP spike that the outermost knot slope cannot express. When every
    sample sits in the window there is no interior value to measure the gap
    against, so they are only pushed off the bound.

    The mirrored side calls this on ``-y`` with ``-bound``; sign flips are
    exact, so both walls get bit-for-bit the same treatment.
    """
    at = y <= bound + eps0
    if not (at.any() and (~at).any()):
        return np.maximum(y, bound + eps0)
    gap = float(np.min(y[~at])) - bound
    y[at] = bound + float(np.clip(0.5 * gap, eps0, max_inset))
    return y


def _make_tail(u_edge: float, y_edge: float, slope: float, alpha: float,
               bound: Optional[float], span: float) -> _MarginalTail:
    """Fit one tail beyond an outermost bulk knot; see :class:`_MarginalTail`.

    Parameters
    ----------
    u_edge, y_edge : float
        Outermost knot of the bulk spline in this side's outward coordinates.
    slope : float
        ``dy/du`` of the bulk spline at that knot (strictly positive).
    alpha : float
        ``tail_alpha``; only used on an unbounded side.
    bound : float or None
        The prior bound in ``y`` coordinates, or None for an unbounded side.
    span : float
        Support span, used only for the float-resolution floors.
    """
    unit = jnp.asarray([0.0, 1.0])
    free = _MarginalTail(
        u_edge=u_edge, y_edge=y_edge, slope=slope, alpha=alpha, y_min=-np.inf,
        wall=False, y_bound=-np.inf, gap=0.0, beta=1.0, log_p_edge=0.0,
        r_floor=0.0, shape_x=unit, shape_y=unit, shape_d=unit)
    if bound is None:
        return free
    # Distance from the outermost knot to the bound. It rounds to exactly
    # zero when the bound's magnitude dwarfs the data span, which would
    # divide by zero below; one ULP at the bound is the smallest gap
    # representable there.
    gap = max(y_edge - bound, float(np.spacing(abs(bound))))
    # Smallest wall offset the tail may produce or resolve. Below ~ULP(bound)
    # the float addition ``bound + tiny`` absorbs to the bound exactly -- an
    # exact-wall sample atom. Saturating there instead relocates that
    # unrepresentable mass to a strictly interior point. Half the knot offset
    # caps it: an offset at or beyond the outermost knot would make the tail
    # decreasing.
    eps = float(np.clip(4.0 * np.spacing(abs(bound)) + 1e-30 * span,
                        1e-30 * span, min(0.01 * span, 0.5 * gap)))
    from scipy.special import log_ndtr as _log_ndtr_np
    log_p_edge = float(_log_ndtr_np(u_edge))
    # S'(1): the bulk's density at the join knot divided by the mean density
    # over the knot-to-bound interval. 1 means the data approach the bound at
    # constant density; larger means the density falls off toward it.
    beta = float(np.clip((gap / slope) * np.exp(_log_phi(jnp.asarray(u_edge))
                                                - log_p_edge),
                         _POS_FLOOR, _WALL_BETA_MAX))
    shape_d = jnp.asarray([1.0, beta])
    # Saturation point: the later of the float-resolution limit above and the
    # wall offset whose u is one margin below the outermost knot. That stops
    # the tail handing the copula flow a z it has no basis for -- extrapolation
    # starts at ``u_edge``, not at some deep absolute |z| -- and it is what
    # keeps the joint density at a bound comparable to the interior: on
    # U(0, 1)^2, n = 1500, 80 epochs, seeds 0-3 the reported log-density on the
    # four box mid-edges was +1.8 to +15.5 nats above the flat truth when the
    # tail ran to |z| = 8 -- above the maximum over 20000 of the fit's own
    # draws in every seed -- and +0.1 to +4.6 at this margin, below that
    # maximum by 0.7 to 3.6 in every seed.
    r_z = float(_rqs_inverse(jnp.asarray(
        np.exp(float(_log_ndtr_np(u_edge - _WALL_Z_MARGIN)) - log_p_edge)),
        unit, unit, shape_d))
    r_floor = min(max(eps / gap, r_z), 0.5)
    return _MarginalTail(
        u_edge=u_edge, y_edge=y_edge, slope=slope, alpha=alpha,
        y_min=bound + gap * r_floor, wall=True, y_bound=bound, gap=gap,
        beta=beta, log_p_edge=log_p_edge, r_floor=r_floor,
        shape_x=unit, shape_y=unit, shape_d=shape_d)


def _tail_r(u: Array, t: _MarginalTail) -> Array:
    """Fractional wall offset ``S^-1(Phi(u) / p_edge)``, floored at the clamp."""
    q = jnp.exp(jnp.minimum(log_ndtr(u) - t.log_p_edge, 0.0))
    return jnp.maximum(_rqs_inverse(q, t.shape_x, t.shape_y, t.shape_d),
                       t.r_floor)


def _tail_y(u: Array, t: _MarginalTail) -> Array:
    """Tail value ``y(u)``, saturating at ``t.y_min``."""
    if t.wall:
        y = t.y_bound + t.gap * _tail_r(u, t)
        # ``y_min`` is computed in float64 at build time. Under a coarser
        # runtime dtype (x64 turned back off) it rounds onto the bound
        # itself, so the saturation guard has to be restated in the dtype
        # actually in use or the map emits exact-wall values.
        return jnp.maximum(y, jnp.nextafter(
            jnp.asarray(t.y_bound, dtype=y.dtype), jnp.inf))
    du = jnp.minimum(u - t.u_edge, 0.0)
    if t.alpha > 0.0:
        return t.y_edge - (t.slope / t.alpha) * jnp.expm1(
            jnp.minimum(-t.alpha * du, _EXP_MAX))
    return t.y_edge + t.slope * du


def _tail_u(y: Array, t: _MarginalTail) -> Array:
    """Exact inverse of :func:`_tail_y` above the saturation point."""
    if t.wall:
        r = jnp.maximum((jnp.maximum(y, t.y_min) - t.y_bound) / t.gap,
                        t.r_floor)
        s = _rqs_forward(r, t.shape_x, t.shape_y, t.shape_d)
        return ndtri(jnp.maximum(jnp.exp(t.log_p_edge) * s, _ndtri_floor(y)))
    dy = y - t.y_edge
    if t.alpha > 0.0:
        return t.u_edge - jnp.log(
            jnp.maximum(1.0 - t.alpha * dy / t.slope, _NDTRI_FLOOR)) / t.alpha
    return t.u_edge + dy / t.slope


def _tail_log_deriv(u: Array, t: _MarginalTail) -> Array:
    """``log |dtheta/dz|`` on the tail; orientation-free, so it needs no sign."""
    if not t.wall:
        return np.log(t.slope) - t.alpha * jnp.minimum(u - t.u_edge, 0.0)
    # dtheta/dz = phi(u) / f(y), with f = p_edge S'(r) / gap the tail density
    s_deriv = _rqs_derivative(_tail_r(u, t), t.shape_x, t.shape_y, t.shape_d)
    return (_log_phi(u) - t.log_p_edge + np.log(t.gap)
            - jnp.log(jnp.maximum(s_deriv, _POS_FLOOR)))


def make_smooth_marginal(samples: np.ndarray,
                         num_points: int = 200,
                         prior_low: Optional[float] = None,
                         prior_high: Optional[float] = None,
                         tail_alpha: float = 0.0,
                         # The three closures are ``Any`` rather than
                         # ``Callable`` on purpose. Both callers splat this
                         # tuple into ``SmoothMarginalToGaussian(...)``, whose
                         # synthesised ``__init__`` mypy reads as starting
                         # with flowjax's ``AbstractVar`` shape/cond_shape
                         # (see the ClassVar note on that class); typing them
                         # turns that existing positional mismatch into four
                         # new arg-type errors in copula_flows and model_io,
                         # which already carry a call-arg ignore for it.
                         ) -> tuple[Any, Any, Any,
                                    tuple[Optional[float], Optional[float]]]:
    """Build a clip-free marginal map ``theta(z)``: all of R onto the support.

    The map is a single monotone C^1 object evaluated consistently in both
    directions (forward, inverse and log-derivative share one
    parameterization), so the flow's sample stream and its reported density
    agree everywhere -- including arbitrarily deep in the tails. This removes
    the legacy Gaussianized pipeline's boundary pathologies at their root:
    there is no CDF clip, no quantile clamp (hence no boundary sample atom)
    and no floored wall density (hence no density cliff), and the density the
    map reports *at* a prior bound is finite and positive rather than 0 or
    infinite (see :class:`SmoothMarginalToGaussian` for what that costs and
    what it does not fix).

    Construction:

    * **Bulk**: knots at the data quantiles ``theta_j`` paired with
      ``z_j = ndtri(p_j)`` at Hazen plotting positions -- all finite -- joined
      by the closed-form monotone rational-quadratic spline (PCHIP slopes).
    * **Tails** beyond the outermost knots (see :class:`_MarginalTail`):
        - *bounded side* (prior bound given): a monotone spline in CDF space
          pinned to reach 0 exactly at the bound, with a *finite, strictly
          positive* wall density equal to the mean density between the
          outermost knot and the bound. The bound is still reached only as
          ``z -> -+inf``, so every finite flow output lands strictly inside
          the prior, but the density there no longer has to diverge or
          vanish.
        - *unbounded side*: ``dtheta/dz = m exp(alpha |z - z_edge|)`` --
          ``tail_alpha = 0`` gives a linear map (Gaussian tails in theta);
          ``tail_alpha > 0`` gives progressively heavier tails. Density
          queries and samples beyond the observed data range are therefore
          honest and smooth (outlier handling), replacing the legacy
          ``tail_extension``/GPD branches.

    Boundary semantics: samples sitting exactly on a prior bound (the
    documented recommended use of ``prior_bounds``, and what a reflecting or
    clipping MCMC kernel produces) are relocated just inside it before the
    knots are built (:func:`_inset_wall_atoms`), so the atom becomes a
    resolvable shoulder rather than a sub-ULP spike. Query points are *not*
    relocated: the map is one bijection in both directions, and a row on the
    bound reads the wall density off the tail like any other point. That
    density is the atom's mass over the inset width, so it is genuinely
    large -- on ``clip(N(0, 1), -1, 1)`` with 16-18% of 1500 rows on the
    lower bound it came out 4.9 to 8.4 nats above the interior median over
    three seeds. A tie block that large also pulls the outermost knot far in,
    to ``z = -0.9`` or so, and the tail is only allowed
    ``_WALL_Z_MARGIN`` past that, so such a row lands at ``|z| ~ 2.4``
    rather than deep in the copula flow's extrapolation region.

    Returns
    -------
    theta_of_z, z_of_theta, log_dtheta_dz : callables (jnp-vectorized)
    x_clamps : tuple of (float or None, float or None)
        Saturation x values of the bounded tails (None on unbounded sides),
        at the float64 wall-offset resolution limit or at the wall offset
        whose z is ``_WALL_Z_MARGIN`` past the outermost knot, whichever is
        further inside. ``theta_of_z``
        returns exactly these floats for z at or beyond the corresponding
        clamp; the sampling layer
        redraws such rows (see ``coppuccino.copula_flows.sample``) so the
        emitted stream matches the reported density everywhere.
    """
    from scipy.special import ndtri as _ndtri

    x = np.asarray(samples, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    num_points = max(20, min(num_points, n // 3))
    if not 0.0 <= tail_alpha <= _TAIL_ALPHA_MAX:
        raise ValueError(
            f"tail_alpha must be in [0, {_TAIL_ALPHA_MAX:g}], got {tail_alpha}. "
            "It is a growth rate per unit z on an unbounded tail, so the map "
            "spans exp(tail_alpha * |z|) decades of theta over the range of z "
            "the base actually visits; at tail_alpha = 1e6 the emitted samples "
            "reached 1.8e298.")
    # a non-finite bound means "unbounded on that side"
    if prior_low is not None and not np.isfinite(prior_low):
        prior_low = None
    if prior_high is not None and not np.isfinite(prior_high):
        prior_high = None

    # The rational-quadratic kernel forms products of two theta-scale terms
    # and then squares them (``b ** 2`` in :func:`_rqs_inverse`), so knot
    # values above ~1e77 overflow float64 and the map silently goes
    # non-finite and non-monotone: standard-normal data scaled by 1e80 gives
    # a NaN round trip, and by 1e120 an all-NaN theta_of_z.
    scale = max(float(np.max(np.abs(x))) if n else 0.0,
                abs(prior_low) if prior_low is not None else 0.0,
                abs(prior_high) if prior_high is not None else 0.0)
    if scale > _MAX_ABS_KNOT:
        raise ValueError(
            f"make_smooth_marginal: largest |sample or prior bound| is "
            f"{scale:.3g}, above the {_MAX_ABS_KNOT:.0e} limit where the "
            "spline kernel's quadratic overflows float64 and the map stops "
            "being finite and monotone. Rescale this parameter (work in "
            "units of its own scale) before fitting.")

    # scale for the strictly-inside boundary clip
    lo_ref = prior_low if prior_low is not None else float(np.min(x))
    hi_ref = prior_high if prior_high is not None else float(np.max(x))
    span = max(hi_ref - lo_ref, _POS_FLOOR)
    # Boundary-atom inset window. Scale it by whichever span is *smaller*. A
    # prior-relative window alone is wrong whenever the prior is much wider
    # than the posterior: under prior_bounds=[[0, 1e12]] on a chain spanning
    # [4e2, 4e3] it reaches to x = 1000 and silently relocates a large
    # fraction of the chain onto one point. What counts as "sitting on the
    # bound" is set by the resolution of the samples, not by how
    # uninformative the prior is. The ULP floor keeps ``bound + eps0``
    # representable when the data span underflows it.
    data_span = max(float(np.max(x)) - float(np.min(x)), _POS_FLOOR)
    eps0 = max(1e-9 * min(span, data_span),
               4.0 * float(np.spacing(max(abs(lo_ref), abs(hi_ref)))))
    if prior_low is not None:
        x = _inset_wall_atoms(x, prior_low, eps0, 0.01 * span)
    if prior_high is not None:
        x = -_inset_wall_atoms(-x, -prior_high, eps0, 0.01 * span)
    xs = np.sort(x)

    # theta knots at *distinct* data quantiles. Tied values (MCMC-rejection
    # duplicates, boundary reflection atoms, ...) must NOT become multiple
    # knots separated by artificial bumps: that makes theta(z) effectively
    # constant across the tie block's whole z-span -- a non-invertible
    # plateau where the sampled point can no longer report its own density.
    # One knot per distinct value, carrying the full Hazen mass of its ties,
    # keeps the map a true diffeomorphism: a data atom becomes a steep but
    # smooth (and exactly invertible) segment over the real gap to its
    # neighbors, and a boundary tie block's mass flows into the analytic
    # tail -- the closest absolutely-continuous statement of an atom.
    thq = np.quantile(xs, np.linspace(0.0, 1.0, num_points))
    # one knot per distinct Hazen count: drops both tied-value knots and
    # interpolated knots that fall between the same adjacent data values
    counts, keep = np.unique(np.searchsorted(xs, thq, side="right"),
                             return_index=True)
    thq = thq[keep]
    if len(thq) < 2:
        raise ValueError(
            "make_smooth_marginal needs at least two distinct sample values.")
    p = np.clip((counts - 0.5) / n, 0.5 / n, 1.0 - 0.5 / n)
    zq = _ndtri(p)

    # bulk: closed-form monotone RQS on (z, theta) knots -- one
    # parameterization supplies forward, inverse and derivative. The edge
    # slope floor keeps the join slopes m1/mK (and hence the tail rates)
    # non-degenerate when PCHIP's endpoint rule returns 0.
    bulk_theta, bulk_z, bulk_deriv = _rqs_bulk(zq, thq, edge_slope_floor=0.05)

    z1, zK = float(zq[0]), float(zq[-1])
    th1, thK = float(thq[0]), float(thq[-1])
    m1 = float(np.maximum(np.asarray(bulk_deriv(jnp.asarray(z1))), _POS_FLOOR))
    mK = float(np.maximum(np.asarray(bulk_deriv(jnp.asarray(zK))), _POS_FLOOR))

    # Analytic tails beyond the outermost knots, each built in its own side's
    # outward coordinates (see :class:`_MarginalTail`), so the right tail is
    # the left one seen through an exact sign flip.
    tail_lo = _make_tail(z1, th1, m1, float(tail_alpha), prior_low, span)
    tail_hi = _make_tail(-zK, -thK, mK, float(tail_alpha),
                         None if prior_high is None else -prior_high, span)

    # x values the tails saturate at (metadata for the sampling-layer
    # rejection): the forward map returns exactly these floats for z at or
    # beyond the clamp, so `x <= x_clamp_lo` / `x >= x_clamp_hi` identifies
    # clamped draws by comparison alone.
    x_clamp_lo = float(tail_lo.y_min) if prior_low is not None else None
    x_clamp_hi = -float(tail_hi.y_min) if prior_high is not None else None
    # z beyond these build-time constants maps onto the tail clamp, so the
    # claimed log-derivative is evaluated at the clamp (a function of the
    # *output point*, matching what z_of_theta reconstructs there)
    z_clamp_lo = (float(_tail_u(jnp.asarray(tail_lo.y_min), tail_lo))
                  if prior_low is not None else -np.inf)
    z_clamp_hi = (-float(_tail_u(jnp.asarray(tail_hi.y_min), tail_hi))
                  if prior_high is not None else np.inf)

    def theta_of_z(z: ArrayLike) -> Array:
        z = jnp.asarray(z)
        bulk = bulk_theta(jnp.clip(z, z1, zK))
        return jnp.where(z < z1, _tail_y(z, tail_lo),
                         jnp.where(z > zK, -_tail_y(-z, tail_hi), bulk))

    def log_dtheta_dz(z: ArrayLike) -> Array:
        z = jnp.clip(jnp.asarray(z), z_clamp_lo, z_clamp_hi)
        bulk = jnp.log(jnp.maximum(bulk_deriv(jnp.clip(z, z1, zK)), _POS_FLOOR))
        return jnp.where(z < z1, _tail_log_deriv(z, tail_lo),
                         jnp.where(z > zK, _tail_log_deriv(-z, tail_hi), bulk))

    def z_of_theta(theta: ArrayLike) -> Array:
        theta = jnp.asarray(theta)
        bulk = bulk_z(jnp.clip(theta, th1, thK))
        return jnp.where(theta < th1, _tail_u(theta, tail_lo),
                         jnp.where(theta > thK, -_tail_u(-theta, tail_hi), bulk))

    return theta_of_z, z_of_theta, log_dtheta_dz, (x_clamp_lo, x_clamp_hi)


class SmoothMarginalToGaussian(AbstractBijection):
    """Clip-free marginal bijection between data space and Gaussian space.

    Wraps the closures from :func:`make_smooth_marginal`. Unlike
    :class:`EmpiricalMarginalToGaussian` there is no probability-integral /
    inverse-normal composition with clipped endpoints: the map ``theta(z)``
    is one analytic monotone object (RQS bulk + analytic tails), so the
    sampling direction and the density direction agree everywhere.
    Consequences:

    * no boundary sample atoms: every finite ``z`` maps strictly inside the
      support;
    * a finite, strictly positive marginal density *at* a prior bound,
      approached smoothly. The bounded tail carries the CDF to exactly 0 at
      the bound with a non-zero slope (see :class:`_MarginalTail`), so the
      wall density it reports is the mean density between the outermost knot
      and the bound. Measured against the analytic truth at the bound and at
      1e-9 / 1e-6 of the span inside it, 8 seeds (``default_rng(0..7)``) x
      both walls x n = 1500: median 1.09 nats and worst 2.03 on ``N(0, 1)``
      truncated to ``[-1, 1]``, median 0.71 / worst 5.50 on ``U(0, 1)`` over
      ``[0, 1]`` (all three worst cases are the lower wall of one seed whose
      smallest order statistic landed far inside its expectation; the worst
      on any other seed is 2.10).
      The residual is the outermost order statistic: the mass the fit assigns
      between it and the bound is ``0.5 / n`` where the truth is
      ``Exp(1) / n``, and that ratio is the whole error. Where the true
      density is 0 or infinite at the bound (``Beta(0.5, 3)``: infinite at 0,
      zero at 1) no finite-density family can follow it -- the reported value
      stays the interval mean, e.g. -5.25 to +1.25 nats against the truth at
      ``1e-9`` from the lower wall over 8 seeds, and far above it arbitrarily
      close to the upper one;
    * a chain with rows sitting exactly *on* a bound (what a clipping or
      reflecting kernel produces) reports a wall density near the interior
      one rather than a spike: on ``clip(N(0.9, 0.25), -1, 1)`` with 32-36%
      of 1500 rows on the upper bound, at-bound minus interior-median
      log-density was -0.37, -0.16, +0.91, +0.64 over four seeds. Those rows
      are inset before the knots are built, so the density does vary steeply
      across the inset window (up to 6.4 nats per decade of wall offset in
      those fits) before settling to its wall value;
    * honest densities for outliers / out-of-data-range queries, with
      Gaussian (``tail_alpha = 0``) or heavier (``tail_alpha > 0``) tails on
      unbounded sides.

    Two resolution caveats. The forward tail saturates at ``x_clamp_lo`` /
    ``x_clamp_hi``, the point whose ``z`` is ``_WALL_Z_MARGIN`` past the
    outermost knot, or the nearest representable offset from the bound,
    whichever is further inside. The first term governs: it is about
    ``Phi(u_edge - margin) / Phi(u_edge)`` of the knot-to-bound distance,
    0.14% of it for an outermost knot at the ``0.5 / n`` plotting position
    with n = 1500; in absolute x that was 5.9e-8 to 2.6e-6 from either bound
    over three seeds of ``U(0, 1)`` on ``[0, 1]``, n = 1500. Queries below
    that report the clamp's density,
    so the reported density is bitwise constant over that last sliver, and
    the sampling entry points (:func:`coppuccino.copula_flows.sample` and
    ``sample_and_log_prob``) redraw any row that lands on a clamp value. The
    emitted stream is then the reported density truncated to the clamp-free
    region and renormalized; that is a constant factor, and it cancels in
    Metropolis-Hastings ratios anyway. Under a standard normal base the
    truncated mass would be ``Phi(u_edge - margin)``, 4.7e-7 per bounded wall
    at n = 1500; the copula's own tail is heavier than that, so the measured
    redraw rate over 20000 draws was 1e-4 to 8e-4 on ``U(0, 1)``, truncated
    ``N(0, 1)`` and ``Beta(0.5, 3)`` fits (3 seeds each, against 0 with the
    uncapped tail). On a chain with wall atoms the float-resolution clamp
    already dominated: ``clip(N(0.5, 0.6), 0, 1)`` redrew 26-30% of raw draws
    before this margin existed and 28-33% with it. The *reported*
    density is not truncated -- the plateau slab carries the clamp's density
    across the last sliver -- so it integrates to 1 only up to the difference
    between that slab and the tail mass it stands in for. That difference is
    second order because the wall shape is pinned to ``S'(0) = 1``: over the
    marginal alone, n = 1500, 4 seeds, the mass came to 1 within 2e-8 on
    ``U(0, 1)``, ``Beta(0.5, 3)`` and ``clip(N(0.9, 0.25), -1, 1)``. It is
    largest where a *lower*-wall tie block pulls the outermost knot in to
    ``z = -0.85`` and the cap with it: on ``clip(N(0.5, 0.6), 0, 1)``, 20% of
    rows on each bound, the mass came to 1 + 3.7e-4 to 1 + 5.6e-4 (against
    1 + 3.5e-3 to 1 + 4.7e-3 at margin 1.0). Second, the deep round trip
    ``z -> theta -> z`` is not exact: it is limited both by ``ULP(bound)``,
    which is all the resolution ``|x - bound|`` has near a bound of large
    magnitude, and by the wall shape's own conditioning where ``beta`` is
    extreme. Measured max ``|dz|`` 2.7e-7 over the whole resolvable ``z``
    range, ``[-2.78, 4.90]``, on a both-walls-spiked ``Beta(0.5, 0.5)`` fit
    with 10% of 1500 rows on each bound; the *density* the two directions
    report at the same point agrees to 1.4e-6 there, which is what the
    sampling contract needs.

    What this class does *not* fix: the copula flow downstream consumes ``z``,
    and a query at a bound hands it a ``z`` one ``_WALL_Z_MARGIN`` past the
    deepest it was trained on, where it extrapolates linearly. That
    extrapolation, not the marginal, dominates the joint density reported
    exactly at a wall. On ``U[0, 1]^2``, 1500 rows, 80 epochs, seeds 0-3, the
    four box mid-edges came out +0.1 to +4.6 nats above the flat truth, where
    the marginals alone put them at -1.9 to +1.7; the four corners, where
    both dimensions sit at their cap at once, ran +0.5 to +11.9. Shrinking the
    margin further trades that against wall resolution and against the
    marginal's own wall-density accuracy above, so bounding it properly
    needs a change to the copula, not to this map.

    Trailing attributes are reconstruction metadata for
    :mod:`coppuccino.model_io`.
    """
    samples: np.ndarray
    theta_of_z: Callable
    z_of_theta: Callable
    log_dtheta_dz_fn: Callable
    num_points: int = 200
    prior_low: Optional[float] = None
    prior_high: Optional[float] = None
    tail_alpha: float = 0.0
    x_clamp_lo: Optional[float] = None
    x_clamp_hi: Optional[float] = None
    marginal: str = "smooth"
    # flowjax declares shape/cond_shape as equinox ``AbstractVar``, which mypy
    # reads as instance variables of the base; a concrete ``ClassVar`` is the
    # intended implementation here (both are per-class constants).
    cond_shape: ClassVar[None] = None  # type: ignore[misc]
    shape: ClassVar[tuple[int, ...]] = ()  # type: ignore[misc]

    def transform_and_log_det(self, z: ArrayLike,
                              condition: Optional[ArrayLike] = None) -> tuple[Array, Array]:
        """Gaussian space -> data space (sampling direction)."""
        x = self.theta_of_z(z)
        return x, jnp.sum(self.log_dtheta_dz_fn(z))

    def inverse_and_log_det(self, x: ArrayLike,
                            condition: Optional[ArrayLike] = None) -> tuple[Array, Array]:
        """Data space -> Gaussian space (training/density direction)."""
        z = self.z_of_theta(x)
        return z, -jnp.sum(self.log_dtheta_dz_fn(z))
