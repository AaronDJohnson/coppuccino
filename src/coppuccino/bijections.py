from typing import Callable, ClassVar, Optional
import jax.numpy as jnp
import numpy as np
from flowjax.bijections import AbstractBijection
from interpax._ppoly import PchipInterpolator
from jax.scipy.special import ndtri, ndtr

__all__ = [
    "make_empirical_cdf_spline",
    "make_smooth_marginal",
    "EmpiricalMarginalToGaussian",
    "EmpiricalMarginalToUniform",
    "SmoothMarginalToGaussian",
]

# Numerical floors, named so the intent is explicit at each use site.
_POS_FLOOR = 1e-30    # keep quantities that sit in denominators strictly positive
_SCALE_FLOOR = 1e-10  # keep fitted scale (sigma) parameters away from zero


def _process_array(arr):
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


def _fit_gpd(excesses):
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


def _gpd_cdf(y, xi, sigma, xi_eps=1e-6):
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


def _gpd_pdf(y, xi, sigma, xi_eps=1e-6):
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


def _gpd_quantile(p, xi, sigma, xi_eps=1e-6):
    """GPD inverse CDF: y(p) for p ∈ [0, 1)."""
    p = jnp.clip(p, 0.0, 1.0 - 1e-15)
    xi_safe = jnp.where(jnp.abs(xi) < xi_eps, 1.0, xi)
    return jnp.where(
        jnp.abs(xi) < xi_eps,
        -sigma * jnp.log1p(-p),
        sigma * (jnp.power(1.0 - p, -xi_safe) - 1.0) / xi_safe,
    )


def _rqs_bulk(xg, cdf_vals, edge_slope_floor=0.0):
    """Build closed-form monotone rational-quadratic spline (RQS) callables.

    Returns ``(cdf, inverse, deriv)`` callables over the bulk knots, matching the
    call interface of the interpax ``PchipInterpolator`` objects they replace in
    :func:`make_empirical_cdf_spline`. Unlike the PCHIP pair (two independently
    constructed splines that are only approximate inverses), the RQS forward,
    inverse and derivative all come from a single parameterization, so the
    transform is self-consistent to machine precision.

    Knot derivatives are taken from the PCHIP slopes, so the RQS reproduces the
    monotone PCHIP shape at the knots while admitting a closed-form inverse
    (a stable quadratic root per bin) and derivative. Formulas follow
    Gregory & Delbourgo (1982) / Durkan et al. (2019).

    Parameters
    ----------
    xg : array_like
        Strictly sorted bulk knot locations (CDF domain).
    cdf_vals : array_like
        Strictly increasing CDF values at ``xg``.
    edge_slope_floor : float, optional
        If positive, floor the two *endpoint* knot derivatives at this
        fraction of their end-bin secant slope. PCHIP's endpoint rule can
        return exactly 0 there (one-sided estimate clamped on sign change),
        which is fatal for callers that extend tails from the boundary slope
        (:func:`make_smooth_marginal`): the C^1-matched tail rate collapses
        to the generic ``_POS_FLOOR`` and the tail map degenerates. The
        default 0.0 preserves the legacy behavior exactly.

    Returns
    -------
    cdf, inverse, deriv : callables
        ``cdf(x)`` maps x -> CDF, ``inverse(u)`` maps CDF -> x, ``deriv(x)``
        is the closed-form CDF derivative (density). All are jnp-vectorized.
    """
    from scipy.interpolate import PchipInterpolator as _ScipyPchip
    xg = np.asarray(xg, dtype=float)
    yg = np.asarray(cdf_vals, dtype=float)
    # RQS needs strictly increasing x knots; drop any ties (cdf_vals is already
    # strictly increasing thanks to the upstream monotonicity bump).
    keep = np.concatenate([[True], np.diff(xg) > 0])
    xk, yk = xg[keep], yg[keep].copy()
    # the tie-drop keeps the FIRST knot of a tied block; the top endpoint's
    # CDF value (exactly 1.0 in unit_range mode) must survive on the kept
    # boundary knot or the spline's range silently tops out below it and the
    # discarded mass reappears as an exact-at-bound sampling atom
    yk[-1] = yg[-1]
    # PCHIP slopes at the knots — monotone by construction, strictly positive
    # floor so they can sit in RQS denominators without dividing by zero.
    dk = np.maximum(_ScipyPchip(xk, yk).derivative()(xk), _POS_FLOOR)
    if edge_slope_floor > 0.0:
        # rescue PCHIP's zeroed endpoint derivatives (see docstring); the RQS
        # stays monotone for any positive knot derivatives
        sec = np.diff(yk) / np.diff(xk)
        dk[0] = max(dk[0], edge_slope_floor * sec[0])
        dk[-1] = max(dk[-1], edge_slope_floor * sec[-1])

    xk_j, yk_j, dk_j = jnp.asarray(xk), jnp.asarray(yk), jnp.asarray(dk)
    w = xk_j[1:] - xk_j[:-1]
    dy = yk_j[1:] - yk_j[:-1]
    s = dy / w  # secant slope per bin
    nbins = len(xk) - 1

    def _bin_x(x):
        return jnp.clip(jnp.searchsorted(xk_j, x, side="right") - 1, 0, nbins - 1)

    def _bin_y(y):
        return jnp.clip(jnp.searchsorted(yk_j, y, side="right") - 1, 0, nbins - 1)

    def cdf(x):
        x = jnp.asarray(x)
        k = _bin_x(x)
        xi = (x - xk_j[k]) / w[k]
        num = dy[k] * (s[k] * xi**2 + dk_j[k] * xi * (1 - xi))
        den = s[k] + (dk_j[k + 1] + dk_j[k] - 2 * s[k]) * xi * (1 - xi)
        return yk_j[k] + num / den

    def deriv(x):
        x = jnp.asarray(x)
        k = _bin_x(x)
        xi = (x - xk_j[k]) / w[k]
        den = s[k] + (dk_j[k + 1] + dk_j[k] - 2 * s[k]) * xi * (1 - xi)
        num = s[k] ** 2 * (dk_j[k + 1] * xi**2 + 2 * s[k] * xi * (1 - xi)
                           + dk_j[k] * (1 - xi) ** 2)
        return num / den**2

    def inverse(u):
        u = jnp.asarray(u)
        k = _bin_y(u)
        alpha = u - yk_j[k]
        coef = dk_j[k + 1] + dk_j[k] - 2 * s[k]
        a = dy[k] * (s[k] - dk_j[k]) + alpha * coef
        b = dy[k] * dk_j[k] - alpha * coef
        c = -s[k] * alpha
        disc = jnp.maximum(b**2 - 4 * a * c, 0.0)
        # Numerically stable root that stays in [0, 1] as a -> 0 (linear bin).
        xi = 2 * c / (-b - jnp.sqrt(disc))
        xi = jnp.clip(xi, 0.0, 1.0)
        return xk_j[k] + xi * w[k]

    return cdf, inverse, deriv


def make_empirical_cdf_spline(
    samples: np.ndarray,
    num_points: int = 200,
    min_eps: float = 1e-7,
    tail_extension: bool = False,
    prior_low: Optional[float] = None,
    prior_high: Optional[float] = None,
    tail_model: str = "gaussian",
    tail_quantile: float = 0.05,
    marginal: str = "pchip",
    unit_range: bool = False,
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
    tail_quantile : float, optional
        Fraction of samples in each GPD tail (only used when
        ``tail_model="gpd"``). Default 0.05 (5%).
    marginal : str, optional
        Bulk interpolant family: ``"pchip"`` (default here) builds two
        independent PCHIP splines for the CDF and its inverse; ``"rqs"`` builds a
        single monotone rational-quadratic spline whose forward, inverse and
        derivative are all closed-form and mutually exact (self-consistent to
        machine precision). Both use the same empirical-quantile knots and the
        same tail / prior-bound handling, so they produce nearly identical
        transforms; ``"rqs"`` additionally guarantees an exact, cheaper inverse,
        whereas the two PCHIP splines are only approximate inverses of each
        other. Note the
        ``normalizing_flows_fit`` entry point resolves its default per
        pipeline (``"smooth"`` for gaussian, ``"rqs"`` for uniform); this
        low-level function keeps ``"pchip"`` as its default for backward
        compatibility.
    unit_range : bool, optional
        If True, build the marginal as an *exact bijection onto* ``[0, 1]``:
        the grid endpoints (the prior bounds when given, else the data bounds)
        map to CDF values of exactly 0 and 1, with a strictly positive density
        all the way to the boundary (no ``min_eps`` clipping and no zeroed wall
        density). This is required by the uniform-copula mode
        (``copula_space="uniform"``), where no inverse-normal transform follows
        and so CDF values of exactly 0/1 are valid. It models *bounded*
        support: requires ``marginal="rqs"`` (self-consistent inverse) and is
        incompatible with ``tail_extension`` and ``tail_model="gpd"``.
        Default is False (original behavior, for the Gaussianized pipeline).

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

    if unit_range:
        # Bounded-support mode: the marginal is an exact [x_lo, x_hi] <-> [0, 1]
        # bijection. Tail models (which extend beyond the bounds) do not apply,
        # and the PCHIP pair's approximate inverse would break the sample/density
        # self-consistency this mode exists to guarantee.
        if marginal == "pchip":
            # (an unknown family falls through to the canonical validation at
            # the bulk-interpolant dispatch below)
            raise ValueError(
                "unit_range=True requires marginal='rqs' (single self-consistent "
                "spline); got marginal='pchip'."
            )
        if tail_extension or use_gpd:
            raise ValueError(
                "unit_range=True models bounded support and is incompatible with "
                "tail_extension=True or tail_model='gpd'."
            )

    # Tail-model parameters below are each bound only inside their guarding
    # branch (GPD or Gaussian tail_extension). Pre-declare them so the
    # cdf/quantile/pdf closures — which reference them under the same guards —
    # can never raise NameError if a future edit shifts a branch boundary.
    F_lo = F_hi = u_lo = u_hi = None
    xi_lo = sigma_lo = xi_hi = sigma_hi = None
    mu = sigma = lo_scale = hi_scale = None

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
        if unit_range:
            # Plotting-position (Hazen) convention: symmetric treatment of the
            # two ends. Plain counts/n hits exactly 1.0 at the top data point,
            # which the min_eps clip then crushes onto the boundary knot --
            # leaving the final segment with ~1e-7 of mass and a near-zero
            # density (a wall cliff). (counts - 1/2)/n keeps every data knot
            # strictly inside (0, 1) with ~1/(2n) of mass beyond the extreme
            # data point on each side.
            cdf_vals = (counts - 0.5) / len(sorted_chain)
        else:
            cdf_vals = counts / len(sorted_chain)

        # Data bounds (may be extended by prior bounds)
        data_min, data_max = np.min(samples), np.max(samples)

    # Extend grid to prior bounds if provided. Skipped under the GPD tail model,
    # whose thresholds live inside the data range and supply the out-of-bulk
    # tails directly (prior bounds would only inject spurious 0/1-clipped knots).
    if not use_gpd:
        if prior_low is not None and prior_low < xg[0]:
            xg = np.concatenate([[prior_low], xg])
            cdf_vals = np.concatenate([[0.0], cdf_vals])
            data_min = prior_low
        if prior_high is not None and prior_high > xg[-1]:
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
    if unit_range:
        # Exact bijection onto [0, 1]: the grid endpoints (prior bounds if given,
        # else data bounds) carry CDF values of exactly 0 and 1. The interior
        # knots remain in [min_eps, 1 - min_eps] from the clip above, so strict
        # monotonicity is preserved. Without a following inverse-normal
        # transform, exact 0/1 endpoint values are well-defined -- this is what
        # removes the boundary atom (no quantile clamping) and the wall density
        # cliff (no zeroed/floored endpoint pdf).
        cdf_vals[0], cdf_vals[-1] = 0.0, 1.0
    # Store actual CDF bounds after clipping (may differ from min_eps due to empirical CDF)
    cdf_min, cdf_max = float(np.min(cdf_vals)), float(np.max(cdf_vals))

    # Bulk interpolant. Both families share the empirical-quantile knots above
    # and the tail / prior-bound closures below; they differ only in how the
    # bulk CDF is interpolated, inverted and differentiated.
    if marginal == "rqs":
        # Single closed-form rational-quadratic spline: forward, inverse and
        # derivative come from one parameterization (machine-precision inverse).
        bulk_cdf, bulk_inverse, bulk_deriv = _rqs_bulk(xg, cdf_vals)
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

        # CDF values of the Gaussian at the data boundaries
        from scipy.stats import norm as _norm
        gauss_cdf_lo = float(_norm.cdf(data_min, loc=mu, scale=sigma))
        gauss_cdf_hi = float(_norm.cdf(data_max, loc=mu, scale=sigma))

        # Empirical CDF at boundaries (from spline)
        emp_cdf_lo = float(cdf_min)
        emp_cdf_hi = float(cdf_max)

        # Scale factors so the Gaussian tail joins the empirical CDF at
        # the boundary.  We want:
        #   tail_cdf(data_min) = emp_cdf_lo
        #   tail_cdf(data_max) = emp_cdf_hi
        # Using:  tail_cdf(x) = emp_cdf_lo * Phi_gauss(x) / Phi_gauss(data_min)   [left]
        #         tail_cdf(x) = 1 - (1 - emp_cdf_hi) * Phi_gauss_sf(x) / Phi_gauss_sf(data_max)  [right]
        lo_scale = emp_cdf_lo / max(gauss_cdf_lo, _POS_FLOOR)
        hi_scale = (1.0 - emp_cdf_hi) / max(1.0 - gauss_cdf_hi, _POS_FLOOR)
    # ------------------------------------------------------------------

    def cdf_fn(u):
        u = jnp.asarray(u)
        u_clipped = jnp.clip(u, data_min, data_max)
        spline_val = bulk_cdf(u_clipped)
        if use_gpd:
            left_tail = F_lo * (1.0 - _gpd_cdf(u_lo - u, xi_lo, sigma_lo))
            right_tail = F_hi + (1.0 - F_hi) * _gpd_cdf(u - u_hi, xi_hi, sigma_hi)
            y = jnp.where(
                u <= u_lo,
                left_tail,
                jnp.where(u >= u_hi, right_tail, spline_val),
            )
        elif tail_extension:
            gauss_cdf = ndtr((u - mu) / sigma)
            left_tail = lo_scale * gauss_cdf
            right_tail = 1.0 - hi_scale * (1.0 - gauss_cdf)
            y = jnp.where(
                u <= data_min,
                left_tail,
                jnp.where(u >= data_max, right_tail, spline_val),
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
                u <= data_min,
                cdf_min,
                jnp.where(u >= data_max, cdf_max, spline_val),
            )
        if unit_range:
            # Exact-0/1 endpoints are valid here (no inverse-normal follows).
            return jnp.clip(y, 0.0, 1.0)
        # Final bounds check to ensure we never return exactly 0 or 1
        return jnp.clip(y, min_eps, 1.0 - min_eps)

    def quantile_fn(u):
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
            # For values below or above the empirical CDF range, invert
            # the Gaussian tail model.
            # Left tail:  u = lo_scale * Phi((x - mu) / sigma)
            #   => x = mu + sigma * Phi^{-1}(u / lo_scale)
            left_x = mu + sigma * ndtri(jnp.clip(u / lo_scale, min_eps, 1.0 - min_eps))
            # Right tail: u = 1 - hi_scale * (1 - Phi((x - mu) / sigma))
            #   => Phi((x - mu) / sigma) = 1 - (1 - u) / hi_scale
            right_x = mu + sigma * ndtri(jnp.clip(1.0 - (1.0 - u) / hi_scale, min_eps, 1.0 - min_eps))

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
            y = jnp.where(u < cdf_min, data_min, jnp.where(u > cdf_max, data_max, y))
        return y

    _inv_sqrt_2pi = 1.0 / jnp.sqrt(2.0 * jnp.pi)

    def pdf_fn(x):
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
            left_pdf = lo_scale * tail_normal_pdf
            right_pdf = hi_scale * tail_normal_pdf
            y = jnp.where(
                x <= data_min,
                left_pdf,
                jnp.where(x >= data_max, right_pdf, mid_pdf),
            )
        elif unit_range:
            # Honest wall density: the endpoint value of the (strictly positive)
            # RQS derivative, extended flat for out-of-support queries. Zeroing
            # or flooring the wall here is exactly what created the boundary
            # density cliff in the Gaussianized pipeline.
            y = mid_pdf
        else:
            y = jnp.where(
                (x <= data_min) | (x >= data_max),
                0.0,
                mid_pdf,
            )
        if unit_range:
            return jnp.maximum(y, _POS_FLOOR)
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
    marginal : str, default="pchip"
        Which bulk interpolant family built ``cdf_fn``/``quantile_fn``/``pdf_fn``
        (``"pchip"`` or ``"rqs"``).

    Examples
    --------
    >>> import numpy as np
    >>> import jax.numpy as jnp
    >>> from coppuccino.bijections import EmpiricalMarginalToGaussian, make_empirical_cdf_spline
    >>> # Generate samples from a non-Gaussian distribution
    >>> samples = np.random.exponential(2.0, 1000)
    >>> cdf_vals, cdf_fn, quantile_fn, pdf_fn = make_empirical_cdf_spline(samples)
    >>> # Create bijection
    >>> bij = EmpiricalMarginalToGaussian(
    ...     samples=samples, cdf_fn=cdf_fn, quantile_fn=quantile_fn, pdf_fn=pdf_fn
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
    marginal: str = "pchip"
    cond_shape: ClassVar[None] = None
    shape: ClassVar[tuple[int, ...]] = ()

    def inverse_and_log_det(self, x, condition=None):
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

    def transform_and_log_det(self, z, condition=None):
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


class EmpiricalMarginalToUniform(AbstractBijection):
    """
    Bijection between an empirical marginal and the uniform distribution on [0, 1].

    This is the probability integral transform (PIT) on its own: ``u = F(x)``
    with no inverse-normal step. It is the marginal stage of the uniform-copula
    pipeline (``normalizing_flows_fit(..., copula_space="uniform")``), where the
    copula is modeled directly on the unit cube by a bounded spline flow over a
    uniform base.

    Built with ``make_empirical_cdf_spline(..., unit_range=True)`` so the
    transform is an *exact* bijection ``[x_lo, x_hi] <-> [0, 1]``: the support
    bounds map to u = 0 and u = 1 exactly, with a strictly positive density all
    the way to the walls. Compared to :class:`EmpiricalMarginalToGaussian` this
    removes the boundary pathologies that the inverse-normal transform forces
    (quantile clamping producing an atom of samples exactly at the bound, and a
    zeroed/floored wall density producing a log-density cliff there), so the
    flow's sampling and reported density agree at the boundaries.

    Attributes mirror :class:`EmpiricalMarginalToGaussian`; the trailing ones are
    reconstruction metadata used by :mod:`coppuccino.model_io`.

    Examples
    --------
    >>> import numpy as np
    >>> import jax.numpy as jnp
    >>> from coppuccino.bijections import EmpiricalMarginalToUniform, make_empirical_cdf_spline
    >>> samples = np.random.uniform(-1.0, 1.0, 1000)
    >>> _, cdf_fn, quantile_fn, pdf_fn = make_empirical_cdf_spline(
    ...     samples, prior_low=-1.0, prior_high=1.0, marginal="rqs", unit_range=True)
    >>> bij = EmpiricalMarginalToUniform(
    ...     samples=samples, cdf_fn=cdf_fn, quantile_fn=quantile_fn, pdf_fn=pdf_fn
    ... )
    >>> u, log_det = bij.inverse_and_log_det(jnp.array(0.3))
    >>> x, log_det_fwd = bij.transform_and_log_det(u)
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
    cond_shape: ClassVar[None] = None
    shape: ClassVar[tuple[int, ...]] = ()

    def inverse_and_log_det(self, x, condition=None):
        """Original space -> unit interval: ``u = F(x)``, ``log|du/dx| = log f(x)``."""
        u = self.cdf_fn(x)
        log_det = jnp.log(self.pdf_fn(x))
        return u, jnp.sum(log_det)

    def transform_and_log_det(self, u, condition=None):
        """Unit interval -> original space: ``x = F^{-1}(u)``, ``log|dx/du| = -log f(x)``."""
        x = self.quantile_fn(u)
        log_det = -jnp.log(self.pdf_fn(x))
        return x, jnp.sum(log_det)


# ----------------------------------------------------------------------
# Smooth-tail marginals: a true diffeomorphism z in R  <->  theta in (lo, hi)
# ----------------------------------------------------------------------

def make_smooth_marginal(samples: np.ndarray,
                         num_points: int = 200,
                         prior_low: Optional[float] = None,
                         prior_high: Optional[float] = None,
                         tail_alpha: float = 0.0):
    """Build a clip-free marginal map ``theta(z)``: all of R onto the support.

    The map is a single monotone C^1 object evaluated consistently in both
    directions (forward, inverse and log-derivative share one
    parameterization), so the flow's sample stream and its reported density
    agree everywhere -- including arbitrarily deep in the tails. This removes
    the legacy Gaussianized pipeline's boundary pathologies at their root:
    there is no CDF clip, no quantile clamp (hence no boundary sample atom)
    and no floored wall density (hence no density cliff).

    Construction:

    * **Bulk**: knots at the data quantiles ``theta_j`` paired with
      ``z_j = ndtri(p_j)`` at Hazen plotting positions -- all finite -- joined
      by the closed-form monotone rational-quadratic spline (PCHIP slopes).
    * **Tails** beyond the outermost knots:
        - *bounded side* (prior bound given): exponential approach to the
          wall, ``theta = lo + (theta_1 - lo) exp(c (z - z_1))`` with the
          slope matched at the join (C^1); the wall is reached only as
          ``z -> -inf``, so every finite flow output lands strictly inside
          the prior.
        - *unbounded side*: ``dtheta/dz = m exp(alpha |z - z_edge|)`` --
          ``tail_alpha = 0`` gives a linear map (Gaussian tails in theta);
          ``tail_alpha > 0`` gives progressively heavier tails. Density
          queries and samples beyond the observed data range are therefore
          honest and smooth (outlier handling), replacing the legacy
          ``tail_extension``/GPD branches.

    Returns
    -------
    theta_of_z, z_of_theta, log_dtheta_dz : callables (jnp-vectorized)
    x_clamps : tuple of (float or None, float or None)
        Saturation x values of the bounded tails at the float64 resolution
        limit (None on unbounded sides). ``theta_of_z`` returns exactly these
        floats for z at/beyond the corresponding clamp; the sampling layer
        redraws such rows (see ``coppuccino.copula_flows.sample``) so the
        emitted stream matches the reported density everywhere.
    """
    from scipy.interpolate import PchipInterpolator as _Pchip  # noqa: F401
    from scipy.special import ndtri as _ndtri

    x = np.asarray(samples, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    num_points = max(20, min(num_points, n // 3))
    if tail_alpha < 0:
        raise ValueError(f"tail_alpha must be >= 0, got {tail_alpha}.")
    # a non-finite bound means "unbounded on that side"
    if prior_low is not None and not np.isfinite(prior_low):
        prior_low = None
    if prior_high is not None and not np.isfinite(prior_high):
        prior_high = None

    # scale for the strictly-inside boundary clip
    lo_ref = prior_low if prior_low is not None else float(np.min(x))
    hi_ref = prior_high if prior_high is not None else float(np.max(x))
    span = max(hi_ref - lo_ref, _POS_FLOOR)
    # Boundary-atom-aware inset. Samples sitting exactly at (or within the
    # clip margin of) a bound -- e.g. reflection/in-bounds artifacts of a
    # bounded MCMC run -- are placed at half the gap to the first distinct
    # interior value rather than at the raw clip margin. The atom's mass
    # then becomes a *resolvable* smooth shoulder: the first knot offset is
    # comparable to the first bin width, so the tail rate c = m/offset stays
    # O(1/dz) instead of exploding and clamping away the whole tail region.
    eps0 = 1e-9 * span
    if prior_low is not None:
        at_lo = x <= prior_low + eps0
        if at_lo.any() and (~at_lo).any():
            gap = float(np.min(x[~at_lo])) - prior_low
            x[at_lo] = prior_low + float(np.clip(0.5 * gap, eps0, 0.01 * span))
        else:
            x = np.maximum(x, prior_low + eps0)
    if prior_high is not None:
        at_hi = x >= prior_high - eps0
        if at_hi.any() and (~at_hi).any():
            gap = prior_high - float(np.max(x[~at_hi]))
            x[at_hi] = prior_high - float(np.clip(0.5 * gap, eps0, 0.01 * span))
        else:
            x = np.minimum(x, prior_high - eps0)
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

    # tail rate constants (build-time Python branching: these are static)
    cL = m1 / (th1 - prior_low) if prior_low is not None else None
    cR = mK / (prior_high - thK) if prior_high is not None else None
    aL = aR = float(tail_alpha)
    _EXP_MAX = 700.0  # exp overflow guard for values that get where()-selected away
    # Smallest wall offset the tails may produce/resolve, per side. Below
    # ~ULP(|bound|) the float addition ``bound + tiny`` absorbs to the bound
    # exactly -- an exact-wall sample atom. Clamping the forward tail there
    # (with the inverse's ratio floor and the log-derivative clip sharing the
    # same constant, so all three directions agree at the clamp point)
    # relocates that unrepresentable mass to a strictly-interior point whose
    # reported density is high, so walkers there escape rather than absorb.
    # rf_log_* is the log of the clamped tail ratio (theta offset / knot
    # offset); the -_EXP_MAX cap doubles as the exp overflow guard.
    def _rf_log(bound, knot_offset):
        # keep eps representable (>= 4 ULP of the bound) unless that would
        # eat more than 1% of the support; in that pathological remainder
        # (span < ~400 ULP of the bound) strictly-interior saturation is
        # unrepresentable and the sampling-layer redraw catches the rows
        eps = float(np.clip(4.0 * np.spacing(abs(bound)) + 1e-30 * span,
                            1e-30 * span, 0.01 * span))
        return max(float(np.log(eps / knot_offset)), -_EXP_MAX)

    rf_log_L = _rf_log(prior_low, th1 - prior_low) if cL is not None else None
    rf_log_R = _rf_log(prior_high, prior_high - thK) if cR is not None else None

    def theta_of_z(z):
        z = jnp.asarray(z)
        bulk = bulk_theta(jnp.clip(z, z1, zK))
        dzl = jnp.minimum(z - z1, 0.0)
        if cL is not None:
            # exponent floor at rf_log_L: even astronomically negative z
            # lands strictly inside the support (no exact-wall atom, ever)
            left = prior_low + (th1 - prior_low) * jnp.exp(
                jnp.maximum(cL * dzl, rf_log_L))
        elif aL > 0:
            left = th1 + (m1 / aL) * (1.0 - jnp.exp(jnp.minimum(-aL * dzl, _EXP_MAX)))
        else:
            left = th1 + m1 * dzl
        dzr = jnp.maximum(z - zK, 0.0)
        if cR is not None:
            right = prior_high - (prior_high - thK) * jnp.exp(
                jnp.maximum(-cR * dzr, rf_log_R))
        elif aR > 0:
            right = thK + (mK / aR) * (jnp.exp(jnp.minimum(aR * dzr, _EXP_MAX)) - 1.0)
        else:
            right = thK + mK * dzr
        return jnp.where(z < z1, left, jnp.where(z > zK, right, bulk))

    # z beyond these build-time constants maps onto the tail clamp, so the
    # claimed log-derivative is evaluated at the clamp (a function of the
    # *output point*, matching what z_of_theta reconstructs there)
    z_clamp_lo = z1 + rf_log_L / cL if cL is not None else -np.inf
    z_clamp_hi = zK - rf_log_R / cR if cR is not None else np.inf

    def log_dtheta_dz(z):
        z = jnp.clip(jnp.asarray(z), z_clamp_lo, z_clamp_hi)
        bulk = jnp.log(jnp.maximum(bulk_deriv(jnp.clip(z, z1, zK)), _POS_FLOOR))
        dzl = jnp.minimum(z - z1, 0.0)
        left = (jnp.log(m1) + cL * dzl) if cL is not None else (jnp.log(m1) - aL * dzl)
        dzr = jnp.maximum(z - zK, 0.0)
        right = (jnp.log(mK) - cR * dzr) if cR is not None else (jnp.log(mK) + aR * dzr)
        return jnp.where(z < z1, left, jnp.where(z > zK, right, bulk))

    def z_of_theta(theta):
        theta = jnp.asarray(theta)
        bulk = bulk_z(jnp.clip(theta, th1, thK))
        if cL is not None:
            # ratio floor matches the forward clamp: inverse(theta_of_z(z))
            # is exact down to the clamp, and wall-or-deeper queries report
            # the clamp point's (high) density rather than exploding
            ratio = jnp.maximum((theta - prior_low) / (th1 - prior_low),
                                np.exp(rf_log_L))
            left = z1 + jnp.log(ratio) / cL
        elif aL > 0:
            arg = jnp.maximum(1.0 + aL * (th1 - theta) / m1, 1e-300)
            left = z1 - jnp.log(arg) / aL
        else:
            left = z1 + (theta - th1) / m1
        if cR is not None:
            ratio = jnp.maximum((prior_high - theta) / (prior_high - thK),
                                np.exp(rf_log_R))
            right = zK - jnp.log(ratio) / cR
        elif aR > 0:
            arg = jnp.maximum(1.0 + aR * (theta - thK) / mK, 1e-300)
            right = zK + jnp.log(arg) / aR
        else:
            right = zK + (theta - thK) / mK
        return jnp.where(theta < th1, left, jnp.where(theta > thK, right, bulk))

    # x values the clamped tails saturate at (metadata for the sampling-layer
    # rejection): the forward map returns exactly these floats for z at or
    # beyond the clamp, so `x <= x_clamp_lo` / `x >= x_clamp_hi` identifies
    # clamped draws by comparison alone.
    x_clamp_lo = (float(prior_low + (th1 - prior_low) * np.exp(rf_log_L))
                  if cL is not None else None)
    x_clamp_hi = (float(prior_high - (prior_high - thK) * np.exp(rf_log_R))
                  if cR is not None else None)

    return theta_of_z, z_of_theta, log_dtheta_dz, (x_clamp_lo, x_clamp_hi)


class SmoothMarginalToGaussian(AbstractBijection):
    """Clip-free marginal bijection between data space and Gaussian space.

    Wraps the closures from :func:`make_smooth_marginal`. Unlike
    :class:`EmpiricalMarginalToGaussian` there is no probability-integral /
    inverse-normal composition with clipped endpoints: the map ``theta(z)``
    is one analytic monotone object (RQS bulk + analytic tails), so the
    sampling direction and the density direction agree to machine precision
    everywhere. Consequences:

    * no boundary sample atoms (every finite ``z`` maps strictly inside the
      support) and no wall density cliffs -- the density decays *smoothly*
      toward a prior bound at exactly the rate the sampler rarifies;
    * honest densities for outliers / out-of-data-range queries, with
      Gaussian (``tail_alpha = 0``) or heavier (``tail_alpha > 0``) tails on
      unbounded sides.

    One caveat at the float64 resolution limit: wall offsets below a few ULP
    of a nonzero bound are not representable, so the forward tail saturates
    at ``x_clamp_lo`` / ``x_clamp_hi``. The trained copula flow's linear
    extrapolation tails can reach such z with non-negligible probability, so
    the sampling entry points (:func:`coppuccino.copula_flows.sample` and
    ``sample_and_log_prob``) redraw any row that lands on a clamp value.
    The emitted stream is then the reported density truncated to the
    clamp-free region and renormalized -- a constant factor that cancels in
    Metropolis-Hastings ratios, leaving no boundary lumps and no absorbing
    states. MCMC code should evaluate proposal densities via ``log_prob``
    for both old and new points -- Eryn's ``FlowMove`` does.

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
    cond_shape: ClassVar[None] = None
    shape: ClassVar[tuple[int, ...]] = ()

    def transform_and_log_det(self, z, condition=None):
        """Gaussian space -> data space (sampling direction)."""
        x = self.theta_of_z(z)
        return x, jnp.sum(self.log_dtheta_dz_fn(z))

    def inverse_and_log_det(self, x, condition=None):
        """Data space -> Gaussian space (training/density direction)."""
        z = self.z_of_theta(x)
        return z, -jnp.sum(self.log_dtheta_dz_fn(z))
