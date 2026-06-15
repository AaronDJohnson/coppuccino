from typing import ClassVar, Callable, Optional
import jax.numpy as jnp
import numpy as np
from flowjax.bijections import AbstractBijection
from interpax._ppoly import PchipInterpolator
from jax.scipy.special import ndtri, ndtr
import jax

__all__ = ["make_empirical_cdf_spline", "EmpiricalMarginalToGaussian"]


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
        return 0.0, max(sigma, 1e-10)
    try:
        xi, _, sigma = genpareto.fit(excesses, floc=0)
    except Exception:
        # Method-of-moments fallback (Hosking & Wallis 1987).
        m = float(np.mean(excesses))
        v = float(np.var(excesses, ddof=1))
        if v <= 0:
            return 0.0, max(m, 1e-10)
        xi = 0.5 * (1.0 - m * m / v)
        sigma = 0.5 * m * (1.0 + m * m / v)
    return float(xi), float(max(sigma, 1e-10))


def _gpd_cdf(y, xi, sigma, xi_eps=1e-6):
    """GPD CDF on excesses y ≥ 0. Handles ξ≈0 (exponential) limit."""
    y = jnp.maximum(y, 0.0)
    z = y / sigma
    # For ξ<0 the support is bounded at y = -σ/ξ; clip to keep argument > 0.
    safe_arg = jnp.maximum(1.0 + xi * z, 1e-30)
    xi_safe = jnp.where(jnp.abs(xi) < xi_eps, 1.0, xi)  # avoid /0 in unused branch
    return jnp.where(
        jnp.abs(xi) < xi_eps,
        1.0 - jnp.exp(-z),
        1.0 - jnp.power(safe_arg, -1.0 / xi_safe),
    )


def _gpd_pdf(y, xi, sigma, xi_eps=1e-6):
    """GPD PDF on excesses y ≥ 0."""
    y = jnp.maximum(y, 0.0)
    z = y / sigma
    safe_arg = jnp.maximum(1.0 + xi * z, 1e-30)
    xi_safe = jnp.where(jnp.abs(xi) < xi_eps, 1.0, xi)
    return jnp.where(
        jnp.abs(xi) < xi_eps,
        jnp.exp(-z) / sigma,
        jnp.power(safe_arg, -1.0 / xi_safe - 1.0) / sigma,
    )


def _gpd_quantile(p, xi, sigma, xi_eps=1e-6):
    """GPD inverse CDF: y(p) for p ∈ [0, 1)."""
    p = jnp.clip(p, 0.0, 1.0 - 1e-15)
    xi_safe = jnp.where(jnp.abs(xi) < xi_eps, 1.0, xi)
    return jnp.where(
        jnp.abs(xi) < xi_eps,
        -sigma * jnp.log1p(-p),
        sigma * (jnp.power(1.0 - p, -xi_safe) - 1.0) / xi_safe,
    )


def _rqs_bulk(xg, cdf_vals):
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
    xk, yk = xg[keep], yg[keep]
    # PCHIP slopes at the knots — monotone by construction, strictly positive
    # floor so they can sit in RQS denominators without dividing by zero.
    dk = np.maximum(_ScipyPchip(xk, yk).derivative()(xk), 1e-30)

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


def make_empirical_cdf_spline(samples, num_points=200, min_eps=1e-7, tail_extension=False,
                              prior_low=None, prior_high=None,
                              tail_model="gaussian", tail_quantile=0.05,
                              marginal="pchip"):
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
        same tail / prior-bound handling, so they are statistically equivalent;
        ``"rqs"`` differs only by having an exact, cheaper inverse. Note the
        ``normalizing_flows_fit`` entry point defaults to ``"rqs"``; this
        low-level function keeps ``"pchip"`` as its default for backward
        compatibility.

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
    >>> # Evaluate CDF at a point
    >>> import jax.numpy as jnp
    >>> cdf_fn(jnp.array(0.0))  # Should be approximately 0.5 for standard normal
    >>> # Get median via quantile function
    >>> quantile_fn(jnp.array(0.5))  # Should be approximately 0.0
    >>> # With prior bounds (recommended for MCMC chains)
    >>> cdf_vals, cdf_fn, quantile_fn, pdf_fn = make_empirical_cdf_spline(
    ...     samples, prior_low=-5.0, prior_high=5.0)
    """
    use_gpd = (tail_model == "gpd")

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
        cdf_vals = counts / len(sorted_chain)

        # Data bounds (may be extended by prior bounds)
        data_min, data_max = np.min(samples), np.max(samples)

    # Extend grid to prior bounds if provided
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
        sigma = max(sigma, 1e-10)  # guard against zero std

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
        lo_scale = emp_cdf_lo / max(gauss_cdf_lo, 1e-30)
        hi_scale = (1.0 - emp_cdf_hi) / max(1.0 - gauss_cdf_hi, 1e-30)
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
            y = jnp.where(
                u <= data_min,
                min_eps,
                jnp.where(u >= data_max, 1.0 - min_eps, spline_val),
            )
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
          * no tail extension: PDF is zero outside data range; floored
            to ``min_eps`` so log(pdf) stays finite.

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
        else:
            y = jnp.where(
                (x <= data_min) | (x >= data_max),
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
    marginal : str, default="pchip"
        Which bulk interpolant family built ``cdf_fn``/``quantile_fn``/``pdf_fn``
        (``"pchip"`` or ``"rqs"``). Stored as metadata so the transform can be
        reconstructed identically by :func:`coppuccino.model_io.load_flow`.

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
        >>> z, log_det = bij.inverse_and_log_det(jnp.array([1.0, 2.0, 3.0]))
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
        >>> z = jnp.array([0.0, 1.0, -1.0])
        >>> x, log_det = bij.transform_and_log_det(z)
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
