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


def make_empirical_cdf_spline(samples, num_points=200, min_eps=1e-7, tail_extension=False,
                              prior_low=None, prior_high=None):
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

    # Create monotonic spline for CDF
    cdf_spline = PchipInterpolator(xg, cdf_vals, extrapolate=False, check=False)
    inverse_cdf_spline = PchipInterpolator(cdf_vals, xg, extrapolate=False, check=False)

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
        if tail_extension:
            # Gaussian tail for left extrapolation
            gauss_cdf = ndtr((u - mu) / sigma)
            left_tail = lo_scale * gauss_cdf
            # Gaussian tail for right extrapolation
            right_tail = 1.0 - hi_scale * (1.0 - gauss_cdf)

            y = jnp.where(
                u <= data_min,
                left_tail,
                jnp.where(
                    u >= data_max,
                    right_tail,
                    cdf_spline(u)
                )
            )
        else:
            y = jnp.where(
                u <= data_min,
                min_eps,
                jnp.where(
                    u >= data_max,
                    1.0 - min_eps,
                    cdf_spline(u)
                )
            )
        # Final bounds check to ensure we never return exactly 0 or 1
        y = jnp.clip(y, min_eps, 1.0 - min_eps)
        return y

    def quantile_fn(u):
        u = jnp.asarray(u)
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
            mid_x = inverse_cdf_spline(u_clipped)

            y = jnp.where(
                u < cdf_min,
                left_x,
                jnp.where(u > cdf_max, right_x, mid_x)
            )
        else:
            u_clipped = jnp.clip(u, cdf_min, cdf_max)
            y = inverse_cdf_spline(u_clipped)
            y = jnp.where(u < cdf_min, data_min, jnp.where(u > cdf_max, data_max, y))
        return y

    _cdf_grad_fn = jax.grad(cdf_fn)

    def pdf_fn(x):
        x = jnp.asarray(x)
        y = _cdf_grad_fn(x)
        y = jnp.maximum(y, min_eps)
        return y

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
