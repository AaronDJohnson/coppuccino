from typing import ClassVar, Callable
import jax.numpy as jnp
import numpy as np
from flowjax.bijections import AbstractBijection
from interpax._ppoly import PchipInterpolator
from jax.scipy.special import ndtri, ndtr
from jax.scipy.stats import norm as jax_norm
import jax

# Precision-dependent constants, computed once at import time from the current
# JAX default dtype (float64 if jax_enable_x64, float32 otherwise).
_DTYPE = jnp.result_type(0.0)

# Maximum |z| before Gaussian CDF rounds to exactly 0 or 1.
# Float64: ~8.13, Float32: ~5.17
_Z_MAX = float(ndtri(1.0 - jnp.finfo(_DTYPE).eps))

# Smallest positive normal float. Used as a floor for PDF values to prevent
# log(0) = -inf at isolated points where the empirical CDF is flat.
# Float64: ~2.2e-308, Float32: ~1.2e-38
_PDF_FLOOR = float(jnp.finfo(_DTYPE).tiny)

def process_array(arr):
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
    >>> from coppuccino.bijections import process_array
    >>> arr = np.array([1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0])
    >>> processed = process_array(arr)
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


def make_empirical_cdf_spline(samples, num_points=200, min_eps=1e-7, tail_percentile=0.02):
    """
    Create empirical CDF, inverse CDF (quantile), and PDF functions from samples.

    This function constructs smooth monotonic spline interpolators for the empirical
    CDF and its inverse based on input samples. The PDF is computed via automatic
    differentiation of the CDF spline. It uses Gaussian tail extrapolation for values
    outside the training data range, with C^1 continuity at the splice points. This
    ensures proper normalization for Bayesian model comparison.

    Parameters
    ----------
    samples : array_like
        Training data samples from which to construct the empirical distribution.
    num_points : int, optional
        Number of grid points to use for spline construction. Default is 200.
    min_eps : float, optional
        Minimum epsilon to avoid CDF values exactly 0 or 1. Default is 1e-7.
    tail_percentile : float, optional
        Percentile from edge to use as splice point for Gaussian tails.
        Default is 0.02 (2nd and 98th percentiles).
        Smaller values = more of the distribution handled by the spline.
        Larger values = Gaussian tails extend further into the data.

    Returns
    -------
    cdf_vals : numpy.ndarray
        Empirical CDF values at the grid points.
    cdf_fn : callable
        Function mapping x → CDF(x), with Gaussian tail extrapolation.
    quantile_fn : callable
        Function mapping u ∈ [0,1] → x (inverse CDF).
    pdf_fn : callable
        Function mapping x → PDF(x) via automatic differentiation.

    Notes
    -----
    The Gaussian tail parameters are determined by C^1 matching at the splice
    points. This means:
    1. The CDF is continuous everywhere
    2. The PDF (derivative of CDF) is continuous at splice points
    3. The PDF integrates to 1 over (-infinity, +infinity)

    For nested sampling / Bayes factor calculations, this ensures that models
    whose spectra extend beyond the training data bounds are not artificially
    penalized or favored due to normalization issues.

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
    """
    x_grid = np.quantile(samples, np.linspace(0, 1, num_points))
    # Sort the grid
    x_grid = np.asarray(x_grid)
    x_grid = np.sort(x_grid)

    # Empirical CDF
    sorted_chain = process_array(samples)
    counts = np.searchsorted(sorted_chain, x_grid, side='right')
    cdf_vals = counts / len(sorted_chain)

    # Data bounds
    data_min, data_max = np.min(samples), np.max(samples)
    data_range = data_max - data_min

    # Clip CDF values to avoid 0 and 1, which cause issues with inverse normal CDF
    cdf_vals = np.clip(cdf_vals, min_eps, 1.0 - min_eps)

    cdf_spline = PchipInterpolator(x_grid, cdf_vals, extrapolate=False, check=False)
    inverse_spline = PchipInterpolator(cdf_vals, x_grid, extrapolate=False, check=False)

    # Splice points: slightly inside data range to get reliable slope estimates
    # (PCHIP can have zero slope at exact data boundaries)
    x_splice_lower = np.quantile(samples, tail_percentile)
    x_splice_upper = np.quantile(samples, 1 - tail_percentile)

    # CDF values at splice points
    cdf_at_lower = cdf_spline(x_splice_lower)
    cdf_at_upper = cdf_spline(x_splice_upper)

    # PDF (slope) at splice points via autodiff
    pdf_at_lower = jax.grad(lambda x: cdf_spline(x))(x_splice_lower)
    pdf_at_upper = jax.grad(lambda x: cdf_spline(x))(x_splice_upper)

    # PCHIP guarantees monotonicity but can have zero slope at flat regions.
    # A zero PDF at the splice point would give sigma = inf (flat tail),
    # defeating the purpose of Gaussian extrapolation. Floor to a small
    # scale-invariant value to prevent this.
    min_pdf = 1.0 / (100 * data_range)
    pdf_at_lower = jnp.max(pdf_at_lower, min_pdf)
    pdf_at_upper = jnp.max(pdf_at_upper, min_pdf)

    # Gaussian sigma from PDF (derivative) continuity:
    # At splice point, we want: pdf_gaussian(x_splice) = pdf_empirical(x_splice)
    #
    # Lower tail: F(x) = 2 * u_lower * Gaussian CDF((x - x_splice) / sigma)
    #             f(x) = 2 * u_lower * Gaussian PDF((x - x_splice) / sigma) / sigma
    #             f(x_splice) = 2 * u_lower * Gaussian PDF(0) / sigma = pdf_at_lower
    #             => sigma = 2 * u_lower * PDF(0) / pdf_at_lower
    #
    # Upper tail: Similar with (1 - u_upper) instead of u_lower

    phi_0 = jax_norm.pdf(0.0)
    sigma_lower = 2 * cdf_at_lower * phi_0 / pdf_at_lower
    sigma_upper = 2 * (1 - cdf_at_upper) * phi_0 / pdf_at_upper

    def cdf_fn(x):
        """
        Evaluate CDF at x with Gaussian tail extrapolation.

        Parameters
        ----------
        x : scalar
            Point at which to evaluate CDF.

        Returns
        -------
        scalar
            CDF value in (min_eps, 1 - min_eps).
        """
        x = jnp.asarray(x, dtype=jnp.float64)

        # Bulk: evaluate spline (clip input to avoid NaN)
        # The clip doesn't affect values in [x_splice_lower, x_splice_upper]
        # For values outside, we use the Gaussian tail anyway, so the clipped
        # spline value is never selected — but this prevents NaN gradient leakage
        x_clipped = jnp.clip(x, x_splice_lower, x_splice_upper)
        cdf_bulk = cdf_spline(x_clipped)

        # Lower Gaussian tail: F(x) = 2 * u_lower * Gaussian CDF((x - x_splice) / sigma)
        # This equals u_lower when x = x_splice (since Gaussian CDF(0) = 0.5)
        z_lower = (x - x_splice_lower) / sigma_lower
        cdf_lower = 2 * cdf_at_lower * jax_norm.cdf(z_lower)

        # Upper Gaussian tail:
        # F(x) = u_upper + 2*(1-u_upper)*(Gaussian CDF((x - x_splice) / sigma) - 0.5)
        # This equals u_upper when x = x_splice (since Gaussian CDF(0) - 0.5 = 0)
        z_upper = (x - x_splice_upper) / sigma_upper
        cdf_upper = cdf_at_upper + 2 * (1 - cdf_at_upper) * (
            jax_norm.cdf(z_upper) - 0.5
        )

        # Select appropriate region
        return jnp.where(
            x < x_splice_lower,
            cdf_lower,
            jnp.where(x > x_splice_upper, cdf_upper, cdf_bulk),
        )

    def quantile_fn(u):
        """
        Evaluate inverse CDF (quantile function) at u.

        Parameters
        ----------
        u : scalar
            Probability value in (0, 1).

        Returns
        -------
        scalar
            Quantile value x such that CDF(x) ~ u.
        """
        u = jnp.asarray(u)
        u_safe = jnp.clip(u, min_eps, 1.0 - min_eps)

        # Bulk: use inverse spline (clip input to valid range)
        u_clipped = jnp.clip(u_safe, cdf_at_lower, cdf_at_upper)
        x_bulk = inverse_spline(u_clipped)

        # Lower tail inverse: u = 2 * u_lower * Gaussian CDF(z)
        #                   => Gaussian CDF(z) = u / (2 * u_lower)
        #                   => z = Gaussian CDF^-1(u / (2 * u_lower))
        #                   => x = x_splice + sigma * z
        arg_lower = jnp.clip(u_safe / (2 * cdf_at_lower), min_eps, 1 - min_eps)
        z_lower = jax_norm.ppf(arg_lower)
        x_lower = x_splice_lower + sigma_lower * z_lower

        # Upper tail inverse: u = u_upper + 2*(1-u_upper)*(Gaussian CDF(z) - 0.5)
        #                   => Gaussian CDF(z) = 0.5 + (u - u_upper) / (2*(1 - u_upper))
        arg_upper = 0.5 + (u_safe - cdf_at_upper) / (2 * (1 - cdf_at_upper))
        arg_upper = jnp.clip(arg_upper, min_eps, 1 - min_eps)
        z_upper = jax_norm.ppf(arg_upper)
        x_upper = x_splice_upper + sigma_upper * z_upper

        # Select appropriate region
        return jnp.where(
            u_safe < cdf_at_lower,
            x_lower,
            jnp.where(u_safe > cdf_at_upper, x_upper, x_bulk),
        )

    def pdf_fn(x):
        """
        Evaluate PDF at x via automatic differentiation of CDF.

        The PDF naturally decays in the tails via Gaussian extrapolation.
        A floor of the smallest positive normal float (_PDF_FLOOR) is applied
        to prevent exactly 0 values at isolated points where the empirical CDF
        may be flat due to data gaps.

        Parameters
        ----------
        x : scalar
            Point at which to evaluate PDF.

        Returns
        -------
        scalar
            PDF value (positive, with tiny floor for numerical stability).
        """
        return jnp.maximum(jax.grad(cdf_fn)(x), _PDF_FLOOR)

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
    z_max: float = _Z_MAX  # Maximum |z| before Gaussian CDF rounds to 0 or 1
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

        # Clip z (not u) to handle floating point limits where CDF rounds to exactly 0 or 1
        # This is better than clipping u because:
        # 1. Allows larger z range (uses precision-appropriate limit from _Z_MAX)
        # 2. More accurate mapping for extreme samples
        # 3. jnp.clip handles inf correctly: clip(inf, -z_max, z_max) = z_max
        z = jnp.clip(z, -self.z_max, self.z_max)

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


# class NormalToUniform(AbstractBijection):
#     r"""Bijection mapping x ∈ [a, b] → z ∈ ℝ via a uniform→normal CDF transform.

#     The forward transform is

#         u = clip((x - a) / (b - a), eps, 1 - eps)
#         z = sqrt(2) * erfinv(2 u - 1)

#     and the inverse is

#         u = 0.5 * (1 + erf(z / sqrt(2)))
#         x = a + u (b - a)

#     Args
#     ----
#     a : array_like or float
#         Lower bound(s) of the uniform support.
#     b : array_like or float
#         Upper bound(s) of the uniform support.
#     eps : float, default=1e-6
#         Clamping parameter to avoid CDF values exactly 0 or 1.
#     """

#     a: float
#     b: float
#     eps: float = 1e-6
#     cond_shape: ClassVar[None] = None

#     shape: ClassVar[tuple[int, ...]] = ()

#     def inverse_and_log_det(self, x, condition=None):
#         # map into (0,1)
#         u = (x - self.a) / (self.b - self.a)
#         u = jnp.clip(u, self.eps, 1.0 - self.eps)
#         # uniform→normal
#         y = jnp.sqrt(2.0) * erfinv(2.0 * u - 1.0)
#         # log|dz/du|
#         log_dz_du = 0.5 * jnp.log(2.0 * jnp.pi) + erfinv(2.0 * u - 1.0) ** 2
#         # log|du/dx|
#         log_du_dx = -jnp.log(self.b - self.a)
#         log_det = jnp.sum(log_dz_du + log_du_dx)
#         return y, log_det

#     def transform_and_log_det(self, y, condition=None):
#         # normal→uniform
#         u = 0.5 * (1.0 + erf(y / jnp.sqrt(2.0)))
#         # uniform→original
#         x = self.a + u * (self.b - self.a)
#         # log|dx/dy|
#         log_du_dy = -0.5 * jnp.log(2.0 * jnp.pi) - 0.5 * y ** 2
#         log_dx_du = jnp.log(self.b - self.a)
#         log_det = jnp.sum(log_dx_du + log_du_dy)
#         return x, log_det


# class InverseStandardize(AbstractBijection):

#     mean: float
#     std: float
#     cond_shape: ClassVar[None] = None

#     shape: ClassVar[tuple[int, ...]] = ()

#     def inverse_and_log_det(self, x, condition=None):
#         u = (x - self.mean) / self.std
#         log_du_dx = -jnp.log(self.std)
#         log_det = jnp.sum(log_du_dx)
#         return u, log_det

#     def transform_and_log_det(self, y, condition=None):
#         u = self.std * y + self.mean
#         log_du_dy = jnp.log(self.std)
#         log_det = jnp.sum(log_du_dy)
#         return u, log_det


# class NormalToUniformInverseStandardize(AbstractBijection):
#     a: float
#     b: float
#     mean: float
#     std: float
#     eps: float = 1e-6
#     cond_shape: ClassVar[None] = None

#     shape: ClassVar[tuple[int, ...]] = ()

#     def inverse_and_log_det(self, x, condition=None):
#         # map into (0,1)
#         u = (x - self.a) / (self.b - self.a)
#         u = jnp.clip(u, self.eps, 1.0 - self.eps)
#         # uniform→normal
#         y = self.mean + self.std * jnp.sqrt(2.0) * erfinv(2.0 * u - 1.0)
#         # standardize
#         z = (y - self.mean) / self.std
#         # log|dz/dy|
#         log_dz_dy = -jnp.log(self.std)
#         # log|dy/du|
#         log_dy_du = jnp.log(self.std) + 0.5 * jnp.log(2.0 * jnp.pi) + erfinv(2.0 * u - 1.0) ** 2
#         # log|du/dx|
#         log_du_dx = -jnp.log(self.b - self.a)
#         log_det = jnp.sum(log_dy_du + log_du_dx + log_dz_dy)
#         return z, log_det

#     def transform_and_log_det(self, z, condition=None):
#         # unstandardize
#         y = self.std * z + self.mean
#         # normal→uniform
#         u = 0.5 * (1.0 + erf((y - self.mean) / (self.std * jnp.sqrt(2.0))))
#         # uniform→original
#         x = self.a + u * (self.b - self.a)
#         # jacobians:
#         log_dz_dy = jnp.log(self.std)
#         log_du_dy = -0.5 * jnp.log(2.0 * jnp.pi) - 0.5 * ((y - self.mean)/self.std)**2 - jnp.log(self.std)
#         log_dx_du = jnp.log(self.b - self.a)
#         log_det = jnp.sum(log_dz_dy + log_dx_du + log_du_dy)
#         return x, log_det
