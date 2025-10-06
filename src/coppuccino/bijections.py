from typing import ClassVar, Callable
import jax.numpy as jnp
import numpy as np
from flowjax.bijections import AbstractBijection
from interpax._ppoly import PchipInterpolator
from jax.scipy.special import ndtri, ndtr
import jax

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


def make_empirical_cdf_spline(samples, num_points=200, min_eps=1e-7):
    """
    Create empirical CDF, inverse CDF (quantile), and PDF functions from samples.

    This function constructs smooth monotonic spline interpolators for the empirical
    CDF and its inverse based on input samples. The PDF is computed via automatic
    differentiation of the CDF spline.

    Parameters
    ----------
    samples : array_like
        Training data samples from which to construct the empirical distribution.
    num_points : int, optional
        Number of grid points to use for spline construction. Default is 200.
    min_eps : float, optional
        Minimum epsilon to avoid CDF values exactly 0 or 1. Default is 1e-7.

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
    """
    x_grid = np.quantile(samples, np.linspace(0, 1, num_points))
    # Sort the grid
    x_grid = np.asarray(x_grid)
    sort_idx = np.argsort(x_grid)
    xg = x_grid[sort_idx]

    # Empirical CDF
    sorted_chain = process_array(samples)
    counts = np.searchsorted(sorted_chain, xg, side='right')
    cdf_vals = counts / len(sorted_chain)

    # Data bounds
    data_min, data_max = np.min(samples), np.max(samples)

    # Clip CDF values to avoid 0 and 1, which cause issues with inverse normal CDF
    cdf_vals = np.clip(cdf_vals, min_eps, 1.0 - min_eps)
    # Store actual CDF bounds after clipping (may differ from min_eps due to empirical CDF)
    cdf_min, cdf_max = float(np.min(cdf_vals)), float(np.max(cdf_vals))

    # Create monotonic spline for CDF
    cdf_spline = PchipInterpolator(xg, cdf_vals, extrapolate=False, check=False)
    inverse_cdf_spline = PchipInterpolator(cdf_vals, xg, extrapolate=False, check=False)

    def cdf_fn(u):
        u = jnp.asarray(u)
        # Handle extrapolation: constant outside domain
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
        # Clip u to the actual CDF range to prevent NaN from extrapolation
        # Use cdf_min/cdf_max (actual bounds) instead of min_eps (which may be smaller)
        u_clipped = jnp.clip(u, cdf_min, cdf_max)
        y = inverse_cdf_spline(u_clipped)
        # Handle extreme values by mapping to data bounds
        y = jnp.where(u < cdf_min, data_min, jnp.where(u > cdf_max, data_max, y))
        return y

    def pdf_fn(x):
        x = jnp.asarray(x)
        grad_fn = jax.grad(cdf_fn)
        y = grad_fn(x)
        # Ensure positive and bounded to avoid log(0)
        y = jnp.maximum(y, min_eps)
        return y

    return cdf_vals[np.argsort(sort_idx)], cdf_fn, quantile_fn, pdf_fn


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
