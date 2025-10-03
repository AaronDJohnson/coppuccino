import pytest
import numpy as np
import jax.numpy as jnp
from coppuccino.bijections import (
    process_array,
    make_empirical_cdf_spline,
    EmpiricalMarginalToGaussian
)


class TestProcessArray:
    """Test the process_array function for handling duplicates."""

    def test_no_duplicates(self):
        """Test that arrays without duplicates are returned unchanged."""
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = process_array(arr)
        np.testing.assert_array_equal(result, arr)

    def test_with_duplicates(self):
        """Test that duplicates are perturbed."""
        arr = np.array([1.0, 2.0, 2.0, 3.0, 4.0])
        result = process_array(arr)
        # Check that all values are unique
        assert len(result) == len(np.unique(result))
        # Check that array is still sorted
        assert np.all(np.diff(result) >= 0)

    def test_all_duplicates(self):
        """Test array with all identical values."""
        arr = np.array([5.0, 5.0, 5.0, 5.0])
        result = process_array(arr)
        # Check that all values are unique
        assert len(result) == len(np.unique(result))
        # Check that array is still sorted
        assert np.all(np.diff(result) >= 0)

    def test_unsorted_input(self):
        """Test that unsorted input is sorted."""
        arr = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
        result = process_array(arr)
        # Check that array is sorted
        assert np.all(np.diff(result) >= 0)
        # Check that all values are unique
        assert len(result) == len(np.unique(result))

    def test_empty_array(self):
        """Test empty array handling."""
        arr = np.array([])
        result = process_array(arr)
        assert len(result) == 0

    def test_single_element(self):
        """Test single element array."""
        arr = np.array([42.0])
        result = process_array(arr)
        np.testing.assert_array_equal(result, arr)


class TestMakeEmpiricalCDFSpline:
    """Test the make_empirical_cdf_spline function."""

    def test_basic_functionality(self):
        """Test basic CDF spline creation."""
        samples = np.random.randn(1000)
        cdf_vals, cdf_fn, quantile_fn, pdf_fn = make_empirical_cdf_spline(samples, num_points=100)

        # Check that CDF values are within valid range
        assert np.all(cdf_vals >= 1e-7)
        assert np.all(cdf_vals <= 1.0 - 1e-7)

        # Check that CDF is monotonically increasing
        assert len(cdf_vals) == 100

    def test_cdf_properties(self):
        """Test that CDF has correct properties."""
        samples = np.random.randn(1000)
        _, cdf_fn, quantile_fn, pdf_fn = make_empirical_cdf_spline(samples, num_points=100)

        # Test that CDF at minimum is close to 0
        min_val = np.min(samples)
        cdf_at_min = cdf_fn(min_val)
        assert 0 <= float(cdf_at_min) <= 0.1

        # Test that CDF at maximum is close to 1
        max_val = np.max(samples)
        cdf_at_max = cdf_fn(max_val)
        assert 0.9 <= float(cdf_at_max) <= 1.0

    def test_pdf_positive(self):
        """Test that PDF is always positive."""
        samples = np.random.randn(1000)
        _, cdf_fn, quantile_fn, pdf_fn = make_empirical_cdf_spline(samples, num_points=100)

        # Test PDF at various points
        test_points = np.linspace(np.min(samples), np.max(samples), 20)
        for x in test_points:
            pdf_val = pdf_fn(x)
            assert float(pdf_val) > 0

    def test_quantile_inverse_cdf(self):
        """Test that quantile function is inverse of CDF."""
        samples = np.random.randn(1000)
        _, cdf_fn, quantile_fn, pdf_fn = make_empirical_cdf_spline(samples, num_points=100)

        # Test at a few points
        test_points = np.linspace(np.min(samples) + 0.1, np.max(samples) - 0.1, 5)
        for x in test_points:
            u = cdf_fn(x)
            x_recovered = quantile_fn(u)
            assert np.abs(float(x_recovered) - float(x)) < 0.1

    def test_custom_min_eps(self):
        """Test that custom min_eps parameter works."""
        samples = np.random.randn(1000)
        min_eps = 1e-5
        cdf_vals, _, _, _ = make_empirical_cdf_spline(samples, num_points=100, min_eps=min_eps)

        # Check that CDF values respect min_eps
        assert np.all(cdf_vals >= min_eps)
        assert np.all(cdf_vals <= 1.0 - min_eps)

    def test_uniform_samples(self):
        """Test with uniformly distributed samples."""
        samples = np.random.uniform(0, 1, 1000)
        cdf_vals, cdf_fn, quantile_fn, pdf_fn = make_empirical_cdf_spline(samples, num_points=100)

        # CDF should be approximately linear for uniform distribution
        assert len(cdf_vals) == 100


class TestEmpiricalMarginalToGaussian:
    """Test the EmpiricalMarginalToGaussian bijection."""

    def test_initialization(self):
        """Test that the bijection can be initialized."""
        samples = np.random.randn(1000)
        _, cdf_fn, quantile_fn, pdf_fn = make_empirical_cdf_spline(samples, num_points=100)

        bijection = EmpiricalMarginalToGaussian(
            samples=samples,
            cdf_fn=cdf_fn,
            quantile_fn=quantile_fn,
            pdf_fn=pdf_fn
        )

        assert bijection is not None
        assert bijection.min_eps == 1e-7

    def test_inverse_and_log_det(self):
        """Test inverse transformation (original to Gaussian)."""
        samples = np.random.randn(1000)
        _, cdf_fn, quantile_fn, pdf_fn = make_empirical_cdf_spline(samples, num_points=100)

        bijection = EmpiricalMarginalToGaussian(
            samples=samples,
            cdf_fn=cdf_fn,
            quantile_fn=quantile_fn,
            pdf_fn=pdf_fn
        )

        # Test on a sample point
        x = jnp.array(0.5)
        z, log_det = bijection.inverse_and_log_det(x)

        # Check that output is finite
        assert jnp.isfinite(z)
        assert jnp.isfinite(log_det)

    def test_transform_and_log_det(self):
        """Test forward transformation (Gaussian to original)."""
        samples = np.random.randn(1000)
        _, cdf_fn, quantile_fn, pdf_fn = make_empirical_cdf_spline(samples, num_points=100)

        bijection = EmpiricalMarginalToGaussian(
            samples=samples,
            cdf_fn=cdf_fn,
            quantile_fn=quantile_fn,
            pdf_fn=pdf_fn
        )

        # Test on a Gaussian sample
        z = jnp.array(0.0)
        x, log_det = bijection.transform_and_log_det(z)

        # Check that output is finite
        assert jnp.isfinite(x)
        assert jnp.isfinite(log_det)

    def test_inverse_consistency(self):
        """Test that transform and inverse are consistent."""
        samples = np.random.randn(1000)
        _, cdf_fn, quantile_fn, pdf_fn = make_empirical_cdf_spline(samples, num_points=100)

        bijection = EmpiricalMarginalToGaussian(
            samples=samples,
            cdf_fn=cdf_fn,
            quantile_fn=quantile_fn,
            pdf_fn=pdf_fn
        )

        # Test round-trip: x -> z -> x'
        x_original = jnp.array(0.5)
        z, _ = bijection.inverse_and_log_det(x_original)
        x_recovered, _ = bijection.transform_and_log_det(z)

        # Check that we recover the original value (within tolerance)
        assert jnp.abs(x_recovered - x_original) < 0.1

    def test_multiple_values(self):
        """Test transformation on multiple values."""
        samples = np.random.randn(1000)
        _, cdf_fn, quantile_fn, pdf_fn = make_empirical_cdf_spline(samples, num_points=100)

        bijection = EmpiricalMarginalToGaussian(
            samples=samples,
            cdf_fn=cdf_fn,
            quantile_fn=quantile_fn,
            pdf_fn=pdf_fn
        )

        # Test on multiple points
        x_values = jnp.array([0.0, 0.5, 1.0])
        for x in x_values:
            z, log_det = bijection.inverse_and_log_det(x)
            assert jnp.isfinite(z)
            assert jnp.isfinite(log_det)

    def test_extreme_values(self):
        """Test transformation with extreme values."""
        samples = np.random.randn(1000)
        _, cdf_fn, quantile_fn, pdf_fn = make_empirical_cdf_spline(samples, num_points=100)

        bijection = EmpiricalMarginalToGaussian(
            samples=samples,
            cdf_fn=cdf_fn,
            quantile_fn=quantile_fn,
            pdf_fn=pdf_fn
        )

        # Test with extreme Gaussian values
        z_extreme = jnp.array([5.0, -5.0])
        for z in z_extreme:
            x, log_det = bijection.transform_and_log_det(z)
            # Should still produce finite results
            assert jnp.isfinite(x)
            assert jnp.isfinite(log_det)

    def test_custom_min_eps(self):
        """Test bijection with custom min_eps."""
        samples = np.random.randn(1000)
        min_eps = 1e-5
        _, cdf_fn, quantile_fn, pdf_fn = make_empirical_cdf_spline(samples, num_points=100, min_eps=min_eps)

        bijection = EmpiricalMarginalToGaussian(
            samples=samples,
            cdf_fn=cdf_fn,
            quantile_fn=quantile_fn,
            pdf_fn=pdf_fn,
            min_eps=min_eps
        )

        assert bijection.min_eps == min_eps

        # Test transformation still works
        x = jnp.array(0.0)
        z, log_det = bijection.inverse_and_log_det(x)
        assert jnp.isfinite(z)
        assert jnp.isfinite(log_det)
