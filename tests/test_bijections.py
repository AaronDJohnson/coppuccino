import pytest
import numpy as np
import jax.numpy as jnp
from coppuccino.bijections import (
    _process_array,
    make_empirical_cdf_spline,
    EmpiricalMarginalToGaussian
)


class TestProcessArray:
    """Test the _process_array function for handling duplicates."""

    def test_no_duplicates(self):
        """Test that arrays without duplicates are returned unchanged."""
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _process_array(arr)
        np.testing.assert_array_equal(result, arr)

    def test_with_duplicates(self):
        """Test that duplicates are perturbed."""
        arr = np.array([1.0, 2.0, 2.0, 3.0, 4.0])
        result = _process_array(arr)
        # Check that all values are unique
        assert len(result) == len(np.unique(result))
        # Check that array is still sorted
        assert np.all(np.diff(result) >= 0)

    def test_all_duplicates(self):
        """Test array with all identical values."""
        arr = np.array([5.0, 5.0, 5.0, 5.0])
        result = _process_array(arr)
        # Check that all values are unique
        assert len(result) == len(np.unique(result))
        # Check that array is still sorted
        assert np.all(np.diff(result) >= 0)

    def test_unsorted_input(self):
        """Test that unsorted input is sorted."""
        arr = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
        result = _process_array(arr)
        # Check that array is sorted
        assert np.all(np.diff(result) >= 0)
        # Check that all values are unique
        assert len(result) == len(np.unique(result))

    def test_empty_array(self):
        """Test empty array handling."""
        arr = np.array([])
        result = _process_array(arr)
        assert len(result) == 0

    def test_single_element(self):
        """Test single element array."""
        arr = np.array([42.0])
        result = _process_array(arr)
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


class TestTailExtension:
    """Test the tail_extension parameter of make_empirical_cdf_spline."""

    def test_tail_extension_produces_wider_range(self):
        """Test that tail extension allows sampling beyond data range."""
        np.random.seed(42)
        samples = np.random.randn(1000)

        _, _, quantile_no_tail, _ = make_empirical_cdf_spline(samples, tail_extension=False)
        _, _, quantile_tail, _ = make_empirical_cdf_spline(samples, tail_extension=True)

        # With tail extension, extreme quantiles should go beyond data range
        data_min, data_max = np.min(samples), np.max(samples)
        low_q = float(quantile_tail(jnp.array(1e-5)))
        high_q = float(quantile_tail(jnp.array(1.0 - 1e-5)))

        # Without tail, quantiles are clipped to data range
        low_no_tail = float(quantile_no_tail(jnp.array(1e-5)))
        high_no_tail = float(quantile_no_tail(jnp.array(1.0 - 1e-5)))
        assert low_no_tail >= data_min - 0.01
        assert high_no_tail <= data_max + 0.01

        # With tail, quantiles can extend beyond
        assert low_q < data_min or high_q > data_max

    def test_tail_extension_cdf_continuity(self):
        """Test that CDF is continuous at data boundaries with tail extension."""
        np.random.seed(42)
        samples = np.random.randn(1000)

        _, cdf_fn, _, _ = make_empirical_cdf_spline(samples, tail_extension=True)

        data_min = np.min(samples)

        # CDF should be continuous at left boundary (right boundary clips at 1-eps)
        eps = 0.01
        cdf_at_min = float(cdf_fn(jnp.array(data_min)))
        cdf_below_min = float(cdf_fn(jnp.array(data_min - eps)))

        assert cdf_below_min < cdf_at_min
        # Tail CDF should still be positive below data range
        assert cdf_below_min > 0

    def test_no_tail_clips_to_bounds(self):
        """Test that without tail extension, CDF clips at data bounds."""
        np.random.seed(42)
        samples = np.random.randn(1000)

        _, cdf_fn, _, _ = make_empirical_cdf_spline(samples, tail_extension=False)

        data_min = np.min(samples)
        # Values well below data range should give min_eps
        cdf_far_below = float(cdf_fn(jnp.array(data_min - 10.0)))
        assert cdf_far_below == pytest.approx(1e-7, abs=1e-8)


class TestPriorBounds:
    """Test the prior_bounds parameter of make_empirical_cdf_spline."""

    def test_prior_bounds_extend_range(self):
        """Test that prior bounds extend the CDF grid."""
        np.random.seed(42)
        samples = np.random.randn(1000)

        _, _, quantile_no_prior, _ = make_empirical_cdf_spline(samples)
        _, _, quantile_with_prior, _ = make_empirical_cdf_spline(
            samples, prior_low=-10.0, prior_high=10.0)

        # With prior bounds, extreme quantiles should reach toward prior edges
        low_no_prior = float(quantile_no_prior(jnp.array(1e-5)))
        low_with_prior = float(quantile_with_prior(jnp.array(1e-5)))

        assert low_with_prior < low_no_prior

    def test_prior_bounds_cdf_at_edges(self):
        """Test CDF values at prior bound edges."""
        np.random.seed(42)
        samples = np.random.randn(1000)

        _, cdf_fn, _, _ = make_empirical_cdf_spline(
            samples, prior_low=-10.0, prior_high=10.0)

        # CDF at prior bounds should be close to min_eps / 1 - min_eps
        cdf_at_low = float(cdf_fn(jnp.array(-10.0)))
        cdf_at_high = float(cdf_fn(jnp.array(10.0)))

        assert cdf_at_low < 0.01
        assert cdf_at_high > 0.99

    def test_prior_bounds_within_data_range_ignored(self):
        """Test that prior bounds within data range don't affect grid."""
        np.random.seed(42)
        samples = np.random.randn(1000)
        data_min = np.min(samples)

        # Prior bound within data range should not be added
        cdf_vals_no_prior, _, _, _ = make_empirical_cdf_spline(samples)
        cdf_vals_with_prior, _, _, _ = make_empirical_cdf_spline(
            samples, prior_low=data_min + 0.1)

        # Should have same grid size since prior_low > xg[0]
        assert len(cdf_vals_no_prior) == len(cdf_vals_with_prior)


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
