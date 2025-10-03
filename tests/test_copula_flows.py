import pytest
import numpy as np
import jax.numpy as jnp
from flowjax.distributions import Transformed
from coppuccino.copula_flows import (
    _create_empirical_transforms,
    fit_chain_entry,
    normalizing_flows_fit,
    sample,
    log_prob,
    sample_and_log_prob
)


class TestCreateEmpiricalTransforms:
    """Test the _create_empirical_transforms function."""

    def test_basic_functionality(self):
        """Test basic creation of empirical transforms."""
        # Create sample data with 2 parameters
        np.random.seed(42)
        samples = np.random.randn(100, 2)

        transform, inverse_log_det = _create_empirical_transforms(samples)

        # Check that transform is created
        assert transform is not None
        assert inverse_log_det is not None

    def test_with_nan_values(self):
        """Test that NaN values are properly filtered."""
        np.random.seed(42)
        samples = np.random.randn(100, 2)
        # Add some NaN values
        samples[0, 0] = np.nan
        samples[5, 1] = np.nan

        transform, inverse_log_det = _create_empirical_transforms(samples)

        # Should still work after filtering NaNs
        assert transform is not None
        assert inverse_log_det is not None

    def test_insufficient_samples(self):
        """Test error handling with insufficient samples."""
        # Create data with too few samples
        samples = np.random.randn(10, 2)

        with pytest.raises(ValueError, match="Insufficient samples"):
            _create_empirical_transforms(samples)

    def test_single_parameter(self):
        """Test with single parameter."""
        np.random.seed(42)
        samples = np.random.randn(100, 1)

        transform, inverse_log_det = _create_empirical_transforms(samples)

        assert transform is not None
        assert inverse_log_det is not None

    def test_multiple_parameters(self):
        """Test with multiple parameters."""
        np.random.seed(42)
        samples = np.random.randn(100, 5)

        transform, inverse_log_det = _create_empirical_transforms(samples)

        assert transform is not None
        assert inverse_log_det is not None

    def test_custom_min_eps(self):
        """Test with custom min_eps parameter."""
        np.random.seed(42)
        samples = np.random.randn(100, 2)
        min_eps = 1e-5

        transform, inverse_log_det = _create_empirical_transforms(samples, min_eps=min_eps)

        assert transform is not None
        assert inverse_log_det is not None

    def test_inverse_log_det_callable(self):
        """Test that inverse_log_det is callable and produces valid output."""
        np.random.seed(42)
        samples = np.random.randn(100, 2)

        transform, inverse_log_det = _create_empirical_transforms(samples)

        # Test that we can call inverse_log_det
        test_data = samples[:10]
        z, log_dets = inverse_log_det(test_data)

        assert z.shape == test_data.shape
        assert log_dets.shape == (test_data.shape[0],)
        assert jnp.all(jnp.isfinite(z))
        assert jnp.all(jnp.isfinite(log_dets))


class TestNormalizingFlowsFit:
    """Test the normalizing_flows_fit function."""

    def test_basic_fit(self):
        """Test basic flow fitting."""
        np.random.seed(42)
        chain = np.random.randn(200, 3)

        flow = normalizing_flows_fit(
            chain,
            rng_seed=999,
            max_epochs=5,  # Use few epochs for testing
            patience=2
        )

        assert isinstance(flow, Transformed)

    def test_different_dimensions(self):
        """Test fitting with different parameter dimensions."""
        np.random.seed(42)

        # Test with 1D
        chain_1d = np.random.randn(200, 1)
        flow_1d = normalizing_flows_fit(chain_1d, max_epochs=5, patience=2)
        assert isinstance(flow_1d, Transformed)

        # Test with 2D
        chain_2d = np.random.randn(200, 2)
        flow_2d = normalizing_flows_fit(chain_2d, max_epochs=5, patience=2)
        assert isinstance(flow_2d, Transformed)

        # Test with 5D
        chain_5d = np.random.randn(200, 5)
        flow_5d = normalizing_flows_fit(chain_5d, max_epochs=5, patience=2)
        assert isinstance(flow_5d, Transformed)

    def test_custom_parameters(self):
        """Test fitting with custom parameters."""
        np.random.seed(42)
        chain = np.random.randn(200, 2)

        flow = normalizing_flows_fit(
            chain,
            rng_seed=123,
            knots=8,
            interval=2,
            nn_depth=1,
            patience=3,
            learning_rate=1e-2,
            max_epochs=5
        )

        assert isinstance(flow, Transformed)

    def test_reproducibility(self):
        """Test that same seed produces same results."""
        np.random.seed(42)
        chain = np.random.randn(200, 2)

        flow1 = normalizing_flows_fit(chain, rng_seed=999, max_epochs=5, patience=2)
        flow2 = normalizing_flows_fit(chain, rng_seed=999, max_epochs=5, patience=2)

        # Sample from both flows and compare
        samples1 = sample(flow1, 100, rng_seed=42)
        samples2 = sample(flow2, 100, rng_seed=42)

        # Should be identical with same seed
        np.testing.assert_array_almost_equal(samples1, samples2)


class TestSample:
    """Test the sample function."""

    def test_basic_sampling(self):
        """Test basic sampling from a fitted flow."""
        np.random.seed(42)
        chain = np.random.randn(200, 2)
        flow = normalizing_flows_fit(chain, max_epochs=5, patience=2)

        samples = sample(flow, n_samples=100, rng_seed=999)

        assert samples.shape == (100, 2)
        assert isinstance(samples, np.ndarray)
        assert np.all(np.isfinite(samples))

    def test_different_sample_sizes(self):
        """Test sampling different numbers of samples."""
        np.random.seed(42)
        chain = np.random.randn(200, 2)
        flow = normalizing_flows_fit(chain, max_epochs=5, patience=2)

        for n in [10, 50, 100, 500]:
            samples = sample(flow, n_samples=n, rng_seed=999)
            assert samples.shape == (n, 2)

    def test_sampling_reproducibility(self):
        """Test that sampling with same seed is reproducible."""
        np.random.seed(42)
        chain = np.random.randn(200, 2)
        flow = normalizing_flows_fit(chain, max_epochs=5, patience=2)

        samples1 = sample(flow, n_samples=100, rng_seed=42)
        samples2 = sample(flow, n_samples=100, rng_seed=42)

        np.testing.assert_array_equal(samples1, samples2)

    def test_different_seeds_produce_different_samples(self):
        """Test that different seeds produce different samples."""
        np.random.seed(42)
        chain = np.random.randn(200, 2)
        flow = normalizing_flows_fit(chain, max_epochs=5, patience=2)

        samples1 = sample(flow, n_samples=100, rng_seed=42)
        samples2 = sample(flow, n_samples=100, rng_seed=123)

        # Samples should be different
        assert not np.array_equal(samples1, samples2)


class TestLogProb:
    """Test the log_prob function."""

    def test_basic_log_prob(self):
        """Test basic log probability computation."""
        np.random.seed(42)
        chain = np.random.randn(200, 2)
        flow = normalizing_flows_fit(chain, max_epochs=5, patience=2)

        test_samples = np.random.randn(50, 2)
        log_probs = log_prob(flow, test_samples)

        assert log_probs.shape == (50,)
        assert isinstance(log_probs, np.ndarray)
        assert np.all(np.isfinite(log_probs))

    def test_log_prob_on_training_data(self):
        """Test log probability on training data."""
        np.random.seed(42)
        chain = np.random.randn(200, 2)
        flow = normalizing_flows_fit(chain, max_epochs=5, patience=2)

        # Compute log prob on training data
        log_probs = log_prob(flow, chain)

        assert log_probs.shape == (200,)
        assert np.all(np.isfinite(log_probs))

    def test_single_sample_log_prob(self):
        """Test log probability for a single sample."""
        np.random.seed(42)
        chain = np.random.randn(200, 2)
        flow = normalizing_flows_fit(chain, max_epochs=5, patience=2)

        single_sample = np.random.randn(1, 2)
        log_probs = log_prob(flow, single_sample)

        assert log_probs.shape == (1,)
        assert np.isfinite(log_probs[0])


class TestSampleAndLogProb:
    """Test the sample_and_log_prob function."""

    def test_basic_functionality(self):
        """Test basic sampling and log probability computation."""
        np.random.seed(42)
        chain = np.random.randn(200, 2)
        flow = normalizing_flows_fit(chain, max_epochs=5, patience=2)

        samples, log_probs = sample_and_log_prob(flow, n_samples=100, rng_seed=999)

        assert samples.shape == (100, 2)
        assert log_probs.shape == (100,)
        assert isinstance(samples, np.ndarray)
        assert isinstance(log_probs, np.ndarray)
        assert np.all(np.isfinite(samples))
        assert np.all(np.isfinite(log_probs))

    def test_consistency_with_separate_calls(self):
        """Test that sample_and_log_prob is consistent with separate calls."""
        np.random.seed(42)
        chain = np.random.randn(200, 2)
        flow = normalizing_flows_fit(chain, max_epochs=5, patience=2)

        # Use sample_and_log_prob
        samples_combined, log_probs_combined = sample_and_log_prob(flow, n_samples=100, rng_seed=42)

        # Use separate calls
        samples_separate = sample(flow, n_samples=100, rng_seed=42)
        log_probs_separate = log_prob(flow, samples_separate)

        # Samples should be identical
        np.testing.assert_array_equal(samples_combined, samples_separate)

    def test_reproducibility(self):
        """Test reproducibility with same seed."""
        np.random.seed(42)
        chain = np.random.randn(200, 2)
        flow = normalizing_flows_fit(chain, max_epochs=5, patience=2)

        samples1, log_probs1 = sample_and_log_prob(flow, n_samples=100, rng_seed=42)
        samples2, log_probs2 = sample_and_log_prob(flow, n_samples=100, rng_seed=42)

        np.testing.assert_array_equal(samples1, samples2)
        np.testing.assert_array_equal(log_probs1, log_probs2)

    def test_different_sample_sizes(self):
        """Test with different sample sizes."""
        np.random.seed(42)
        chain = np.random.randn(200, 2)
        flow = normalizing_flows_fit(chain, max_epochs=5, patience=2)

        for n in [10, 50, 100]:
            samples, log_probs = sample_and_log_prob(flow, n_samples=n, rng_seed=999)
            assert samples.shape == (n, 2)
            assert log_probs.shape == (n,)
