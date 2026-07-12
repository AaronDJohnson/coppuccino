import pytest
import numpy as np
import jax.numpy as jnp
from flowjax.distributions import Transformed
from paramax import unwrap
from coppuccino.copula_flows import (
    _create_empirical_transforms,
    normalizing_flows_fit,
    sample,
    log_prob,
    sample_and_log_prob
)


def _marginal_families(flow):
    """The marginal family recorded on each empirical transform of a flow."""
    return [str(unwrap(getattr(b, "marginal"))) for b in flow.bijection.bijections]


class TestMarginalSelection:
    """Test the marginal='rqs'|'pchip' choice on normalizing_flows_fit."""

    def test_default_marginal_is_smooth(self):
        """normalizing_flows_fit defaults to the gaussian pipeline's smooth marginal."""
        np.random.seed(42)
        chain = np.random.randn(200, 3)
        flow = normalizing_flows_fit(chain, max_epochs=5, patience=2)
        assert _marginal_families(flow) == ["smooth", "smooth", "smooth"]

    def test_uniform_pipeline_marginal_is_rqs(self):
        """The uniform pipeline resolves the default marginal to RQS."""
        np.random.seed(42)
        chain = np.random.randn(200, 3)
        flow = normalizing_flows_fit(chain, max_epochs=5, patience=2,
                                     copula_space="uniform")
        assert _marginal_families(flow) == ["rqs", "rqs", "rqs"]

    def test_pchip_marginal_is_selectable(self):
        """Passing marginal='pchip' uses the PCHIP marginal on every dimension."""
        np.random.seed(42)
        chain = np.random.randn(200, 3)
        flow = normalizing_flows_fit(chain, max_epochs=5, patience=2, marginal="pchip",
                                     copula_space="gaussian")
        assert isinstance(flow, Transformed)
        assert _marginal_families(flow) == ["pchip", "pchip", "pchip"]

    def test_both_marginals_fit_and_sample(self):
        """Both marginal families fit and produce finite samples/log-probs."""
        np.random.seed(42)
        chain = np.random.randn(200, 2)
        for marginal in ("rqs", "pchip"):
            flow = normalizing_flows_fit(chain, max_epochs=5, patience=2, marginal=marginal,
                                         copula_space="gaussian")
            s = sample(flow, n_samples=100, rng_seed=0)
            lp = log_prob(flow, chain)
            assert s.shape == (100, 2) and np.all(np.isfinite(s))
            assert np.all(np.isfinite(lp))

    def test_invalid_marginal_raises(self):
        """An unknown marginal family raises a clear error."""
        np.random.seed(42)
        chain = np.random.randn(200, 2)
        with pytest.raises(ValueError, match="marginal must be"):
            normalizing_flows_fit(chain, max_epochs=5, patience=2, marginal="bogus")

    def test_create_empirical_transforms_default_rqs(self):
        """_create_empirical_transforms defaults to RQS marginals."""
        np.random.seed(42)
        samples = np.random.randn(200, 2)
        transform, _ = _create_empirical_transforms(samples)
        assert all(str(unwrap(b.marginal)) == "rqs" for b in transform.bijections)


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
        """sample_and_log_prob and sample must return the *same draws* for a
        fixed seed.

        Note: the log-probs returned by sample_and_log_prob (the generation
        path) are the honest densities to use for importance weights; they are
        not asserted equal to log_prob(flow, samples) here because, with the
        default clipped empirical marginals, a sample drawn beyond ~ndtri(1/N)
        sigma is clipped and cannot be re-inverted exactly (supply prior_bounds
        to push that threshold out to ~5 sigma).
        """
        np.random.seed(42)
        chain = np.random.randn(200, 2)
        flow = normalizing_flows_fit(chain, max_epochs=5, patience=2)

        samples_combined, _ = sample_and_log_prob(flow, n_samples=100, rng_seed=42)
        samples_separate = sample(flow, n_samples=100, rng_seed=42)

        np.testing.assert_array_almost_equal(samples_combined, samples_separate)

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


class TestLogProbAccuracy:
    """Validate the public log_prob against a known analytic density."""

    def test_log_prob_tracks_correlated_gaussian(self):
        """Fit a flow to a correlated 2D Gaussian and check that log_prob tracks
        the analytic log-density up to an additive normalization constant.

        This is the one end-to-end accuracy check: it catches sign errors,
        varying offsets and gross miscalibration that shape/isfinite smoke tests
        cannot.
        """
        from scipy.stats import multivariate_normal
        rng = np.random.default_rng(0)
        mean = np.array([0.0, 0.0])
        cov = np.array([[1.0, 0.6], [0.6, 1.0]])
        data = rng.multivariate_normal(mean, cov, size=8000)

        flow = normalizing_flows_fit(
            data, rng_seed=0, max_epochs=120, patience=25, flow_layers=6
        )

        # Evaluate on in-support test points drawn from the same distribution.
        test = rng.multivariate_normal(mean, cov, size=400)
        lo, hi = data.min(axis=0), data.max(axis=0)
        test = test[np.all((test >= lo) & (test <= hi), axis=1)]

        flow_lp = log_prob(flow, test)
        true_lp = multivariate_normal(mean, cov).logpdf(test)

        assert np.all(np.isfinite(flow_lp))
        # Shape tracks the truth (a sign error would destroy this correlation).
        corr = float(np.corrcoef(flow_lp, true_lp)[0, 1])
        assert corr > 0.9
        # After removing the (near-constant) normalization offset, the residual
        # spread is small — the density is right, not merely monotone-related.
        resid = np.asarray(flow_lp) - true_lp
        assert float(np.std(resid)) < 0.5
