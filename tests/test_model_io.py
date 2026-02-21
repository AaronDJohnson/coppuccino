import pytest
import numpy as np
import tempfile
import os
from pathlib import Path

from flowjax.distributions import Transformed

from coppuccino.copula_flows import (
    normalizing_flows_fit,
    sample,
    log_prob,
    sample_and_log_prob
)
from coppuccino.model_io import (
    save_flow,
    load_flow,
    _extract_spline_data,
    _reconstruct_empirical_transforms,
)


class TestSaveFlow:
    """Test the save_flow function."""

    def test_basic_save(self):
        """Test basic flow saving."""
        np.random.seed(42)
        chain = np.random.randn(200, 2)
        flow = normalizing_flows_fit(chain, max_epochs=5, patience=2)

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            path = f.name

        try:
            save_flow(flow, path)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)

    def test_save_with_path_object(self):
        """Test saving with Path object."""
        np.random.seed(42)
        chain = np.random.randn(200, 2)
        flow = normalizing_flows_fit(chain, max_epochs=5, patience=2)

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            path = Path(f.name)

        try:
            save_flow(flow, path)
            assert path.exists()
        finally:
            path.unlink()

    def test_save_different_dimensions(self):
        """Test saving flows with different dimensions."""
        np.random.seed(42)

        with tempfile.TemporaryDirectory() as tmpdir:
            # 1D flow
            chain_1d = np.random.randn(200, 1)
            flow_1d = normalizing_flows_fit(chain_1d, max_epochs=5, patience=2)
            save_flow(flow_1d, os.path.join(tmpdir, "flow_1d.pkl"))
            assert os.path.exists(os.path.join(tmpdir, "flow_1d.pkl"))

            # 3D flow
            chain_3d = np.random.randn(200, 3)
            flow_3d = normalizing_flows_fit(chain_3d, max_epochs=5, patience=2)
            save_flow(flow_3d, os.path.join(tmpdir, "flow_3d.pkl"))
            assert os.path.exists(os.path.join(tmpdir, "flow_3d.pkl"))

            # 5D flow
            chain_5d = np.random.randn(200, 5)
            flow_5d = normalizing_flows_fit(chain_5d, max_epochs=5, patience=2)
            save_flow(flow_5d, os.path.join(tmpdir, "flow_5d.pkl"))
            assert os.path.exists(os.path.join(tmpdir, "flow_5d.pkl"))


class TestLoadFlow:
    """Test the load_flow function."""

    def test_basic_load(self):
        """Test basic flow loading."""
        np.random.seed(42)
        chain = np.random.randn(200, 2)
        flow = normalizing_flows_fit(chain, max_epochs=5, patience=2)

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            path = f.name

        try:
            save_flow(flow, path)
            loaded_flow = load_flow(path)
            assert isinstance(loaded_flow, Transformed)
        finally:
            os.unlink(path)

    def test_load_with_path_object(self):
        """Test loading with Path object."""
        np.random.seed(42)
        chain = np.random.randn(200, 2)
        flow = normalizing_flows_fit(chain, max_epochs=5, patience=2)

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            path = Path(f.name)

        try:
            save_flow(flow, path)
            loaded_flow = load_flow(path)
            assert isinstance(loaded_flow, Transformed)
        finally:
            path.unlink()


class TestSaveLoadRoundTrip:
    """Test save/load round-trip functionality."""

    def test_samples_match_after_reload(self):
        """Test that samples from loaded flow match original."""
        np.random.seed(42)
        chain = np.random.randn(200, 2)
        flow = normalizing_flows_fit(chain, max_epochs=5, patience=2)

        # Sample from original flow
        original_samples = sample(flow, n_samples=100, rng_seed=999)

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            path = f.name

        try:
            save_flow(flow, path)
            loaded_flow = load_flow(path)

            # Sample from loaded flow with same seed
            loaded_samples = sample(loaded_flow, n_samples=100, rng_seed=999)

            # Samples should be identical
            np.testing.assert_array_almost_equal(original_samples, loaded_samples)
        finally:
            os.unlink(path)

    def test_log_prob_matches_after_reload(self):
        """Test that log probabilities match after reload."""
        np.random.seed(42)
        chain = np.random.randn(200, 2)
        flow = normalizing_flows_fit(chain, max_epochs=5, patience=2)

        # Compute log prob with original flow
        test_data = np.random.randn(50, 2)
        original_log_probs = log_prob(flow, test_data)

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            path = f.name

        try:
            save_flow(flow, path)
            loaded_flow = load_flow(path)

            # Compute log prob with loaded flow
            loaded_log_probs = log_prob(loaded_flow, test_data)

            # Log probs should be very close (numerical precision may differ slightly)
            np.testing.assert_array_almost_equal(
                original_log_probs, loaded_log_probs, decimal=5
            )
        finally:
            os.unlink(path)

    def test_sample_and_log_prob_matches_after_reload(self):
        """Test that sample_and_log_prob matches after reload."""
        np.random.seed(42)
        chain = np.random.randn(200, 2)
        flow = normalizing_flows_fit(chain, max_epochs=5, patience=2)

        # Use sample_and_log_prob with original flow
        original_samples, original_log_probs = sample_and_log_prob(
            flow, n_samples=100, rng_seed=42
        )

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            path = f.name

        try:
            save_flow(flow, path)
            loaded_flow = load_flow(path)

            # Use sample_and_log_prob with loaded flow
            loaded_samples, loaded_log_probs = sample_and_log_prob(
                loaded_flow, n_samples=100, rng_seed=42
            )

            # Results should match
            np.testing.assert_array_almost_equal(original_samples, loaded_samples)
            np.testing.assert_array_almost_equal(
                original_log_probs, loaded_log_probs, decimal=5
            )
        finally:
            os.unlink(path)

    def test_round_trip_1d(self):
        """Test save/load round-trip for 1D flow."""
        np.random.seed(42)
        chain = np.random.randn(200, 1)
        flow = normalizing_flows_fit(chain, max_epochs=5, patience=2)

        original_samples = sample(flow, n_samples=100, rng_seed=999)

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            path = f.name

        try:
            save_flow(flow, path)
            loaded_flow = load_flow(path)
            loaded_samples = sample(loaded_flow, n_samples=100, rng_seed=999)

            np.testing.assert_array_almost_equal(original_samples, loaded_samples)
        finally:
            os.unlink(path)

    def test_round_trip_5d(self):
        """Test save/load round-trip for 5D flow."""
        np.random.seed(42)
        chain = np.random.randn(200, 5)
        flow = normalizing_flows_fit(chain, max_epochs=5, patience=2)

        original_samples = sample(flow, n_samples=100, rng_seed=999)

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            path = f.name

        try:
            save_flow(flow, path)
            loaded_flow = load_flow(path)
            loaded_samples = sample(loaded_flow, n_samples=100, rng_seed=999)

            np.testing.assert_array_almost_equal(original_samples, loaded_samples)
        finally:
            os.unlink(path)

    def test_multiple_save_load_cycles(self):
        """Test that multiple save/load cycles preserve the flow."""
        np.random.seed(42)
        chain = np.random.randn(200, 2)
        flow = normalizing_flows_fit(chain, max_epochs=5, patience=2)

        original_samples = sample(flow, n_samples=100, rng_seed=999)

        with tempfile.TemporaryDirectory() as tmpdir:
            # First save/load
            path1 = os.path.join(tmpdir, "flow1.pkl")
            save_flow(flow, path1)
            flow_v1 = load_flow(path1)

            # Second save/load
            path2 = os.path.join(tmpdir, "flow2.pkl")
            save_flow(flow_v1, path2)
            flow_v2 = load_flow(path2)

            # Third save/load
            path3 = os.path.join(tmpdir, "flow3.pkl")
            save_flow(flow_v2, path3)
            flow_v3 = load_flow(path3)

            # Samples should still match
            final_samples = sample(flow_v3, n_samples=100, rng_seed=999)
            np.testing.assert_array_almost_equal(original_samples, final_samples)


class TestExtractSplineData:
    """Test the _extract_spline_data helper function."""

    def test_extracts_correct_number_of_dimensions(self):
        """Test that correct number of dimensions is extracted."""
        np.random.seed(42)

        for n_dims in [1, 2, 3, 5]:
            chain = np.random.randn(200, n_dims)
            flow = normalizing_flows_fit(chain, max_epochs=5, patience=2)

            spline_data = _extract_spline_data(flow)

            assert spline_data['num_dims'] == n_dims
            assert len(spline_data['samples']) == n_dims
            assert len(spline_data['min_eps']) == n_dims

    def test_samples_are_preserved(self):
        """Test that sample arrays are correctly extracted."""
        np.random.seed(42)
        chain = np.random.randn(200, 2)
        flow = normalizing_flows_fit(chain, max_epochs=5, patience=2)

        spline_data = _extract_spline_data(flow)

        # Samples should have the right shape
        for i, samples in enumerate(spline_data['samples']):
            assert isinstance(samples, np.ndarray)
            assert len(samples.shape) == 1  # 1D array per dimension

    def test_min_eps_is_float(self):
        """Test that min_eps values are floats."""
        np.random.seed(42)
        chain = np.random.randn(200, 2)
        flow = normalizing_flows_fit(chain, max_epochs=5, patience=2)

        spline_data = _extract_spline_data(flow)

        for eps in spline_data['min_eps']:
            assert isinstance(eps, float)
            assert eps > 0


class TestReconstructEmpiricalTransforms:
    """Test the _reconstruct_empirical_transforms helper function."""

    def test_reconstruction_creates_valid_transforms(self):
        """Test that reconstruction creates valid transforms."""
        np.random.seed(42)
        chain = np.random.randn(200, 2)
        flow = normalizing_flows_fit(chain, max_epochs=5, patience=2)

        spline_data = _extract_spline_data(flow)
        reconstructed = _reconstruct_empirical_transforms(spline_data)

        # Should have correct number of bijections
        assert len(reconstructed.bijections) == spline_data['num_dims']

    def test_reconstructed_transforms_work(self):
        """Test that reconstructed transforms can be used."""
        import jax
        import jax.numpy as jnp

        np.random.seed(42)
        chain = np.random.randn(200, 2)
        flow = normalizing_flows_fit(chain, max_epochs=5, patience=2)

        spline_data = _extract_spline_data(flow)
        reconstructed = _reconstruct_empirical_transforms(spline_data)

        # Test that we can apply the transform
        test_data = chain[:10]
        inverse_log_det = jax.jit(jax.vmap(reconstructed.inverse_and_log_det))

        z, log_dets = inverse_log_det(test_data)

        assert z.shape == test_data.shape
        assert log_dets.shape == (test_data.shape[0],)
        assert jnp.all(jnp.isfinite(z))
        assert jnp.all(jnp.isfinite(log_dets))


class TestSaveLoadWithNonDefaultParams:
    """Test save/load round-trip with non-default parameters."""

    def test_round_trip_with_prior_bounds(self):
        """Test that prior_bounds are preserved through save/load."""
        np.random.seed(42)
        chain = np.random.randn(200, 2)
        bounds = np.array([[-5.0, 5.0], [-5.0, 5.0]])
        flow = normalizing_flows_fit(chain, max_epochs=5, patience=2, prior_bounds=bounds)

        original_samples = sample(flow, n_samples=100, rng_seed=999)

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            path = f.name

        try:
            save_flow(flow, path)
            loaded_flow = load_flow(path)
            loaded_samples = sample(loaded_flow, n_samples=100, rng_seed=999)

            np.testing.assert_array_almost_equal(original_samples, loaded_samples)
        finally:
            os.unlink(path)

    def test_spline_data_preserves_metadata(self):
        """Test that _extract_spline_data preserves new metadata fields."""
        np.random.seed(42)
        chain = np.random.randn(200, 2)
        bounds = np.array([[-5.0, 5.0], [-5.0, 5.0]])
        flow = normalizing_flows_fit(chain, max_epochs=5, patience=2, prior_bounds=bounds)

        spline_data = _extract_spline_data(flow)

        assert 'num_points' in spline_data
        assert 'tail_extension' in spline_data
        assert 'prior_low' in spline_data
        assert 'prior_high' in spline_data
        assert len(spline_data['num_points']) == 2
        assert len(spline_data['tail_extension']) == 2
        # Prior bounds should be preserved
        for i in range(2):
            assert spline_data['prior_low'][i] == pytest.approx(-5.0)
            assert spline_data['prior_high'][i] == pytest.approx(5.0)
            assert spline_data['tail_extension'][i] is False

    def test_round_trip_with_tail_extension(self):
        """Test that tail_extension is preserved through save/load."""
        np.random.seed(42)
        chain = np.random.randn(200, 2)
        flow = normalizing_flows_fit(chain, max_epochs=5, patience=2, tail_extension=True)

        original_samples = sample(flow, n_samples=100, rng_seed=999)

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            path = f.name

        try:
            save_flow(flow, path)

            spline_data = _extract_spline_data(flow)
            for i in range(2):
                assert spline_data['tail_extension'][i] is True

            loaded_flow = load_flow(path)
            loaded_samples = sample(loaded_flow, n_samples=100, rng_seed=999)

            np.testing.assert_array_almost_equal(original_samples, loaded_samples)
        finally:
            os.unlink(path)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_nonexistent_file_raises_error(self):
        """Test that loading non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_flow("/nonexistent/path/flow.pkl")

    def test_save_creates_parent_directories(self):
        """Test that save works even with nested paths."""
        np.random.seed(42)
        chain = np.random.randn(200, 2)
        flow = normalizing_flows_fit(chain, max_epochs=5, patience=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save to a nested path (parent exists)
            path = os.path.join(tmpdir, "flow.pkl")
            save_flow(flow, path)
            assert os.path.exists(path)

    def test_overwrite_existing_file(self):
        """Test that save overwrites existing file."""
        np.random.seed(42)
        chain = np.random.randn(200, 2)
        flow = normalizing_flows_fit(chain, max_epochs=5, patience=2)

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            path = f.name
            # Write dummy content
            f.write(b"dummy content")

        try:
            # Save should overwrite
            save_flow(flow, path)

            # Verify by loading
            loaded_flow = load_flow(path)
            assert isinstance(loaded_flow, Transformed)
        finally:
            os.unlink(path)

    def test_loaded_flow_produces_finite_samples(self):
        """Test that loaded flow produces finite samples."""
        np.random.seed(42)
        chain = np.random.randn(200, 3)
        flow = normalizing_flows_fit(chain, max_epochs=5, patience=2)

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            path = f.name

        try:
            save_flow(flow, path)
            loaded_flow = load_flow(path)

            # Generate many samples and check they're all finite
            samples = sample(loaded_flow, n_samples=1000, rng_seed=42)
            assert np.all(np.isfinite(samples))
        finally:
            os.unlink(path)

    def test_loaded_flow_produces_finite_log_probs(self):
        """Test that loaded flow produces finite log probabilities."""
        np.random.seed(42)
        chain = np.random.randn(200, 3)
        flow = normalizing_flows_fit(chain, max_epochs=5, patience=2)

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            path = f.name

        try:
            save_flow(flow, path)
            loaded_flow = load_flow(path)

            # Evaluate log probs on training data
            log_probs = log_prob(loaded_flow, chain)
            assert np.all(np.isfinite(log_probs))
        finally:
            os.unlink(path)
