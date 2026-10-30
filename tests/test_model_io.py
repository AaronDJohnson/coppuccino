import os
import stat
import tempfile
from pathlib import Path

import cloudpickle
import numpy as np
import pytest
from flowjax.distributions import Transformed

import coppuccino.model_io as model_io
from coppuccino.copula_flows import log_prob, normalizing_flows_fit, sample, sample_and_log_prob
from coppuccino.model_io import (
    FLEET_KIND,
    FLOW_KIND,
    FORMAT_VERSION,
    _atomic_write_bytes,
    _extract_spline_data,
    _reconstruct_empirical_transforms,
    load_flow,
    save_flow,
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


class TestPayloadSizeLaw:
    """Pin the file-size scaling that `save_flow`'s Notes quote figures for."""

    def test_size_is_one_float64_per_row_per_dim_plus_a_fixed_part(self):
        """Growing the chain must cost exactly the retained training columns.

        The docstring's MB figures are only reproducible while this holds: if a
        future writer stores something else that scales with ``n`` (a knot grid
        per row, a cached z column) the quoted sizes stop following from the
        chain, and the advice to thin the chain stops being the whole story.
        """
        dim = 2
        rng = np.random.default_rng(0)
        sizes = {}
        with tempfile.TemporaryDirectory() as tmpdir:
            for n in (300, 2400):
                chain = rng.standard_normal((n, dim))
                flow = normalizing_flows_fit(chain, max_epochs=3, patience=2,
                                             show_progress=False)
                path = os.path.join(tmpdir, f"flow_{n}.pkl")
                save_flow(flow, path)
                sizes[n] = os.path.getsize(path)

        grew_by = sizes[2400] - sizes[300]
        expected = 8 * dim * (2400 - 300)
        # Slack absorbs pickle's variable-width length prefixes only: measured
        # at 36 bytes of drift across n=100..500,000, so anything of the order
        # of a second per-row array would blow straight through it.
        assert abs(grew_by - expected) < 128, (
            f"payload grew by {grew_by} bytes for 2100 extra rows x {dim} "
            f"dims; one float64 each is {expected}")

        fixed = sizes[300] - 8 * dim * 300
        assert 0 < fixed < 64_000, (
            f"fixed (chain-independent) part of the payload is {fixed} bytes")


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
            assert spline_data['marginal'] == ['smooth'] * n_dims

    def test_samples_are_preserved(self):
        """Test that sample arrays are correctly extracted."""
        np.random.seed(42)
        chain = np.random.randn(200, 2)
        flow = normalizing_flows_fit(chain, max_epochs=5, patience=2)

        spline_data = _extract_spline_data(flow)

        # Samples should have the right shape
        for samples in spline_data['samples']:
            assert isinstance(samples, np.ndarray)
            assert len(samples.shape) == 1  # 1D array per dimension

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
        assert 'prior_low' in spline_data
        assert 'prior_high' in spline_data
        assert len(spline_data['num_points']) == 2
        # Prior bounds should be preserved
        for i in range(2):
            assert spline_data['prior_low'][i] == pytest.approx(-5.0)
            assert spline_data['prior_high'][i] == pytest.approx(5.0)
        assert spline_data['tail_alpha'] == [0.0, 0.0]
        # the removed marginal families' metadata is no longer written
        assert not {'min_eps', 'tail_extension', 'tail_model',
                    'tail_quantile'} & spline_data.keys()

    def test_round_trip_preserves_tail_alpha(self):
        """A non-default tail_alpha survives save/load and the reloaded flow
        reproduces the original samples exactly."""
        from paramax import unwrap
        np.random.seed(42)
        chain = np.random.randn(200, 2)
        flow = normalizing_flows_fit(chain, max_epochs=5, patience=2,
                                     tail_alpha=0.5)
        assert _extract_spline_data(flow)['tail_alpha'] == [0.5, 0.5]

        original_samples = sample(flow, n_samples=100, rng_seed=999)
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            path = f.name
        try:
            save_flow(flow, path)
            loaded_flow = load_flow(path)
            assert [float(unwrap(b.tail_alpha))
                    for b in loaded_flow.bijection.bijections] == [0.5, 0.5]
            loaded_samples = sample(loaded_flow, n_samples=100, rng_seed=999)
            np.testing.assert_array_almost_equal(original_samples, loaded_samples)
        finally:
            os.unlink(path)

    @pytest.mark.parametrize("families, named", [
        (None, "'pchip'"),               # predates the 'marginal' key
        (["rqs", "smooth"], "'rqs'"),    # 1.0.0's default family
    ])
    def test_a_removed_marginal_family_is_refused_by_name(self, families, named):
        """The removed families cannot be rebuilt, so their files must fail
        naming the file and the family, not on a missing key mid-rebuild."""
        np.random.seed(42)
        chain = np.random.randn(200, 2)
        flow = normalizing_flows_fit(chain, max_epochs=5, patience=2)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "flow.pkl"
            save_flow(flow, path)
            with open(path, "rb") as f:
                payload = cloudpickle.load(f)
            if families is None:
                del payload['spline_data']['marginal']
            else:
                payload['spline_data']['marginal'] = families
            with open(path, "wb") as f:
                cloudpickle.dump(payload, f)
            with pytest.raises(ValueError, match=f"{named} marginal") as exc:
                load_flow(path)
            assert str(path) in str(exc.value)
            assert "coppuccino<2" in str(exc.value)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_nonexistent_file_raises_error(self):
        """Test that loading non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_flow("/nonexistent/path/flow.pkl")

    def test_save_into_existing_directory(self):
        """Saving into a directory that already exists writes the file."""
        np.random.seed(42)
        chain = np.random.randn(200, 2)
        flow = normalizing_flows_fit(chain, max_epochs=5, patience=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "flow.pkl")
            save_flow(flow, path)
            assert os.path.exists(path)

    def test_save_does_not_create_missing_parent_directory(self):
        """save_flow does not mkdir: a missing parent is a clear, actionable
        FileNotFoundError rather than a silently created directory tree."""
        np.random.seed(42)
        chain = np.random.randn(200, 2)
        flow = normalizing_flows_fit(chain, max_epochs=5, patience=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "nope", "flow.pkl")
            with pytest.raises(FileNotFoundError):
                save_flow(flow, path)
            assert not os.path.exists(os.path.join(tmpdir, "nope"))

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


class TestRemovedUniformPipeline:
    """The unit-cube pipeline was removed in 3.0.0; its files must fail clearly."""

    def test_loading_a_uniform_mode_file_raises(self):
        """A v2 uniform-mode payload is detected by its 'inner_arch' key."""
        import cloudpickle

        # A v2 uniform save_flow payload: architecture spec instead of a
        # pickled inner-flow skeleton. Only the marker key matters here.
        payload = {
            'spline_data': {'num_dims': 0},
            'inner_flow_bytes': b'',
            'inner_arch': ('uniform', 2, 4, 6),
        }
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            path = f.name
        try:
            with open(path, 'wb') as f:
                cloudpickle.dump(payload, f)
            with pytest.raises(ValueError, match="removed in coppuccino 3.0.0"):
                load_flow(path)
        finally:
            os.unlink(path)

    def test_current_files_do_not_carry_the_marker(self):
        """A flow saved today reconstructs without tripping the guard."""
        np.random.seed(42)
        chain = np.random.randn(200, 2)
        flow = normalizing_flows_fit(chain, max_epochs=5, patience=2)

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            path = f.name
        try:
            save_flow(flow, path)
            reloaded = load_flow(path)
            assert np.allclose(log_prob(reloaded, chain), log_prob(flow, chain))
        finally:
            os.unlink(path)


@pytest.fixture(scope="module")
def small_flow():
    """A cheap fitted flow shared by the persistence-mechanics tests."""
    np.random.seed(42)
    chain = np.random.randn(200, 2)
    return normalizing_flows_fit(chain, max_epochs=5, patience=2)


class _FullDisk:
    """A cloudpickle stand-in whose serialization fails with ENOSPC.

    ``dump`` writes half the payload before failing, which is what a writer
    that opens the destination directly leaves behind; ``dumps`` fails before
    producing any bytes at all. Injecting the same fault at both seams keeps
    the test independent of which call `save_flow` happens to use.
    """

    @staticmethod
    def dump(obj, f, *args, **kwargs):
        blob = cloudpickle.dumps(obj, *args, **kwargs)
        f.write(blob[: len(blob) // 2])
        raise OSError(28, "No space left on device")

    @staticmethod
    def dumps(obj, *args, **kwargs):
        raise OSError(28, "No space left on device")


class TestAtomicSave:
    """A failed save must not destroy the model already on disk."""

    def test_failed_save_leaves_previous_model_intact(self, small_flow, monkeypatch):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "flow.pkl"
            save_flow(small_flow, path)
            before = path.read_bytes()

            monkeypatch.setattr(model_io, "cloudpickle", _FullDisk)
            with pytest.raises(OSError):
                save_flow(small_flow, path)
            monkeypatch.undo()

            assert path.read_bytes() == before
            assert isinstance(load_flow(path), Transformed)

    def test_failed_save_leaves_no_temporary_files(self, small_flow, monkeypatch):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "flow.pkl"
            save_flow(small_flow, path)

            monkeypatch.setattr(model_io, "cloudpickle", _FullDisk)
            with pytest.raises(OSError):
                save_flow(small_flow, path)
            monkeypatch.undo()

            assert sorted(os.listdir(tmpdir)) == ["flow.pkl"]

    def test_overwrite_preserves_file_permissions(self, small_flow):
        """Publishing through a temp file must not tighten the mode: mkstemp
        creates 0o600, so the destination's own permissions are reapplied."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "flow.pkl"
            save_flow(small_flow, path)
            os.chmod(path, 0o640)

            save_flow(small_flow, path)

            assert stat.S_IMODE(os.stat(path).st_mode) == 0o640

    def test_atomic_write_bytes_replaces_content_wholesale(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "payload.bin"
            path.write_bytes(b"old contents, longer than the new ones")

            _atomic_write_bytes(path, b"new")

            assert path.read_bytes() == b"new"
            assert sorted(os.listdir(tmpdir)) == ["payload.bin"]

    def test_atomic_write_bytes_keeps_the_old_file_on_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "payload.bin"
            path.write_bytes(b"original")

            # A str is rejected by the underlying binary write, so the failure
            # lands after the temp file exists but before publication.
            with pytest.raises(TypeError):
                _atomic_write_bytes(path, "not bytes")

            assert path.read_bytes() == b"original"
            assert sorted(os.listdir(tmpdir)) == ["payload.bin"]

    def test_symlinked_destination_updates_the_target(self):
        """os.replace would swap out the link itself, where the direct
        open(path, 'wb') it replaced wrote *through* it -- so a published
        model would stop being visible at the shared store it points into."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = Path(tmpdir) / "store"
            store.mkdir()
            target = store / "model.bin"
            target.write_bytes(b"original")
            link = Path(tmpdir) / "latest.bin"
            link.symlink_to(target)

            _atomic_write_bytes(link, b"published")

            assert link.is_symlink()
            assert os.readlink(link) == str(target)
            assert target.read_bytes() == b"published"
            # the temp file belongs next to the target, not next to the link
            assert sorted(os.listdir(store)) == ["model.bin"]

    def test_save_flow_through_a_symlink_updates_the_target(self, small_flow):
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "model.pkl"
            save_flow(small_flow, target)
            link = Path(tmpdir) / "latest.pkl"
            link.symlink_to(target)

            save_flow(small_flow, link)

            assert link.is_symlink()
            assert isinstance(load_flow(target), Transformed)


class TestWriteErrorsNameTheDestination:
    """Every publication failure must quote the path the caller passed, never
    the ``<name>.<random>.tmp`` sibling it cannot find again."""

    def test_missing_parent_names_the_destination(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nope" / "flow.pkl"
            with pytest.raises(FileNotFoundError) as excinfo:
                _atomic_write_bytes(path, b"payload")

            message = str(excinfo.value)
            assert str(path) in message
            assert ".tmp" not in message

    def test_save_flow_missing_parent_names_the_destination(self, small_flow):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nope" / "flow.pkl"
            with pytest.raises(FileNotFoundError) as excinfo:
                save_flow(small_flow, path)

            message = str(excinfo.value)
            assert str(path) in message
            assert ".tmp" not in message

    def test_directory_destination_fails_before_writing_anything(self, monkeypatch):
        """A directory destination can never be published, so it is caught up
        front rather than after the whole payload has been written to a temp
        sibling that os.replace then fails on."""
        def _refuse(*args, **kwargs):
            raise AssertionError("mkstemp must not run for a directory destination")

        monkeypatch.setattr(model_io.tempfile, "mkstemp", _refuse)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "adir"
            path.mkdir()
            with pytest.raises(IsADirectoryError) as excinfo:
                _atomic_write_bytes(path, b"payload")

            message = str(excinfo.value)
            assert str(path) in message
            assert ".tmp" not in message
            assert sorted(os.listdir(tmpdir)) == ["adir"]

    @pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0,
                        reason="root ignores directory write permissions")
    def test_unwritable_directory_names_the_destination(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            locked = Path(tmpdir) / "locked"
            locked.mkdir()
            path = locked / "flow.pkl"
            path.write_bytes(b"original")
            os.chmod(locked, 0o500)
            try:
                with pytest.raises(PermissionError) as excinfo:
                    _atomic_write_bytes(path, b"payload")
            finally:
                os.chmod(locked, 0o700)

            message = str(excinfo.value)
            assert str(path) in message
            assert ".tmp" not in message
            assert path.read_bytes() == b"original"

    def test_empty_path_is_rejected_without_touching_the_cwd(self, monkeypatch):
        """Path('') is Path('.'), whose parent is a directory too, so the
        write used to land a temp file in the current directory and fail with
        EINVAL on a path the caller never wrote."""
        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.chdir(tmpdir)
            with pytest.raises(ValueError, match="not a file"):
                _atomic_write_bytes("", b"payload")
            assert sorted(os.listdir(".")) == []


class TestSaveFlowArgumentChecks:
    """save_flow validates `flow` instead of dying on an internal attribute."""

    @pytest.mark.parametrize("bad", [
        np.zeros((10, 2)),
        None,
        {"spline_data": {}},
        "not a flow",
    ])
    def test_non_flow_raises_typeerror(self, bad):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "flow.pkl"
            with pytest.raises(TypeError) as excinfo:
                save_flow(bad, path)

            message = str(excinfo.value)
            assert "save_flow" in message
            assert type(bad).__name__ in message
            # the old failure was an AttributeError about .bijection
            assert "bijection" not in message
            assert sorted(os.listdir(tmpdir)) == []


class TestLoadFlowFailureMessages:
    """Unreadable files must fail by name, not by internal key."""

    def _write(self, tmpdir, payload_bytes):
        path = Path(tmpdir) / "flow.pkl"
        path.write_bytes(payload_bytes)
        return path

    def test_missing_payload_key_names_the_file_and_the_key(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # a flow-shaped payload with one key removed: used to raise a bare
            # KeyError: 'inner_flow_like'
            path = self._write(tmpdir, cloudpickle.dumps({
                "spline_data": {"num_dims": 0},
                "inner_flow_bytes": b"",
            }))
            with pytest.raises(ValueError) as excinfo:
                load_flow(path)

            message = str(excinfo.value)
            assert str(path) in message
            assert "inner_flow_like" in message

    def test_truncated_file_names_the_file(self):
        good = cloudpickle.dumps({
            "spline_data": {"num_dims": 0},
            "inner_flow_bytes": b"",
            "inner_flow_like": None,
        })
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write(tmpdir, good[: len(good) // 2])
            with pytest.raises(ValueError) as excinfo:
                load_flow(path)
            assert str(path) in str(excinfo.value)

    @pytest.mark.parametrize("payload", [b"", b"not a pickle at all\n" * 4])
    def test_garbage_file_names_the_file(self, payload):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write(tmpdir, payload)
            with pytest.raises(ValueError) as excinfo:
                load_flow(path)
            assert str(path) in str(excinfo.value)

    def test_missing_file_still_raises_filenotfound(self):
        """The deserialization guard must not swallow a plain missing file."""
        with tempfile.TemporaryDirectory() as tmpdir, pytest.raises(FileNotFoundError):
            load_flow(Path(tmpdir) / "absent.pkl")


class TestFormatStamps:
    """Saved payloads carry a format version and a kind tag."""

    def test_saved_payload_carries_version_and_kind(self, small_flow):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "flow.pkl"
            save_flow(small_flow, path)
            with open(path, "rb") as f:
                payload = cloudpickle.load(f)

            assert payload["format_version"] == FORMAT_VERSION
            assert payload["kind"] == FLOW_KIND

    def test_saved_payload_records_the_writer_version(self, small_flow):
        """A pickled model outlives its session; the payload must say which
        install wrote it, without that becoming a compatibility gate."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "flow.pkl"
            save_flow(small_flow, path)
            with open(path, "rb") as f:
                payload = cloudpickle.load(f)

            assert payload["writer_version"] == model_io._writer_version()
            assert isinstance(payload["writer_version"], str)
            assert payload["writer_version"]

            # dropping it must not change what load_flow does: older files
            # have no such key
            del payload["writer_version"]
            with open(path, "wb") as f:
                cloudpickle.dump(payload, f)
            assert isinstance(load_flow(path), Transformed)

    def test_unstamped_legacy_file_loads_as_version_1(self, small_flow):
        """Every .pkl written before the stamps existed lacks both keys, so a
        missing version must read as 1 and a missing kind must be inferred."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "flow.pkl"
            save_flow(small_flow, path)
            with open(path, "rb") as f:
                payload = cloudpickle.load(f)
            legacy = {k: v for k, v in payload.items()
                      if k not in ("format_version", "kind")}
            with open(path, "wb") as f:
                cloudpickle.dump(legacy, f)

            reloaded = load_flow(path)

            chain = np.random.randn(20, 2)
            np.testing.assert_allclose(
                log_prob(reloaded, chain), log_prob(small_flow, chain), rtol=1e-6
            )

    def test_fleet_file_points_at_the_fleet_loader(self):
        """A fleet payload used to die on `KeyError: 'spline_data'`."""
        fleet_payload = {
            "arch": (2, 8, 6),
            "params_bytes": b"",
            "xk": np.zeros((1, 2, 3)),
            "yk": np.zeros((1, 2, 3)),
            "dk": np.zeros((1, 2, 3)),
            "lo": np.zeros((1, 2)),
            "hi": np.ones((1, 2)),
        }
        for stamped in (False, True):
            payload = dict(fleet_payload)
            if stamped:
                payload["format_version"] = FORMAT_VERSION
                payload["kind"] = FLEET_KIND
            with tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir) / "fleet.pkl"
                with open(path, "wb") as f:
                    cloudpickle.dump(payload, f)
                with pytest.raises(ValueError, match="FleetFlow.load"):
                    load_flow(path)

    def test_future_format_version_is_rejected(self, small_flow):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "flow.pkl"
            save_flow(small_flow, path)
            with open(path, "rb") as f:
                payload = cloudpickle.load(f)
            payload["format_version"] = FORMAT_VERSION + 1
            with open(path, "wb") as f:
                cloudpickle.dump(payload, f)

            with pytest.raises(ValueError, match="save format version"):
                load_flow(path)

    def test_unrecognised_payload_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "junk.pkl"
            with open(path, "wb") as f:
                cloudpickle.dump({"something": "else"}, f)
            with pytest.raises(ValueError, match="'unknown' model"):
                load_flow(path)

            with open(path, "wb") as f:
                cloudpickle.dump([1, 2, 3], f)
            with pytest.raises(ValueError, match="does not hold a coppuccino model"):
                load_flow(path)


class TestPickleTrustBoundary:
    """load_flow deserializes with cloudpickle, which executes code from the
    file; the docstring must say so."""

    def test_load_flow_documents_the_trust_boundary(self):
        doc = load_flow.__doc__
        assert ".. warning::" in doc
        lowered = doc.lower()
        for phrase in ("pickle", "arbitrary", "trust"):
            assert phrase in lowered
