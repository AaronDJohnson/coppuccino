import pytest
import numpy as np
from flowjax.distributions import Transformed
from coppuccino.hdr import compute_injection_hdr, check_in_support


class TestCheckInSupport:
    """Test the check_in_support function."""

    def test_inside_support(self):
        """Test that parameters inside support return True."""
        samples = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        injection = np.array([1.0, 1.0])
        assert check_in_support(samples, injection) is True

    def test_outside_support(self):
        """Test that parameters outside support return False."""
        samples = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        injection = np.array([5.0, 5.0])
        assert check_in_support(samples, injection) is False

    def test_at_boundary(self):
        """Test that parameters at boundary return True."""
        samples = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        injection = np.array([0.0, 2.0])
        assert check_in_support(samples, injection) is True

    def test_partially_outside(self):
        """Test that parameters partially outside return False."""
        samples = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        injection = np.array([1.0, 5.0])  # 2nd param outside
        assert check_in_support(samples, injection) is False

    def test_wrong_ndim_raises(self):
        """Test that non-1D injection raises ValueError."""
        samples = np.array([[0.0, 0.0], [1.0, 1.0]])
        injection = np.array([[1.0, 1.0]])  # 2D
        with pytest.raises(ValueError, match="injection_params must be a 1D array"):
            check_in_support(samples, injection)

    def test_wrong_length_raises(self):
        """Test that mismatched length raises ValueError."""
        samples = np.array([[0.0, 0.0], [1.0, 1.0]])
        injection = np.array([1.0, 1.0, 1.0])  # 3 params vs 2
        with pytest.raises(ValueError, match="injection_params length must match"):
            check_in_support(samples, injection)


class TestComputeInjectionHDR:
    """Test the compute_injection_hdr function."""

    def test_basic_functionality_1d_injection(self):
        """Test basic HDR computation with 1D injection parameters."""
        np.random.seed(42)
        samples = np.random.randn(200, 2)
        injection_params = np.array([0.0, 0.0])

        hdr = compute_injection_hdr(
            samples,
            injection_params,
            num_samples=1000,
            max_epochs=5,
            patience=2
        )

        assert isinstance(hdr, np.ndarray)
        assert hdr.shape == (1,)
        assert 0.0 <= hdr[0] <= 1.0

    def test_multiple_injections(self):
        """Test HDR computation with multiple injection parameters."""
        np.random.seed(42)
        samples = np.random.randn(200, 2)
        injection_params = np.array([
            [0.0, 0.0],
            [0.5, 0.5],
            [1.0, 1.0]
        ])

        hdr = compute_injection_hdr(
            samples,
            injection_params,
            num_samples=1000,
            max_epochs=5,
            patience=2
        )

        assert isinstance(hdr, np.ndarray)
        assert hdr.shape == (3,)
        assert np.all((hdr >= 0.0) & (hdr <= 1.0))

    def test_return_flow_option(self):
        """Test that return_flow option works."""
        np.random.seed(42)
        samples = np.random.randn(200, 2)
        injection_params = np.array([0.0, 0.0])

        hdr, flow = compute_injection_hdr(
            samples,
            injection_params,
            num_samples=1000,
            return_flow=True,
            max_epochs=5,
            patience=2
        )

        assert isinstance(hdr, np.ndarray)
        assert isinstance(flow, Transformed)
        assert hdr.shape == (1,)

    def test_0d_injection_raises_error(self):
        """Test that 0D injection parameters raise an error."""
        np.random.seed(42)
        samples = np.random.randn(200, 2)
        injection_params = np.array(0.0)  # 0D array

        with pytest.raises(ValueError, match="injection_params must be at least 1D"):
            compute_injection_hdr(
                samples,
                injection_params,
                num_samples=1000,
                max_epochs=5,
                patience=2
            )

    def test_different_num_samples(self):
        """Test with different numbers of generated samples."""
        np.random.seed(42)
        samples = np.random.randn(200, 2)
        injection_params = np.array([0.0, 0.0])

        for num_samples in [500, 1000, 5000]:
            hdr = compute_injection_hdr(
                samples,
                injection_params,
                num_samples=num_samples,
                max_epochs=5,
                patience=2
            )
            assert isinstance(hdr, np.ndarray)
            assert hdr.shape == (1,)
            assert 0.0 <= hdr[0] <= 1.0

    def test_custom_nf_kwargs(self):
        """Test with custom normalizing flow parameters."""
        np.random.seed(42)
        samples = np.random.randn(200, 2)
        injection_params = np.array([0.0, 0.0])

        custom_kwargs = {
            'knots': 8,
            'patience': 3,
            'learning_rate': 1e-3,
            'max_epochs': 10,
        }

        hdr = compute_injection_hdr(
            samples,
            injection_params,
            num_samples=1000,
            **custom_kwargs
        )

        assert isinstance(hdr, np.ndarray)
        assert hdr.shape == (1,)
        assert 0.0 <= hdr[0] <= 1.0

    def test_injection_at_mode_has_lower_hdr_than_tail(self):
        """HDR credibility is the fraction of the fit *more probable* than the
        point, so a point at the mode scores near 0 and a point out in the tail
        scores higher. Assert that ordering — the module's core semantic
        property, which the old ``>= 0.0`` assertions did not test at all."""
        np.random.seed(42)
        # Fit to a 2D standard normal; the mode is [0, 0].
        samples = np.random.randn(3000, 2)

        # A point at the mode and a point well out in the (still in-support) tail.
        injection_mode = np.array([0.0, 0.0])
        injection_tail = np.array([2.0, 2.0])

        hdr_mode = compute_injection_hdr(
            samples, injection_mode, num_samples=10000,
            rng_seed=0, max_epochs=50, patience=15,
        )
        hdr_tail = compute_injection_hdr(
            samples, injection_tail, num_samples=10000,
            rng_seed=0, max_epochs=50, patience=15,
        )

        # Neither is the out-of-support 1.0 sentinel.
        assert hdr_mode[0] < hdr_tail[0]
        assert hdr_mode[0] < 0.5   # mode is among the most probable points
        assert hdr_tail[0] > 0.5   # tail point is among the least probable

    def test_1d_parameter_space(self):
        """Test with 1D parameter space."""
        np.random.seed(42)
        samples = np.random.randn(200, 1)
        injection_params = np.array([0.0])

        hdr = compute_injection_hdr(
            samples,
            injection_params,
            num_samples=1000,
            max_epochs=5,
            patience=2
        )

        assert isinstance(hdr, np.ndarray)
        assert hdr.shape == (1,)
        assert 0.0 <= hdr[0] <= 1.0

    def test_high_dimensional_parameter_space(self):
        """Test with higher dimensional parameter space."""
        np.random.seed(42)
        samples = np.random.randn(200, 5)
        injection_params = np.zeros(5)

        hdr = compute_injection_hdr(
            samples,
            injection_params,
            num_samples=1000,
            max_epochs=5,
            patience=2
        )

        assert isinstance(hdr, np.ndarray)
        assert hdr.shape == (1,)
        assert 0.0 <= hdr[0] <= 1.0

    def test_consistent_results_with_seed(self):
        """Test that results are consistent when using same random seed in flow."""
        np.random.seed(42)
        samples = np.random.randn(200, 2)
        injection_params = np.array([0.0, 0.0])

        hdr1 = compute_injection_hdr(
            samples,
            injection_params,
            num_samples=1000,
            rng_seed=123,
            max_epochs=5,
            patience=2
        )

        np.random.seed(42)  # Reset numpy seed
        samples2 = np.random.randn(200, 2)
        hdr2 = compute_injection_hdr(
            samples2,
            injection_params,
            num_samples=1000,
            rng_seed=123,
            max_epochs=5,
            patience=2
        )

        # Results should be very similar (allowing for numerical differences)
        np.testing.assert_array_almost_equal(hdr1, hdr2, decimal=2)

    def test_uniform_distribution(self):
        """Test HDR computation with uniformly distributed samples."""
        np.random.seed(42)
        samples = np.random.uniform(-1, 1, (200, 2))
        injection_params = np.array([0.0, 0.0])

        hdr = compute_injection_hdr(
            samples,
            injection_params,
            num_samples=1000,
            max_epochs=5,
            patience=2
        )

        assert isinstance(hdr, np.ndarray)
        assert hdr.shape == (1,)
        assert 0.0 <= hdr[0] <= 1.0

    def test_multiple_injections_different_locations(self):
        """Test HDR with multiple injection parameters at different locations."""
        np.random.seed(42)
        samples = np.random.randn(200, 2)

        # Create injections at various locations
        injection_params = np.array([
            [0.0, 0.0],   # At mean
            [-1.0, -1.0], # Below mean
            [1.0, 1.0],   # Above mean
            [0.0, 1.0],   # Mixed
            [-0.5, 0.5]   # Mixed
        ])

        hdr = compute_injection_hdr(
            samples,
            injection_params,
            num_samples=1000,
            max_epochs=5,
            patience=2
        )

        assert isinstance(hdr, np.ndarray)
        assert hdr.shape == (5,)
        assert np.all((hdr >= 0.0) & (hdr <= 1.0))
