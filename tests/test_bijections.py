import jax
import jax.numpy as jnp
import numpy as np
import pytest

from coppuccino import bijections, calibration, fleet
from coppuccino.bijections import (
    _POS_FLOOR,
    EmpiricalMarginalToGaussian,
    SmoothMarginalToGaussian,
    _process_array,
    _rqs_bulk,
    _rqs_knot_derivatives,
    make_empirical_cdf_spline,
    make_smooth_marginal,
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
        rng = np.random.default_rng(101)
        samples = rng.standard_normal(1000)
        cdf_vals, cdf_fn, quantile_fn, pdf_fn = make_empirical_cdf_spline(samples, num_points=100)

        # Check that CDF values are within valid range
        assert np.all(cdf_vals >= 1e-7)
        assert np.all(cdf_vals <= 1.0 - 1e-7)

        # Check that CDF is monotonically increasing
        assert len(cdf_vals) == 100

    def test_cdf_properties(self):
        """Test that CDF has correct properties."""
        rng = np.random.default_rng(102)
        samples = rng.standard_normal(1000)
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
        rng = np.random.default_rng(103)
        samples = rng.standard_normal(1000)
        _, cdf_fn, quantile_fn, pdf_fn = make_empirical_cdf_spline(samples, num_points=100)

        # Test PDF at various points
        test_points = np.linspace(np.min(samples), np.max(samples), 20)
        for x in test_points:
            pdf_val = pdf_fn(x)
            assert float(pdf_val) > 0

    def test_quantile_inverse_cdf(self):
        """Test that quantile function is inverse of CDF."""
        rng = np.random.default_rng(104)
        samples = rng.standard_normal(1000)
        _, cdf_fn, quantile_fn, pdf_fn = make_empirical_cdf_spline(samples, num_points=100)

        # Test at a few points
        test_points = np.linspace(np.min(samples) + 0.1, np.max(samples) - 0.1, 5)
        for x in test_points:
            u = cdf_fn(x)
            x_recovered = quantile_fn(u)
            assert np.abs(float(x_recovered) - float(x)) < 0.1

    def test_custom_min_eps(self):
        """Test that custom min_eps parameter works."""
        rng = np.random.default_rng(105)
        samples = rng.standard_normal(1000)
        min_eps = 1e-5
        cdf_vals, _, _, _ = make_empirical_cdf_spline(samples, num_points=100, min_eps=min_eps)

        # Check that CDF values respect min_eps
        assert np.all(cdf_vals >= min_eps)
        assert np.all(cdf_vals <= 1.0 - min_eps)

    def test_uniform_samples(self):
        """Test with uniformly distributed samples."""
        rng = np.random.default_rng(106)
        samples = rng.uniform(0, 1, 1000)
        cdf_vals, cdf_fn, quantile_fn, pdf_fn = make_empirical_cdf_spline(samples, num_points=100)

        # CDF should be approximately linear for uniform distribution
        assert len(cdf_vals) == 100


class TestTailExtension:
    """Test the tail_extension parameter of make_empirical_cdf_spline."""

    def test_tail_extension_produces_wider_range(self):
        """Test that tail extension allows sampling beyond data range."""
        rng = np.random.default_rng(42)
        samples = rng.standard_normal(1000)

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
        rng = np.random.default_rng(42)
        samples = rng.standard_normal(1000)

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
        """Without tail extension the CDF clips *continuously* to the boundary
        CDF value (no spurious jump to min_eps at the data edges, which used to
        map each dimension's extreme sample to a z ~= -5.2 outlier)."""
        rng = np.random.default_rng(42)
        samples = rng.standard_normal(1000)

        _, cdf_fn, _, _ = make_empirical_cdf_spline(samples, tail_extension=False)

        data_min, data_max = np.min(samples), np.max(samples)

        # Below the data range the CDF is flat and equals its value AT data_min.
        cdf_at_min = float(cdf_fn(jnp.array(data_min)))
        cdf_far_below = float(cdf_fn(jnp.array(data_min - 10.0)))
        assert cdf_far_below == pytest.approx(cdf_at_min, abs=1e-9)
        # It is the empirical ~1/N boundary value: small, but well above min_eps.
        assert 1e-7 < cdf_far_below < 0.05

        # Symmetric behavior at the upper edge.
        cdf_at_max = float(cdf_fn(jnp.array(data_max)))
        cdf_far_above = float(cdf_fn(jnp.array(data_max + 10.0)))
        assert cdf_far_above == pytest.approx(cdf_at_max, abs=1e-9)
        assert 0.95 < cdf_far_above < 1.0


class TestPriorBounds:
    """Test the prior_bounds parameter of make_empirical_cdf_spline."""

    def test_prior_bounds_extend_range(self):
        """Test that prior bounds extend the CDF grid."""
        rng = np.random.default_rng(42)
        samples = rng.standard_normal(1000)

        _, _, quantile_no_prior, _ = make_empirical_cdf_spline(samples)
        _, _, quantile_with_prior, _ = make_empirical_cdf_spline(
            samples, prior_low=-10.0, prior_high=10.0)

        # With prior bounds, extreme quantiles should reach toward prior edges
        low_no_prior = float(quantile_no_prior(jnp.array(1e-5)))
        low_with_prior = float(quantile_with_prior(jnp.array(1e-5)))

        assert low_with_prior < low_no_prior

    def test_prior_bounds_cdf_at_edges(self):
        """Test CDF values at prior bound edges."""
        rng = np.random.default_rng(42)
        samples = rng.standard_normal(1000)

        _, cdf_fn, _, _ = make_empirical_cdf_spline(
            samples, prior_low=-10.0, prior_high=10.0)

        # CDF at prior bounds should be close to min_eps / 1 - min_eps
        cdf_at_low = float(cdf_fn(jnp.array(-10.0)))
        cdf_at_high = float(cdf_fn(jnp.array(10.0)))

        assert cdf_at_low < 0.01
        assert cdf_at_high > 0.99

    def test_prior_bounds_within_data_range_ignored(self):
        """Test that prior bounds within data range don't affect grid."""
        rng = np.random.default_rng(42)
        samples = rng.standard_normal(1000)
        data_min = np.min(samples)

        # Prior bound within data range should not be added
        cdf_vals_no_prior, _, _, _ = make_empirical_cdf_spline(samples)
        cdf_vals_with_prior, _, _, _ = make_empirical_cdf_spline(
            samples, prior_low=data_min + 0.1)

        # Should have same grid size since prior_low > xg[0]
        assert len(cdf_vals_no_prior) == len(cdf_vals_with_prior)

    @pytest.mark.parametrize("marginal", ["rqs", "pchip"])
    def test_quantile_never_lands_on_a_prior_wall(self, marginal):
        """The legacy marginals must not emit exact-at-wall sample atoms.

        ``quantile_fn`` saturated at ``data_min`` / ``data_max`` themselves,
        which under prior bounds *are* the walls: 4 of 20000 draws from a
        fitted U(0, 1) flow with prior_bounds [0, 1] came back as exactly 0.0
        for both families. The saturation points are now one representable
        notch inside, which is invisible to every CDF and density value here.
        """
        rng = np.random.default_rng(3)
        samples = rng.uniform(0.0, 1.0, 2000)
        _, cdf_fn, quantile_fn, pdf_fn = make_empirical_cdf_spline(
            samples, prior_low=0.0, prior_high=1.0, marginal=marginal)

        u = jnp.asarray(np.concatenate([[0.0, 1e-30, 1e-12, 1e-8],
                                        np.linspace(1e-3, 1 - 1e-3, 999),
                                        [1 - 1e-8, 1 - 1e-12, 1.0]]))
        x = np.asarray(quantile_fn(u))
        assert np.all(x > 0.0) and np.all(x < 1.0), (x.min(), x.max())
        assert np.all(np.diff(x) >= 0.0)
        # still a faithful inverse in the bulk. "pchip" builds the CDF and its
        # inverse as two independent splines, so it only inverts to ~1e-3
        # (measured 1.2e-3 here); "rqs" is one parameterization and exact.
        mid = np.asarray(u)[4:-3]
        tol = 1e-6 if marginal == "rqs" else 5e-3
        assert np.max(np.abs(np.asarray(cdf_fn(x[4:-3])) - mid)) < tol
        assert np.all(np.asarray(pdf_fn(x)) > 0.0)

    @pytest.mark.parametrize("marginal", ["rqs", "pchip"])
    def test_the_boundary_bins_carry_the_mass_they_should(self, marginal):
        """An appended prior bound must leave its boundary bin real mass.

        The empirical CDF is ``counts / n``, so the largest sample sits at
        exactly 1.0. With ``prior_high`` appended at 1.0 too, both knots
        clipped to ``1 - min_eps`` and the tie bump then separated them by
        ``min_eps / (10 K)``: the whole interval from the largest sample to
        the upper bound carried 5e-11 of CDF instead of the ``0.5 / n`` its
        plotting position asks for, the spline slope there collapsed with it,
        and ``log_prob`` at the upper wall of a U(0, 1)^2 fit (n = 1500, 80
        epochs) read -14.6 to -11.6 over four seeds against a truth of 0 --
        while the *lower* wall, whose smallest sample sits at ``1 / n`` and
        so never tied, read +2.0 to +4.5. It also mapped that one extreme
        training row to ``z = ndtri(1 - min_eps) = 5.2``, the spurious
        outlier the no-prior-bounds path already guards against.
        """
        rng = np.random.default_rng(0)
        x = rng.uniform(0.0, 1.0, 1500)
        _, cdf_fn, _, pdf_fn = make_empirical_cdf_spline(
            x, prior_low=0.0, prior_high=1.0, marginal=marginal)
        lo_bin = (float(cdf_fn(jnp.array(float(x.min()))))
                  - float(cdf_fn(jnp.array(0.0))))
        hi_bin = (float(cdf_fn(jnp.array(1.0)))
                  - float(cdf_fn(jnp.array(float(x.max())))))
        # 1/n at the bottom against 0.5/n at the top: exactly 0.5, and the
        # ratio was 7.5e-8 before the top knot was held back
        assert 0.3 < hi_bin / lo_bin < 1.0, (lo_bin, hi_bin)
        # and the density just inside the upper bound is of ordinary size
        # rather than the 2.5e-7 the collapsed slope produced
        assert float(pdf_fn(jnp.array(1.0))) > 0.01


class TestEmpiricalMarginalToGaussian:
    """Test the EmpiricalMarginalToGaussian bijection."""

    def test_initialization(self):
        """Test that the bijection can be initialized."""
        rng = np.random.default_rng(201)
        samples = rng.standard_normal(1000)
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
        rng = np.random.default_rng(202)
        samples = rng.standard_normal(1000)
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
        rng = np.random.default_rng(203)
        samples = rng.standard_normal(1000)
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
        rng = np.random.default_rng(204)
        samples = rng.standard_normal(1000)
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
        rng = np.random.default_rng(205)
        samples = rng.standard_normal(1000)
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
        rng = np.random.default_rng(206)
        samples = rng.standard_normal(1000)
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
        rng = np.random.default_rng(207)
        samples = rng.standard_normal(1000)
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

    def test_log_det_matches_finite_difference(self):
        """The Jacobian log-det must equal the numerical derivative of the
        transform. Guards against a sign error or constant offset in the
        log-det (which directly corrupts log_prob) that the isfinite checks
        cannot catch."""
        rng = np.random.default_rng(0)
        samples = rng.standard_normal(3000)
        _, cdf_fn, quantile_fn, pdf_fn = make_empirical_cdf_spline(
            samples, marginal="rqs")
        bij = EmpiricalMarginalToGaussian(
            samples=samples, cdf_fn=cdf_fn, quantile_fn=quantile_fn,
            pdf_fn=pdf_fn, marginal="rqs")

        h = 1e-5
        # inverse_and_log_det: x -> z, returned log_det == log|dz/dx|.
        for xv in [-1.2, -0.4, 0.0, 0.5, 1.1]:
            z_hi, _ = bij.inverse_and_log_det(jnp.array(xv + h))
            z_lo, _ = bij.inverse_and_log_det(jnp.array(xv - h))
            fd = float((z_hi - z_lo) / (2 * h))
            _, log_det = bij.inverse_and_log_det(jnp.array(xv))
            assert float(log_det) == pytest.approx(float(np.log(abs(fd))), abs=2e-3)

        # transform_and_log_det: z -> x, returned log_det == log|dx/dz|.
        for zv in [-1.0, -0.3, 0.0, 0.4, 1.0]:
            x_hi, _ = bij.transform_and_log_det(jnp.array(zv + h))
            x_lo, _ = bij.transform_and_log_det(jnp.array(zv - h))
            fd = float((x_hi - x_lo) / (2 * h))
            _, log_det = bij.transform_and_log_det(jnp.array(zv))
            assert float(log_det) == pytest.approx(float(np.log(abs(fd))), abs=2e-3)


class TestGPDTailModel:
    """Test the tail_model='gpd' peaks-over-threshold path."""

    def test_gpd_recovers_shape_parameter(self):
        """_fit_gpd should recover the true ξ on a known GPD sample."""
        from scipy.stats import genpareto

        from coppuccino.bijections import _fit_gpd
        rng = np.random.default_rng(0)
        excesses = genpareto.rvs(0.3, scale=1.0, size=3000, random_state=rng)
        xi, sigma = _fit_gpd(excesses)
        assert abs(xi - 0.3) < 0.05
        assert abs(sigma - 1.0) < 0.1

    def test_gpd_basic_roundtrip(self):
        """CDF/quantile/PDF are finite and consistent with tail_model='gpd'."""
        rng = np.random.default_rng(42)
        samples = rng.standard_t(3, size=5000)
        _, cdf_fn, quantile_fn, pdf_fn = make_empirical_cdf_spline(
            samples, tail_model="gpd"
        )
        # Middle round-trip
        x = jnp.array(0.5)
        u = cdf_fn(x)
        x_back = quantile_fn(u)
        assert abs(float(x_back) - 0.5) < 0.1
        # Tails are finite, monotone, positive density
        for xv in [-50.0, -10.0, 10.0, 50.0]:
            xj = jnp.array(xv)
            assert jnp.isfinite(cdf_fn(xj))
            assert jnp.isfinite(quantile_fn(jnp.array(1e-5)))
            assert float(pdf_fn(xj)) > 0
        # CDF is monotone
        xs = jnp.linspace(-30.0, 30.0, 200)
        cs = jax.vmap(cdf_fn)(xs)
        assert bool(jnp.all(jnp.diff(cs) >= -1e-10))

    def test_gpd_nll_close_to_true_on_heavy_tails(self):
        """GPD tails on Student-t data should produce held-out NLL within
        a small additive gap of the analytic NLL — confirms the tail model
        is at least as accurate as the empirical estimate."""
        from scipy.stats import t as student_t
        rng = np.random.default_rng(1)
        train = student_t.rvs(3, size=8000, random_state=rng)
        test = student_t.rvs(3, size=8000, random_state=rng)

        _, _, _, pdf = make_empirical_cdf_spline(train, tail_model="gpd")
        nll_gpd = -float(jnp.mean(jnp.log(jax.vmap(pdf)(jnp.asarray(test)))))
        true_nll = -float(np.mean(student_t.logpdf(test, 3)))

        # Empirical estimator floor is roughly 1/N — give 0.05 headroom.
        assert nll_gpd - true_nll < 0.05

    def test_gpd_falls_back_on_constant_data(self):
        """Degenerate samples shouldn't crash the GPD path."""
        rng = np.random.default_rng(301)
        samples = np.zeros(500) + 1e-10 * rng.standard_normal(500)
        # Just needs to not raise:
        make_empirical_cdf_spline(samples, tail_model="gpd")


class TestRQSMarginal:
    """Test the marginal='rqs' (rational-quadratic spline) bulk interpolant."""

    def test_invalid_marginal_raises(self):
        """An unknown marginal family should raise a clear error."""
        rng = np.random.default_rng(302)
        samples = rng.standard_normal(500)
        with pytest.raises(ValueError, match="marginal must be"):
            make_empirical_cdf_spline(samples, marginal="bogus")

    def test_rqs_roundtrip_machine_precision(self):
        """RQS forward/inverse come from one parameterization, so the round-trip
        is exact to ~machine precision — and far tighter than the PCHIP pair."""
        rng = np.random.default_rng(0)
        samples = rng.standard_normal(2000)
        x = jnp.asarray(np.quantile(samples, np.linspace(0.05, 0.95, 60)))

        _, cdf_r, q_r, _ = make_empirical_cdf_spline(samples, marginal="rqs")
        _, cdf_p, q_p, _ = make_empirical_cdf_spline(samples, marginal="pchip")

        rt_rqs = float(jnp.max(jnp.abs(q_r(cdf_r(x)) - x)))
        rt_pchip = float(jnp.max(jnp.abs(q_p(cdf_p(x)) - x)))

        assert rt_rqs < 1e-9
        assert rt_rqs < rt_pchip  # RQS is the self-consistent one

    def test_rqs_cdf_monotone_and_pdf_positive(self):
        """RQS CDF is monotone increasing and its density is strictly positive."""
        rng = np.random.default_rng(1)
        samples = rng.standard_normal(2000)
        _, cdf_fn, _, pdf_fn = make_empirical_cdf_spline(samples, marginal="rqs")

        xs = jnp.linspace(float(np.min(samples)), float(np.max(samples)), 300)
        cs = jax.vmap(cdf_fn)(xs)
        assert bool(jnp.all(jnp.diff(cs) >= -1e-10))
        assert bool(jnp.all(jax.vmap(pdf_fn)(xs) > 0))

    def test_rqs_cdf_bounds(self):
        """CDF stays within (min_eps, 1 - min_eps) and clips *continuously* to the
        boundary CDF value outside the data range."""
        rng = np.random.default_rng(2)
        samples = rng.standard_normal(2000)
        _, cdf_fn, _, _ = make_empirical_cdf_spline(samples, marginal="rqs")
        lo = float(cdf_fn(jnp.array(np.min(samples) - 10.0)))
        hi = float(cdf_fn(jnp.array(np.max(samples) + 10.0)))
        # Continuous with the boundary value, and strictly inside (min_eps, 1-min_eps).
        assert lo == pytest.approx(float(cdf_fn(jnp.array(np.min(samples)))), abs=1e-9)
        assert hi == pytest.approx(float(cdf_fn(jnp.array(np.max(samples)))), abs=1e-9)
        assert 1e-7 <= lo < 0.05
        assert 0.95 < hi <= 1.0 - 1e-7

    @pytest.mark.parametrize("kwargs", [
        {"tail_extension": True},
        {"prior_low": -8.0, "prior_high": 8.0},
        {"tail_model": "gpd"},
    ])
    def test_rqs_feature_parity(self, kwargs):
        """RQS shares the tail / prior-bound closures, so tail_extension,
        prior_bounds and GPD tails all work with it (finite, monotone)."""
        rng = np.random.default_rng(3)
        samples = rng.standard_normal(3000)
        _, cdf_fn, quantile_fn, pdf_fn = make_empirical_cdf_spline(
            samples, marginal="rqs", **kwargs)
        xs = jnp.linspace(-6.0, 6.0, 300)
        cs = jax.vmap(cdf_fn)(xs)
        assert bool(jnp.all(jnp.isfinite(cs)))
        assert bool(jnp.all(jnp.diff(cs) >= -1e-9))
        assert jnp.isfinite(quantile_fn(jnp.array(1e-4)))
        assert jnp.isfinite(quantile_fn(jnp.array(1.0 - 1e-4)))
        assert float(pdf_fn(jnp.array(0.0))) > 0

    def test_rqs_bijection_inverse_consistency(self):
        """Through EmpiricalMarginalToGaussian, the RQS marginal round-trips
        x -> z -> x' essentially exactly (the motivation for adding it)."""
        rng = np.random.default_rng(4)
        samples = rng.standard_normal(2000)
        _, cdf_fn, quantile_fn, pdf_fn = make_empirical_cdf_spline(samples, marginal="rqs")
        bij = EmpiricalMarginalToGaussian(
            samples=samples, cdf_fn=cdf_fn, quantile_fn=quantile_fn,
            pdf_fn=pdf_fn, marginal="rqs")
        for xv in [-1.0, -0.3, 0.0, 0.4, 1.2]:
            z, _ = bij.inverse_and_log_det(jnp.array(xv))
            x_back, _ = bij.transform_and_log_det(z)
            assert jnp.abs(x_back - xv) < 1e-6


class TestEdgeKnotDensity:
    """PCHIP's one-sided endpoint rule returns exactly 0 on empirical-quantile
    grids, which floors the outermost knot's density at ``_POS_FLOOR``."""

    def test_density_at_extreme_sample_is_not_collapsed(self):
        """With no prior bounds the outermost knot *is* each dimension's most
        extreme sample, so a zeroed endpoint slope puts a multi-nat hole in the
        density exactly where the tail should be describing the data."""
        from scipy.stats import norm
        rng = np.random.default_rng(7)
        samples = rng.standard_normal(2000)
        _, _, _, pdf_fn = make_empirical_cdf_spline(samples, marginal="rqs")

        # just inside the knot: pdf_fn reports the "outside the data range"
        # floor exactly at it, so the interpolant is probed a hair inward
        probe = float(np.min(samples)) + 1e-6
        err = float(np.log(float(pdf_fn(jnp.array(probe)))) - np.log(norm.pdf(probe)))
        assert abs(err) < 3.0, f"log-density error at the extreme sample: {err}"

    def test_density_error_at_extreme_sample_across_seeds(self):
        """PCHIP zeroes that endpoint on most empirical-quantile grids, not
        all, so pin the typical case rather than a single draw.

        Thresholds sit just above the measured spread on these eight seeds
        (median 0.807, max 1.446 nats); over 40 other seeds the same protocol
        gives median 1.51 and max 2.64, so treat these as pins on *this*
        sample of seeds, not as a distributional claim.
        """
        from scipy.stats import norm
        errors = []
        for seed in range(8):
            rng = np.random.default_rng(20 + seed)
            samples = rng.standard_normal(2000)
            _, _, _, pdf_fn = make_empirical_cdf_spline(samples, marginal="rqs")
            probe = float(np.min(samples)) + 1e-6
            errors.append(abs(float(np.log(float(pdf_fn(jnp.array(probe))))
                                    - np.log(norm.pdf(probe)))))
        assert np.median(errors) < 1.2
        assert max(errors) < 2.0

    @pytest.mark.parametrize("marginal", ["rqs", "pchip"])
    def test_density_at_the_endpoint_itself_is_the_interpolant(self, marginal):
        """The endpoints are training rows, and the density is defined there.

        ``pdf_fn``'s no-tail branch tested ``x <= data_min | x >= data_max``
        and returned 0.0, floored to ``min_eps``: 1.0e-7 exactly at
        ``data_min`` against 1.845e-4 one picometre inside it, a 7.6-nat hole
        one ULP wide at a point that is in the training set. Over 20 seeds of
        N(0, 1) with n = 3000 the error against the true density at
        ``data_min`` was median 9.26 / max 10.24 nats before and median 1.30 /
        max 2.00 after, for ``marginal="rqs"``. ``"pchip"`` gains only the
        continuity: its endpoint slope is PCHIP's own one-sided estimate,
        which cannot be overridden and is 0 on these grids, so both sides of
        the endpoint sit on the ``min_eps`` floor together.
        """
        from scipy.stats import norm
        errors = []
        for seed in range(20):
            samples = np.random.default_rng(seed).standard_normal(3000)
            _, _, _, pdf_fn = make_empirical_cdf_spline(samples,
                                                        marginal=marginal)
            for edge, step in ((float(np.min(samples)), +1e-12),
                               (float(np.max(samples)), -1e-12)):
                at = float(pdf_fn(jnp.array(edge)))
                just_inside = float(pdf_fn(jnp.array(edge + step)))
                assert at == pytest.approx(just_inside, rel=1e-6)
                if marginal == "rqs":
                    assert at > 1e-6, (edge, at)   # not the min_eps floor
                errors.append(abs(np.log(at) - np.log(float(norm.pdf(edge)))))
        if marginal == "rqs":
            assert np.median(errors) < 2.0, np.median(errors)
            assert max(errors) < 4.0, max(errors)


class TestTailExtensionMonotonicity:
    """quantile_fn must stay a valid inverse CDF on heavy-tailed data."""

    U_GRID = np.concatenate([
        np.logspace(-7, -1, 400),
        np.linspace(0.1, 0.9, 200),
        1.0 - np.logspace(-1, -7, 400),
    ])

    @pytest.mark.parametrize("dist", ["normal_with_outliers", "t2", "lognormal",
                                      "exponential"])
    def test_quantile_is_nondecreasing(self, dist):
        """A min_eps floor on the ndtri argument pins each Gaussian tail at
        mu -/+ 5.199 sigma. Past that the tail branch returns a value *inside*
        the data range, so quantile_fn folds back on itself at the join and the
        marginal stops being invertible."""
        from scipy.stats import expon, lognorm
        from scipy.stats import t as student_t
        rng = np.random.default_rng(12)
        if dist == "normal_with_outliers":
            samples = np.concatenate([rng.standard_normal(4990),
                                      rng.standard_normal(10) * 3.0 - 12.0])
        elif dist == "t2":
            samples = student_t.rvs(2, size=5000, random_state=rng)
        elif dist == "lognormal":
            samples = lognorm.rvs(1.0, size=5000, random_state=rng)
        else:
            samples = expon.rvs(size=5000, random_state=rng)

        _, _, quantile_fn, _ = make_empirical_cdf_spline(
            samples, tail_extension=True, marginal="rqs")
        x = np.asarray(jax.vmap(quantile_fn)(jnp.asarray(self.U_GRID)))
        steps = np.diff(x)
        assert np.all(steps >= -1e-9), (
            f"{dist}: {int((steps < -1e-9).sum())} decreasing steps, worst "
            f"{steps.min()}")

    def test_bijection_round_trip_past_five_sigma(self):
        """The pinned tail collapses a whole band of z onto one x, so the
        round-trip z -> x -> z' silently loses the input."""
        rng = np.random.default_rng(13)
        samples = np.concatenate([rng.standard_normal(4990),
                                  rng.standard_normal(10) * 3.0 - 12.0])
        _, cdf_fn, quantile_fn, pdf_fn = make_empirical_cdf_spline(
            samples, tail_extension=True, marginal="rqs")
        bij = EmpiricalMarginalToGaussian(
            samples=samples, cdf_fn=cdf_fn, quantile_fn=quantile_fn,
            pdf_fn=pdf_fn, tail_extension=True, marginal="rqs")

        xs = []
        for zv in [-5.0, -4.5, -4.0, -3.0]:
            x, _ = bij.transform_and_log_det(jnp.array(zv))
            z_back, _ = bij.inverse_and_log_det(x)
            assert float(z_back) == pytest.approx(zv, abs=1e-6)
            xs.append(float(x))
        assert np.all(np.diff(xs) > 0)


class TestGPDJoinContinuity:
    """The GPD threshold is an interior point, so the density must not step."""

    def test_density_is_continuous_across_both_joins(self):
        from scipy.stats import t as student_t
        rng = np.random.default_rng(11)
        samples = student_t.rvs(3, size=5000, random_state=rng)
        _, _, _, pdf_fn = make_empirical_cdf_spline(
            samples, tail_model="gpd", marginal="rqs")

        h = 1e-9
        for u0 in (float(np.quantile(samples, 0.05)),
                   float(np.quantile(samples, 0.95))):
            inside = float(pdf_fn(jnp.array(u0 - h)))
            outside = float(pdf_fn(jnp.array(u0 + h)))
            jump = abs(np.log(inside) - np.log(outside))
            assert jump < 1e-3, f"|log density jump| at {u0}: {jump}"

    def test_cdf_stays_monotone_with_the_matched_join(self):
        """Overriding the bulk endpoint slope must not cost monotonicity."""
        from scipy.stats import t as student_t
        rng = np.random.default_rng(14)
        samples = student_t.rvs(3, size=5000, random_state=rng)
        _, cdf_fn, _, pdf_fn = make_empirical_cdf_spline(
            samples, tail_model="gpd", marginal="rqs")
        xs = jnp.linspace(-60.0, 60.0, 2000)
        cs = jax.vmap(cdf_fn)(xs)
        assert bool(jnp.all(jnp.diff(cs) >= -1e-12))
        assert bool(jnp.all(jax.vmap(pdf_fn)(xs) > 0))


class TestReconstructionMetadataDefaults:
    """The recorded family must match the closures the builder actually made."""

    def test_class_marginal_default_matches_builder_default(self):
        import inspect
        builder_default = inspect.signature(
            make_empirical_cdf_spline).parameters["marginal"].default
        rng = np.random.default_rng(9)
        samples = rng.standard_normal(500)
        _, cdf_fn, quantile_fn, pdf_fn = make_empirical_cdf_spline(samples)
        bij = EmpiricalMarginalToGaussian(
            samples=samples, cdf_fn=cdf_fn, quantile_fn=quantile_fn,
            pdf_fn=pdf_fn)
        assert bij.marginal == builder_default


class TestSmoothMarginalBoundaryAtoms:
    """Samples sitting exactly on a prior bound -- the documented recommended
    use of ``prior_bounds``, and what a clipping/reflecting MCMC kernel
    produces."""

    @staticmethod
    def _chain_with_upper_atom(n=1500, lo=-1.0, hi=1.0):
        rng = np.random.default_rng(0)
        return np.clip(rng.normal(0.9, 0.25, n), lo, hi), lo, hi

    @staticmethod
    def _chain_with_lower_atom(n=1500, lo=-1.0, hi=1.0):
        rng = np.random.default_rng(0)
        return np.clip(rng.normal(-0.9, 0.25, n), lo, hi), lo, hi

    @staticmethod
    def _log_q(z_of_theta, log_dtheta_dz):
        def log_q(theta):
            z = z_of_theta(jnp.asarray(theta))
            return float(-0.5 * z ** 2 - 0.5 * np.log(2 * np.pi)
                         - log_dtheta_dz(z))
        return log_q

    @pytest.mark.parametrize("side,max_excess", [("lower", 12.0), ("upper", 8.0)])
    def test_bound_row_reports_a_bounded_wall_density(self, side, max_excess):
        """A row on the bound must report a density of interior magnitude.

        The constructor insets a wall atom before building the knots, so the
        atom's mass becomes a resolvable shoulder and the tail below it
        carries a finite wall density. Both halves matter: without the
        finite-density tail the wall value runs off to +-infinity (the
        exponential wall approach reported -7.6e5 nats here), and without the
        inset the shoulder is only ``eps0`` = 1e-9 of the span wide, so the
        same mass is reported over a 1e9-times-narrower window.

        Measured excess of the at-bound log-density over the interior median,
        8 seeds (``default_rng(0..7)``) of ``clip(N(-+0.9, 0.25), -1, 1)``
        (33% of 1500 rows on the bound): 6.34 to 11.03 nats at the lower wall
        and -0.66 to 3.84 at the upper one. Reverting the inset to the raw
        ``eps0`` margin spreads the same atom over a window 1e9 times
        narrower, which pushes both far higher. The two walls differ because
        the Hazen tie count puts a tie block's plotting position at the top
        of the block, so a lower-wall atom lands in the tail (with its
        outermost knot dragged in to z = -0.4 or so, and the wall therefore
        reached at |z| = 1.82 to 1.96) and an upper-wall one in the outermost bulk bin
        (knot at the usual ndtri(0.5/n), wall at |z| = 4.90). The lower
        branch had no test at all before this one -- deleting it left the
        suite green.
        """
        x, lo, hi = (self._chain_with_lower_atom() if side == "lower"
                     else self._chain_with_upper_atom())
        bound = lo if side == "lower" else hi
        on_bound = x <= lo if side == "lower" else x >= hi
        assert on_bound.sum() > 0.2 * len(x)
        _, z_of_theta, log_dtheta_dz, _ = make_smooth_marginal(
            x, prior_low=lo, prior_high=hi)
        log_q = self._log_q(z_of_theta, log_dtheta_dz)

        z_bound = float(z_of_theta(jnp.array(bound)))
        assert abs(z_bound) < 10.0, f"z at the bound is {z_bound}"
        interior = float(np.median(x[~on_bound]))
        assert log_q(bound) - log_q(interior) < max_excess, (
            log_q(bound), log_q(interior))
        assert log_q(bound) - log_q(interior) > -5.0, (
            log_q(bound), log_q(interior))
        # and it is a genuine limit, not a clamp: the density converges as the
        # query approaches the bound instead of stepping at a window edge
        sign = 1.0 if side == "lower" else -1.0
        ladder = np.array([log_q(bound + sign * 10.0 ** -k)
                           for k in range(6, 13)])
        assert np.max(np.abs(np.diff(ladder))) < 1.0, ladder
        assert abs(ladder[-1] - log_q(bound)) < 0.1, ladder

    def test_bound_row_and_nan_free_log_det(self):
        """Both directions stay finite and mutually consistent at the bound."""
        x, lo, hi = self._chain_with_upper_atom()
        theta_of_z, z_of_theta, log_dtheta_dz, clamps = make_smooth_marginal(
            x, prior_low=lo, prior_high=hi)
        bij = SmoothMarginalToGaussian(
            samples=x, theta_of_z=theta_of_z, z_of_theta=z_of_theta,
            log_dtheta_dz_fn=log_dtheta_dz, prior_low=lo, prior_high=hi,
            x_clamp_lo=clamps[0], x_clamp_hi=clamps[1])

        # the tail saturates one _WALL_Z_MARGIN below its outermost knot, so
        # the last sliver of the support does share one z; probe outside it
        inside = hi - 10.0 * (hi - clamps[1])
        zs = []
        for xv in (hi, inside, hi + 1.0):
            z, log_det = bij.inverse_and_log_det(jnp.array(xv))
            assert bool(jnp.isfinite(z)) and bool(jnp.isfinite(log_det))
            assert abs(float(z)) < 10.0
            zs.append(float(z))
        # a value beyond the bound saturates at the wall, not extrapolated
        assert zs[2] == pytest.approx(zs[0], abs=1e-12)
        # and the saturated sliver is exactly the one the builder reports:
        # everything above ``x_clamp_hi`` shares the bound's z, everything
        # below it is resolved separately
        assert float(bij.inverse_and_log_det(jnp.array(clamps[1]))[0]) == \
            pytest.approx(zs[0], abs=1e-12)
        assert zs[1] < zs[0]

    def test_unclamped_forward_draws_are_never_absorbed(self):
        """``sample`` redraws exactly the rows that land on ``x_clamp_hi``, so
        every x the forward map emits *below* the clamp must still invert
        through the tail. Otherwise a kept draw would silently report the knot
        density instead of the density it was drawn from."""
        x, lo, hi = self._chain_with_upper_atom()
        theta_of_z, z_of_theta, _, (_, clamp_hi) = make_smooth_marginal(
            x, prior_low=lo, prior_high=hi)
        assert float(theta_of_z(jnp.array(1e6))) == clamp_hi

        zs = np.linspace(4.0, 2000.0, 400)
        xs = np.asarray(theta_of_z(jnp.asarray(zs)))
        kept = xs < clamp_hi
        assert kept.any()
        z_back = np.asarray(z_of_theta(jnp.asarray(xs[kept])))
        assert np.allclose(z_back, zs[kept], rtol=1e-6)

    def test_unbounded_and_atom_free_sides_are_untouched(self):
        """Absorption is tied to an atom actually being relocated: a bounded
        side with no wall atom keeps its analytic tail over the whole region
        the tail resolves, which is everything up to ``x_clamp_hi``.

        The saturated sliver beyond it is the ``_WALL_Z_MARGIN`` trade, not
        absorption: it is set by the outermost knot's z, not by any atom, and
        it is the same fraction of the knot-to-bound distance on a chain with
        no atom at all. Here it is 9.4e-5 wide against a 6.6e-2 knot-to-bound
        gap (0.14% of it), so probes an order of magnitude inside it are
        still resolved separately.
        """
        rng = np.random.default_rng(5)
        x = rng.normal(0.0, 0.3, 1500)
        assert x.min() > -1.0 + 1e-6 and x.max() < 1.0 - 1e-6  # no wall atom
        _, z_of_theta, _, clamps = make_smooth_marginal(x, prior_low=-1.0,
                                                        prior_high=1.0)
        sliver = 1.0 - clamps[1]
        assert sliver < 0.01 * (1.0 - x.max())
        # strictly monotone right up to the saturation point
        probes = jnp.array([1.0 - 1e-2, 1.0 - 1e-3,
                            1.0 - 10.0 * sliver, 1.0 - sliver])
        zs = np.asarray(z_of_theta(probes))
        assert np.all(np.diff(zs) > 0)


class TestWallDepthTracksTheTrainingEdge:
    """A bounded tail stops one margin past its own outermost knot.

    That knot sits on the most extreme sample, at Hazen position
    ``0.5 / n``, so it is exactly the deepest ``|z|`` the copula flow
    downstream was ever trained on in this dimension; everything past it is
    extrapolation, and ``_WALL_Z_MARGIN`` is how much of it the marginal is
    willing to hand over. No constant can express that, because
    ``|ndtri(0.5 / n)|`` runs from 2.94 at n = 300 to 4.42 at n = 100000: a
    constant is either far past the edge on a small chain or inside it on a
    large one, and once it is inside, the half-gap guard takes over and the
    reported depth drifts with n regardless of the number named (the 8.0 this
    replaced sat 4.6 past the edge at n = 1500; a 4.0 would have been 0.6
    past it there and *inside* it by n = 6000).
    """

    @pytest.mark.parametrize("n", [300, 1500, 6000, 100000])
    def test_the_wall_z_is_one_margin_past_the_outermost_knot(self, n):
        from scipy.special import ndtri

        rng = np.random.default_rng(0)
        x = rng.uniform(0.0, 1.0, n)
        _, z_of_theta, _, clamps = make_smooth_marginal(
            x, prior_low=0.0, prior_high=1.0)
        edge = float(ndtri(0.5 / n))  # the outermost Hazen plotting position
        assert float(z_of_theta(jnp.array(0.0))) == pytest.approx(
            edge - bijections._WALL_Z_MARGIN, abs=1e-6)
        # and the wall resolution therefore *improves* with more data rather
        # than staying pinned: 2.8e-3 of the knot-to-bound gap at n = 300
        # down to 3.3e-4 at n = 100000
        assert clamps[0] / float(x.min()) < 3e-3

    def test_more_data_resolves_further_into_the_wall(self):
        fracs = []
        for n in (300, 100000):
            x = np.random.default_rng(0).uniform(0.0, 1.0, n)
            _, _, _, clamps = make_smooth_marginal(x, prior_low=0.0,
                                                   prior_high=1.0)
            fracs.append(clamps[0] / float(x.min()))
        assert fracs[1] < 0.25 * fracs[0], fracs


class TestWallShapeJoinPin:
    """The wall shape's ``S'(1) = beta`` pin closes the bulk/tail join.

    :class:`bijections._MarginalTail` presents that pin as what makes the
    bounded tail's density continuous with the bulk where they meet. Nothing
    was checking it: replacing the fitted ``beta`` with 1.0 in ``_make_tail``
    -- deleting the pin, so the wall shape becomes the identity and the tail's
    join density is off by exactly that factor -- left every other test in
    this file passing.
    """

    LO, HI = 0.0, 1.0

    @staticmethod
    def _join_jumps(x, lo, hi):
        """``log dtheta/dz`` across each bulk/tail join, tail minus bulk."""
        _, z_of_theta, log_dtheta_dz, _ = make_smooth_marginal(
            x, prior_low=lo, prior_high=hi)
        out = []
        for theta in (float(np.min(x)), float(np.max(x))):
            z_join = float(z_of_theta(jnp.array(theta)))
            out.append(float(log_dtheta_dz(jnp.array(z_join - 1e-7)))
                       - float(log_dtheta_dz(jnp.array(z_join + 1e-7))))
        return out

    def test_the_density_is_continuous_across_both_wall_joins(self):
        rng = np.random.default_rng(0)
        x = rng.beta(3.0, 3.0, 1500)
        for jump in self._join_jumps(x, self.LO, self.HI):
            assert abs(jump) < 1e-3, jump

    def test_dropping_the_pin_opens_both_joins(self, monkeypatch):
        """The fixture has teeth: without the pin these joins step by nats.

        Beta(3, 3) puts the outermost knots where the density is falling
        steeply, so the join-side derivative the pin carries is far from 1 on
        both sides and the mutant cannot pass by luck.
        """
        real = bijections._make_tail

        def unpinned(*args, **kwargs):
            tail = real(*args, **kwargs)
            if not tail.wall:
                return tail
            return tail._replace(beta=1.0, shape_d=jnp.asarray([1.0, 1.0]))

        monkeypatch.setattr(bijections, "_make_tail", unpinned)
        rng = np.random.default_rng(0)
        x = rng.beta(3.0, 3.0, 1500)
        jumps = self._join_jumps(x, self.LO, self.HI)
        assert min(abs(j) for j in jumps) > 0.5, jumps


class TestSharedRQSKernel:
    """One rational-quadratic-spline kernel, used by all three RQS call sites.

    The marginal splines here, the fleet's stacked marginals and the edge
    calibration windows used to carry three transcriptions of the same
    Gregory-Delbourgo algebra with three different out-of-range policies.
    """

    @staticmethod
    def _shared_knots(seed=0, n=12):
        """Knots the three call sites can all be evaluated on."""
        rng = np.random.default_rng(seed)
        xk = np.sort(rng.normal(size=n))
        yk = np.sort(rng.uniform(size=n))
        yk = (yk - yk[0]) / (yk[-1] - yk[0])
        return xk, yk

    def test_the_three_call_sites_hold_one_kernel(self):
        """A re-fork must be caught here, not by luck of a differing guard."""
        assert fleet._rqs_forward is bijections._rqs_forward
        assert fleet._rqs_inverse is bijections._rqs_inverse
        assert fleet._rqs_derivative is bijections._rqs_derivative
        assert calibration._rqs_forward is bijections._rqs_forward
        assert calibration._rqs_inverse is bijections._rqs_inverse
        assert calibration._rqs_derivative is bijections._rqs_derivative
        assert calibration._POS_FLOOR is bijections._POS_FLOOR
        assert fleet._POS_FLOOR is bijections._POS_FLOOR
        # calibration used to restate the same 1e-30 under its own name
        assert not hasattr(calibration, "_POSITIVE_FLOOR")

    def test_bulk_fleet_and_calibration_paths_agree_bitwise(self):
        """Forward, inverse and log-derivative agree to exactly 0.0.

        Each module is driven through its own entry point, on knots shared by
        all three, and including queries outside the knot range — where the
        three copies used to disagree by construction.
        """
        xk, yk = self._shared_knots()
        cdf, inverse, deriv = _rqs_bulk(xk, yk)
        xk_j, yk_j = jnp.asarray(xk), jnp.asarray(yk)
        dk_j = jnp.asarray(_rqs_knot_derivatives(xk, yk))

        inside = jnp.asarray(np.linspace(xk[0], xk[-1], 257))
        outside = jnp.asarray(np.concatenate([xk[0] - np.geomspace(1e-6, 5.0, 8),
                                              xk[-1] + np.geomspace(1e-6, 5.0, 8)]))
        u = jnp.asarray(np.linspace(yk[0], yk[-1], 257))

        # fleet drives one source with one dimension: x is (b, d), knots (d, K)
        def fleet_forward(x):
            return fleet._u_and_logdet(x[:, None], xk_j[None], yk_j[None],
                                       dk_j[None])[0][:, 0]

        for x in (inside, outside):
            assert jnp.all(cdf(x) == fleet_forward(x))
            assert jnp.all(cdf(x) == calibration._rqs_forward_and_log_det(
                x, xk_j, yk_j, dk_j)[0])

        # the log-det is only comparable where the fleet reports one: outside
        # the prior box it deliberately returns -inf (a support test)
        log_deriv = jnp.log(jnp.maximum(deriv(inside), _POS_FLOOR))
        assert jnp.all(log_deriv == fleet._u_and_logdet(
            inside[:, None], xk_j[None], yk_j[None], dk_j[None])[1])
        assert jnp.all(log_deriv == calibration._rqs_forward_and_log_det(
            inside, xk_j, yk_j, dk_j)[1])

        assert jnp.all(inverse(u) == fleet._x_from_u(
            u[:, None], xk_j[None], yk_j[None], dk_j[None])[:, 0])
        assert jnp.all(inverse(u) == calibration._rqs_inverse_and_log_det(
            u, xk_j, yk_j, dk_j)[0])

    def test_out_of_range_queries_saturate(self):
        """Extrapolating the rational quadratic turns it over: the CDF falls
        below zero and rises again, and the density goes negative. Queries
        outside the knot range are evaluated at the nearest end knot instead.
        """
        xk, yk = self._shared_knots(seed=1)
        cdf, _, deriv = _rqs_bulk(xk, yk)

        span = xk[-1] - xk[0]
        probe = jnp.asarray(np.concatenate([
            xk[0] - np.linspace(3.0 * span, 0.0, 200)[:-1],
            np.linspace(xk[0], xk[-1], 200),
            xk[-1] + np.linspace(0.0, 3.0 * span, 200)[1:],
        ]))
        values = np.asarray(cdf(probe))
        densities = np.asarray(deriv(probe))

        assert np.all(np.diff(values) >= 0.0)
        assert values.min() >= yk[0] and values.max() <= yk[-1]
        assert np.all(densities >= 0.0)
        # constant beyond each end knot, i.e. genuinely saturated
        assert cdf(jnp.asarray(xk[0] - 1e3)) == cdf(jnp.asarray(xk[0] - 1.0))
        assert cdf(jnp.asarray(xk[-1] + 1e3)) == cdf(jnp.asarray(xk[-1] + 1.0))
        assert deriv(jnp.asarray(xk[-1] + 1e3)) == deriv(jnp.asarray(xk[-1]))

    def test_knot_derivative_floors_are_parameters(self):
        """One knot-slope builder, with each caller's floor policy as an argument."""
        xk = np.array([0.0, 1.0, 2.0, 3.0])
        yk = np.array([0.0, 0.5, 0.5 + 1e-14, 1.0])  # near-flat middle bin

        assert np.all(_rqs_knot_derivatives(xk, yk) >= _POS_FLOOR)
        assert np.all(_rqs_knot_derivatives(xk, yk, floor=1e-8) >= 1e-8)

        edged = _rqs_knot_derivatives(xk, yk, edge_slopes=(0.25, None))
        assert edged[0] == 0.25
        assert edged[-1] == _rqs_knot_derivatives(xk, yk)[-1]

        # PCHIP's one-sided endpoint rule zeroes the first knot derivative
        # whenever the second bin is more than three times as steep
        steepening = np.array([0.0, 0.01, 1.0, 1.5])
        assert _rqs_knot_derivatives(xk, steepening)[0] == _POS_FLOOR
        floored = _rqs_knot_derivatives(xk, steepening, edge_slope_floor=0.05)
        assert floored[0] == 0.05 * (steepening[1] - steepening[0]) / (xk[1] - xk[0])

    def test_calibration_window_keeps_its_own_floor_policy(self):
        """Sharing the builder must not move calibration to the generic floor."""
        xk = np.array([0.0, 1.0, 2.0, 3.0])
        yk = np.array([0.0, 0.5, 0.5 + 1e-14, 1.0])
        derivatives = calibration._window_derivatives(xk, yk)
        assert derivatives[0] == 1.0 and derivatives[-1] == 1.0
        assert np.all(derivatives >= 1e-8)


class TestFloat32Path:
    """The x64-disabled path had no coverage anywhere in tests/.

    ``coppuccino`` enables x64 on import, but that is a process-wide setting
    a caller can turn back off, and several constants in this module are
    float64 literals that underflow below it. Everything here runs inside
    ``jax.experimental.disable_x64()``.
    """

    def test_tail_quantile_stays_finite_without_x64(self):
        """``ndtri``'s domain floor has to follow the runtime dtype.

        With the float64 literal 1e-300 as the clip floor, the argument
        underflows to 0 in float32 and ``ndtri`` returns -inf, so the fitted
        marginal hands the flow an infinite quantile. Heavy-tailed data with
        a Gaussian tail model is where it bites: the boundary sits ~13 fitted
        sigma out, so the tail scale factor is ~1e31 and u / lo_scale lands
        below the float32 subnormal edge.
        """
        samples = np.random.default_rng(0).standard_t(1.5, 4000)
        u = jnp.asarray(np.array([1e-7, 1e-5], dtype=np.float32))
        with jax.experimental.disable_x64():
            _, _, quantile_fn, _ = make_empirical_cdf_spline(
                samples, tail_extension=True)
            x = np.asarray(quantile_fn(u))
        assert np.all(np.isfinite(x)), x
        # and it still saturates at the data range rather than folding back in
        assert np.all(x <= float(np.min(samples)) + 1e-3)

    def test_smooth_marginal_stays_a_finite_monotone_map_without_x64(self):
        """The bounded tails must not go NaN or non-monotone in float32."""
        rng = np.random.default_rng(1)
        x = np.clip(rng.beta(0.5, 3.0, 3000), 0.0, 1.0)
        with jax.experimental.disable_x64():
            theta_of_z, z_of_theta, log_dtheta_dz, clamps = (
                make_smooth_marginal(x, prior_low=0.0, prior_high=1.0))
            z = jnp.asarray(np.linspace(-7.0, 7.0, 1001, dtype=np.float32))
            theta = np.asarray(theta_of_z(z))
            log_det = np.asarray(log_dtheta_dz(z))
            back = np.asarray(z_of_theta(jnp.asarray(theta)))
        assert np.all(np.isfinite(theta)) and np.all(np.isfinite(log_det))
        assert np.all(np.diff(theta) >= 0.0)
        assert theta.min() > 0.0 and theta.max() < 1.0
        core = np.abs(np.asarray(z)) <= 3.0
        assert np.max(np.abs(back[core] - np.asarray(z)[core])) < 1e-3
        assert clamps[0] > 0.0 and clamps[1] < 1.0
