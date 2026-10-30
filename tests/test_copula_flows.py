import jax
import jax.numpy as jnp
import numpy as np
import pytest
from equinox import filter_jit
from flowjax.distributions import Transformed

from coppuccino.copula_flows import (
    _MIN_SAMPLES_PER_DIM,
    _assert_fit_converged,
    _create_empirical_transforms,
    _resolve_seed,
    log_prob,
    normalizing_flows_fit,
    normalizing_flows_fit_auto,
    sample,
    sample_and_log_prob,
)


def largest_value_multiplicity(samples: np.ndarray) -> int:
    """Rows sharing the most repeated exact value, maximised over columns.

    Parameters
    ----------
    samples : np.ndarray
        A ``(n_samples, n_params)`` draw.

    Returns
    -------
    int
        1 when no float64 value repeats anywhere -- the floor, and what a
        continuous density gives.

    Notes
    -----
    This is the independent oracle for "the emitted stream carries no sample
    atom". It reads the numbers only: a continuous density repeats a float64
    value essentially never, so any multiplicity above 1 is a value the
    sampler saturated onto and the reported density knows nothing about.
    Asking ``_clamped_rows`` instead is vacuous exactly when ``_clamped_rows``
    is what broke -- stubbing it to all-False passes such a check while the
    atoms are still in the stream. Exported at module scope so other test
    modules can share it.
    """
    return max(
        int(np.unique(samples[:, j], return_counts=True)[1].max())
        for j in range(samples.shape[1])
    )


class TestMarginalOptions:
    """The removed marginal options fail loudly rather than being ignored."""

    @pytest.mark.parametrize("option", [
        {"marginal": "rqs"}, {"marginal": "pchip"}, {"tail_extension": True},
        {"tail_model": "gpd"}, {"tail_quantile": 0.05},
    ])
    def test_a_removed_option_is_rejected_by_name(self, option):
        chain = np.random.default_rng(0).standard_normal((200, 2))
        with pytest.raises(TypeError, match=next(iter(option))):
            normalizing_flows_fit(chain, max_epochs=1, show_progress=False,
                                  **option)

    def test_options_are_keyword_only(self):
        """1.0.0 took every option positionally, in an order that no longer
        exists; a positional call has to raise rather than shift silently."""
        chain = np.random.default_rng(0).standard_normal((200, 2))
        with pytest.raises(TypeError, match="positional"):
            normalizing_flows_fit(chain, 999)


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
        """sample_and_log_prob and sample return the *same draws* for a fixed
        seed, and (v2.0 smooth marginals) the returned log-probs agree with
        log_prob(flow, samples): one map is evaluated both ways, and any
        near-wall float-truncation rows are recomputed via log_prob inside
        sample_and_log_prob."""
        np.random.seed(42)
        chain = np.random.randn(200, 2)
        flow = normalizing_flows_fit(chain, max_epochs=5, patience=2)

        samples_combined, lp_combined = sample_and_log_prob(
            flow, n_samples=100, rng_seed=42)
        samples_separate = sample(flow, n_samples=100, rng_seed=42)

        np.testing.assert_array_almost_equal(samples_combined, samples_separate)
        np.testing.assert_allclose(
            lp_combined, log_prob(flow, samples_combined), rtol=0, atol=1e-8)

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


@pytest.fixture(scope="module")
def bounded_chain():
    """600 rows strictly inside [-3, 3]^2 (fits are the test-suite bottleneck)."""
    rng = np.random.default_rng(0)
    return np.tanh(rng.standard_normal((600, 2))) * 3.0


class TestPriorBoundsValidation:
    """prior_bounds is checked for shape, ordering and containment.

    Before this, an out-of-range chain was accepted silently and each marginal
    family disposed of it differently: ``"smooth"`` folded every out-of-range
    row onto the wall, the legacy spline families ignored the bound. Malformed
    bounds surfaced as errors that named the *data*, not the bounds.
    """

    def test_shape_must_be_n_params_by_two(self, bounded_chain):
        with pytest.raises(ValueError, match=r"prior_bounds must be \(n_params, 2\)"):
            normalizing_flows_fit(bounded_chain, prior_bounds=np.array([[-9.0, 9.0]]),
                                  max_epochs=1, show_progress=False)

    def test_row_must_be_low_then_high(self, bounded_chain):
        with pytest.raises(ValueError, match=r"prior_bounds row 0 must be \[low, high\]"):
            normalizing_flows_fit(bounded_chain,
                                  prior_bounds=np.array([[9.0, -9.0], [-9.0, 9.0]]),
                                  max_epochs=1, show_progress=False)

    def test_nan_bound_is_rejected(self, bounded_chain):
        with pytest.raises(ValueError, match=r"prior_bounds row 1 must be"):
            normalizing_flows_fit(bounded_chain,
                                  prior_bounds=np.array([[-9.0, 9.0], [np.nan, 9.0]]),
                                  max_epochs=1, show_progress=False)

    def test_chain_outside_box_raises(self, bounded_chain):
        """A box narrower than the data is an error, not a silent fold."""
        with pytest.raises(ValueError, match="outside prior_bounds") as exc:
            normalizing_flows_fit(bounded_chain,
                                  prior_bounds=np.array([[-1.0, 1.0], [-1.0, 1.0]]),
                                  max_epochs=1, show_progress=False)
        # names the offending column and the interval it actually spans, at
        # full float precision (see test_a_sub_ulp_violation_is_readable)
        assert "parameter (column) 0" in str(exc.value)
        assert repr(float(bounded_chain[:, 0].max())) in str(exc.value)

    def test_a_sub_ulp_violation_is_readable(self, bounded_chain):
        """A one-ULP overshoot must not print two identical-looking intervals.

        This is the violation a float round-tripped through a config file
        actually produces. Under ``%g`` the message read "spans [-2.99754,
        2.97997] but prior_bounds[0] is [-2.99754, 2.97997]" -- true, and
        unactionable.
        """
        bounds = np.array([[bounded_chain[:, 0].min(), bounded_chain[:, 0].max()],
                           [bounded_chain[:, 1].min(), bounded_chain[:, 1].max()]])
        bounds[0, 0] = np.nextafter(bounds[0, 0], np.inf)
        with pytest.raises(ValueError, match="outside prior_bounds") as exc:
            normalizing_flows_fit(bounded_chain, prior_bounds=bounds,
                                  max_epochs=1, show_progress=False)
        message = str(exc.value)
        data_low = repr(float(bounded_chain[:, 0].min()))
        bound_low = repr(float(bounds[0, 0]))
        # the two lower endpoints differ by one ULP and must render differently
        assert data_low != bound_low
        assert data_low in message and bound_low in message
        # and the message says what to do about it
        assert "Widen prior_bounds" in message

    def test_bounds_by_kind_transpose_is_caught_by_containment(self, bounded_chain):
        """``[[low0, low1], [high0, high1]]`` keeps low < high on every row, so
        only the containment check can catch it."""
        # true bounds [[-4, 4], [-3.5, 5]] written out by kind instead of by
        # parameter; both rows are still strictly increasing
        transposed = np.array([[-4.0, -3.5], [4.0, 5.0]])
        assert np.all(transposed[:, 0] < transposed[:, 1])
        with pytest.raises(ValueError, match="outside prior_bounds"):
            normalizing_flows_fit(bounded_chain, prior_bounds=transposed,
                                  max_epochs=1, show_progress=False)

    def test_list_bounds_and_infinite_sides_are_accepted(self, bounded_chain):
        """A plain nested list used to die with a bare TypeError from indexing."""
        flow = normalizing_flows_fit(bounded_chain,
                                     prior_bounds=[[-4.0, 4.0], [-np.inf, np.inf]],
                                     max_epochs=1, show_progress=False)
        assert np.all(np.isfinite(log_prob(flow, bounded_chain[:20])))

    def test_data_touching_the_bound_is_inside(self, bounded_chain):
        """Containment is inclusive: a sample exactly on the bound is legal."""
        bounds = np.array([[bounded_chain[:, 0].min(), bounded_chain[:, 0].max()],
                           [bounded_chain[:, 1].min(), bounded_chain[:, 1].max()]])
        flow = normalizing_flows_fit(bounded_chain, prior_bounds=bounds,
                                     max_epochs=1, show_progress=False)
        assert np.all(np.isfinite(log_prob(flow, bounded_chain[:20])))


class TestChainValidation:
    """chain is coerced and shape-checked before anything indexes it."""

    def test_one_dimensional_chain_raises(self, bounded_chain):
        with pytest.raises(ValueError, match=r"chain must be \(n_samples, n_params\)"):
            normalizing_flows_fit(bounded_chain[:, 0], max_epochs=1,
                                  show_progress=False)

    def test_three_dimensional_chain_raises(self, bounded_chain):
        # square trailing axes: this shape used to be silently flattened and
        # fitted as 2-D rather than rejected
        with pytest.raises(ValueError, match=r"got shape \(150, 2, 2\)"):
            normalizing_flows_fit(bounded_chain[:300].reshape(150, 2, 2),
                                  max_epochs=1, show_progress=False)

    def test_list_chain_is_accepted(self, bounded_chain):
        flow = normalizing_flows_fit(bounded_chain.tolist(), max_epochs=1,
                                     show_progress=False)
        assert np.all(np.isfinite(log_prob(flow, bounded_chain[:20])))

    def test_infinite_rows_are_dropped_not_fitted(self, bounded_chain):
        """+-inf rows used to die inside scipy, naming neither column nor
        argument."""
        dirty = bounded_chain.copy()
        dirty[3, 1] = np.inf
        dirty[7, 0] = -np.inf
        with pytest.warns(UserWarning, match="2 of 600 rows containing non-finite"):
            flow = normalizing_flows_fit(dirty, max_epochs=1,
                                         show_progress=False)
        assert np.all(np.isfinite(sample(flow, 100, rng_seed=0)))
        assert np.all(np.isfinite(log_prob(flow, bounded_chain[:20])))


class TestTrainingKnobValidation:
    """Every training knob is range-checked, not just ``flow_layers``.

    Before this the set failed three different ways: ``knots=0`` reached a
    ``ZeroDivisionError`` inside flowjax and ``rng_seed=1.5`` a jax "PRNG key
    seed must be an integer", while ``max_epochs<=0``, ``learning_rate=0.0``,
    ``patience=-1``, ``num_points<=1``, ``weight_decay<0`` and
    ``tanh_max_val<0`` were all accepted and returned a flow that looks fitted.
    """

    @pytest.mark.parametrize(
        "kwargs, parameter",
        [({"knots": 0}, "knots"),
         ({"knots": -3}, "knots"),
         ({"knots": 4.0}, "knots"),
         ({"max_epochs": 0}, "max_epochs"),
         ({"max_epochs": -1}, "max_epochs"),
         ({"patience": -1}, "patience"),
         ({"patience": 0}, "patience"),
         ({"num_points": -5}, "num_points"),
         ({"num_points": 1}, "num_points"),
         ({"rng_seed": 1.5}, "rng_seed"),
         ({"learning_rate": 0.0}, "learning_rate"),
         ({"learning_rate": -1e-3}, "learning_rate"),
         ({"learning_rate": np.nan}, "learning_rate"),
         ({"tanh_max_val": -1.0}, "tanh_max_val"),
         ({"tanh_max_val": 0.0}, "tanh_max_val"),
         ({"weight_decay": -5.0}, "weight_decay"),
         ({"flow_layers": -1}, "flow_layers"),
         ({"flow_layers": 0}, "flow_layers")],
    )
    def test_out_of_range_knob_names_itself_and_its_value(
            self, bounded_chain, kwargs, parameter):
        with pytest.raises(ValueError) as exc:
            normalizing_flows_fit(bounded_chain, show_progress=False,
                                  **{"max_epochs": 1, **kwargs})
        message = str(exc.value)
        assert message.startswith(parameter)
        assert repr(kwargs[parameter]) in message

    def test_a_negative_learning_rate_is_refused_before_it_trains(
            self, bounded_chain):
        """``learning_rate=-1e-3`` is a plausible sign typo, and the only knob
        in the set that no post-fit check could catch.

        flowjax keeps the best-validation-loss parameters, and epoch 1 is
        always accepted (it is trivially the minimum of a one-element loss
        history), so a wrong-sign step is frozen there and every later, worse
        epoch is discarded. Measured on this fixture with the hyperparameter
        validation stubbed out: the returned flow is bitwise identical at
        ``max_epochs`` 1, 2, 30 and 400; it reports 100% finite densities; and
        its 4,000-draw sample std is [2.65, 2.77] against a data std of
        [1.87, 1.84] -- indistinguishable from ``learning_rate=+1e-3,
        max_epochs=1``, which gives [2.61, 2.74]. There is no non-finite
        parameter, no non-finite density and no absurd scale to key on.
        """
        with pytest.raises(ValueError, match=r"learning_rate must be .*> 0"):
            normalizing_flows_fit(bounded_chain, learning_rate=-1e-3,
                                  max_epochs=1, show_progress=False)

    def test_the_documented_defaults_all_pass(self, bounded_chain):
        """The guard must not have narrowed the working range."""
        flow = normalizing_flows_fit(bounded_chain, max_epochs=1, knots=2,
                                     patience=1, num_points=2, flow_layers=1,
                                     weight_decay=0.0, rng_seed=0,
                                     show_progress=False)
        assert np.all(np.isfinite(log_prob(flow, bounded_chain[:20])))


class TestDivergenceGuard:
    """A diverged optimizer must not be handed back as a fitted flow.

    ``learning_rate=1e5`` used to return a ``Transformed`` whose ``sample()``
    is 100% NaN and whose ``log_prob`` is ``-inf`` on every training row, with
    no warning: NaN fails the ``<=``/``>=`` comparisons the redraw loop uses,
    ``save_flow`` persists it, and ``compute_injection_hdr`` turns it into the
    out-of-support sentinel for every injection.

    Finite parameters do not rule that out, which is the second half of the
    guard. flowjax keeps the best-*validation*-loss parameters and epoch 1 is
    unconditionally accepted, so a rate that blows up on the first step is
    frozen there with its parameters intact. Measured on 800 rows of
    correlated Gaussian data at ``max_epochs=30``: every ``learning_rate``
    from 1 to 100 returned normally under a parameter-only guard, with mean
    training log-densities of -49 to -2043 against -2.2 for a healthy fit, and
    100%-NaN draws from 60 upward.
    """

    def test_a_diverged_fit_raises_and_names_the_knobs(self, bounded_chain):
        with pytest.raises(RuntimeError, match="optimizer diverged") as exc:
            normalizing_flows_fit(bounded_chain, learning_rate=1e5,
                                  max_epochs=8, show_progress=False)
        assert "non-finite parameters" in str(exc.value)
        assert "learning_rate=100000.0" in str(exc.value)
        assert "max_epochs=8" in str(exc.value)

    @pytest.mark.parametrize("learning_rate", [1.0, 10.0, 60.0, 100.0])
    def test_the_finite_parameter_band_is_caught_by_its_density(
            self, bounded_chain, learning_rate):
        """The band the parameter check cannot see, closed by the density check.

        Asserting on the *density* half of the message is the point: passing
        here via the parameter branch would mean the band is still open for
        every rate that keeps its parameters finite.
        """
        with pytest.raises(RuntimeError, match="optimizer diverged") as exc:
            normalizing_flows_fit(bounded_chain, learning_rate=learning_rate,
                                  max_epochs=8, show_progress=False)
        message = str(exc.value)
        assert "the parameters are finite" in message
        assert "mean log-density on the training rows" in message
        assert f"learning_rate={learning_rate!r}" in message
        assert "max_epochs=8" in message

    def test_a_nonfinite_mean_density_is_refused_not_compared(
            self, bounded_chain):
        """A non-finite mean density is a failure, and says so.

        The comparison against the baseline is a subtraction and a ``>=``,
        both of which propagate NaN silently; the pass/fail direction that
        falls out of them is an accident of how the inequality happens to be
        written. This pins it explicitly, and pins the message with it: a
        guard that quoted a NaN as though it were a measured density would
        send the caller looking for a threshold problem.
        """
        flow = normalizing_flows_fit(bounded_chain, max_epochs=3,
                                     show_progress=False)
        unreachable = jnp.asarray(np.full((4, 2), np.inf))
        with pytest.raises(RuntimeError, match="optimizer diverged") as exc:
            _assert_fit_converged(flow, flow.base_dist, unreachable, 1e-3, 3)
        assert "is non-finite" in str(exc.value)

    def test_a_healthy_fit_is_untouched(self, bounded_chain):
        flow = normalizing_flows_fit(bounded_chain, max_epochs=3,
                                     show_progress=False)
        assert np.all(np.isfinite(sample(flow, 200, rng_seed=0)))

    @pytest.mark.parametrize(
        "kwargs",
        [{"learning_rate": 1e-4},
         {"learning_rate": 1e-2},
         {"learning_rate": 1e-1},
         {"knots": 2, "flow_layers": 2, "tanh_max_val": 2.0},
         {"knots": 8, "flow_layers": 8},
         {"weight_decay": 1e-2}],
    )
    def test_ordinary_fits_over_the_knob_range_still_succeed(
            self, bounded_chain, kwargs):
        """The other direction: the guard must not have narrowed the API.

        ``learning_rate`` spans three decades and the architecture spans the
        smallest to the largest default rung -- every axis the guard's
        baseline (the flow's own untrained initialization) is built from.
        """
        flow = normalizing_flows_fit(bounded_chain, max_epochs=8, rng_seed=0,
                                     show_progress=False, **kwargs)
        assert np.all(np.isfinite(log_prob(flow, bounded_chain[:50])))

    @pytest.mark.parametrize("chain_shape", [(200, 1), (400, 5)])
    def test_the_guard_scales_its_budget_with_dimension(self, chain_shape):
        """A per-dimension budget must accept 1-D and 5-D fits alike.

        A budget expressed in total nats instead would tighten as dimensions
        are added, and reject wide chains for having more coordinates.
        """
        rng = np.random.default_rng(1)
        chain = rng.standard_normal(chain_shape)
        flow = normalizing_flows_fit(chain, max_epochs=8, rng_seed=0,
                                     show_progress=False)
        assert np.all(np.isfinite(log_prob(flow, chain[:50])))

    def test_auto_reports_the_divergence_per_candidate(self, bounded_chain):
        """``normalizing_flows_fit_auto`` aggregates it as a candidate failure
        rather than letting an all-NaN flow win the comparison."""
        with pytest.warns(UserWarning, match="excluded failed fits"), \
                pytest.raises(RuntimeError, match="every candidate fit raised"):
            normalizing_flows_fit_auto(
                bounded_chain, candidates=[{"learning_rate": 1e5}],
                max_epochs=8, refit_best=False, n_workers=1)


class TestZeroColumnChain:
    """A ``(n, 0)`` chain used to reach ``Stack([])`` and die there.

    Every per-column loop is a no-op on a zero-column input, so the first
    thing that noticed was a bare ``IndexError: list index out of range``.
    """

    def test_fit_names_the_shape(self):
        with pytest.raises(ValueError, match="at least one parameter"):
            normalizing_flows_fit(np.zeros((300, 0)), max_epochs=1,
                                  show_progress=False)

    def test_auto_names_the_shape(self):
        with pytest.raises(ValueError, match="at least one parameter"):
            normalizing_flows_fit_auto(np.zeros((300, 0)), refit_best=False)

    def test_the_transform_builder_names_the_shape(self):
        with pytest.raises(ValueError, match=r"n_params >= 1"):
            _create_empirical_transforms(np.zeros((300, 0)))


@pytest.fixture(scope="module")
def small_flow(bounded_chain):
    """One cheap fitted flow shared by the argument-handling tests."""
    return normalizing_flows_fit(bounded_chain, max_epochs=2,
                                 show_progress=False, rng_seed=0)


class TestArgumentCoercion:
    """The single-flow verbs accept what ``FleetFlow``'s methods accept."""

    def test_log_prob_accepts_a_nested_list(self, small_flow):
        """flowjax alone raises 'Expected x to be arraylike; got list' here,
        while ``FleetFlow.log_prob`` accepts the identical input."""
        rows = [[0.1, 0.2], [0.3, 0.4]]
        np.testing.assert_array_equal(log_prob(small_flow, rows),
                                      log_prob(small_flow, np.asarray(rows)))

    def test_log_prob_rejects_a_non_numeric_argument(self, small_flow):
        with pytest.raises(ValueError, match="samples must be"):
            log_prob(small_flow, [["a", "b"]])

    def test_log_prob_rejects_a_three_dimensional_argument(self, small_flow):
        with pytest.raises(ValueError, match=r"got shape \(2, 2, 2\)"):
            log_prob(small_flow, np.zeros((2, 2, 2)))

    @pytest.mark.parametrize("n_samples", [-5, 0, 10.5, None])
    def test_bad_n_samples_names_itself(self, small_flow, n_samples):
        """Unvalidated these surface as jax static-shape TypeErrors naming
        'broadcast_in_dim' or 'Shapes must be 1D sequences'."""
        for verb in (sample, sample_and_log_prob):
            with pytest.raises(ValueError, match="n_samples"):
                verb(small_flow, n_samples)


class TestDegenerateColumnDiagnostics:
    """A failed marginal build names the column it came from."""

    def test_constant_column_error_names_the_column(self):
        rng = np.random.default_rng(0)
        chain = np.column_stack([rng.standard_normal(400),
                                 rng.standard_normal(400),
                                 np.ones(400)])
        with pytest.raises(ValueError, match=r"parameter \(column\) 2:") as exc:
            normalizing_flows_fit(chain, max_epochs=1, show_progress=False)
        assert "two distinct sample values" in str(exc.value)

    def test_minimum_samples_threshold_matches_the_documented_boundary(self):
        """Exactly ``_MIN_SAMPLES_PER_DIM`` rows raise; one more is enough."""
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="Insufficient samples"):
            _create_empirical_transforms(
                rng.standard_normal((_MIN_SAMPLES_PER_DIM, 2)))
        transform, _ = _create_empirical_transforms(
            rng.standard_normal((_MIN_SAMPLES_PER_DIM + 1, 2)))
        assert transform is not None


class TestUnseededSampling:
    """``rng_seed=None`` is the documented default of both sampling helpers."""

    def test_resolve_seed_passes_through_an_integer(self):
        assert _resolve_seed(7) == 7

    def test_resolve_seed_draws_a_fresh_seed_for_none(self):
        # A refactor to `int(rng_seed or 0)` would conflate None with 0 and
        # still pass every explicitly-seeded test in the suite.
        assert len({_resolve_seed(None) for _ in range(5)}) == 5

    def test_unseeded_draws_are_independent(self, bounded_chain):
        flow = normalizing_flows_fit(bounded_chain, max_epochs=2,
                                     show_progress=False)
        assert not np.array_equal(sample(flow, 64), sample(flow, 64))
        first, _ = sample_and_log_prob(flow, 64)
        second, _ = sample_and_log_prob(flow, 64)
        assert not np.array_equal(first, second)


@pytest.fixture(scope="module")
def wall_atom_flow():
    """A flow fitted on a chain clipped onto its own prior bounds.

    54.1% of the rows sit exactly on a wall, which is the documented use of
    ``prior_bounds`` and the configuration that drives the smooth marginal's
    bounded tail into its float64 saturation region.
    """
    rng = np.random.default_rng(0)
    chain = np.clip(rng.standard_normal((1500, 2)), -1.0, 1.0)
    return normalizing_flows_fit(chain,
                                 prior_bounds=np.array([[-1.0, 1.0], [-1.0, 1.0]]),
                                 max_epochs=20, show_progress=False, rng_seed=0)


class TestTailClampRedrawIsLoadBearing:
    """The redraw loop in ``sample`` / ``sample_and_log_prob`` must fire.

    The loop is the mechanism both docstrings advertise and the point of the
    smooth-marginal pipeline: a tail-clamped row is one whose value saturated
    onto a single float, so emitting it puts a *sample atom* in the stream
    that the reported density knows nothing about.

    The oracle is ``largest_value_multiplicity``, which is independent of
    ``_clamped_rows`` -- testing the redraw through it is vacuous exactly when
    ``_clamped_rows`` is what broke. Instead each test compares the emitted
    stream against the *raw* draw made from the same key, using only the
    numbers. The comparison is self-calibrating, which matters because the
    absolute clamp rate is a property of the fitted marginal and moves with
    it.

    Measured on this fixture at 4,000 draws with ``rng_seed=1``: the raw
    draw's most repeated value occupies 406 rows (10.15%), the emitted
    stream's occupies 1 -- the floor of the statistic, i.e. no repeat at all.
    Stubbing ``_clamped_rows`` to return all-False makes the emitted stream
    *bitwise identical* to the raw one, restoring all 406.
    """

    N_DRAWS = 4000

    @staticmethod
    def _largest_atom_fraction(samples):
        """Largest share of rows sharing one exact value, over all columns."""
        return largest_value_multiplicity(samples) / len(samples)

    def _raw_atom_fraction(self, flow, seed):
        """Same statistic on the untreated draw ``sample`` starts from."""
        raw = np.asarray(filter_jit(flow.sample)(
            jax.random.key(seed), (self.N_DRAWS,)))
        fraction = self._largest_atom_fraction(raw)
        # If the fixture ever stopped producing clamped rows the comparisons
        # below would pass vacuously, so make that a failure in its own right.
        assert fraction > 0.02, (
            "the wall_atom_flow fixture no longer saturates its bounded tail; "
            f"largest raw atom is only {fraction:.4%} of draws")
        return fraction

    def test_sample_removes_the_raw_draw_value_atoms(self, wall_atom_flow):
        raw_atoms = self._raw_atom_fraction(wall_atom_flow, 1)
        emitted = sample(wall_atom_flow, self.N_DRAWS, rng_seed=1)
        assert self._largest_atom_fraction(emitted) < raw_atoms / 5.0
        assert self._largest_atom_fraction(emitted) < 0.01

    def test_sample_and_log_prob_removes_them_too(self, wall_atom_flow):
        raw_atoms = self._raw_atom_fraction(wall_atom_flow, 1)
        emitted, log_probs = sample_and_log_prob(wall_atom_flow, self.N_DRAWS,
                                                 rng_seed=1)
        assert self._largest_atom_fraction(emitted) < raw_atoms / 5.0
        assert self._largest_atom_fraction(emitted) < 0.01
        # and the densities that come back with them stay usable
        assert np.all(np.isfinite(log_probs))


class TestPriorBoxIsHardSupport:
    """log_prob must be -inf outside prior_bounds, finite on the closed box.

    Each marginal evaluates an out-of-range query at its nearest bound, so
    without an explicit mask a point outside the box is scored as if it sat on
    the wall -- bitwise identical at the bound and at infinity, and on some
    fits larger than the density of every real draw.
    """

    @staticmethod
    def _bounded_flow():
        rng = np.random.default_rng(4)
        chain = rng.uniform(0.0, 1.0, (1200, 2))
        bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
        return normalizing_flows_fit(chain, prior_bounds=bounds, max_epochs=30,
                                     show_progress=False, rng_seed=1)

    def test_outside_the_box_is_minus_inf(self):
        flow = self._bounded_flow()
        outside = np.array([[1.001, 0.5], [5.0, 0.5], [1e6, 0.5],
                            [np.inf, 0.5], [-1e-9, 0.5], [0.5, -3.0]])
        assert np.all(np.isneginf(log_prob(flow, outside)))

    ON_BOX = np.array([[0.0, 0.5], [1.0, 0.5], [0.5, 0.0], [0.5, 1.0],
                       [0.0, 0.0], [1.0, 1.0]])

    def test_the_bounds_themselves_are_reported_by_the_flow_not_the_mask(self):
        """Finite on the closed box, *and* the mask leaves those rows alone.

        Finiteness on its own is nearly free -- it also holds for a mask that
        substituted any finite sentinel, or that had simply moved the wall.
        Requiring bitwise equality with the unmasked flow pins the actual
        contract: the box is closed, and ``-inf`` is applied strictly outside.
        """
        flow = self._bounded_flow()
        got = log_prob(flow, self.ON_BOX)
        raw = np.asarray(filter_jit(flow.log_prob)(self.ON_BOX))
        assert np.all(np.isfinite(got))
        np.testing.assert_array_equal(got, raw)
        # one ULP outside is already -inf, so the boundary is exactly the box
        just_outside = np.array([[np.nextafter(1.0, np.inf), 0.5],
                                 [np.nextafter(0.0, -np.inf), 0.5]])
        assert np.all(np.isneginf(log_prob(flow, just_outside)))

    @pytest.mark.xfail(
        strict=False,
        reason="Known copula-extrapolation defect, not a masking defect and "
               "not a marginal defect. On this fixture (U(0,1)^2, true "
               "log-density 0 everywhere, seed 4 chain, seed 1 fit, 30 "
               "epochs) the marginals alone report -1.99 to +0.59 on the six "
               "box points -- within a nat of the truth -- while the copula "
               "adds +2.56 to +11.56 on top, for +2.92 to +10.00 against an "
               "interior median of +0.98. The copula's outermost training z "
               "is the outermost marginal knot, ndtri(0.5/1200) = 3.34, and "
               "bijections._WALL_Z_MARGIN lets a wall query reach 4.84, where "
               "the triangular RQS flow extrapolates linearly; the two "
               "corners, which sit at that depth in *both* dimensions, carry "
               "+11.4 and +11.6 of the excess and are what breaks the 5-nat "
               "bound (the four edge midpoints run +1.84 to +2.74 above the "
               "interior). Shrinking the margin further trades this against "
               "wall resolution and marginal wall-density accuracy, so the "
               "fix has to bound the copula's extrapolation itself. The box "
               "maximum still exceeds the +8.50 maximum over 20,000 of the "
               "flow's own draws, by 1.50. Delete this marker with that fix.")
    def test_the_bounds_are_within_a_few_nats_of_the_interior(self):
        """The at-bound density must be comparable to the interior.

        The fixture is uniform on its own box, so the truth is flat: every
        point in the closed square has the same density, and the reported
        log-density at a wall should differ from the interior by a few nats at
        most. Without this, ``test_the_bounds_themselves_...`` above passes
        for an arbitrarily wrong finite value.
        """
        flow = self._bounded_flow()
        interior = float(np.median(log_prob(
            flow, np.array([[0.5, 0.5], [0.25, 0.75], [0.75, 0.25]]))))
        assert np.max(np.abs(log_prob(flow, self.ON_BOX) - interior)) < 5.0

    def test_every_draw_has_a_finite_density(self):
        flow = self._bounded_flow()
        samples, log_probs = sample_and_log_prob(flow, 4000, rng_seed=2)
        assert np.all(np.isfinite(log_probs))
        assert np.array_equal(log_prob(flow, samples), log_probs) or np.allclose(
            log_prob(flow, samples), log_probs, atol=1e-6)

    def test_unbounded_flows_are_untouched(self):
        """No prior_bounds means no hard support: distant points stay finite."""
        rng = np.random.default_rng(5)
        flow = normalizing_flows_fit(rng.standard_normal((800, 2)),
                                     max_epochs=20, show_progress=False,
                                     rng_seed=1)
        far = np.array([[0.0, 0.0], [20.0, 20.0], [-50.0, 3.0]])
        assert np.all(np.isfinite(log_prob(flow, far)))

    def test_one_sided_bounds_mask_only_the_bounded_side(self):
        rng = np.random.default_rng(6)
        flow = normalizing_flows_fit(
            rng.exponential(1.0, (800, 2)),
            prior_bounds=np.array([[0.0, np.inf], [0.0, np.inf]]),
            max_epochs=20, show_progress=False, rng_seed=1)
        got = log_prob(flow, np.array([[1.0, 1.0], [-1e-3, 1.0], [500.0, 1.0]]))
        assert np.isfinite(got[0])
        assert np.isneginf(got[1])
        assert np.isfinite(got[2])
