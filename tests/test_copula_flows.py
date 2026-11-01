import contextlib
import logging
import re
from types import SimpleNamespace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from equinox import filter_jit
from flowjax.distributions import Normal, Transformed

from coppuccino.copula_flows import (
    _MIN_SAMPLES_PER_DIM,
    _assert_fit_converged,
    _create_empirical_transforms,
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


# Fitting is the cost of this module, and on test-sized data it is almost all
# JAX tracing and compilation, paid once per *distinct* flow whatever
# ``max_epochs`` says. Tests that only need some fitted flow therefore share
# the module-scoped fixtures below rather than fitting their own; a test fits
# its own flow only when its premise is the data or the arguments of the fit.

# The smallest value each training knob's validator accepts.
_MINIMUM_LEGAL_KNOBS = {"max_epochs": 1, "knots": 2, "patience": 1,
                        "num_points": 2, "flow_layers": 1,
                        "weight_decay": 0.0, "rng_seed": 0}


@pytest.fixture(scope="module")
def bounded_chain():
    """600 rows strictly inside [-3, 3]^2."""
    rng = np.random.default_rng(0)
    return np.tanh(rng.standard_normal((600, 2))) * 3.0


@pytest.fixture(scope="module")
def small_flow(bounded_chain):
    """One cheap unbounded flow shared by every test that only needs a flow.

    It is fitted at ``_MINIMUM_LEGAL_KNOBS``, so building it is also the
    check that the knob validators accept the bottom of their documented
    range (``test_the_minimum_legal_knob_values_are_accepted``). Keep it at
    those values.
    """
    return normalizing_flows_fit(bounded_chain, show_progress=False,
                                 **_MINIMUM_LEGAL_KNOBS)


@pytest.fixture(scope="module")
def list_input_flow(bounded_chain):
    """A flow fitted from plain nested lists, with a one-sided box.

    Both ``chain`` and ``prior_bounds`` arrive as lists, and the box bounds
    column 0 from below only and column 1 not at all, so one fit serves the
    list-coercion tests and the one-sided masking test.
    """
    return normalizing_flows_fit(
        bounded_chain.tolist(), prior_bounds=[[-4.0, np.inf], [-np.inf, np.inf]],
        max_epochs=1, show_progress=False)


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
            normalizing_flows_fit(chain, 999, show_progress=False)


class TestCreateEmpiricalTransforms:
    """The marginal stack, exercised on its own without a copula fit."""

    def test_inverse_log_det_callable(self):
        """The jitted inverse returns one z row and one log-det per input row."""
        np.random.seed(42)
        samples = np.random.randn(100, 2)

        transform, inverse_log_det = _create_empirical_transforms(samples)

        test_data = samples[:10]
        z, log_dets = inverse_log_det(test_data)

        assert z.shape == test_data.shape
        assert log_dets.shape == (test_data.shape[0],)
        assert jnp.all(jnp.isfinite(z))
        assert jnp.all(jnp.isfinite(log_dets))


class TestVerbs:
    """The three public verbs on one fitted flow."""

    def test_verbs_return_the_documented_shapes(self, small_flow):
        """Each verb's return type and shape, for every input form it takes.

        One flow serves all three verbs: the shapes do not depend on the fit,
        so a fit per verb (or per draw count) would only pay compilation
        again.
        """
        draws = sample(small_flow, 100, rng_seed=0)
        assert isinstance(draws, np.ndarray) and draws.shape == (100, 2)
        assert np.all(np.isfinite(draws))

        scores = log_prob(small_flow, draws[:50])
        assert isinstance(scores, np.ndarray) and scores.shape == (50,)
        assert np.all(np.isfinite(scores))
        assert log_prob(small_flow, draws[:1]).shape == (1,)

        both, both_lp = sample_and_log_prob(small_flow, 100, rng_seed=0)
        assert isinstance(both, np.ndarray) and both.shape == (100, 2)
        assert isinstance(both_lp, np.ndarray) and both_lp.shape == (100,)
        assert np.all(np.isfinite(both)) and np.all(np.isfinite(both_lp))

    def test_the_same_seed_refits_the_same_flow(self, bounded_chain, small_flow):
        """Two independent fits with one ``rng_seed`` give the same flow.

        ``small_flow`` is the first fit; this refits from scratch with the
        identical arguments and compares draws made with one sampling seed.
        """
        refit = normalizing_flows_fit(bounded_chain, show_progress=False,
                                      **_MINIMUM_LEGAL_KNOBS)
        np.testing.assert_array_almost_equal(sample(small_flow, 100, rng_seed=42),
                                             sample(refit, 100, rng_seed=42))

    def test_consistency_with_separate_calls(self, small_flow):
        """sample_and_log_prob and sample return the *same draws* for a fixed
        seed, and the returned log-probs agree with log_prob(flow, samples):
        one map is evaluated both ways.

        Same-seed determinism of both verbs follows: neither could match the
        other draw for draw if either ignored its seed. The flow is unbounded,
        so the redraw loop and the near-wall recompute never run here; those
        are pinned on ``wall_atom_flow`` below.
        """
        samples_combined, lp_combined = sample_and_log_prob(
            small_flow, n_samples=100, rng_seed=42)
        samples_separate = sample(small_flow, n_samples=100, rng_seed=42)

        np.testing.assert_array_almost_equal(samples_combined, samples_separate)
        np.testing.assert_allclose(
            lp_combined, log_prob(small_flow, samples_combined), rtol=0, atol=1e-8)


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

        # 40 epochs, not more: measured at this seed and at seeds 1 and 2
        # (data and fit alike), the 40-epoch fit keeps corr >= 0.960 and
        # residual std <= 0.279 on both the current and the oldest supported
        # dependencies. On current ones the best validation epoch is 22
        # (seeds 1, 2: 25, 39), so 40 returns the same flow as 120. On the
        # oldest (jax 0.4.38, optax 0.2.2) it is 55 (65, 49), so 40 stops
        # short, but the 120-epoch fit scores no better: corr 0.961 to 0.969,
        # std 0.237 to 0.276. Rows are what buy the margin: on current
        # dependencies 4000 rows give 0.944 / 0.331 and 2000 rows fail the
        # residual bound outright (0.526).
        flow = normalizing_flows_fit(
            data, rng_seed=0, max_epochs=40, patience=25, flow_layers=6,
            show_progress=False
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
        only the containment check can catch it -- and here only its upper
        half.

        The two columns live on disjoint ranges, [0, 1] and [3, 4], with true
        bounds [-1, 2] and [0.5, 6]. Written out by kind, column 0's row
        becomes [-1, 0.5]: still below the data, so only ``high < max``
        flags it, and column 1's row [2, 6] still contains its data. Every
        other containment test also breaks a lower side, which left the
        ``highs > pb[:, 1]`` half of the check unpinned.
        """
        chain = np.column_stack([(bounded_chain[:, 0] + 3.0) / 6.0,
                                 (bounded_chain[:, 1] + 3.0) / 6.0 + 3.0])
        true_bounds = np.array([[-1.0, 2.0], [0.5, 6.0]])
        transposed = true_bounds.T.copy()
        assert np.all(transposed[:, 0] < transposed[:, 1])
        assert np.all(chain.min(axis=0) >= true_bounds[:, 0])
        assert np.all(chain.max(axis=0) <= true_bounds[:, 1])
        # only column 0's upper side is violated
        assert np.all(chain.min(axis=0) >= transposed[:, 0])
        assert chain[:, 0].max() > transposed[0, 1]
        assert chain[:, 1].max() <= transposed[1, 1]
        with pytest.raises(ValueError, match="outside prior_bounds") as exc:
            normalizing_flows_fit(chain, prior_bounds=transposed,
                                  max_epochs=1, show_progress=False)
        assert "parameter (column) 0" in str(exc.value)

    def test_list_bounds_and_infinite_sides_are_accepted(self, list_input_flow,
                                                         bounded_chain):
        """A plain nested list used to die with a bare TypeError from indexing."""
        assert np.all(np.isfinite(log_prob(list_input_flow, bounded_chain[:20])))


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

    def test_list_chain_is_accepted(self, list_input_flow, bounded_chain):
        assert np.all(np.isfinite(log_prob(list_input_flow, bounded_chain[:20])))

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
         # ``True`` is an ``int`` to isinstance, and would otherwise be read
         # as seed 1
         ({"rng_seed": True}, "rng_seed"),
         ({"learning_rate": 0.0}, "learning_rate"),
         # A plausible sign typo, and the only knob in the set that no
         # post-fit check could catch. flowjax keeps the best-validation-loss
         # parameters and epoch 1 is always accepted, so a wrong-sign step is
         # frozen there. Measured on ``bounded_chain`` with the validation
         # stubbed out: the returned flow is bitwise identical at max_epochs
         # 1, 2, 30 and 400, reports 100% finite densities, and its sample std
         # [2.65, 2.77] is indistinguishable from a 1-epoch ``+1e-3`` fit's
         # [2.61, 2.74].
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

    def test_the_minimum_legal_knob_values_are_accepted(self, small_flow,
                                                        bounded_chain):
        """The guard must not have narrowed the working range.

        ``small_flow`` is fitted at ``_MINIMUM_LEGAL_KNOBS``, the smallest value
        each validator accepts, so the fixture building at all is most of
        this test.
        """
        assert np.all(np.isfinite(log_prob(small_flow, bounded_chain[:20])))


class _StopBeforeTraining(Exception):
    """Raised by a ``fit_to_data`` spy once it has seen its arguments."""


class TestWeightDecayReachesTheOptimizer:
    """``weight_decay`` is wired through to the optimizer flowjax trains with.

    The only fit that exercises it otherwise asserts a finite density, which
    plain Adam gives too, so dropping the AdamW branch in ``_fit_chain_entry``
    left the suite green. A spy on the ``fit_to_data`` that module calls
    captures its keyword arguments and stops before any training happens.
    """

    @staticmethod
    def _captured_fit_kwargs(monkeypatch, chain, **kwargs):
        captured = {}

        def spy(key, dist, x, **fit_kwargs):
            captured.update(fit_kwargs)
            raise _StopBeforeTraining

        monkeypatch.setattr("coppuccino.copula_flows.fit_to_data", spy)
        with pytest.raises(_StopBeforeTraining):
            normalizing_flows_fit(chain, show_progress=False, **kwargs)
        return captured

    def test_a_positive_weight_decay_decays_the_weights(self, monkeypatch,
                                                        bounded_chain):
        """The optimizer handed over is AdamW with the requested decay.

        At zero gradient Adam's update is exactly zero, so any update at all
        is the decoupled decay term, ``-learning_rate * weight_decay * w``.
        """
        fit_kwargs = self._captured_fit_kwargs(
            monkeypatch, bounded_chain, learning_rate=1e-3, weight_decay=1e-2)
        optimizer = fit_kwargs["optimizer"]
        weights = jnp.ones(3)
        updates, _ = optimizer.update(jnp.zeros(3), optimizer.init(weights),
                                      weights)
        np.testing.assert_allclose(np.asarray(updates), -1e-3 * 1e-2 * np.ones(3))

    def test_no_weight_decay_trains_with_plain_adam(self, monkeypatch,
                                                    bounded_chain):
        """Without decay the optimizer is Adam at the requested learning rate.

        That is what flowjax would build itself from ``learning_rate``. The
        optimizer is handed over anyway so that every fit at one learning
        rate shares a single optimizer object, which the compiled training
        step is cached against.
        """
        import optax

        fit_kwargs = self._captured_fit_kwargs(
            monkeypatch, bounded_chain, learning_rate=1e-3, weight_decay=0.0)
        optimizer, adam = fit_kwargs["optimizer"], optax.adam(1e-3)
        weights, gradient = jnp.ones(3), jnp.array([0.5, -2.0, 1e-3])
        updates, _ = optimizer.update(gradient, optimizer.init(weights), weights)
        expected, _ = adam.update(gradient, adam.init(weights), weights)
        np.testing.assert_array_equal(np.asarray(updates), np.asarray(expected))


@contextlib.contextmanager
def _compile_log():
    """Collect JAX's "Compiling <program> ..." messages while the block runs.

    JAX logs one per program it lowers, i.e. on every tracing-cache miss,
    whether or not a persistent compilation cache then supplies the binary.
    """
    messages: list = []

    class _Collect(logging.Handler):
        def emit(self, record):
            messages.append(record.getMessage())

    handler = _Collect()
    logger = logging.getLogger("jax")
    logger.addHandler(handler)
    previous = jax.config.jax_log_compiles
    jax.config.update("jax_log_compiles", True)
    try:
        yield messages
    finally:
        jax.config.update("jax_log_compiles", previous)
        logger.removeHandler(handler)


# flowjax's jitted training step. JAX names the program "step" up to 0.4.x and
# "jit(step)" from 0.5 on.
_TRAINING_STEP_COMPILE = re.compile(r"Compiling (?:jit\()?step\b")
# Any jitted program whose name ends in log_prob: the divergence check's pass
# over the training rows, whatever function wraps it.
_DENSITY_PASS_COMPILE = re.compile(r"Compiling (?:jit\()?\w*log_prob\b")


class TestCompiledProgramsAreReused:
    """A fit compiles each program once, and later fits reuse them.

    On test-sized data compiling is most of what a fit costs, so a program
    compiled twice, or compiled afresh by every fit, is the cost to avoid.
    """

    def test_one_fit_compiles_the_training_step_once(self, bounded_chain):
        """Not once per input type.

        jit compiles a function again whenever an argument's type changes
        between calls. One spline parameter of a fresh triangular spline flow
        is built from a Python float, so JAX marks it weakly typed, and the
        first optimizer update hands back an ordinary float64 array in its
        place: the second call of the step no longer matched the first and
        every fit compiled it twice, about 0.7 s each time on an 8-column
        chain.
        """
        # Start from empty caches, so the count is this fit's own whatever
        # ran earlier in the process.
        jax.clear_caches()
        with _compile_log() as compiled:
            normalizing_flows_fit(bounded_chain, max_epochs=2, flow_layers=1,
                                  show_progress=False)
        step_compiles = [m for m in compiled if _TRAINING_STEP_COMPILE.match(m)]
        assert len(step_compiles) == 1, step_compiles

    def test_a_second_fit_of_one_architecture_reuses_the_compiled_programs(
            self, bounded_chain):
        """Fitting the same architecture again compiles no training step.

        flowjax caches its compiled step against the loss object, the
        optimizer and every function stored in the flow, all by identity, and
        it makes a fresh loss and optimizer per call while every new flow
        carries freshly made functions, so each fit used to compile from
        scratch. The divergence check's density pass is reused the same way
        when the chain has the same shape.
        """
        knobs = {"max_epochs": 2, "flow_layers": 1, "show_progress": False}
        jax.clear_caches()
        with _compile_log() as first:
            normalizing_flows_fit(bounded_chain, rng_seed=0, **knobs)
        with _compile_log() as second:
            normalizing_flows_fit(bounded_chain, rng_seed=1, **knobs)
        # The first fit shows both programs compiling, so an empty match in
        # the second is not just a pattern that never matches.
        for pattern in (_TRAINING_STEP_COMPILE, _DENSITY_PASS_COMPILE):
            assert [m for m in first if pattern.match(m)]
            assert not [m for m in second if pattern.match(m)], pattern.pattern

    def test_a_new_chain_of_the_same_shape_compiles_nothing(self):
        """Another chain, same shape and settings: no program is compiled.

        The marginal maps used to keep their spline and tail constants inside
        Python closures built per chain, so the map from data to copula space
        compiled afresh for every chain (about 0.7 s on 10 000 x 8). They now
        hold those numbers as arrays, which jit traces instead of baking in.
        Both chains have a bounded and an unbounded column, so both tail
        kinds are covered.
        """
        def chain(seed):
            rng = np.random.default_rng(seed)
            return np.tanh(rng.standard_normal((600, 2))) * 3.0

        knobs = {"max_epochs": 2, "flow_layers": 1, "show_progress": False,
                 "prior_bounds": [[-3.0, 3.0], [-np.inf, np.inf]]}
        jax.clear_caches()
        with _compile_log() as first:
            normalizing_flows_fit(chain(11), rng_seed=0, **knobs)
        with _compile_log() as second:
            normalizing_flows_fit(chain(12), rng_seed=1, **knobs)
        assert [m for m in first if m.startswith("Compiling")]
        assert not [m for m in second if m.startswith("Compiling")], second

    def test_sharing_the_template_functions_does_not_change_the_fit(
            self, monkeypatch, bounded_chain):
        """A new flow takes its stored functions from a cached template.

        Those functions are what flowjax's step cache compares by identity,
        and they are rebuilt per flow. Swapping in the template's copies is
        only sound if they carry nothing of a particular flow (its seed, its
        permutations), so the trained flow must not depend on the swap.
        """
        def trained_leaves(flow):
            return [np.asarray(leaf) for leaf in jax.tree_util.tree_leaves(flow)
                    if eqx.is_inexact_array(leaf)]

        knobs = {"max_epochs": 3, "flow_layers": 2, "rng_seed": 7,
                 "show_progress": False}
        shared = trained_leaves(normalizing_flows_fit(bounded_chain, **knobs))
        monkeypatch.setattr("coppuccino.copula_flows._share_template_functions",
                            lambda flow, architecture: flow)
        unshared = trained_leaves(normalizing_flows_fit(bounded_chain, **knobs))
        assert len(shared) == len(unshared)
        for a, b in zip(shared, unshared):
            np.testing.assert_array_equal(a, b)


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

    # the two edges of the band: 1.0 sits nearest the density budget, 100.0
    # nearest the rates whose parameters go non-finite
    @pytest.mark.parametrize("learning_rate", [1.0, 100.0])
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

    def test_a_nonfinite_mean_density_is_refused_not_compared(self, small_flow):
        """A non-finite mean density is a failure, and says so.

        The comparison against the baseline is a subtraction and a ``>=``,
        both of which propagate NaN silently; the pass/fail direction that
        falls out of them is an accident of how the inequality happens to be
        written. This pins it explicitly, and pins the message with it: a
        guard that quoted a NaN as though it were a measured density would
        send the caller looking for a threshold problem.
        """
        unreachable = jnp.asarray(np.full((4, 2), np.inf))
        with pytest.raises(RuntimeError, match="optimizer diverged") as exc:
            _assert_fit_converged(small_flow, small_flow.base_dist, unreachable,
                                  1e-3, 3)
        assert "is non-finite" in str(exc.value)

    def test_the_budget_is_per_dimension(self):
        """The density budget is 3 nats *per dimension*, measured exactly.

        Two Gaussians stand in for the untrained and the trained copula, and
        every training row sits at their shared mean, where a scale of
        ``exp(k)`` costs exactly ``k`` nats per dimension against the
        unit-scale baseline. A budget written in total nats would refuse the
        5-D, 2-nats-per-dimension case; a looser per-dimension constant would
        let the 4-nat case through.
        """
        for dim in (1, 5):
            z_train = jnp.zeros((8, dim))
            baseline = Normal(jnp.zeros(dim))

            def worse_by(nats, dim=dim):
                return SimpleNamespace(base_dist=Normal(
                    jnp.zeros(dim), jnp.full(dim, np.exp(nats))))

            _assert_fit_converged(worse_by(2.0), baseline, z_train, 1e-3, 8)
            with pytest.raises(RuntimeError, match="optimizer diverged"):
                _assert_fit_converged(worse_by(4.0), baseline, z_train, 1e-3, 8)

    @pytest.mark.parametrize(
        "kwargs",
        [{"learning_rate": 1e-4},
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
    def test_one_and_five_dimensional_fits_pass_the_guard(self, chain_shape):
        """Smoke: a 1-D copula (the narrowest) and a 5-D one fit and score.

        This does not pin the per-dimension budget: a healthy fit beats its
        own initialization, so its deficit is negative and any non-negative
        budget lets it through. ``test_the_budget_is_per_dimension`` pins
        that.
        """
        rng = np.random.default_rng(1)
        chain = rng.standard_normal(chain_shape)
        flow = normalizing_flows_fit(chain, max_epochs=8, rng_seed=0,
                                     show_progress=False)
        assert np.all(np.isfinite(log_prob(flow, chain[:50])))


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

    # ``True`` passes ``isinstance(n, int)``; without its own rejection it
    # would draw a single row
    @pytest.mark.parametrize("n_samples", [-5, 0, 10.5, None, True])
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

    def test_unseeded_draws_are_independent(self, small_flow):
        """A refactor to ``int(rng_seed or 0)`` would conflate None with 0 and
        still pass every explicitly-seeded test in the suite."""
        assert not np.array_equal(sample(small_flow, 64), sample(small_flow, 64))
        first, _ = sample_and_log_prob(small_flow, 64)
        second, _ = sample_and_log_prob(small_flow, 64)
        assert not np.array_equal(first, second)


@pytest.fixture(scope="module")
def wall_atom_flow():
    """A flow fitted on a chain clipped onto its own prior bounds.

    54.1% of the rows sit exactly on a wall, which is the documented use of
    ``prior_bounds`` and the configuration that drives the smooth marginal's
    bounded tail into its float64 saturation region. Building it is also the
    check that containment is inclusive: a chain touching its box is inside.
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
    draw's most repeated value occupies 436 rows (10.90%), the emitted
    stream's occupies 1 -- the floor of the statistic, i.e. no repeat at all.
    Stubbing ``_clamped_rows`` to return all-False makes the emitted stream
    *bitwise identical* to the raw one, restoring all 436.
    """

    N_DRAWS = 4000

    @staticmethod
    def _largest_atom_fraction(samples):
        """Largest share of rows sharing one exact value, over all columns."""
        return largest_value_multiplicity(samples) / len(samples)

    def _raw_atom_fraction(self, flow, seed):
        """Same statistic on the untreated draw ``sample`` starts from."""
        # ``sample`` splits the seed's key and draws from the first half
        draw_key = jax.random.split(jax.random.key(seed))[0]
        raw = np.asarray(filter_jit(flow.sample)(draw_key, (self.N_DRAWS,)))
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
        """The atoms go, and the densities returned with the redrawn rows are
        the ones ``log_prob`` reports for them (measured to 8e-11 here).

        The near-wall recompute inside ``sample_and_log_prob`` does not fire
        on this fixture -- no kept row comes within its 1e6-ULP window of a
        clamp -- so ``TestNearWallDensitiesAreRecomputed`` pins that.
        """
        raw_atoms = self._raw_atom_fraction(wall_atom_flow, 1)
        emitted, log_probs = sample_and_log_prob(wall_atom_flow, self.N_DRAWS,
                                                 rng_seed=1)
        assert self._largest_atom_fraction(emitted) < raw_atoms / 5.0
        assert self._largest_atom_fraction(emitted) < 0.01
        # and the densities that come back with them stay usable
        assert np.all(np.isfinite(log_probs))
        np.testing.assert_allclose(
            log_probs, log_prob(wall_atom_flow, emitted), rtol=0, atol=1e-8)

    @pytest.mark.parametrize("verb", [sample, sample_and_log_prob])
    def test_no_draw_reuses_a_key_another_draw_consumed(
            self, wall_atom_flow, monkeypatch, verb):
        """Each draw, raw or replacement, gets keys no other draw has touched.

        flowjax splits a draw's key into one key per row. The redraw loop used
        to split the raw draw's key again, so under jax's default
        partitionable threefry its first replacement block came from row 1's
        key, and the key it carried forward was row 0's. The dependence that
        left was below noise when measured, but it breaks jax's rule that a key
        is used once, so the assertion is on the keys themselves: every draw's
        key, and every per-row key split from it, is distinct.
        """
        from coppuccino import copula_flows as cf_mod

        draws = []
        real_jit = cf_mod.filter_jit

        def recording_jit(fn):
            jitted = real_jit(fn)

            def call(*args, **kwargs):
                dtype = getattr(args[0], "dtype", None)
                if dtype is not None and jax.dtypes.issubdtype(dtype, jax.dtypes.prng_key):
                    draws.append((args[0], int(np.prod(args[1]))))
                return jitted(*args, **kwargs)
            return call

        monkeypatch.setattr(cf_mod, "filter_jit", recording_jit)
        verb(wall_atom_flow, self.N_DRAWS, rng_seed=1)
        assert len(draws) > 1, "the fixture no longer triggers a redraw"

        def key_bytes(keys):
            data = np.asarray(jax.random.key_data(keys)).reshape(-1, 2)
            return {tuple(row) for row in data.tolist()}

        seen: set = set()
        for key, n_rows in draws:
            used = key_bytes(key) | key_bytes(jax.random.split(key, n_rows))
            assert not used & seen
            seen |= used

    def test_a_redraw_that_never_clears_gives_up_with_a_warning(
            self, wall_atom_flow, monkeypatch):
        """The loop is bounded, and running out of rounds is reported.

        With a single round allowed, the replacement block clamps at the same
        rate as the raw draw, so rows are still clamped when the rounds run
        out. ``sample`` must return the last draw and say so rather than loop
        on or hand the atoms back silently. The loop and its warning are one
        helper shared by both verbs, so one verb covers it.
        """
        monkeypatch.setattr("coppuccino.copula_flows._MAX_REDRAW_ROUNDS", 1)
        with pytest.warns(UserWarning, match="persisted after 1 redraw rounds"):
            emitted = sample(wall_atom_flow, self.N_DRAWS, rng_seed=1)
        assert emitted.shape == (self.N_DRAWS, 2)
        assert largest_value_multiplicity(emitted) > 1

    def test_a_last_round_that_clears_every_row_does_not_warn(self, monkeypatch):
        """The give-up warning is for rows that are *still* clamped.

        The loop used to check before each redraw and then warn once the
        rounds ran out, so a final round that cleared every row still reported
        that clamped rows persisted.
        """
        from coppuccino import copula_flows as cf_mod

        monkeypatch.setattr(cf_mod, "_MAX_REDRAW_ROUNDS", 1)
        limits = (np.array([-1.0]), np.array([1.0]))
        samples = np.array([[0.0], [1.0], [0.5]])  # row 1 sits on the clamp

        def clean_draw(key, block):
            return (np.zeros((block, 1)),)

        cf_mod._redraw_clamped_rows(clean_draw, (samples,), limits,
                                    jax.random.key(0), len(samples), "sample")
        np.testing.assert_array_equal(samples, [[0.0], [0.0], [0.5]])


class TestNearWallDensitiesAreRecomputed:
    """``sample_and_log_prob`` rescores rows within 1e6 ULPs of a tail clamp.

    The forward pass reports the log-det at the generated z, ``log_prob`` the
    one reconstructed from the float-truncated x, and where a bounded support
    is only a few dozen ULPs wide the two differ by nats. No fitted flow in
    this module puts a kept row inside that window (``wall_atom_flow``'s
    closest sits about 3e6 ULPs from its clamp), so the flow here is the
    marginal stack over an untrained standard-normal copula: nothing is
    trained, and every draw lands in the window. Without the recompute its
    densities missed ``log_prob`` by up to 3.4 nats.
    """

    def test_draws_in_a_ulp_wide_support_report_their_log_prob(self):
        rng = np.random.default_rng(0)
        # about 50 ULPs wide at this magnitude, half the column on its bound
        lo, hi = 1e10, 1e10 + 1e-4
        chain = np.column_stack([
            np.concatenate([np.full(300, hi), rng.uniform(lo, hi, 300)]),
            rng.standard_normal(600)])
        transform, _ = _create_empirical_transforms(
            chain, prior_bounds=np.array([[lo, hi], [-np.inf, np.inf]]))
        flow = Transformed(Normal(jnp.zeros(2)), transform)
        samples, log_probs = sample_and_log_prob(flow, 4000, rng_seed=1)
        np.testing.assert_allclose(log_probs, log_prob(flow, samples),
                                   rtol=0, atol=1e-8)


@pytest.fixture(scope="module")
def box_flow():
    """Uniform data on its own unit box, whose true log-density is 0."""
    rng = np.random.default_rng(4)
    chain = rng.uniform(0.0, 1.0, (1200, 2))
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
    return normalizing_flows_fit(chain, prior_bounds=bounds, max_epochs=30,
                                 show_progress=False, rng_seed=1)


class TestPriorBoxIsHardSupport:
    """log_prob must be -inf outside prior_bounds, finite on the closed box.

    Each marginal evaluates an out-of-range query at its nearest bound, so
    without an explicit mask a point outside the box is scored as if it sat on
    the wall -- bitwise identical at the bound and at infinity, and on some
    fits larger than the density of every real draw.
    """

    def test_outside_the_box_is_minus_inf(self, box_flow):
        outside = np.array([[1.001, 0.5], [5.0, 0.5], [1e6, 0.5],
                            [np.inf, 0.5], [-1e-9, 0.5], [0.5, -3.0]])
        assert np.all(np.isneginf(log_prob(box_flow, outside)))

    def test_a_single_point_scores_as_a_scalar_with_or_without_a_box(
            self, box_flow, small_flow):
        """A (n_params,) query returns shape (), and the mask must not change it.

        The mask is built row-wise from ``np.atleast_2d``, and broadcasting its
        (1,) against the flow's () used to hand back (1,) for a bounded flow
        only, so the same call changed shape with ``prior_bounds``.
        """
        point = np.array([0.5, 0.5])
        assert np.shape(log_prob(small_flow, point)) == ()
        assert np.shape(log_prob(box_flow, point)) == ()
        outside = log_prob(box_flow, np.array([1.5, 0.5]))
        assert np.shape(outside) == ()
        assert np.isneginf(outside)

    ON_BOX =np.array([[0.0, 0.5], [1.0, 0.5], [0.5, 0.0], [0.5, 1.0],
                       [0.0, 0.0], [1.0, 1.0]])

    def test_the_bounds_themselves_are_reported_by_the_flow_not_the_mask(
            self, box_flow):
        """Finite on the closed box, *and* the mask leaves those rows alone.

        Finiteness on its own is nearly free -- it also holds for a mask that
        substituted any finite sentinel, or that had simply moved the wall.
        Requiring bitwise equality with the unmasked flow pins the actual
        contract: the box is closed, and ``-inf`` is applied strictly outside.
        """
        got = log_prob(box_flow, self.ON_BOX)
        raw = np.asarray(filter_jit(box_flow.log_prob)(self.ON_BOX))
        assert np.all(np.isfinite(got))
        np.testing.assert_array_equal(got, raw)
        # one ULP outside is already -inf, so the boundary is exactly the box
        just_outside = np.array([[np.nextafter(1.0, np.inf), 0.5],
                                 [np.nextafter(0.0, -np.inf), 0.5]])
        assert np.all(np.isneginf(log_prob(box_flow, just_outside)))

    def test_unbounded_flows_are_untouched(self, small_flow):
        """No prior_bounds means no hard support: distant points stay finite."""
        far = np.array([[0.0, 0.0], [20.0, 20.0], [-50.0, 3.0]])
        assert np.all(np.isfinite(log_prob(small_flow, far)))

    def test_one_sided_bounds_mask_only_the_bounded_side(self, list_input_flow):
        """Column 0 is bounded below at -4 only, column 1 not at all."""
        got = log_prob(list_input_flow,
                       np.array([[1.0, 1.0], [-4.001, 1.0], [500.0, 1.0],
                                 [1.0, -500.0]]))
        assert np.isfinite(got[0])
        assert np.isneginf(got[1])
        assert np.isfinite(got[2])
        assert np.isfinite(got[3])
