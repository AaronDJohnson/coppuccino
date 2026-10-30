import numpy as np
import pytest
from flowjax.distributions import Transformed
from scipy.stats import chi2, kstest

from coppuccino.copula_flows import log_prob, sample_and_log_prob
from coppuccino.hdr import _in_support_mask, check_in_support, compute_injection_hdr

# A throwaway fit, for tests whose assertion does not depend on fit quality.
_FAST_FIT = {"max_epochs": 5, "patience": 2, "show_progress": False}
# Enough training that the fitted density is monotone along a ray from the
# mode; the ordering assertions below were verified over seeds 0-5 with it.
_TRAINED_FIT = {"max_epochs": 150, "patience": 25, "show_progress": False}


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

    def test_accepts_nested_sequences(self):
        """Lists must be coerced, not met with an AttributeError about .ndim.

        The function is exported at package level next to compute_injection_hdr,
        which has always accepted lists for both arguments.
        """
        samples = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]
        assert check_in_support(samples, [1.0, 1.0]) is True
        assert check_in_support(samples, [5.0, 1.0]) is False
        assert check_in_support(samples, (0.0, 2.0)) is True

    def test_one_dimensional_samples_are_rejected_by_name(self):
        """A flat sample array is a shape mistake, not an IndexError."""
        with pytest.raises(ValueError, match=r"samples must be \(n_samples, n_params\)"):
            check_in_support(np.zeros(50), np.array([0.0]))

    def test_non_finite_injection_is_out_of_support(self):
        """NaN/Inf lie inside no hull, so the predicate answers False."""
        samples = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        assert check_in_support(samples, [np.nan, 1.0]) is False
        assert check_in_support(samples, [np.inf, 1.0]) is False
        assert check_in_support(samples, [-np.inf, 1.0]) is False


class TestInSupportMask:
    """The batched hull test that compute_injection_hdr actually uses."""

    def test_matches_check_in_support_row_for_row(self):
        """The batched form is the implementation; the scalar form wraps it.

        Includes rows sitting exactly on the hull, which is where a >=/> slip
        changes the answer, and rows outside in each coordinate separately.
        """
        rng = np.random.default_rng(0)
        samples = rng.standard_normal((500, 3))
        mins = samples.min(axis=0)
        maxs = samples.max(axis=0)
        injections = np.vstack(
            [
                rng.standard_normal((200, 3)),
                mins,
                maxs,
                np.column_stack([mins, maxs, mins])[0][None, :],
                mins - 1e-9,
                maxs + 1e-9,
            ]
        )

        mask = _in_support_mask(samples, injections)
        expected = np.array([check_in_support(samples, row) for row in injections])
        assert np.array_equal(mask, expected)

    def test_the_hull_is_closed(self):
        """Exactly-on-the-hull rows are in support; one ulp outside is not."""
        samples = np.array([[0.0, -1.0], [1.0, 0.0], [2.0, 3.0]])
        corners = np.array([[0.0, -1.0], [2.0, 3.0], [0.0, 3.0], [2.0, -1.0]])
        assert np.all(_in_support_mask(samples, corners))

        just_outside = np.array(
            [
                [np.nextafter(0.0, -np.inf), 0.0],
                [np.nextafter(2.0, np.inf), 0.0],
                [1.0, np.nextafter(-1.0, -np.inf)],
                [1.0, np.nextafter(3.0, np.inf)],
            ]
        )
        assert not np.any(_in_support_mask(samples, just_outside))


class TestComputeInjectionHDR:
    """Test the compute_injection_hdr function."""

    def test_basic_functionality_1d_injection(self):
        """The returned value is exactly the rank of the injection's density.

        Recomputed from the returned flow through the public API, so it pins the
        estimator's definition (which tail, and the denominator) rather than the
        0 <= hdr <= 1 that holds by construction.
        """
        rng = np.random.default_rng(42)
        samples = rng.standard_normal((200, 2))
        injection_params = np.array([0.0, 0.0])

        hdr, flow = compute_injection_hdr(
            samples,
            injection_params,
            num_samples=4000,
            rng_seed=7,
            return_flow=True,
            **_FAST_FIT,
        )

        assert isinstance(hdr, np.ndarray)
        assert hdr.shape == (1,)
        _, generated_log_probs = sample_and_log_prob(flow, n_samples=4000, rng_seed=7)
        injection_log_prob = float(np.asarray(log_prob(flow, injection_params[None, :]))[0])
        expected = float(np.mean(np.asarray(generated_log_probs) >= injection_log_prob))
        assert hdr[0] == expected

    def test_multiple_injections(self):
        """HDR rises monotonically along a ray leaving the mode.

        Radii 0, 1.5 and 3.0 of a 2-D standard normal; the outermost point's
        analytic credibility is 0.989 and the estimate landed within 0.01 of it
        on seeds 0, 1 and 42.
        """
        rng = np.random.default_rng(42)
        samples = rng.standard_normal((3000, 2))
        radii = np.array([0.0, 1.5, 3.0])
        injection_params = np.column_stack([radii, radii]) / np.sqrt(2.0)

        hdr = compute_injection_hdr(
            samples,
            injection_params,
            num_samples=20000,
            rng_seed=42,
            **_TRAINED_FIT,
        )

        assert hdr.shape == (3,)
        assert np.all(np.diff(hdr) > 0.0)
        assert hdr[0] < 0.6  # nothing at the mode may look like a tail point
        assert abs(hdr[2] - chi2.cdf(9.0, 2)) < 0.06

    def test_return_flow_option(self):
        """The returned flow is the one that produced the values.

        Scoring a second injection through it by hand must reproduce what a
        second call with the same seed reports, so the flow is not a separate
        fit and the values are not recomputed differently on the two paths.
        """
        rng = np.random.default_rng(42)
        samples = rng.standard_normal((200, 2))
        injection_params = np.array([0.0, 0.0])
        shared = dict(num_samples=4000, rng_seed=7, **_FAST_FIT)

        hdr, flow = compute_injection_hdr(
            samples, injection_params, return_flow=True, **shared
        )
        assert isinstance(flow, Transformed)
        assert np.array_equal(hdr, compute_injection_hdr(samples, injection_params, **shared))

        other = np.array([0.7, -0.4])
        _, generated_log_probs = sample_and_log_prob(flow, n_samples=4000, rng_seed=7)
        by_hand = float(
            np.mean(
                np.asarray(generated_log_probs)
                >= float(np.asarray(log_prob(flow, other[None, :]))[0])
            )
        )
        assert compute_injection_hdr(samples, other, **shared)[0] == by_hand

    def test_different_num_samples(self):
        """num_samples sets the resolution of the estimate, not its value.

        Every result is an exact multiple of 1/num_samples, and the three
        estimates of the same quantity agree to 0.0068 -- they are 0.646,
        0.6392 and 0.64428 in this configuration -- because the fit is held
        fixed by rng_seed. The assertion allows 0.03.
        """
        rng = np.random.default_rng(42)
        samples = rng.standard_normal((200, 2))
        injection_params = np.array([0.0, 0.0])

        values = {}
        for num_samples in [500, 5000, 50000]:
            hdr = compute_injection_hdr(
                samples,
                injection_params,
                num_samples=num_samples,
                rng_seed=3,
                **_FAST_FIT,
            )
            assert hdr.shape == (1,)
            counts = hdr[0] * num_samples
            assert abs(counts - round(counts)) < 1e-6
            values[num_samples] = hdr[0]

        assert max(values.values()) - min(values.values()) < 0.03

    def test_custom_nf_kwargs(self):
        """nf_kwargs reach normalizing_flows_fit rather than being swallowed."""
        rng = np.random.default_rng(42)
        samples = rng.standard_normal((200, 2))
        injection_params = np.array([0.0, 0.0])

        _, sparse = compute_injection_hdr(
            samples, injection_params, num_samples=2000, rng_seed=3,
            knots=4, return_flow=True, **_FAST_FIT,
        )
        _, flexible = compute_injection_hdr(
            samples, injection_params, num_samples=2000, rng_seed=3,
            knots=12, return_flow=True, **_FAST_FIT,
        )
        # Same data, same seed: only the forwarded knots can move the density.
        assert not np.allclose(
            np.asarray(log_prob(sparse, samples)), np.asarray(log_prob(flexible, samples))
        )

        with pytest.raises(TypeError, match="definitely_not_a_kwarg"):
            compute_injection_hdr(
                samples, injection_params, num_samples=2000, definitely_not_a_kwarg=1,
            )

    def test_non_finite_injections_are_rejected(self):
        """NaN/Inf injections must not be laundered into the 1.0 sentinel.

        They fail the hull test, so before this check they short-circuited to
        exactly 1.0 -- the value a legitimately out-of-support injection gets.
        """
        rng = np.random.default_rng(42)
        samples = rng.standard_normal((200, 2))

        for bad in ([np.nan, 0.0], [np.inf, 0.0], [-np.inf, 0.0]):
            with pytest.raises(ValueError, match="injection_params contains 1 non-finite"):
                compute_injection_hdr(samples, np.array(bad), num_samples=1000, **_FAST_FIT)

        # The message locates the offending cell of a batch.
        batch = np.array([[0.0, 0.0], [1.0, np.nan], [0.5, 0.5]])
        with pytest.raises(ValueError, match=r"first at row 1, column 1 with value nan"):
            compute_injection_hdr(samples, batch, num_samples=1000, **_FAST_FIT)

    def test_non_finite_samples_are_rejected(self):
        """The same check on the posterior samples names the offending cell."""
        rng = np.random.default_rng(42)
        samples = rng.standard_normal((200, 2))
        samples[13, 1] = np.inf
        with pytest.raises(ValueError, match=r"samples contains 1 non-finite.*row 13, column 1"):
            compute_injection_hdr(samples, np.zeros(2), num_samples=1000, **_FAST_FIT)

    def test_diverged_fit_raises_instead_of_returning_the_sentinel(self):
        """A diverged flow scores everything NaN, and NaN loses every comparison.

        Unguarded, that returns 1.0 for every in-support injection, which in a
        P-P study is indistinguishable from three out-of-support events. The
        fitter now refuses to return such a flow at all; this asserts the
        end-to-end behaviour, whichever layer catches it.
        """
        rng = np.random.default_rng(42)
        samples = rng.standard_normal((300, 2))
        injections = np.array([[0.0, 0.0], [1.0, 1.0], [-1.0, 0.5]])

        with pytest.raises((RuntimeError, ValueError), match="diverged"):
            compute_injection_hdr(
                samples,
                injections,
                num_samples=2000,
                rng_seed=0,
                learning_rate=1e5,
                max_epochs=10,
                patience=5,
                show_progress=False,
            )

    def test_non_finite_reference_scores_raise(self, monkeypatch):
        """HDR's own guard, isolated from the fitter's.

        Any route to a NaN reference score -- a diverged fit, a flow loaded
        from an older release, a marginal that underflows -- would otherwise
        make searchsorted return 0 for every injection and hand back a column
        of 1.0 sentinels.
        """
        import coppuccino.hdr as hdr_module

        rng = np.random.default_rng(0)
        samples = rng.standard_normal((200, 2))

        def nan_scores(flow, n_samples, rng_seed=None):
            return np.zeros((n_samples, 2)), np.full(n_samples, np.nan)

        monkeypatch.setattr(hdr_module, "sample_and_log_prob", nan_scores)
        with pytest.raises(RuntimeError, match="1000 of 1000 reference points non-finitely"):
            compute_injection_hdr(samples, np.zeros(2), num_samples=1000, **_FAST_FIT)

    def test_nan_injection_scores_raise(self, monkeypatch):
        """The injection-side guard, which neither of the other two reaches.

        ``test_diverged_fit_raises_instead_of_returning_the_sentinel`` is
        satisfied by the fitter's own divergence check, and the reference guard
        above only sees the reference array, so a flow that scores its own
        draws finitely but an in-support injection NaN reaches this guard
        alone. It has to raise, because NaN sorts *after* every finite
        reference point: the count of more-probable points is zero and the
        unguarded answer is HDR 0.0, i.e. the injection is reported as sitting
        at the mode. With this one guard disabled and nothing else changed,
        this exact call returns [0.0, 0.236] instead of raising.
        """
        import coppuccino.hdr as hdr_module

        # The premise of the guard: searchsorted places NaN past every finite
        # score, so the unguarded count of more-probable reference points is 0.
        reference = np.sort(np.array([-3.0, -1.0, 0.5, 2.0]))
        assert np.searchsorted(reference, np.nan, side="right") == len(reference)

        rng = np.random.default_rng(0)
        samples = rng.standard_normal((200, 2))
        injections = np.array([[0.0, 0.0], [1.0, 1.0]])
        real_log_prob = hdr_module.log_prob

        def nan_first_row(flow, x):
            scores = np.asarray(real_log_prob(flow, x), dtype=float).copy()
            scores[0] = np.nan
            return scores

        monkeypatch.setattr(hdr_module, "log_prob", nan_first_row)
        with pytest.raises(RuntimeError, match="NaN log-density for 1 of 2 in-support"):
            compute_injection_hdr(
                samples, injections, num_samples=1000, rng_seed=0, **_FAST_FIT
            )

    def test_num_samples_and_reference_are_validated(self):
        """Bad knobs are named, not turned into nan or an obscure failure."""
        rng = np.random.default_rng(42)
        samples = rng.standard_normal((200, 2))
        injection = np.zeros(2)

        with pytest.raises(ValueError, match="num_samples must be a positive integer"):
            compute_injection_hdr(samples, injection, num_samples=0, **_FAST_FIT)
        with pytest.raises(ValueError, match="num_samples must be a positive integer"):
            compute_injection_hdr(samples, injection, num_samples=-5, **_FAST_FIT)
        with pytest.raises(ValueError, match="reference must be 'flow' or 'held_out'"):
            compute_injection_hdr(samples, injection, reference="heldout", **_FAST_FIT)
        with pytest.raises(ValueError, match="at least 40 rows; samples has 30"):
            compute_injection_hdr(
                rng.standard_normal((30, 2)), injection, reference="held_out", **_FAST_FIT
            )

    def test_held_out_reference_ranks_against_the_unused_half(self):
        """The cross-fit reference set is the half of samples the fit never saw.

        Its size is what the resolution of the estimate is: with 201 rows the
        values are exact multiples of 1/101, and the split is a function of
        rng_seed alone.
        """
        rng = np.random.default_rng(1)
        samples = rng.standard_normal((201, 2))
        shared = dict(reference="held_out", rng_seed=11, **_FAST_FIT)

        hdr = compute_injection_hdr(samples, np.array([0.0, 0.0]), **shared)
        counts = hdr[0] * 101
        assert abs(counts - round(counts)) < 1e-9
        assert np.array_equal(hdr, compute_injection_hdr(samples, np.array([0.0, 0.0]), **shared))

    def test_injection_at_mode_has_lower_hdr_than_tail(self):
        """HDR credibility is the fraction of the fit *more probable* than the
        point, so a point at the mode scores near 0 and a point out in the tail
        scores higher. Assert that ordering — the module's core semantic
        property, which the old ``>= 0.0`` assertions did not test at all."""
        rng = np.random.default_rng(42)
        # Fit to a 2D standard normal; the mode is [0, 0].
        samples = rng.standard_normal((3000, 2))

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

    def test_out_of_support_injection_returns_exact_sentinel(self):
        """Injections outside the coordinate-wise hull short-circuit to exactly 1.0.

        That exact value is a calibration convention, not a density result: the
        flow's smooth tails give such a point a finite density that scores well
        below 1.0, so the short-circuit has to be observable."""
        rng = np.random.default_rng(42)
        samples = rng.standard_normal((300, 2))
        # One point just outside the hull of dimension 0 and one just inside it,
        # at a comparable density, so only the short-circuit separates them.
        outside = samples[:, 0].max() + 0.05
        inside = samples[:, 0].max() - 0.05

        injection_params = np.array([[outside, 0.5], [inside, 0.5], [0.0, 0.0]])
        hdrs = compute_injection_hdr(
            samples, injection_params, num_samples=10000,
            rng_seed=0, max_epochs=50, patience=15,
        )

        assert hdrs[0] == 1.0
        # Its in-hull neighbour is scored by the flow and lands strictly below
        # the sentinel, which is what makes the sentinel a real choice.
        assert hdrs[1] < 1.0
        assert hdrs[0] > hdrs[1] > hdrs[2]

    def test_batched_injections_match_single_injection_calls(self):
        """One batched call scores each injection exactly as a lone call does."""
        rng = np.random.default_rng(42)
        samples = rng.standard_normal((300, 2))
        injection_params = np.array(
            [[samples[:, 0].max() + 0.05, 0.5], [1.0, -1.0], [-4.0, 0.0], [0.0, 0.0]]
        )
        kwargs = {"num_samples": 2000, "rng_seed": 0, "max_epochs": 5,
                  "patience": 2}

        batched = compute_injection_hdr(samples, injection_params, **kwargs)
        one_at_a_time = np.concatenate(
            [compute_injection_hdr(samples, injection, **kwargs) for injection in injection_params]
        )

        assert np.array_equal(batched, one_at_a_time)
        # The batch really does mix both kinds of row.
        assert batched[0] == 1.0 and batched[2] == 1.0
        assert np.all(batched[[1, 3]] < 1.0)

    def test_0d_injection_raises_error(self):
        """Test that 0D injection parameters raise an error."""
        rng = np.random.default_rng(42)
        samples = rng.standard_normal((200, 2))
        injection_params = np.array(0.0)  # 0D array

        with pytest.raises(ValueError, match="injection_params must be at least 1D"):
            compute_injection_hdr(
                samples,
                injection_params,
                num_samples=1000,
                max_epochs=5,
                patience=2
            )

    def test_injection_width_mismatch_raises(self):
        """A 2D injection array with the wrong parameter count is rejected."""
        rng = np.random.default_rng(42)
        samples = rng.standard_normal((50, 2))

        with pytest.raises(ValueError, match="injection_params length must match"):
            compute_injection_hdr(samples, np.zeros((3, 5)), num_samples=1000)

    def test_3d_injection_raises(self):
        """Anything beyond a batch of injections is rejected by shape."""
        rng = np.random.default_rng(42)
        samples = rng.standard_normal((50, 2))

        with pytest.raises(ValueError, match="must be 1D or 2D"):
            compute_injection_hdr(samples, np.zeros((2, 2, 2)), num_samples=1000)

    def test_1d_parameter_space(self):
        """In one dimension the credibility has a closed form to check against.

        For a standard normal the HDR of x is chi2_1.cdf(x**2): 0 at the mode,
        0.683 at x=1, 0.954 at x=2. The estimate at x=2 was within 0.038 of the
        truth on seeds 0-5 (the assertion allows 0.06); the interior points
        carry the estimator bias documented in the module and are only checked
        for ordering.
        """
        rng = np.random.default_rng(42)
        samples = rng.standard_normal((2000, 1))
        injection_params = np.array([[0.0], [1.0], [2.0]])

        hdr = compute_injection_hdr(
            samples,
            injection_params,
            num_samples=20000,
            rng_seed=42,
            **_TRAINED_FIT,
        )

        assert hdr.shape == (3,)
        assert np.all(np.diff(hdr) > 0.0)
        assert abs(hdr[2] - chi2.cdf(4.0, 1)) < 0.06

    def test_high_dimensional_parameter_space(self):
        """The same radial ordering has to survive five dimensions.

        Points at squared radius 0, 4.05 and 9.8 of a 5-D standard normal; the
        outermost analytic credibility is 0.919 and the estimate stayed within
        0.052 of it on seeds 0, 1 and 42 (the assertion allows 0.06).
        """
        rng = np.random.default_rng(42)
        samples = rng.standard_normal((2000, 5))
        injection_params = np.array([np.zeros(5), np.full(5, 0.9), np.full(5, 1.4)])

        hdr = compute_injection_hdr(
            samples,
            injection_params,
            num_samples=20000,
            rng_seed=42,
            **_TRAINED_FIT,
        )

        assert hdr.shape == (3,)
        assert np.all(np.diff(hdr) > 0.0)
        assert abs(hdr[2] - chi2.cdf(9.8, 5)) < 0.06

    def test_consistent_results_with_seed(self):
        """Identical data and rng_seed reproduce the value bitwise.

        The old assertion compared to two decimals, which a re-fit on a
        different key would still have satisfied.
        """
        rng = np.random.default_rng(42)
        samples = rng.standard_normal((200, 2))
        injection_params = np.array([0.0, 0.0])
        shared = dict(num_samples=1000, rng_seed=123, **_FAST_FIT)

        hdr1 = compute_injection_hdr(samples, injection_params, **shared)
        hdr2 = compute_injection_hdr(samples.copy(), injection_params.copy(), **shared)

        assert np.array_equal(hdr1, hdr2)

    def test_uniform_distribution(self):
        """The cross-fit reference restores uniformity; the flow reference does not.

        Injections drawn from the same law as the posterior samples must have
        uniform credibility. Ranking them against draws from the *fitted*
        density instead makes every unit of estimator error inflate HDR: on
        seeds 0, 1 and 42 of this configuration the flow reference gave mean
        0.634-0.653 (KS 0.198-0.232) where the held-out reference gave
        0.491-0.515 (KS 0.028-0.040).
        """
        rng = np.random.default_rng(42)
        samples = rng.uniform(-1.0, 1.0, (2000, 2))
        injections = rng.uniform(-1.0, 1.0, (500, 2))
        shared = {
            "num_samples": 20000, "rng_seed": 42, "max_epochs": 100,
            "patience": 20, "show_progress": False,
        }

        held_out = compute_injection_hdr(samples, injections, reference="held_out", **shared)
        from_flow = compute_injection_hdr(samples, injections, reference="flow", **shared)

        assert held_out.shape == injections.shape[:1]
        assert abs(held_out.mean() - 0.5) < 0.05
        held_out_ks = kstest(held_out, "uniform").statistic
        assert held_out_ks < 0.10
        # The default estimator is materially further from uniform on the same
        # data and the same fit budget; that gap is the bias being corrected.
        assert held_out_ks < 0.5 * kstest(from_flow, "uniform").statistic

    def test_multiple_injections_different_locations(self):
        """HDR must be a strictly decreasing function of the fitted log-density.

        This is the definition, and it holds whatever the fit looks like, so it
        catches a reversed comparison or a mis-sorted reference that the old
        0 <= hdr <= 1 assertion could not.
        """
        rng = np.random.default_rng(42)
        samples = rng.standard_normal((2000, 2))
        injection_params = np.array([
            [0.0, 0.0],   # At mean
            [-1.0, -1.0], # Below mean
            [1.0, 1.0],   # Above mean
            [0.0, 1.0],   # Mixed
            [-0.5, 0.5]   # Mixed
        ])

        hdr, flow = compute_injection_hdr(
            samples,
            injection_params,
            num_samples=20000,
            rng_seed=5,
            return_flow=True,
            max_epochs=60,
            patience=15,
            show_progress=False,
        )

        assert hdr.shape == (5,)
        assert len(set(hdr)) == 5
        densities = np.asarray(log_prob(flow, injection_params))
        assert np.array_equal(np.argsort(hdr), np.argsort(-densities))
