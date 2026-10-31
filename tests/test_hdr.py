from types import SimpleNamespace

import numpy as np
import pytest
from flowjax.distributions import Transformed
from paramax import unwrap
from scipy.stats import chi2, kstest

import coppuccino.hdr as hdr_module
from coppuccino.copula_flows import log_prob
from coppuccino.hdr import _in_support_mask, check_in_support, compute_injection_hdr

# A throwaway fit, for tests whose assertion does not depend on fit quality.
# A fit costs about two seconds of tracing and compiling whatever its epoch
# count, so what these tests spend is the number of compute_injection_hdr
# calls, each of which fits a flow.
_FAST_FIT = {"max_epochs": 5, "patience": 2, "show_progress": False}
# Enough training that the fitted density is monotone along a ray from the
# mode; each test using it names the seeds its assertions were verified on.
# A 30-epoch fit kept margin on the newest dependencies but not on the oldest
# supported ones, which train differently.
_TRAINED_FIT = {"max_epochs": 150, "patience": 25, "show_progress": False}


def _split(samples, rng_seed):
    """The (fit, reference) halves compute_injection_hdr uses, rebuilt by hand."""
    order = np.random.default_rng(rng_seed).permutation(len(samples))
    shuffled = samples[order]
    return shuffled[: len(samples) // 2], shuffled[len(samples) // 2:]


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


# The anchor call. 201 rows split 100 / 101, so the reference half is one row
# larger than the fit half and the denominator says which half was used.
_ANCHOR_SEED = 7


@pytest.fixture(scope="module")
def anchor():
    """One compute_injection_hdr call whose every value is known in advance.

    The injections are the held-out reference half itself, in its own order,
    with two rows outside the hull of the samples interleaved. Their in-support
    subset is then exactly the reference array, scored by the same compiled
    ``log_prob``, so every in-support injection ties *bitwise* with itself in
    the reference set. Shared by several tests because each call is a fit.
    """
    samples = np.random.default_rng(42).standard_normal((201, 2))
    _, reference = _split(samples, _ANCHOR_SEED)
    above = [samples[:, 0].max() + 0.05, 0.5]
    below = [-4.0, 0.0]
    injections = np.vstack([[above], reference[:50], [below], reference[50:]])
    sentinel = np.zeros(len(injections), dtype=bool)
    sentinel[[0, 51]] = True
    hdr, flow = compute_injection_hdr(
        samples, injections, rng_seed=_ANCHOR_SEED, return_flow=True, **_FAST_FIT
    )
    return SimpleNamespace(samples=samples, injections=injections,
                           sentinel=sentinel, hdr=hdr, flow=flow)


class TestComputeInjectionHDR:
    """Test the compute_injection_hdr function."""

    def test_values_are_the_strict_rank_against_the_held_out_half(self, anchor):
        """Each value is the fraction of the held-out half the flow scores
        strictly higher than the injection, written to the injection's row.

        The injections are the reference rows themselves (see ``anchor``), so
        the values must be exactly 0, 1/101, ..., 100/101 in some order:

        - strictly higher: each row's tie with itself is not counted, so the
          most probable reference row scores exactly 0; counting ties (>=)
          would shift every value up by 1/101;
        - the held-out half: ranking the rows against the fit half (100 rows),
          or against any other 101 rows (the wrong slice, or fresh draws from
          the flow as coppuccino 1.0.0 did) cannot give that set;
        - higher, not lower: each value also matches the count recomputed from
          the returned flow, which a reversed comparison would mirror;
        - row bookkeeping: the two out-of-hull rows sit before and inside the
          in-support block, so a value written to the wrong row shows up.
        """
        assert isinstance(anchor.flow, Transformed)
        values = anchor.hdr[~anchor.sentinel]
        assert np.array_equal(np.sort(values), np.arange(101) / 101)

        _, reference = _split(anchor.samples, _ANCHOR_SEED)
        reference_log_probs = np.asarray(log_prob(anchor.flow, reference))
        by_hand = np.mean(reference_log_probs[None, :] > reference_log_probs[:, None], axis=1)
        assert np.array_equal(values, by_hand)

    def test_the_support_hull_is_that_of_every_sample(self, anchor):
        """Rows outside the hull of all the samples get exactly 1.0, and rows
        inside it are scored even where they leave the fit half's hull.

        The sentinel rows lie past the sample maximum of coordinate 0 and far
        below its minimum. The held-out half holds extreme rows of some
        coordinate, which lie outside the hull of the fit half (asserted here,
        as the premise). A support test against the fit half alone would send
        those rows to the 1.0 sentinel instead of their rank, which is below
        1.0 by construction.
        """
        sentinel_rows = anchor.injections[anchor.sentinel]
        assert not np.any(_in_support_mask(anchor.samples, sentinel_rows))
        assert np.all(anchor.hdr[anchor.sentinel] == 1.0)

        fit_rows, reference = _split(anchor.samples, _ANCHOR_SEED)
        outside_fit_hull = ~_in_support_mask(fit_rows, reference)
        assert outside_fit_hull.any()
        assert np.all(anchor.hdr[~anchor.sentinel][outside_fit_hull] < 1.0)

    def test_a_second_call_with_the_same_seed_refits_the_same_flow(self, anchor):
        """Identical data and rng_seed reproduce the anchor's ranking exactly.

        This is a second, independent fit. It differs from the anchor only in
        what it is asked -- one injection, as a 1-D array, without
        return_flow -- and its value must equal the count recomputed from the
        *anchor's* flow. So rng_seed alone decides the split and the fit, the
        single-injection path agrees with the batched one, and return_flow does
        not change the values. The old assertion compared to two decimals,
        which a re-fit on a different key would still have satisfied.
        """
        injection = np.array([0.7, -0.4])
        hdr = compute_injection_hdr(
            anchor.samples.copy(), injection.copy(), rng_seed=_ANCHOR_SEED, **_FAST_FIT
        )

        assert isinstance(hdr, np.ndarray)
        assert hdr.shape == (1,)
        _, reference = _split(anchor.samples, _ANCHOR_SEED)
        reference_log_probs = np.asarray(log_prob(anchor.flow, reference))
        injection_log_prob = float(np.asarray(log_prob(anchor.flow, injection[None, :]))[0])
        assert hdr[0] == np.mean(reference_log_probs > injection_log_prob)

    def test_multiple_injections(self):
        """HDR rises monotonically along a ray leaving the mode.

        Radii 0, 1.5 and 3.0 of a 2-D standard normal; the outermost point's
        analytic credibility is 0.989 and the estimate landed within 0.015 of
        it on seeds 0, 1 and 42, on both the newest and the oldest supported
        dependencies. A 5-epoch fit missed by 0.12 on seed 42, and a 30-epoch
        one by 0.087 on seed 1 under the oldest dependencies.
        """
        rng = np.random.default_rng(42)
        samples = rng.standard_normal((3000, 2))
        radii = np.array([0.0, 1.5, 3.0])
        injection_params = np.column_stack([radii, radii]) / np.sqrt(2.0)

        hdr = compute_injection_hdr(
            samples,
            injection_params,
            rng_seed=42,
            **_TRAINED_FIT,
        )

        assert hdr.shape == (3,)
        assert np.all(np.diff(hdr) > 0.0)
        assert hdr[0] < 0.6  # nothing at the mode may look like a tail point
        assert abs(hdr[2] - chi2.cdf(9.0, 2)) < 0.06

    def test_non_finite_injections_are_rejected(self):
        """NaN/Inf injections must not be laundered into the 1.0 sentinel.

        They fail the hull test, so before this check they short-circuited to
        exactly 1.0 -- the value a legitimately out-of-support injection gets.
        """
        rng = np.random.default_rng(42)
        samples = rng.standard_normal((200, 2))

        for bad in ([np.nan, 0.0], [np.inf, 0.0], [-np.inf, 0.0]):
            with pytest.raises(ValueError, match="injection_params contains 1 non-finite"):
                compute_injection_hdr(samples, np.array(bad), **_FAST_FIT)

        # The message locates the offending cell of a batch.
        batch = np.array([[0.0, 0.0], [1.0, np.nan], [0.5, 0.5]])
        with pytest.raises(ValueError, match=r"first at row 1, column 1 with value nan"):
            compute_injection_hdr(samples, batch, **_FAST_FIT)

    def test_non_finite_samples_are_rejected(self):
        """The same check on the posterior samples names the offending cell."""
        rng = np.random.default_rng(42)
        samples = rng.standard_normal((200, 2))
        samples[13, 1] = np.inf
        with pytest.raises(ValueError, match=r"samples contains 1 non-finite.*row 13, column 1"):
            compute_injection_hdr(samples, np.zeros(2), **_FAST_FIT)

    def test_non_finite_reference_scores_raise(self, anchor, monkeypatch):
        """HDR's own guard, isolated from the fitter's.

        Any route to a NaN reference score -- a diverged fit, a marginal that
        underflows -- would otherwise make searchsorted return 0 for every
        injection and hand back a column of 1.0 sentinels. The reference half
        is scored first, so NaN-ing only that first ``log_prob`` call reaches
        this guard alone. The fit is stubbed with the anchor's flow: the guard
        sits downstream of it, and the patched ``log_prob`` decides what the
        guard sees.
        """
        rng = np.random.default_rng(0)
        samples = rng.standard_normal((200, 2))
        real_log_prob = hdr_module.log_prob
        calls = []

        def nan_reference(flow, x):
            calls.append(len(x))
            scores = np.asarray(real_log_prob(flow, x), dtype=float)
            return np.full_like(scores, np.nan) if len(calls) == 1 else scores

        monkeypatch.setattr(hdr_module, "normalizing_flows_fit",
                            lambda rows, **kwargs: anchor.flow)
        monkeypatch.setattr(hdr_module, "log_prob", nan_reference)
        with pytest.raises(RuntimeError, match="100 of 100 reference points non-finitely"):
            compute_injection_hdr(samples, np.zeros(2), **_FAST_FIT)
        assert calls == [100]

    def test_nan_injection_scores_raise(self, anchor, monkeypatch):
        """The injection-side guard, which neither of the other two reaches.

        A real diverged fit is refused by the fitter's own divergence check
        (test_copula_flows' TestDivergenceGuard) before HDR sees it, and the
        reference guard above only sees the reference array, so a flow that
        scores its own draws finitely but an in-support injection NaN reaches
        this guard alone. It has to raise, because NaN sorts *after* every
        finite reference point: the count of more-probable points is zero and
        the unguarded answer is HDR 0.0, i.e. the injection is reported as
        sitting at the mode. With this one guard disabled and nothing else
        changed, the NaN row of this exact call comes back as 0.0. The fit is
        stubbed with the anchor's flow, as in the reference-guard test.
        """
        # The premise of the guard: searchsorted places NaN past every finite
        # score, so the unguarded count of more-probable reference points is 0.
        reference = np.sort(np.array([-3.0, -1.0, 0.5, 2.0]))
        assert np.searchsorted(reference, np.nan, side="right") == len(reference)

        rng = np.random.default_rng(0)
        samples = rng.standard_normal((200, 2))
        injections = np.array([[0.0, 0.0], [1.0, 1.0]])
        real_log_prob = hdr_module.log_prob

        def nan_first_injection(flow, x):
            # The held-out reference goes through log_prob too; leave it finite
            # so the reference guard stays quiet and this one is reached.
            scores = np.asarray(real_log_prob(flow, x), dtype=float).copy()
            if len(x) == len(injections):
                scores[0] = np.nan
            return scores

        monkeypatch.setattr(hdr_module, "normalizing_flows_fit",
                            lambda rows, **kwargs: anchor.flow)
        monkeypatch.setattr(hdr_module, "log_prob", nan_first_injection)
        with pytest.raises(RuntimeError, match="NaN log-density for 1 of 2 in-support"):
            compute_injection_hdr(samples, injections, rng_seed=0, **_FAST_FIT)

    def test_a_batch_entirely_outside_the_hull_never_scores_an_injection(
            self, anchor, monkeypatch):
        """Every row gets the 1.0 sentinel, and the flow scores only the
        reference half.

        The docstring promises out-of-hull injections are never evaluated, and
        a batch with no in-support row is the case where scoring the empty
        remainder would still reach the flow. The fit is stubbed with the
        anchor's flow, as in the guard tests above.
        """
        samples = np.random.default_rng(0).standard_normal((200, 2))
        real_log_prob = hdr_module.log_prob
        calls = []

        def counting_log_prob(flow, x):
            calls.append(len(x))
            return real_log_prob(flow, x)

        monkeypatch.setattr(hdr_module, "normalizing_flows_fit",
                            lambda rows, **kwargs: anchor.flow)
        monkeypatch.setattr(hdr_module, "log_prob", counting_log_prob)
        far = np.array([[10.0, 0.0], [0.0, -10.0]])
        hdr = compute_injection_hdr(samples, far, **_FAST_FIT)
        np.testing.assert_array_equal(hdr, [1.0, 1.0])
        assert calls == [100]

    def test_too_few_rows_and_removed_options_are_rejected(self):
        """The split needs 42 rows, and 1.0.0's num_samples cannot sneak through.

        Every option after ``injection_params`` is keyword-only, so a 1.0.0
        positional ``num_samples`` raises instead of becoming the seed, and a
        ``num_samples=`` keyword reaches ``normalizing_flows_fit``, which
        rejects it by name -- which is also what shows that ``nf_kwargs`` are
        forwarded to the fit rather than swallowed.
        """
        rng = np.random.default_rng(42)
        samples = rng.standard_normal((200, 2))
        injection = np.zeros(2)

        with pytest.raises(ValueError, match="at least 42 rows; samples has 30"):
            compute_injection_hdr(rng.standard_normal((30, 2)), injection, **_FAST_FIT)
        with pytest.raises(TypeError, match="positional"):
            compute_injection_hdr(samples, injection, 50_000)
        with pytest.raises(TypeError, match="num_samples"):
            compute_injection_hdr(samples, injection, num_samples=50_000, **_FAST_FIT)

    def test_the_row_floor_is_the_one_the_fit_half_needs(self):
        """Each half has to clear normalizing_flows_fit's own 21-row floor.

        At 40 and 41 rows the fit half holds 20, so a 40-row floor let these
        through only for the fit to refuse them with "Insufficient samples",
        which names neither the split nor the row count the caller passed. 41
        is the largest count refused; the next test fits 42.
        """
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="at least 42 rows; samples has 41"):
            compute_injection_hdr(rng.standard_normal((41, 2)), np.zeros(2), **_FAST_FIT)

    def test_the_smallest_accepted_input_fits_with_valid_prior_bounds(self):
        """42 rows are fitted and scored, and valid prior_bounds reach the fit.

        compute_injection_hdr checks prior_bounds against every row itself and
        then forwards the validated array, so the fitted marginals have to
        carry the bounds; every other prior_bounds test here is a rejection.
        One finite side is enough to see the forwarding (an infinite side is
        recorded as None), and each finite wall adds compile time to the fit.
        The reference half holds 21 rows, so the value is a multiple of 1/21.
        """
        rng = np.random.default_rng(0)
        bounds = [[-5.0, np.inf], [-np.inf, np.inf]]
        hdrs, flow = compute_injection_hdr(
            rng.standard_normal((42, 2)), np.zeros(2),
            return_flow=True, prior_bounds=bounds, **_FAST_FIT,
        )

        assert hdrs.shape == (1,)
        assert 0.0 <= hdrs[0] < 1.0
        assert abs(hdrs[0] * 21 - round(hdrs[0] * 21)) < 1e-9
        marginals = [unwrap(b) for b in flow.bijection.bijections]
        assert [(m.prior_low, m.prior_high) for m in marginals] == [
            (-5.0, None), (None, None)]

    def test_prior_bounds_are_checked_against_every_row(self):
        """A row outside prior_bounds is a ValueError wherever the split puts it.

        Checked on the fit half only, the row raised the right error when it
        landed there (seeds 0 and 2) and otherwise reached the reference half,
        which scored it -inf and raised "the fit likely diverged" (seeds 1 and
        3): the wrong exception type, with advice to lower learning_rate. The
        check now runs before the split, so only a seed of the second kind can
        tell it apart from the fit's own check; seed 1 is one.
        """
        samples = np.random.default_rng(0).uniform(-1.0, 1.0, (200, 2))
        samples[0, 0] = 1.5
        with pytest.raises(ValueError, match="outside prior_bounds"):
            compute_injection_hdr(samples, np.zeros(2), rng_seed=1,
                                  prior_bounds=[[-1.0, 1.0], [-1.0, 1.0]], **_FAST_FIT)

    def test_0d_injection_raises_error(self):
        """Test that 0D injection parameters raise an error."""
        rng = np.random.default_rng(42)
        samples = rng.standard_normal((200, 2))
        injection_params = np.array(0.0)  # 0D array

        with pytest.raises(ValueError, match="injection_params must be at least 1D"):
            compute_injection_hdr(samples, injection_params, **_FAST_FIT)

    def test_injection_width_mismatch_raises(self):
        """A 2D injection array with the wrong parameter count is rejected."""
        rng = np.random.default_rng(42)
        samples = rng.standard_normal((50, 2))

        with pytest.raises(ValueError, match="injection_params length must match"):
            compute_injection_hdr(samples, np.zeros((3, 5)), **_FAST_FIT)

    def test_3d_injection_raises(self):
        """Anything beyond a batch of injections is rejected by shape."""
        rng = np.random.default_rng(42)
        samples = rng.standard_normal((50, 2))

        with pytest.raises(ValueError, match="must be 1D or 2D"):
            compute_injection_hdr(samples, np.zeros((2, 2, 2)), **_FAST_FIT)

    def test_1d_parameter_space(self):
        """In one dimension the credibility has a closed form to check against.

        For a standard normal the HDR of x is chi2_1.cdf(x**2): 0 at the mode,
        0.683 at x=1, 0.954 at x=2. The estimate at x=2 was within 0.018 of the
        truth on seeds 0-5 and 42 (the assertion allows 0.06). The near-mode
        points are only checked to rank below it: a 1-D normal's log-density
        falls just 0.125 nats by |x|=0.5, so small wiggles in the fit move a
        near-mode point's HDR a long way. At x=0 it ranged 0.05-0.71 over
        those seeds; it stayed below x=1 on all of them, but by only 0.004 on
        seed 42.

        In 1-D the marginal map does all the work and the copula has nothing
        to learn, so a 5-epoch fit returns the same three values on seed 42 as
        150 epochs did. The row count is what sets the accuracy: at 2,000 or
        4,000 rows seed 42 still passed, but seeds 1 and 5 missed by 0.16 and
        0.07, so the 8,000 rows stay.
        """
        rng = np.random.default_rng(42)
        samples = rng.standard_normal((8000, 1))
        injection_params = np.array([[0.0], [1.0], [2.0]])

        hdr = compute_injection_hdr(
            samples,
            injection_params,
            rng_seed=42,
            **_FAST_FIT,
        )

        assert hdr.shape == (3,)
        assert hdr[2] > max(hdr[0], hdr[1])
        assert abs(hdr[2] - chi2.cdf(4.0, 1)) < 0.06

    def test_high_dimensional_parameter_space(self):
        """The same radial ordering has to survive five dimensions.

        Points at squared radius 0, 4.05 and 9.8 of a 5-D standard normal; the
        outermost analytic credibility is 0.919 and the estimate stayed within
        0.027 of it on seeds 0, 1 and 42, on both the newest and the oldest
        supported dependencies (the assertion allows 0.06). A 5-epoch fit
        missed by 0.11 on seed 42, and a 30-epoch one reached 0.060 there under
        the oldest dependencies.
        """
        rng = np.random.default_rng(42)
        samples = rng.standard_normal((2000, 5))
        injection_params = np.array([np.zeros(5), np.full(5, 0.9), np.full(5, 1.4)])

        hdr = compute_injection_hdr(
            samples,
            injection_params,
            rng_seed=42,
            **_TRAINED_FIT,
        )

        assert hdr.shape == (3,)
        assert np.all(np.diff(hdr) > 0.0)
        assert abs(hdr[2] - chi2.cdf(9.8, 5)) < 0.06

    def test_uniform_distribution(self):
        """Injections drawn from the posterior's own law get uniform credibility.

        That holds for any fitted density because the reference half is a set
        of posterior draws the fit never saw, so a 5-epoch fit is enough. On
        seeds 0, 1 and 42 of this configuration the mean was 0.500-0.531 (KS
        0.029-0.061); ranking the same injections against fresh draws from the
        fitted flow instead (the method coppuccino 1.0.0 used) gave 0.824-0.887
        (KS 0.51-0.66).
        """
        rng = np.random.default_rng(42)
        samples = rng.uniform(-1.0, 1.0, (2000, 2))
        injections = rng.uniform(-1.0, 1.0, (500, 2))

        hdrs = compute_injection_hdr(samples, injections, rng_seed=42, **_FAST_FIT)

        assert hdrs.shape == injections.shape[:1]
        assert abs(hdrs.mean() - 0.5) < 0.05
        assert kstest(hdrs, "uniform").statistic < 0.10
