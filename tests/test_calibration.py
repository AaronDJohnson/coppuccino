"""Tests for post-fit Gaussian edge-bin calibration."""

import os
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from equinox import filter_jit
from flowjax.bijections import Stack
from flowjax.distributions import MultivariateNormal, Transformed
from paramax import non_trainable, unwrap
from scipy.stats import norm, spearmanr

from coppuccino import (
    calibrate_edge_bins,
    load_flow,
    log_prob,
    normalizing_flows_fit,
    normalizing_flows_fit_fleet,
    sample,
    sample_and_log_prob,
    save_flow,
)
from coppuccino._stats import wilson_interval
from coppuccino.bijections import SmoothMarginalToGaussian, make_smooth_marginal
from coppuccino.calibration import (
    _EdgeMarginalCalibration,
    _window_derivatives,
)

_WILSON_PROBABILITIES = [0.0, 1e-6, 1e-3, 0.01, 0.05, 0.2, 0.5, 0.8, 0.95, 0.99, 1.0]
_WILSON_SIZES = [1.0, 5.0, 20.0, 137.5, 1000.0, 1e5]
_WILSON_CONFIDENCES = [0.5, 0.68, 0.9, 0.95, 0.99]


def _wilson_by_quadratic_roots(probability, effective_n, confidence_level):
    """The Wilson endpoints as roots of the score equation, not as centre +/- radius.

    Wilson's interval is by definition the set of ``q`` satisfying
    ``(p - q)^2 <= z^2 q(1 - q)/n``, i.e. the roots of
    ``(1 + z^2/n) q^2 - (2p + z^2/n) q + p^2``.  Solving that quadratic is
    algebraically independent of the closed form the helper evaluates, so it
    catches a mis-stated centre shift or half-width rather than merely
    restating the implementation.
    """
    z = float(norm.ppf(0.5 + confidence_level / 2.0))
    a = 1.0 + z**2 / effective_n
    b = -(2.0 * probability + z**2 / effective_n)
    c = probability**2
    root = np.sqrt(b**2 - 4.0 * a * c)
    return np.clip((-b - root) / (2.0 * a), 0.0, 1.0), np.clip(
        (-b + root) / (2.0 * a), 0.0, 1.0
    )


def test_wilson_interval_has_exactly_one_implementation():
    """Both consumers must resolve to the shared leaf helper, not private copies.

    The interval formula was previously inlined in ``calibration`` and again in
    ``copula_flows``, and only the ``copula_flows`` copy was ever exercised.
    """
    import coppuccino.calibration as calibration_module
    import coppuccino.copula_flows as copula_flows_module

    assert calibration_module.wilson_interval is wilson_interval
    assert copula_flows_module.wilson_interval is wilson_interval
    assert not hasattr(calibration_module, "_wilson_interval")
    assert not hasattr(copula_flows_module, "_wilson_interval")


@pytest.mark.parametrize("probability", _WILSON_PROBABILITIES)
def test_wilson_interval_matches_the_score_equation_roots(probability):
    """Sweep every effective size and confidence against an independent oracle."""
    for effective_n in _WILSON_SIZES:
        for confidence_level in _WILSON_CONFIDENCES:
            case = f"p={probability!r} n={effective_n!r} conf={confidence_level!r}"
            lower, upper = wilson_interval(probability, effective_n, confidence_level)
            expected_lower, expected_upper = _wilson_by_quadratic_roots(
                probability, effective_n, confidence_level
            )
            # The quadratic formula loses digits to cancellation at extreme p/n,
            # so the oracle is only good to ~1e-11; a real formula error moves
            # the endpoints by more than 1e-3.
            assert float(lower) == pytest.approx(expected_lower, abs=1e-9), case
            assert float(upper) == pytest.approx(expected_upper, abs=1e-9), case
            assert 0.0 <= float(lower) <= float(upper) <= 1.0, case
            # The observed proportion lies inside its own interval; at p of
            # exactly 0 or 1 the containing endpoint meets p to within rounding.
            assert float(lower) <= probability + 1e-15, case
            assert probability <= float(upper) + 1e-15, case


@pytest.mark.parametrize("confidence_level", _WILSON_CONFIDENCES)
def test_wilson_interval_stays_in_the_unit_interval_at_p_zero_and_one(confidence_level):
    """Degenerate proportions must not leak outside [0, 1] through rounding.

    Both endpoints are analytically in [0, 1], but the unclamped closed form
    returns e.g. 1.0000000000000002 at ``probability=1``, which downstream code
    treating the result as a probability has no reason to expect.
    """
    for effective_n in _WILSON_SIZES:
        case = f"n={effective_n!r} conf={confidence_level!r}"
        zero_lower, zero_upper = wilson_interval(0.0, effective_n, confidence_level)
        one_lower, one_upper = wilson_interval(1.0, effective_n, confidence_level)

        assert 0.0 <= float(zero_lower) <= float(zero_upper) <= 1.0, case
        assert 0.0 <= float(one_lower) <= float(one_upper) <= 1.0, case
        assert float(zero_lower) == pytest.approx(0.0, abs=4e-16), case
        assert float(one_upper) == pytest.approx(1.0, abs=4e-16), case
        # A zero count is not certainty: the interval must stay non-degenerate.
        assert 0.0 < float(zero_upper) < 1.0, case
        assert 0.0 < float(one_lower) < 1.0, case
        assert float(zero_upper) == pytest.approx(1.0 - float(one_lower), rel=1e-8), case


def test_wilson_interval_broadcasts_elementwise():
    """The array form must agree bitwise with a scalar loop, and keep shape."""
    probabilities = np.array(_WILSON_PROBABILITIES)
    lower, upper = wilson_interval(probabilities, 137.5, 0.95)
    assert lower.shape == probabilities.shape
    assert upper.shape == probabilities.shape
    for index, probability in enumerate(_WILSON_PROBABILITIES):
        scalar_lower, scalar_upper = wilson_interval(probability, 137.5, 0.95)
        assert float(lower[index]) == float(scalar_lower)
        assert float(upper[index]) == float(scalar_upper)

    grid = wilson_interval(probabilities.reshape(-1, 1), 137.5, 0.95)
    assert grid[0].shape == (len(probabilities), 1)


def test_wilson_interval_width_shrinks_as_the_root_of_effective_n():
    """A tenfold ESS penalty must widen the gate by sqrt(10).

    This is the relationship ``reference_ess`` leans on: passing an effective
    size instead of a raw row count is only a meaningful correction if the gate
    scales the way an independent sample of that size would.
    """
    narrow_lower, narrow_upper = wilson_interval(0.05, 10_000.0, 0.95)
    wide_lower, wide_upper = wilson_interval(0.05, 1_000.0, 0.95)
    ratio = float(wide_upper - wide_lower) / float(narrow_upper - narrow_lower)
    assert ratio == pytest.approx(np.sqrt(10.0), rel=0.01)


def _bounded_reference_and_flow(
    n=8000,
    seed=0,
    inner_loc=(0.10, -0.10),
    inner_covariance=((1.44, 0.72), (0.72, 0.81)),
):
    """Gaussian copula with uniform marginals and a deliberately biased inner law.

    ``inner_loc``/``inner_covariance`` select how the latent law drifts away from
    the reference, which is what decides per-boundary activity.
    """
    rng = np.random.default_rng(seed)
    latent = rng.multivariate_normal(np.zeros(2), np.array([[1.0, 0.7], [0.7, 1.0]]), size=n)
    reference = norm.cdf(latent)

    marginals = []
    for dim in range(2):
        theta_of_z, z_of_theta, log_dtheta_dz, clamps = make_smooth_marginal(
            reference[:, dim], prior_low=0.0, prior_high=1.0
        )
        marginals.append(
            non_trainable(
                SmoothMarginalToGaussian(
                    reference[:, dim],
                    theta_of_z,
                    z_of_theta,
                    log_dtheta_dz,
                    prior_low=0.0,
                    prior_high=1.0,
                    x_clamp_lo=clamps[0],
                    x_clamp_hi=clamps[1],
                )
            )
        )

    # The copula remains useful, but these shifted/scaled latent marginals make
    # the physical outer bins visibly wrong before calibration.
    inner = MultivariateNormal(
        loc=jnp.asarray(inner_loc),
        covariance=jnp.asarray(inner_covariance),
    )
    flow = Transformed(inner, Stack(marginals))
    edges = [np.linspace(0.0, 1.0, 21), np.linspace(0.0, 1.0, 21)]
    return reference, flow, edges


def test_compact_rqs_roundtrip_anchors_and_identity():
    lower_x = np.array([-4.0, -2.2, -1.0])
    lower_y = np.array([-4.0, -1.9, -1.0])
    upper_x = np.array([1.0, 2.0, 4.0])
    upper_y = np.array([1.0, 2.3, 4.0])
    calibration = _EdgeMarginalCalibration(
        lower_x,
        lower_y,
        _window_derivatives(lower_x, lower_y),
        upper_x,
        upper_y,
        _window_derivatives(upper_x, upper_y),
    )

    anchors = np.array([-2.2, 2.0])
    mapped = np.asarray(jax.vmap(calibration.transform)(jnp.asarray(anchors)))
    assert np.max(np.abs(mapped - np.array([-1.9, 2.3]))) < 1e-12

    grid = np.linspace(-6.0, 6.0, 4001)
    values, forward_ld = jax.vmap(calibration.transform_and_log_det)(grid)
    recovered, inverse_ld = jax.vmap(calibration.inverse_and_log_det)(values)
    assert np.all(np.diff(np.asarray(values)) > 0.0)
    assert np.max(np.abs(np.asarray(recovered) - grid)) < 1e-10
    assert np.max(np.abs(np.asarray(forward_ld + inverse_ld))) < 1e-10

    check = jnp.array([-3.0, -1.6, 1.5, 3.0])
    h = 1e-6
    numerical = (
        jax.vmap(calibration.transform)(check + h) - jax.vmap(calibration.transform)(check - h)
    ) / (2.0 * h)
    _, claimed_log_det = jax.vmap(calibration.transform_and_log_det)(check)
    assert np.max(np.abs(np.asarray(numerical / jnp.exp(claimed_log_det) - 1.0))) < 1e-8

    identity = np.array([-5.0, 0.0, 5.0])
    identity_values, identity_ld = jax.vmap(calibration.transform_and_log_det)(identity)
    assert np.array_equal(np.asarray(identity_values), identity)
    assert np.array_equal(np.asarray(identity_ld), np.zeros(3))


@pytest.fixture(scope="module")
def calibrated_case():
    reference, flow, edges = _bounded_reference_and_flow()
    calibrated, report = calibrate_edge_bins(
        flow,
        reference,
        edges,
        edge_bins=1,
        core_quantiles=(0.05, 0.95),
        n_calibration_samples=80_000,
        rng_seed=4,
        return_report=True,
    )
    return reference, flow, calibrated, edges, report


def test_edge_masses_improve_without_changing_copula(calibrated_case):
    reference, flow, calibrated, edges, report = calibrated_case
    before = sample(flow, 120_000, rng_seed=10)
    after = sample(calibrated, 120_000, rng_seed=10)

    for dim, edge in enumerate(edges):
        lower, upper = edge[1], edge[-2]
        target = np.array([np.mean(reference[:, dim] < lower), np.mean(reference[:, dim] < upper)])
        pre = np.array([np.mean(before[:, dim] < lower), np.mean(before[:, dim] < upper)])
        post = np.array([np.mean(after[:, dim] < lower), np.mean(after[:, dim] < upper)])
        dimension_report = report["dimensions"][dim]
        intervals = np.vstack(
            [
                dimension_report["lower_confidence_interval"],
                dimension_report["upper_confidence_interval"],
            ]
        )
        active = np.concatenate(
            [dimension_report["active_lower"], dimension_report["active_upper"]]
        )
        # Independent evaluation Monte Carlo gets a small allowance around the
        # reference interval selected during calibration.
        assert np.all(post >= intervals[:, 0] - 0.003)
        assert np.all(post <= intervals[:, 1] + 0.003)
        assert np.all(np.abs(post[active] - target[active]) < np.abs(pre[active] - target[active]))

    # A componentwise monotone map leaves the sample copula unchanged.
    rho_before = spearmanr(before).statistic
    rho_after = spearmanr(after).statistic
    assert abs(rho_before - rho_after) < 1e-12


def test_identity_core_and_density_contract(calibrated_case):
    _, flow, calibrated, _, report = calibrated_case
    calibration_stack = calibrated.base_dist.bijection
    raw_inner = calibrated.base_dist.base_dist
    z = np.asarray(filter_jit(raw_inner.sample)(jax.random.key(21), (20_000,)))
    corrected, calibration_ld = jax.vmap(calibration_stack.transform_and_log_det)(z)
    corrected = np.asarray(corrected)
    calibration_ld = np.asarray(calibration_ld)

    core_limits = np.asarray(
        [dimension["identity_core_latent"] for dimension in report["dimensions"]]
    )
    in_core = np.all(
        (z >= core_limits[:, 0][None, :]) & (z <= core_limits[:, 1][None, :]),
        axis=1,
    )
    assert in_core.sum() > 1000
    assert np.array_equal(corrected[in_core], z[in_core])
    assert np.array_equal(calibration_ld[in_core], np.zeros(in_core.sum()))

    for calibration_bij, marginal_bij in zip(
        calibration_stack.bijections, flow.bijection.bijections
    ):
        calibration = unwrap(calibration_bij)
        marginal = unwrap(marginal_bij)
        clamp_low = float(marginal.z_of_theta(jnp.asarray(marginal.x_clamp_lo)))
        clamp_high = float(marginal.z_of_theta(jnp.asarray(marginal.x_clamp_hi)))
        assert float(calibration.lower_x[0]) > clamp_low
        assert float(calibration.upper_x[-1]) < clamp_high

    physical_core = np.asarray(jax.vmap(flow.bijection.transform)(jnp.asarray(z[in_core][:500])))
    before_lp = log_prob(flow, physical_core)
    after_lp = log_prob(calibrated, physical_core)
    assert np.array_equal(after_lp, before_lp)

    samples, sampled_lp = sample_and_log_prob(calibrated, n_samples=10_000, rng_seed=22)
    recomputed_lp = log_prob(calibrated, samples)
    assert np.max(np.abs(sampled_lp - recomputed_lp)) < 1e-8


def test_save_load_roundtrip(calibrated_case):
    reference, _, calibrated, edges, _ = calibrated_case
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "calibrated.pkl")
        save_flow(calibrated, path)
        loaded = load_flow(path)

    original_samples = sample(calibrated, 2000, rng_seed=30)
    loaded_samples = sample(loaded, 2000, rng_seed=30)
    assert np.array_equal(loaded_samples, original_samples)
    assert np.array_equal(
        log_prob(loaded, original_samples), log_prob(calibrated, original_samples)
    )
    stack = loaded.base_dist.bijection
    assert all(
        isinstance(unwrap(bijection), _EdgeMarginalCalibration) for bijection in stack.bijections
    )
    with pytest.raises(ValueError, match="already"):
        calibrate_edge_bins(
            loaded,
            reference,
            edges,
            n_calibration_samples=1000,
        )


def test_guards_and_recalibration(calibrated_case):
    reference, flow, calibrated, edges, _ = calibrated_case
    with pytest.raises(ValueError, match="already"):
        calibrate_edge_bins(
            calibrated,
            reference,
            edges,
            n_calibration_samples=1000,
        )
    with pytest.raises(ValueError, match="one edge array per dimension"):
        calibrate_edge_bins(
            flow,
            reference,
            np.linspace(0.0, 1.0, 21),
            n_calibration_samples=1000,
        )


def test_non_smooth_marginals_are_rejected_by_what_is_actually_required():
    """The message must name the marginal family, not the removed copula_space.

    ``copula_space`` was dropped in 3.0.0 and survives only as a shim that
    accepts ``"gaussian"`` as a no-op and raises for everything else, so the
    old text told the user to pass a deprecated argument that would not have
    changed the outcome.
    """
    rng = np.random.default_rng(3)
    reference = norm.cdf(
        rng.multivariate_normal(np.zeros(2), np.array([[1.0, 0.7], [0.7, 1.0]]), size=600)
    )
    flow = normalizing_flows_fit(
        reference,
        marginal="rqs",
        prior_bounds=np.array([[0.0, 1.0], [0.0, 1.0]]),
        max_epochs=2,
        patience=1,
        rng_seed=1,
        show_progress=False,
    )

    with pytest.raises(ValueError) as caught:
        calibrate_edge_bins(
            flow, reference, [np.linspace(0.0, 1.0, 21)] * 2, n_calibration_samples=1000
        )

    message = str(caught.value)
    assert "copula_space" not in message
    assert "marginal='smooth'" in message
    # The offending coordinate and what it actually holds are both named.
    assert "Dimension 0" in message
    assert "EmpiricalMarginalToGaussian" in message


def test_a_fleet_gets_the_same_redirect_as_the_other_verbs():
    """calibrate_edge_bins(fleet, ...) must say why a fleet cannot be calibrated.

    sample/log_prob/sample_and_log_prob/save_flow all redirect a FleetFlow by
    name; this verb answered with the generic "requires a fitted Transformed
    flow", which reads like the caller passed the wrong kind of array.
    """
    rng = np.random.default_rng(0)
    chains = rng.normal(size=(3, 300, 2))
    fleet = normalizing_flows_fit_fleet(
        chains, prior_bounds=np.array([[-6.0, 6.0], [-6.0, 6.0]]), max_epochs=1, rng_seed=0
    )

    with pytest.raises(TypeError) as caught:
        calibrate_edge_bins(
            fleet,
            rng.normal(size=(200, 2)),
            [np.linspace(-6.0, 6.0, 21)] * 2,
            n_calibration_samples=1000,
        )

    message = str(caught.value)
    assert "FleetFlow of 3 sources" in message
    assert "normalizing_flows_fit(" in message


def test_statistically_compatible_flow_is_exact_noop():
    reference, template, edges = _bounded_reference_and_flow()
    matching_inner = MultivariateNormal(
        loc=jnp.zeros(2),
        covariance=jnp.array([[1.0, 0.7], [0.7, 1.0]]),
    )
    matching = Transformed(matching_inner, template.bijection)
    calibrated, report = calibrate_edge_bins(
        matching,
        reference,
        edges,
        n_calibration_samples=40_000,
        rng_seed=40,
        return_report=True,
    )
    assert not any(
        np.any(dimension["active_lower"]) or np.any(dimension["active_upper"])
        for dimension in report["dimensions"]
    )
    before = sample(matching, 10_000, rng_seed=41)
    after = sample(calibrated, 10_000, rng_seed=41)
    assert np.array_equal(after, before)
    assert np.array_equal(log_prob(calibrated, before), log_prob(matching, before))


def test_sparse_reference_and_unresolved_calibration_are_rejected():
    reference, flow, _ = _bounded_reference_and_flow()
    sparse_edges = [np.array([0.0, 0.001, 0.5, 0.999, 1.0])] * 2
    with pytest.raises(ValueError, match="effective count"):
        calibrate_edge_bins(
            flow,
            reference,
            sparse_edges,
            n_calibration_samples=1000,
            rng_seed=42,
        )

    unresolved_edges = [np.array([0.0, 0.01, 0.5, 0.99, 1.0])] * 2
    with pytest.raises(ValueError, match="expected calibration draws"):
        calibrate_edge_bins(
            flow,
            reference,
            unresolved_edges,
            n_calibration_samples=1000,
            rng_seed=43,
        )


def test_force_hard_fits_raw_targets_and_downgrades_guards():
    """force=True must warn where the gated path raises, and skip the Wilson gate."""
    reference, flow, _ = _bounded_reference_and_flow()
    sparse_edges = [np.array([0.0, 0.001, 0.5, 0.999, 1.0])] * 2

    with pytest.warns(UserWarning) as caught:
        _, report = calibrate_edge_bins(
            flow,
            reference,
            sparse_edges,
            n_calibration_samples=1000,
            rng_seed=42,
            force=True,
            return_report=True,
        )
    messages = " ".join(str(warning.message) for warning in caught)
    assert "effective count" in messages
    assert "expected calibration draws" in messages

    assert report["force"] is True
    for dimension_report in report["dimensions"]:
        # Every boundary is hard-fitted to the raw empirical CDF, so all sides
        # are active and no target is pulled back to a Wilson edge.
        assert np.all(dimension_report["active_lower"])
        assert np.all(dimension_report["active_upper"])
        assert np.array_equal(
            dimension_report["fitted_lower_cdf"], dimension_report["target_lower_cdf"]
        )
        assert np.array_equal(
            dimension_report["fitted_upper_cdf"], dimension_report["target_upper_cdf"]
        )


def test_reference_ess_widens_intervals_and_moves_targets():
    """A correlated-sample ESS must widen the Wilson gate by the usual sqrt factor."""
    reference, flow, edges = _bounded_reference_and_flow()
    ess = len(reference) // 10
    _, independent = calibrate_edge_bins(
        flow, reference, edges, n_calibration_samples=40_000, rng_seed=4, return_report=True
    )
    _, correlated = calibrate_edge_bins(
        flow,
        reference,
        edges,
        n_calibration_samples=40_000,
        rng_seed=4,
        reference_ess=ess,
        return_report=True,
    )

    for independent_dim, correlated_dim in zip(
        independent["dimensions"], correlated["dimensions"]
    ):
        assert independent_dim["reference_ess"] == float(len(reference))
        assert correlated_dim["reference_ess"] == float(ess)
        for key in ("lower_confidence_interval", "upper_confidence_interval"):
            widths = np.diff(correlated_dim[key], axis=1) / np.diff(independent_dim[key], axis=1)
            assert np.all(np.abs(widths - np.sqrt(10.0)) < 0.1)
        # A wider gate moves every corrected boundary to a different target.
        assert not np.array_equal(
            correlated_dim["fitted_lower_cdf"], independent_dim["fitted_lower_cdf"]
        )
        assert not np.array_equal(
            correlated_dim["fitted_upper_cdf"], independent_dim["fitted_upper_cdf"]
        )


def test_reference_ess_is_validated():
    reference, flow, edges = _bounded_reference_and_flow(n=800)
    for bad, message in (
        ([1000.0, 1000.0, 1000.0], "scalar or have shape"),
        (0.0, "finite and positive"),
        (np.inf, "finite and positive"),
        (10 * len(reference), "cannot exceed"),
    ):
        with pytest.raises(ValueError, match=message):
            calibrate_edge_bins(
                flow,
                reference,
                edges,
                n_calibration_samples=1000,
                reference_ess=bad,
            )


def test_resolution_guard_measures_gaps_between_all_anchors():
    """Inactive anchors are still knots, so their gaps count toward resolution.

    The tuned inner law leaves dimension 0's outermost lower boundary compatible
    (inactive) while the next one in is corrected, so the guard's anchor list and
    the spline's knot list disagree.
    """
    reference, flow, _ = _bounded_reference_and_flow(
        inner_loc=(-0.4935, 0.0),
        inner_covariance=((0.49, 0.35), (0.35, 1.0)),
    )
    edges = [np.array([0.0, 0.05, 0.10, 0.80, 0.90, 1.0])] * 2
    n_calibration_samples = 20_000
    shared = {'edge_bins': 2, 'n_calibration_samples': n_calibration_samples,
              'rng_seed': 44}

    _, report = calibrate_edge_bins(
        flow, reference, edges, min_calibration_tail_count=1, return_report=True, **shared
    )
    biased = report["dimensions"][0]
    assert np.array_equal(biased["active_lower"], np.array([False, True]))
    lower_gaps = np.diff(np.concatenate([[0.0], biased["fitted_lower_cdf"]]))
    lower_counts = lower_gaps * n_calibration_samples
    # Merging the two gaps (what the masked anchor list does) hides the fact
    # that the active anchor is resolved by only the second one.
    assert lower_counts[1] < 1500 < lower_counts.sum()

    with pytest.raises(ValueError, match="expected calibration draws"):
        calibrate_edge_bins(
            flow, reference, edges, min_calibration_tail_count=1500, **shared
        )

    # The gap below the inactive anchor is narrower still, but no quantile is
    # estimated there, so it must not trip the guard.
    assert lower_counts[0] < 1000
    calibrate_edge_bins(flow, reference, edges, min_calibration_tail_count=1000, **shared)


def test_two_edge_bins_are_calibrated():
    reference, flow, edges = _bounded_reference_and_flow()
    calibrated, report = calibrate_edge_bins(
        flow,
        reference,
        edges,
        edge_bins=2,
        n_calibration_samples=40_000,
        rng_seed=44,
        return_report=True,
    )
    draws = sample(calibrated, 60_000, rng_seed=45)
    for dim, dimension_report in enumerate(report["dimensions"]):
        boundaries = np.concatenate(
            [dimension_report["lower_boundaries"], dimension_report["upper_boundaries"]]
        )
        intervals = np.vstack(
            [
                dimension_report["lower_confidence_interval"],
                dimension_report["upper_confidence_interval"],
            ]
        )
        observed = np.asarray([np.mean(draws[:, dim] < boundary) for boundary in boundaries])
        assert np.all(observed >= intervals[:, 0] - 0.004)
        assert np.all(observed <= intervals[:, 1] + 0.004)


def test_actual_fitted_gaussian_flow_smoke():
    rng = np.random.default_rng(12)
    latent = rng.multivariate_normal(np.zeros(2), np.array([[1.0, 0.75], [0.75, 1.0]]), size=1500)
    reference = norm.cdf(latent)
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
    flow = normalizing_flows_fit(
        reference,
        prior_bounds=bounds,
        max_epochs=2,
        patience=1,
        rng_seed=12,
        show_progress=False,
    )
    edges = [np.linspace(0.0, 1.0, 21), np.linspace(0.0, 1.0, 21)]
    calibrated, report = calibrate_edge_bins(
        flow,
        reference,
        edges,
        n_calibration_samples=10_000,
        rng_seed=13,
        return_report=True,
    )
    samples, sampled_lp = sample_and_log_prob(calibrated, 2000, rng_seed=14)
    assert np.all(np.isfinite(samples))
    assert np.max(np.abs(sampled_lp - log_prob(calibrated, samples))) < 1e-8

    before = sample(flow, 20_000, rng_seed=15)
    after = sample(calibrated, 20_000, rng_seed=15)
    for dim, dimension_report in enumerate(report["dimensions"]):
        boundaries = np.concatenate(
            [dimension_report["lower_boundaries"], dimension_report["upper_boundaries"]]
        )
        target = np.concatenate(
            [dimension_report["target_lower_cdf"], dimension_report["target_upper_cdf"]]
        )
        active = np.concatenate(
            [dimension_report["active_lower"], dimension_report["active_upper"]]
        )
        pre = np.asarray([np.mean(before[:, dim] < boundary) for boundary in boundaries])
        post = np.asarray([np.mean(after[:, dim] < boundary) for boundary in boundaries])
        assert np.all(np.abs(post[active] - target[active]) < np.abs(pre[active] - target[active]))
