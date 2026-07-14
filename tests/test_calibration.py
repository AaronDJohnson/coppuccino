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
    sample,
    sample_and_log_prob,
    save_flow,
)
from coppuccino.bijections import SmoothMarginalToGaussian, make_smooth_marginal
from coppuccino.calibration import (
    _EdgeMarginalCalibration,
    _window_derivatives,
)


def _bounded_reference_and_flow(n=8000, seed=0):
    """Gaussian copula with uniform marginals and a deliberately biased inner law."""
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
        loc=jnp.array([0.10, -0.10]),
        covariance=jnp.array([[1.44, 0.72], [0.72, 0.81]]),
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
        copula_space="gaussian",
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
