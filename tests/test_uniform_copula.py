"""Tests for the uniform-copula pipeline (``copula_space="uniform"``).

The uniform mode exists to make the flow's sample stream and its reported
density agree at hard support boundaries. An independence MCMC sampler that
proposes from true density ``s`` but corrects with reported density ``q`` has
stationary distribution ``pi * s / q`` -- so any boundary region where
``s != q`` is a bias amplified by ``exp(log s - log q)``, and a sampling atom
sitting on a reported-density cliff is an absorbing state. These tests pin the
boundary behavior that the Gaussianized pipeline could not guarantee:

* no samples outside the prior bounds and no exact-boundary atoms;
* near-uniform shell occupancy against a flat target (sampling side);
* no wall cliff in ``log_prob`` (density side);
* exact save/load round-trip (the coupling-flow skeleton is rebuilt from its
  architecture spec rather than pickled).
"""

import os
import tempfile

import numpy as np
import pytest

from coppuccino import (
    CopulaFlow,
    load_flow,
    log_prob,
    normalizing_flows_fit,
    sample,
    save_flow,
)

BOUNDS = np.array([[-1.0, 1.0], [-1.0, 1.0]])


@pytest.fixture(scope="module")
def uniform_flow():
    """Fit once, reuse across tests (fits dominate test runtime)."""
    rng = np.random.default_rng(0)
    n = 4000
    a = rng.uniform(-1.0, 1.0, n)                            # flat to both walls
    b = np.clip(0.5 * a + 0.25 * rng.standard_normal(n), -1, 1)
    data = np.stack([a, b], 1)
    flow = normalizing_flows_fit(
        data, prior_bounds=BOUNDS, copula_space="uniform",
        knots=8, flow_layers=4, max_epochs=100, rng_seed=1,
    )
    return flow, data


def test_returns_copula_flow_with_arch(uniform_flow):
    flow, _ = uniform_flow
    assert isinstance(flow, CopulaFlow)
    assert flow.arch is not None and flow.arch[0] == "uniform"


def test_support_exactly_prior_box_no_atoms(uniform_flow):
    flow, _ = uniform_flow
    d = np.concatenate([sample(flow, 200_000, rng_seed=10 + k) for k in range(3)])
    # support is exactly the box
    assert np.sum((d < -1.0) | (d > 1.0)) == 0
    # no exact-boundary atoms (the Gaussianized pipeline's quantile clamping
    # produced draws exactly at the bound)
    for j in range(2):
        assert np.sum(d[:, j] == -1.0) == 0
        assert np.sum(d[:, j] == 1.0) == 0


def test_flat_marginal_shell_occupancy(uniform_flow):
    """Sampling side: outer shells of a flat-to-the-wall marginal stay ~flat."""
    flow, _ = uniform_flow
    d = np.concatenate([sample(flow, 200_000, rng_seed=20 + k) for k in range(3)])
    for sh in (1e-3, 1e-2):
        w = sh * 2.0  # shell width in x units (box width 2)
        for side_frac in (np.mean(d[:, 0] < -1.0 + w), np.mean(d[:, 0] > 1.0 - w)):
            # within a factor of ~3 of flat (fit quality; the Gaussianized
            # pipeline gave 0.0x here at 1e-3 with a matching density cliff)
            assert side_frac / sh > 0.3, (sh, side_frac / sh)
            assert side_frac / sh < 3.0, (sh, side_frac / sh)


def test_no_wall_density_cliff(uniform_flow):
    """Density side: log_prob at the exact walls stays within a few nats of
    the near-wall value (the Gaussianized pipeline dropped ~48 nats)."""
    flow, _ = uniform_flow
    for side in (-1.0, 1.0):
        g = np.zeros((4, 2))
        g[:, 0] = side - np.sign(side) * np.array([0.0, 1e-9, 1e-6, 1e-3])
        lp = log_prob(flow, g)
        assert np.all(np.isfinite(lp))
        # exact wall vs 1e-3 inside: no cliff
        assert abs(lp[0] - lp[3]) < 5.0, lp
        # exact wall vs 1e-9 inside: continuous
        assert abs(lp[0] - lp[1]) < 0.1, lp


def test_sample_density_consistency_shells(uniform_flow):
    """s vs q agree shell-by-shell: empirical shell mass matches the density.

    This is the test that would have caught the boundary defect: it compares
    what the flow *does* (sample occupancy) with what it *says* (log_prob),
    marginalized over the correlated partner by using the sampled pairs.
    """
    flow, _ = uniform_flow
    d = np.concatenate([sample(flow, 200_000, rng_seed=30 + k) for k in range(3)])
    lp = log_prob(flow, d)
    # bin dim0 into edge and interior regions; compare mass ratios to density
    # ratios via importance identity: for regions A, B,
    #   mass(A)/mass(B) should match the sampled counts ratio.  Since lp is the
    #   joint density of the actual draws, exp(lp) integrates correctly by
    #   construction -- a gross s/q mismatch shows up as bins where the count
    #   is far from the density-implied mass.  We check the outer 1% bins are
    #   populated in proportion to their mean density.
    for lo_bin, hi_bin in (((-1.0, -0.98), (-0.01, 0.01)),
                           ((0.98, 1.0), (-0.01, 0.01))):
        in_a = (d[:, 0] >= lo_bin[0]) & (d[:, 0] < lo_bin[1])
        in_b = (d[:, 0] >= hi_bin[0]) & (d[:, 0] < hi_bin[1])
        assert in_a.sum() > 100 and in_b.sum() > 100
        # mean density in each slab (same width) vs occupancy ratio
        ratio_counts = in_a.sum() / in_b.sum()
        ratio_density = np.exp(
            np.log(np.mean(np.exp(lp[in_a] - lp.max())))
            - np.log(np.mean(np.exp(lp[in_b] - lp.max())))
        )
        # agreement within a factor ~2 (Monte Carlo + fit wiggle)
        assert 0.4 < ratio_counts / ratio_density < 2.5, (
            ratio_counts, ratio_density)


def test_save_load_roundtrip_exact(uniform_flow):
    flow, _ = uniform_flow
    rng = np.random.default_rng(3)
    x = rng.uniform(-0.99, 0.99, (200, 2))
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "uflow.pkl")
        save_flow(flow, p)
        loaded = load_flow(p)
    assert isinstance(loaded, CopulaFlow)
    assert loaded.arch == flow.arch
    assert np.max(np.abs(log_prob(flow, x) - log_prob(loaded, x))) == 0.0
    s1 = sample(flow, 50, rng_seed=7)
    s2 = sample(loaded, 50, rng_seed=7)
    assert np.max(np.abs(s1 - s2)) == 0.0


def test_uniform_mode_guards():
    rng = np.random.default_rng(1)
    data = rng.uniform(-1, 1, (200, 2))
    with pytest.raises(ValueError, match="rqs"):
        normalizing_flows_fit(data, prior_bounds=BOUNDS, copula_space="uniform",
                              marginal="pchip", max_epochs=1)
    with pytest.raises(ValueError, match="bounded support"):
        normalizing_flows_fit(data, prior_bounds=BOUNDS, copula_space="uniform",
                              tail_extension=True, max_epochs=1)
    with pytest.raises(ValueError, match="bounded support"):
        normalizing_flows_fit(data, prior_bounds=BOUNDS, copula_space="uniform",
                              tail_model="gpd", max_epochs=1)
    with pytest.raises(ValueError, match="copula_space"):
        normalizing_flows_fit(data, prior_bounds=BOUNDS, copula_space="banana",
                              max_epochs=1)


def test_dependence_recovered(uniform_flow):
    """The copula stage actually learns the correlation, not just marginals."""
    flow, data = uniform_flow
    d = sample(flow, 100_000, rng_seed=40)
    r_fit = np.corrcoef(d[:, 0], d[:, 1])[0, 1]
    r_true = np.corrcoef(data[:, 0], data[:, 1])[0, 1]
    assert abs(r_fit - r_true) < 0.1, (r_fit, r_true)
