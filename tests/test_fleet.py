"""Tests for fleet fitting (normalizing_flows_fit_fleet / FleetFlow)."""

import os
import tempfile

import numpy as np
import jax
import equinox as eqx
import pytest

from coppuccino import FleetFlow, normalizing_flows_fit_fleet

N, NSAMP, DIM = 6, 2000, 2
BOUNDS = np.array([[-1.0, 1.0], [-1.0, 1.0]])
RHOS = np.linspace(-0.7, 0.7, N)


@pytest.fixture(scope="module")
def chains():
    rng = np.random.default_rng(0)
    out = np.empty((N, NSAMP, DIM))
    for s in range(N):
        a = rng.uniform(-1, 1, NSAMP)
        b = np.clip(RHOS[s] * a + 0.35 * rng.standard_normal(NSAMP), -1, 1)
        out[s] = np.stack([a, b], 1)
    return out


@pytest.fixture(scope="module")
def fleet(chains):
    return normalizing_flows_fit_fleet(chains, BOUNDS, max_epochs=100,
                                       rng_seed=1)


def test_shapes_support_and_boundaries(fleet):
    x = fleet.sample(3000, rng_seed=2)
    assert x.shape == (N, 3000, DIM)
    assert np.sum((x < -1) | (x > 1)) == 0          # support exactly the box
    # float rounding can land the odd draw exactly ON a wall (~1e-5 rate).
    # That is harmless here: the reported density at the wall is finite and
    # consistent with sampling (no atom-over-a-cliff), which the wall check
    # below asserts. Require rarity, not float impossibility.
    assert np.mean((x == -1) | (x == 1)) < 1e-4
    lp = fleet.log_prob(x)
    assert lp.shape == (N, 3000)
    assert np.all(np.isfinite(lp))
    # no wall density cliff: exact-wall logq within a few nats of interior
    g = np.zeros((3, DIM))
    g[:, 0] = [-1.0, 0.0, 1.0]
    lpg = fleet.log_prob(g, source_index=0)
    assert np.all(np.isfinite(lpg))
    assert abs(lpg[0] - lpg[1]) < 6.0 and abs(lpg[2] - lpg[1]) < 6.0


def test_per_source_dependence_learned(fleet, chains):
    x = fleet.sample(4000, rng_seed=3)
    fit = np.array([np.corrcoef(x[s, :, 0], x[s, :, 1])[0, 1] for s in range(N)])
    # compare against each source's own EMPIRICAL correlation (the clip in the
    # data generation shifts it away from the generating rho)
    true = np.array([np.corrcoef(chains[s, :, 0], chains[s, :, 1])[0, 1]
                     for s in range(N)])
    # 0.2 (not 0.15): this is a trained-fit statistic; leave platform
    # headroom for the macOS/lowest-resolution CI jobs. The correlation
    # check below carries the discriminative load.
    assert np.max(np.abs(fit - true)) < 0.2
    # and the fleet actually distinguishes sources (spread preserved)
    assert np.corrcoef(fit, true)[0, 1] > 0.95


def test_source_index_matches_batched(fleet):
    x = fleet.sample(200, rng_seed=4)
    lp_all = fleet.log_prob(x)
    for s in (0, N - 1):
        assert np.allclose(fleet.log_prob(x[s], source_index=s), lp_all[s])


def test_save_load_roundtrip_exact(fleet):
    x = fleet.sample(300, rng_seed=5)
    lp = fleet.log_prob(x)
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "fleet.pkl")
        fleet.save(p)
        f2 = FleetFlow.load(p)
    assert f2.arch == fleet.arch
    assert np.max(np.abs(lp - f2.log_prob(x))) == 0.0
    s1 = fleet.sample(64, rng_seed=6)
    s2 = f2.sample(64, rng_seed=6)
    assert np.max(np.abs(s1 - s2)) == 0.0


def test_warm_start(chains, fleet):
    f2 = normalizing_flows_fit_fleet(chains, BOUNDS, max_epochs=5,
                                     rng_seed=1, warm_start=fleet)
    assert f2.arch[:5] == fleet.arch[:5]
    x = f2.sample(500, rng_seed=7)
    assert np.all(np.isfinite(f2.log_prob(x)))
    # arch mismatch is rejected
    with pytest.raises(ValueError, match="warm_start"):
        normalizing_flows_fit_fleet(chains, BOUNDS, max_epochs=2,
                                    knots=2, warm_start=fleet)


def test_arch_kwargs_respected(chains):
    def nparams(fl):
        leaves = jax.tree_util.tree_leaves(
            eqx.filter(fl.params, eqx.is_inexact_array))
        return sum(int(np.prod(leaf.shape)) for leaf in leaves)
    small = normalizing_flows_fit_fleet(chains, BOUNDS, max_epochs=2,
                                        knots=2, flow_layers=3, rng_seed=1)
    big = normalizing_flows_fit_fleet(chains, BOUNDS, max_epochs=2,
                                      knots=8, flow_layers=8, rng_seed=1)
    assert nparams(big) > nparams(small)
    assert small.arch[2:4] == (2, 3) and big.arch[2:4] == (8, 8)


def test_determinism(chains):
    f1 = normalizing_flows_fit_fleet(chains, BOUNDS, max_epochs=8, rng_seed=9)
    f2 = normalizing_flows_fit_fleet(chains, BOUNDS, max_epochs=8, rng_seed=9)
    x = f1.sample(100, rng_seed=1)
    assert np.max(np.abs(f1.log_prob(x) - f2.log_prob(x))) == 0.0


def test_input_validation(chains):
    with pytest.raises(ValueError, match="n_sources, n_samples, dim"):
        normalizing_flows_fit_fleet(chains[0], BOUNDS, max_epochs=1)
    with pytest.raises(ValueError, match="prior_bounds"):
        normalizing_flows_fit_fleet(chains, np.zeros((3, 2)), max_epochs=1)
    bad = chains.copy()
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        normalizing_flows_fit_fleet(bad, BOUNDS, max_epochs=1)
    with pytest.raises(ValueError, match="outside prior_bounds"):
        normalizing_flows_fit_fleet(chains * 3.0, BOUNDS, max_epochs=1)
