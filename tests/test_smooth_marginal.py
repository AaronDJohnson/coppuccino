"""Tests for the smooth-tail marginal (``copula_space="gaussian"`` default).

``marginal="smooth"`` replaces the legacy clipped PIT/ndtri marginal with a
single clip-free diffeomorphism ``theta(z)``: R onto the support (RQS bulk on
(z, theta) knots + analytic tails). Sampling and density directions share one
parameterization, so the boundary defects of the legacy Gaussianized pipeline
(exact-bound sample atoms, wall density cliffs/floors) are removed at the
root, and out-of-data-range queries get honest smoothly-decaying densities
(``tail_alpha`` controls heaviness on unbounded sides).

The tests pin:

* marginal-level exactness: round trip, true log-derivative, and an exact
  s == q shell comparison (z drawn from the *known* N(0,1) base, so expected
  shell masses are analytic Phi differences at any wall depth);
* flow-level hygiene: no exact-bound atoms / out-of-bounds / NaN draws, no
  wall cliff or floor in log_prob, bounded per-decade ladder rates;
* the sample_and_log_prob contract: the sampling layer redraws rows whose
  bounded tails saturated at the float64 wall-resolution limit, so the
  reported density is exactly log_prob for every emitted row;
* guards, default resolution per copula_space, and save/load round trip.
"""

import os
import tempfile

import numpy as np
import pytest
from scipy.stats import norm

from coppuccino import (
    load_flow,
    log_prob,
    normalizing_flows_fit,
    sample,
    save_flow,
)
from coppuccino.bijections import (
    SmoothMarginalToGaussian,
    make_smooth_marginal,
)
from coppuccino.copula_flows import sample_and_log_prob

BOUNDS = np.array([[0.0, 1.0],
                   [-np.inf, np.inf],
                   [-np.inf, np.inf],
                   [-2.0, 2.0]])


def _testbed_chain(n=5000, seed=7):
    """dim0: density spike at the low wall of [0,1] (Beta(0.5,3));
    dim1/dim2: correlated unbounded pair; dim3: pile-up at the high wall
    of [-2,2]."""
    rng = np.random.default_rng(seed)
    d0 = rng.beta(0.5, 3.0, n)
    d12 = rng.multivariate_normal([0.0, 0.0], [[1.0, 0.8], [0.8, 1.0]], n)
    d3 = rng.normal(1.8, 0.6, n)
    keep = (d3 > -2) & (d3 < 2)
    d3 = np.concatenate([d3[keep], rng.uniform(-2, 2, n - keep.sum())])[:n]
    return np.column_stack([d0, d12, d3])


@pytest.fixture(scope="module")
def smooth_flow():
    """Fit once, reuse across tests (fits dominate test runtime)."""
    chain = _testbed_chain()
    flow = normalizing_flows_fit(
        chain, prior_bounds=BOUNDS, copula_space="gaussian",
        max_epochs=60, rng_seed=3, show_progress=False,
    )
    return flow, chain


# ---------------------------------------------------------------------------
# marginal-level: the bijection itself is exact
# ---------------------------------------------------------------------------

def _clamp_plateaus(theta_of_z, z_of_theta):
    """Self-reported clamp depths: fixed points of the deep round trip.

    Bounded tails clamp once the wall offset reaches the float64 resolution
    limit; the round trip is exact above these depths and constant below."""
    lo = float(z_of_theta(theta_of_z(np.asarray(-1e12))))
    hi = float(z_of_theta(theta_of_z(np.asarray(+1e12))))
    return lo, hi


def test_marginal_round_trip_and_derivative():
    rng = np.random.default_rng(0)
    x = rng.beta(0.5, 3.0, 4000)
    theta_of_z, z_of_theta, log_ddz, _ = make_smooth_marginal(
        x, prior_low=0.0, prior_high=1.0)
    p_lo, p_hi = _clamp_plateaus(theta_of_z, z_of_theta)
    # the clamp lives far below any probabilistically relevant z
    assert p_lo < -4.0 and p_hi > 4.0, (p_lo, p_hi)
    # round trip: exact in the mass-bearing region; toward a nonzero bound
    # the offset (bound - theta) loses relative precision to ULP(|bound|)
    # (inherent to float64 x-space), so bound the induced *log-density*
    # error |dz|*|z| instead -- vs ~49 nats for the legacy cliff.
    z = np.linspace(max(p_lo + 1e-3, -8.0), min(p_hi - 1e-3, 8.0), 4001)
    th = np.asarray(theta_of_z(z))
    dz = np.abs(np.asarray(z_of_theta(th)) - z)
    core = np.abs(z) <= 4.0
    assert np.max(dz[core]) < 1e-9
    assert np.max(dz * np.maximum(np.abs(z), 1.0)) < 1e-3
    # below the clamp, relocated points report the clamp z (consistency)
    z_deep = np.asarray(z_of_theta(theta_of_z(np.array([p_lo - 5.0, p_hi + 5.0]))))
    assert np.allclose(z_deep, [p_lo, p_hi], atol=1e-9)
    # claimed log-derivative is the true derivative of theta_of_z (checked
    # where the theta spacing is float64-resolvable)
    h = 1e-6
    zc = z[core]
    num = (np.asarray(theta_of_z(zc + h)) - np.asarray(theta_of_z(zc - h))) / (2 * h)
    rel = np.abs(num / np.exp(np.asarray(log_ddz(zc))) - 1.0)
    assert np.max(rel) < 1e-4, np.max(rel)


def test_marginal_exact_shell_occupancy():
    """s == q at any wall depth: draw z from the *known* N(0,1) base, push
    through theta(z), and compare shell counts against the analytic mass
    Phi(z(edge_hi)) - Phi(z(edge_lo)). This is exact -- no fitted copula in
    the way -- and is the test the legacy marginal fails (atom over cliff)."""
    rng = np.random.default_rng(1)
    x = rng.beta(0.5, 3.0, 5000)
    theta_of_z, z_of_theta, _, _ = make_smooth_marginal(
        x, prior_low=0.0, prior_high=1.0)
    n = 2_000_000
    th = np.asarray(theta_of_z(rng.standard_normal(n)))
    # hygiene at scale: strictly inside, no exact-bound atom
    assert th.min() > 0.0 and th.max() < 1.0
    assert np.sum(th == 0.0) == 0 and np.sum(th == 1.0) == 0
    # log-spaced shells toward the low (spike) wall
    edges = 10.0 ** np.arange(-9.0, 0.1, 1.0)
    z_edges = np.asarray(z_of_theta(edges))
    exp_mass = np.diff(norm.cdf(z_edges))
    obs = np.histogram(th, np.asarray(edges))[0]
    exp = exp_mass * n
    for o, e in zip(obs, exp):
        if e >= 25:
            assert abs(o - e) / np.sqrt(e) < 5.0, (o, e)
        else:
            assert o <= 3 * e + 10, (o, e)  # no unexplained pileup
    # deepest region: mass below the first knot must also match analytically
    z1 = float(z_of_theta(np.asarray(edges[0])))
    e0 = norm.cdf(z1) * n
    o0 = int(np.sum(th < edges[0]))
    assert abs(o0 - e0) / np.sqrt(max(e0, 1.0)) < 5.0, (o0, e0)


def test_marginal_tail_alpha_controls_unbounded_tails():
    rng = np.random.default_rng(2)
    x = rng.standard_normal(3000)
    t0, _, l0, cl0 = make_smooth_marginal(x, tail_alpha=0.0)
    t1, _, l1, _ = make_smooth_marginal(x, tail_alpha=1.0)
    assert cl0 == (None, None)   # unbounded: no tail clamps
    # alpha=0: linear map beyond the knots (Gaussian tails in theta)
    z = np.array([5.0, 7.0, 9.0])
    d2 = np.diff(np.asarray(t0(z)), 2)
    assert np.max(np.abs(d2)) < 1e-8
    # alpha>0: strictly heavier -- larger theta and larger derivative far out
    assert float(t1(np.array(9.0))) > float(t0(np.array(9.0)))
    assert float(l1(np.array(9.0))) > float(l0(np.array(9.0)))
    with pytest.raises(ValueError):
        make_smooth_marginal(x, tail_alpha=-0.5)


def test_marginal_nonfinite_bounds_mean_unbounded():
    rng = np.random.default_rng(3)
    x = rng.standard_normal(3000)
    t_inf, _, _, _ = make_smooth_marginal(x, prior_low=-np.inf, prior_high=np.inf)
    t_none, _, _, _ = make_smooth_marginal(x, prior_low=None, prior_high=None)
    z = np.linspace(-6, 6, 101)
    assert np.allclose(np.asarray(t_inf(z)), np.asarray(t_none(z)))


def test_marginal_handles_ties_and_spikes_both_walls():
    rng = np.random.default_rng(4)
    x = rng.beta(0.5, 0.5, 4000)          # spikes at BOTH walls
    x[:200] = 0.25                          # heavy tie block
    theta_of_z, z_of_theta, _, _ = make_smooth_marginal(
        x, prior_low=0.0, prior_high=1.0)
    p_lo, p_hi = _clamp_plateaus(theta_of_z, z_of_theta)
    z = np.linspace(max(-7, p_lo + 1e-3), min(7, p_hi - 1e-3), 2001)
    th = np.asarray(theta_of_z(z))
    assert np.all(np.diff(th) >= 0)
    assert th.min() > 0.0 and th.max() < 1.0
    dz = np.abs(np.asarray(z_of_theta(th)) - z)
    assert np.max(dz[np.abs(z) <= 4.0]) < 1e-8
    assert np.max(dz * np.maximum(np.abs(z), 1.0)) < 1e-3


# ---------------------------------------------------------------------------
# API: defaults and guards
# ---------------------------------------------------------------------------

def test_gaussian_default_resolves_to_smooth(smooth_flow):
    from paramax import unwrap
    flow, _ = smooth_flow
    for bij in flow.bijection.bijections:
        m = unwrap(bij)
        assert isinstance(m, SmoothMarginalToGaussian)
        assert m.marginal == "smooth"


def test_guards():
    chain = _testbed_chain(600)
    with pytest.raises(ValueError, match="uniform"):
        normalizing_flows_fit(chain, copula_space="uniform", marginal="smooth",
                              max_epochs=1)
    with pytest.raises(ValueError, match="tail_alpha"):
        normalizing_flows_fit(chain, copula_space="gaussian", marginal="smooth",
                              tail_extension=True, max_epochs=1)
    with pytest.raises(ValueError, match="tail_alpha"):
        normalizing_flows_fit(chain, copula_space="gaussian", marginal="smooth",
                              tail_model="gpd", max_epochs=1)
    with pytest.raises(ValueError, match="marginal"):
        normalizing_flows_fit(chain, marginal="nope", max_epochs=1)


def test_legacy_marginals_still_available():
    chain = _testbed_chain(1500)
    finite = np.array([[0., 1.], [-6., 6.], [-6., 6.], [-2., 2.]])
    flow = normalizing_flows_fit(chain, prior_bounds=finite,
                                 copula_space="gaussian", marginal="rqs",
                                 max_epochs=2, show_progress=False)
    assert np.all(np.isfinite(log_prob(flow, chain[:50])))


# ---------------------------------------------------------------------------
# flow-level: sampling hygiene and density behavior
# ---------------------------------------------------------------------------

def test_no_atoms_no_oob_no_nan(smooth_flow):
    flow, _ = smooth_flow
    d = np.concatenate([sample(flow, 200_000, rng_seed=10 + k) for k in range(3)])
    assert np.sum(~np.isfinite(d)) == 0
    assert np.sum((d[:, 0] < 0) | (d[:, 0] > 1) | (d[:, 3] < -2) | (d[:, 3] > 2)) == 0
    for j, b in ((0, 0.0), (0, 1.0), (3, -2.0), (3, 2.0)):
        assert np.sum(d[:, j] == b) == 0


def test_wall_ladder_no_cliff_no_floor(smooth_flow):
    """log q toward a hard wall changes at a bounded, smoothly varying rate:
    no cliff (legacy: ~10-nat drop) and no floor (legacy: exactly constant
    plateau below the last knot)."""
    flow, chain = smooth_flow
    med = np.median(chain, axis=0)
    ks = np.arange(1, 16)
    for dim, wall, sign in ((0, 0.0, +1), (3, 2.0, -1)):
        lad = []
        for k in ks:
            pt = med.copy()
            pt[dim] = wall + sign * 10.0 ** (-float(k))
            lad.append(float(log_prob(flow, pt[None, :])[0]))
        lad = np.array(lad)
        d = np.diff(lad)
        assert np.all(np.isfinite(lad))
        assert np.all(np.abs(d) < 2.6), d          # pdf grows at most ~1/(wall-x)
        assert np.all(np.abs(np.diff(d[-8:])) < 2.5), d  # no cliff
        assert not np.allclose(d[-5:], 0.0, atol=1e-3)   # no exact floor


def test_sample_logprob_contract(smooth_flow):
    """sample_and_log_prob == log_prob(samples) exactly: rows that saturated
    a bounded tail at the float64 wall-resolution limit are redrawn by the
    sampling layer, so the emitted stream carries its own true density."""
    flow, _ = smooth_flow
    s, lq = sample_and_log_prob(flow, n_samples=20_000, rng_seed=13)
    s = np.asarray(s)
    d = np.abs(np.asarray(lq) - np.asarray(log_prob(flow, s)))
    assert np.max(d) < 1e-8, np.max(d)
    # and no emitted row sits at/beyond a tail clamp
    from coppuccino.copula_flows import _smooth_clamp_limits, _clamped_rows
    limits = _smooth_clamp_limits(flow)
    assert limits is not None
    assert not _clamped_rows(s, limits).any()


def test_outlier_queries_finite_and_decaying(smooth_flow):
    flow, chain = smooth_flow
    med = np.median(chain, axis=0)
    hi, sd = float(chain[:, 1].max()), float(chain[:, 1].std())
    lq = []
    for f in (1.0, 2.0, 5.0, 10.0, 20.0):
        pt = med.copy()
        pt[1] = hi + f * sd
        lq.append(float(log_prob(flow, pt[None, :])[0]))
    lq = np.array(lq)
    assert np.all(np.isfinite(lq))
    assert np.all(np.diff(lq) < 0)


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------

def test_save_load_round_trip(smooth_flow):
    flow, chain = smooth_flow
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "smooth.pkl")
        save_flow(flow, path)
        loaded = load_flow(path)
    pts = chain[::37]
    assert np.max(np.abs(np.asarray(log_prob(flow, pts))
                         - np.asarray(log_prob(loaded, pts)))) < 1e-9
    # near-wall agreement too (the analytic-tail constants must survive)
    pt = np.median(chain, axis=0)
    pt[0] = 1e-12
    a = float(log_prob(flow, pt[None, :])[0])
    b = float(log_prob(loaded, pt[None, :])[0])
    assert abs(a - b) < 1e-6
    # metadata round-trips
    from paramax import unwrap
    m = unwrap(loaded.bijection.bijections[0])
    assert isinstance(m, SmoothMarginalToGaussian)
    assert m.tail_alpha == 0.0 and m.prior_low == 0.0 and m.prior_high == 1.0


def test_load_predates_tail_alpha_key():
    """Files saved before tail_alpha existed reconstruct with the default."""
    from coppuccino.model_io import (_extract_spline_data,
                                     _reconstruct_empirical_transforms)
    chain = _testbed_chain(1500)
    flow = normalizing_flows_fit(chain, prior_bounds=BOUNDS,
                                 copula_space="gaussian", max_epochs=2,
                                 show_progress=False)
    data = _extract_spline_data(flow)
    data.pop('tail_alpha')
    stack = _reconstruct_empirical_transforms(data)
    from paramax import unwrap
    assert all(unwrap(b).tail_alpha == 0.0 for b in stack.bijections)
