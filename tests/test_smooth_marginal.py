"""Tests for the smooth-tail marginal (every ``normalizing_flows_fit`` marginal).

The marginal is a single clip-free diffeomorphism ``theta(z)``: R onto the
support (RQS bulk on (z, theta) knots + analytic tails). Sampling and density
directions share one parameterization, so a clipped PIT/ndtri marginal's
boundary defects (exact-bound sample atoms, wall density cliffs/floors) do
not arise, and out-of-data-range queries get honest smoothly-decaying
densities (``tail_alpha`` controls heaviness on unbounded sides).

The tests pin:

* marginal-level exactness: round trip, true log-derivative, and an exact
  s == q shell comparison (z drawn from the *known* N(0,1) base, so expected
  shell masses are analytic Phi differences at any wall depth);
* wall fidelity: the density reported at a prior bound against the analytic
  truth for laws whose wall density is finite and positive, and that the
  density integrates to 1 by grid-free Monte Carlo;
* flow-level hygiene: no exact-bound atoms / out-of-bounds / NaN draws, no
  wall cliff or floor in log_prob, bounded per-decade ladder rates;
* the sample_and_log_prob contract: the sampling layer redraws rows whose
  bounded tails saturated at the float64 wall-resolution limit, so the
  reported density is exactly log_prob for every emitted row;
* guards and save/load round trip.
"""

import os
import tempfile

import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats
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
        chain, prior_bounds=BOUNDS,
        max_epochs=60, rng_seed=3, show_progress=False,
    )
    return flow, chain


# ---------------------------------------------------------------------------
# marginal-level: the bijection itself is exact
# ---------------------------------------------------------------------------

def _log_ddz(log_ddz, z):
    """``log dtheta/dz`` as a plain numpy array."""
    return np.asarray(log_ddz(jnp.asarray(z)))


def _marginal_log_pdf(z_of_theta, log_ddz, x):
    """Marginal log-density the map reports at data-space point(s) ``x``."""
    z = np.asarray(z_of_theta(jnp.asarray(x)))
    return -0.5 * z ** 2 - 0.5 * np.log(2 * np.pi) - _log_ddz(log_ddz, z)


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
    the way -- and is the test a clipped PIT/ndtri marginal fails (atom over
    cliff)."""
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
    theta_of_z, z_of_theta, log_ddz, _ = make_smooth_marginal(
        x, prior_low=0.0, prior_high=1.0)
    p_lo, p_hi = _clamp_plateaus(theta_of_z, z_of_theta)
    z = np.linspace(max(-7, p_lo + 1e-3), min(7, p_hi - 1e-3), 2001)
    th = np.asarray(theta_of_z(z))
    assert np.all(np.diff(th) >= 0)
    assert th.min() > 0.0 and th.max() < 1.0
    z_back = np.asarray(z_of_theta(th))
    dz = np.abs(z_back - z)
    assert np.max(dz[np.abs(z) <= 4.0]) < 1e-8
    # Deeper than that ``theta`` itself runs out of float64: within 1e-13 of a
    # bound of magnitude 1 the wall offset is only representable to a part in
    # 1e-3, so the round trip cannot be exact there and |dz| reaches 9.3e-4 on
    # this fit. What the sampling contract actually needs is that the two
    # directions report the same *density* at the same point, and that holds
    # to 3.4e-10 -- the density is flat in that region, which is exactly why
    # the z it is read off is imprecise.
    lq_out = -0.5 * z ** 2 - _log_ddz(log_ddz, z)
    lq_back = -0.5 * z_back ** 2 - _log_ddz(log_ddz, z_back)
    assert np.max(np.abs(lq_out - lq_back)) < 1e-7


# ---------------------------------------------------------------------------
# API: defaults and guards
# ---------------------------------------------------------------------------

def test_every_fitted_marginal_is_smooth(smooth_flow):
    from paramax import unwrap
    flow, _ = smooth_flow
    for bij in flow.bijection.bijections:
        assert isinstance(unwrap(bij), SmoothMarginalToGaussian)


def test_guards():
    """An out-of-range tail_alpha is refused at fit time, naming the column."""
    chain = _testbed_chain(600)
    with pytest.raises(ValueError, match=r"parameter \(column\) 0: tail_alpha must be in"):
        normalizing_flows_fit(chain, tail_alpha=-0.5, max_epochs=1,
                              show_progress=False)


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
    """log q toward a hard wall: finite, no cliff, and one declared plateau.

    The marginal now carries a finite wall density, so its own ladder settles
    onto a constant (checked to 0.05 nats below, and against the analytic
    truth in ``test_wall_density_matches_the_analytic_truth``). What still
    moves is the *copula*: past its outermost knot -- which sits on the most
    extreme sample, so it is the deepest |z| the copula ever saw -- it
    extrapolates linearly, and the bounded tail is allowed to run
    ``_WALL_Z_MARGIN`` further before it saturates. Measured on this fixture
    the joint ladder is monotone in |z| and its largest per-decade step is
    4.15 nats (dim 0) and 1.79 (dim 3), so the bound below is a pin on that
    extrapolation, not on the marginal. The per-decade bound over the first
    six decades -- where |z| stays inside the copula's knots -- is the part
    that tests the marginal.

    The last rungs repeat, and that is the declared trade: below
    ``x_clamp_lo`` / ``x_clamp_hi`` the tail saturates and every query reads
    the same density. What must hold is that the plateau starts exactly where
    the builder says it does, and nowhere earlier. Letting the tail run to
    ``|z| = 8`` instead made all 15 rungs distinct but drove the wall value to
    +12.87 (dim 0) and +11.96 (dim 3) against +9.91 and +2.51 here, above the
    density of any draw the flow emits.
    """
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
        assert np.all(np.abs(d[:6]) < 2.6), d   # pdf grows at most ~1/(wall-x)
        assert np.all(np.abs(d) < 5.0), d       # no cliff anywhere

        # the marginal alone -- the part this module owns -- converges
        theta_of_z, z_of_theta, log_ddz, clamps = make_smooth_marginal(
            chain[:, dim], prior_low=BOUNDS[dim, 0], prior_high=BOUNDS[dim, 1])
        # the plateau is exactly the saturated sliver the builder declares:
        # every rung outside it is a distinct query, every rung inside it
        # repeats the wall value
        offsets = 10.0 ** (-ks.astype(float))
        clamp_offset = (clamps[0] - wall) if sign > 0 else (wall - clamps[1])
        resolved = offsets > clamp_offset
        # the plateau must not reach into the decades the steps above test:
        # measured clamp offsets are 3.4e-12 (dim 0) and 5.8e-8 (dim 3)
        assert resolved.sum() >= 7, clamp_offset
        assert len(np.unique(lad[resolved])) == int(resolved.sum()), lad
        assert np.all(lad[~resolved] == lad[-1]), lad
        xs = wall + sign * 10.0 ** (-ks.astype(float))
        marg = _marginal_log_pdf(z_of_theta, log_ddz, xs)
        assert np.all(np.isfinite(marg))
        assert np.max(np.abs(np.diff(marg[-5:]))) < 0.05, marg
        assert abs(marg[-1] - float(_marginal_log_pdf(
            z_of_theta, log_ddz, np.asarray(float(wall))))) < 0.05


@pytest.mark.parametrize("law,lo,hi,truth,med_tol,max_tol", [
    ("truncnorm", -1.0, 1.0,
     lambda t: stats.truncnorm.logpdf(t, -1, 1), 1.5, 3.0),
    ("uniform", 0.0, 1.0, lambda t: np.zeros_like(np.asarray(t, float)),
     1.5, 3.0),
])
def test_wall_density_matches_the_analytic_truth(law, lo, hi, truth,
                                                 med_tol, max_tol):
    """At a wall where the truth is finite and positive, so is the report.

    The bounded tail used to be ``theta = lo + gap * exp(c (z - z1))``, which
    makes ``theta - lo`` log-normal: its density at the bound is 0 or
    infinity depending on the fitted rate, never a positive constant. Over
    the six probe points below (each bound, and 1e-9 / 1e-6 of the span
    inside it) on 8 seeds x n = 1500, the old family gave median |error|
    173.5 nats and worst 5.6e4 on truncated N(0, 1) over [-1, 1], and median
    179.6 / worst 2.3e4 on U(0, 1) over [0, 1]. The current family gives
    median 0.83 / worst 1.78 and median 0.74 / worst 1.49 respectively; the
    thresholds here sit at roughly twice that, since the residual is set by
    the outermost order statistic and moves with the seed family.
    """
    span = hi - lo
    errors = []
    for seed in range(8):
        rng = np.random.default_rng(4000 + seed)
        x = (stats.truncnorm.rvs(-1, 1, size=1500, random_state=rng)
             if law == "truncnorm" else rng.uniform(lo, hi, 1500))
        _, z_of_theta, log_ddz, _ = make_smooth_marginal(
            x, prior_low=lo, prior_high=hi)
        probes = np.array([lo, lo + 1e-9 * span, lo + 1e-6 * span,
                           hi - 1e-6 * span, hi - 1e-9 * span, hi])
        got = _marginal_log_pdf(z_of_theta, log_ddz, probes)
        ref = np.array([float(truth(np.clip(p, lo + 1e-12 * span,
                                            hi - 1e-12 * span)))
                        for p in probes])
        assert np.all(np.isfinite(got))
        errors.extend(np.abs(got - ref))
    assert np.median(errors) < med_tol, np.median(errors)
    assert np.max(errors) < max_tol, np.max(errors)


def test_bounded_density_integrates_to_one():
    """Grid-free check that the marginal is a probability density.

    Uniform importance sampling over the box, not quadrature: an earlier
    trapezoid estimate of the same quantity was retracted as a grid artifact
    (the boundary grid line carried the wall spike).
    """
    rng = np.random.default_rng(11)
    for x, lo, hi in ((rng.beta(0.5, 3.0, 2000), 0.0, 1.0),
                      (np.clip(rng.normal(0.9, 0.25, 2000), -1, 1), -1.0, 1.0),
                      (rng.uniform(0.0, 1.0, 2000), 0.0, 1.0)):
        _, z_of_theta, log_ddz, _ = make_smooth_marginal(
            x, prior_low=lo, prior_high=hi)
        u = rng.uniform(lo, hi, 400_000)
        mass = np.mean(np.exp(_marginal_log_pdf(z_of_theta, log_ddz, u)))
        mass *= (hi - lo)
        assert abs(mass - 1.0) < 0.05, mass


def test_guard_rails_on_scale_and_tail_alpha():
    """Silently-broken maps become named errors.

    Data far above the spline kernel's float64 headroom used to return a
    non-finite, non-monotone map (standard normals scaled by 1e80: NaN round
    trip; by 1e120: theta_of_z all NaN), and ``tail_alpha`` was checked only
    for being non-negative, so 1e6 fitted silently and emitted samples up to
    1.8e298.
    """
    x = np.random.default_rng(6).standard_normal(600)
    with pytest.raises(ValueError, match="tail_alpha must be in"):
        make_smooth_marginal(x, tail_alpha=1e6)
    with pytest.raises(ValueError, match="tail_alpha must be in"):
        make_smooth_marginal(x, tail_alpha=-0.5)
    # the documented range still fits, and stays representable
    theta = np.asarray(make_smooth_marginal(x, tail_alpha=10.0)[0](
        np.random.default_rng(7).standard_normal(20000)))
    assert np.all(np.isfinite(theta)) and np.max(np.abs(theta)) < 1e8

    for scale in (1e80, 1e160):
        with pytest.raises(ValueError, match=r"above the 1e\+75 limit"):
            make_smooth_marginal(x * scale)
    with pytest.raises(ValueError, match=r"above the 1e\+75 limit"):
        make_smooth_marginal(x, prior_low=-1e80, prior_high=1e80)
    # just under the limit still builds a finite monotone map
    z = np.linspace(-6.0, 6.0, 401)
    theta = np.asarray(make_smooth_marginal(x * 1e70)[0](z))
    assert np.all(np.isfinite(theta)) and np.all(np.diff(theta) > 0)


def test_sample_logprob_contract(smooth_flow):
    """sample_and_log_prob == log_prob(samples) exactly: rows that saturated
    a bounded tail at the float64 wall-resolution limit are redrawn by the
    sampling layer, so the emitted stream carries its own true density."""
    flow, _ = smooth_flow
    s, lq = sample_and_log_prob(flow, n_samples=20_000, rng_seed=13)
    s = np.asarray(s)
    d = np.abs(np.asarray(lq) - np.asarray(log_prob(flow, s)))
    assert np.max(d) < 1e-8, np.max(d)
    # And no emitted row sits at or beyond a tail clamp. The oracle is read
    # straight off the marginals, not from the sampler's own rejection helper:
    # reusing that helper would make this assertion pass for any helper at all,
    # including one that never rejects anything.
    from paramax import unwrap
    los, his = [], []
    for b in flow.bijection.bijections:
        m = unwrap(b)
        los.append(-np.inf if m.x_clamp_lo is None else float(m.x_clamp_lo))
        his.append(np.inf if m.x_clamp_hi is None else float(m.x_clamp_hi))
    los, his = np.asarray(los), np.asarray(his)
    assert np.any(np.isfinite(los) | np.isfinite(his)), "fixture has no bounds"
    assert not np.any((s <= los[None, :]) | (s >= his[None, :]))


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
    from coppuccino.model_io import _extract_spline_data, _reconstruct_empirical_transforms
    chain = _testbed_chain(1500)
    flow = normalizing_flows_fit(chain, prior_bounds=BOUNDS,
                                 max_epochs=2,
                                 show_progress=False)
    data = _extract_spline_data(flow)
    data.pop('tail_alpha')
    stack = _reconstruct_empirical_transforms(data)
    from paramax import unwrap
    assert all(unwrap(b).tail_alpha == 0.0 for b in stack.bijections)


def test_bounds_far_from_the_origin_with_a_wall_atom():
    """A bound whose magnitude dwarfs the span still builds a usable map.

    ``prior_high - thK`` rounds to exactly zero once ``bound + offset``
    absorbs back to ``bound`` in float, which used to divide by zero in the
    tail rate. The gap is floored at one ULP of the bound instead.
    """
    lo, hi = 1e10, 1e10 + 1e-4
    rng = np.random.default_rng(0)
    samples = np.concatenate([np.full(300, hi), rng.uniform(lo, hi, 300)])

    theta_of_z, z_of_theta, log_dtheta_dz, *_ = make_smooth_marginal(
        samples, prior_low=lo, prior_high=hi)

    z = np.linspace(-6.0, 6.0, 401)
    theta = np.asarray(theta_of_z(z))
    assert np.all(np.isfinite(theta))
    assert np.all((theta >= lo) & (theta <= hi))
    # The whole support is ~50 ULPs wide at this magnitude, so monotonicity
    # holds only to the float grid it is expressed on -- that is a property of
    # the bounds, not of the map.
    assert np.all(np.diff(theta) >= -np.spacing(hi))
    assert np.all(np.isfinite(np.asarray(log_dtheta_dz(z))))
    # the wall atom itself must not blow up
    assert np.isfinite(float(z_of_theta(np.asarray(hi))))


def test_absorption_window_does_not_scale_with_an_uninformative_prior():
    """A wide prior must not swallow the interior of the chain.

    The boundary-atom window was scaled by the *prior* span, so a diffuse
    prior_bounds=[[0, 1e12]] on a chain spanning ~[5e2, 4e3] reached out to
    x = 1000 and relocated every sample below it onto a single point --
    z_of_theta was then bit-identical across two decades of real support.
    """
    rng = np.random.default_rng(0)
    samples = rng.gamma(9.0, 200.0, 3000)
    probe = np.array([50.0, 200.0, 400.0, 700.0, 1000.0])

    _, z_wide, *_ = make_smooth_marginal(samples, prior_low=0.0,
                                         prior_high=1e12)
    z = np.asarray(z_wide(probe))
    assert np.all(np.diff(z) > 0.0), "interior points collapsed onto one value"

    # and the map must not depend on how uninformative the prior is
    _, z_snug, *_ = make_smooth_marginal(samples, prior_low=0.0,
                                         prior_high=5000.0)
    assert np.allclose(z, np.asarray(z_snug(probe)), rtol=0, atol=0)
