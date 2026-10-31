"""Tests for fleet fitting (normalizing_flows_fit_fleet / FleetFlow)."""

import builtins
import errno
import os
import tempfile

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from coppuccino import FleetFlow, normalizing_flows_fit_fleet
from coppuccino import fleet as fleet_mod

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
    """The shared trained fleet.

    100 epochs, not fewer: the worst per-source correlation error the
    dependence test reads (against its 0.2 bound) is 0.141 here and 0.107 on
    the oldest supported dependencies. At 30 epochs it is 0.141 here and 0.145
    there, but a 30-epoch fit at ``rng_seed=3`` reaches 0.180 here.
    """
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


def test_density_integrates_to_the_mass_its_draws_put_there(fleet):
    """``log_prob`` is a normalized density, per source.

    The other density checks here are finiteness, masking, agreement between
    call paths and relative ordering, all of which a constant offset in the
    marginal log-determinant (a dropped log of the box width, say) leaves
    intact. A uniform Monte Carlo integral over the box does not.

    The reference is the fraction of the fleet's own draws in the box inset
    by 1e-6, rather than 1: each source's clipped second column puts its
    lower-wall atom (0.2% to 2.8% of the mass) inside the 2e-9 inset of the
    knot grid, a sliver uniform points essentially never land in, while the
    sampler -- an independent path through the inverse spline and the copula's
    own sampler -- reaches it. With 1e5 points the integral's standard error is
    at most 0.0037 (the most peaked source) and the fraction's 0.0004, so 0.03
    is about eight combined standard errors; the largest measured gap is
    0.0042 here and 0.0037 on the oldest supported dependencies.
    """
    inset = 1e-6
    lo, hi = -1.0 + inset, 1.0 - inset
    m = 100_000
    u = np.random.default_rng(5).uniform(lo, hi, (m, DIM))
    integral = (hi - lo) ** DIM * np.mean(
        np.exp(fleet.log_prob(np.broadcast_to(u, (N, m, DIM)))), axis=1)
    x = fleet.sample(m, rng_seed=6)
    inside = np.mean(np.all((x > lo) & (x < hi), axis=-1), axis=1)
    assert np.max(np.abs(integral - inside)) < 0.03, (integral, inside)


def test_log_prob_is_minus_inf_outside_the_prior_box(fleet):
    """Finiteness of log_prob is a support test: the closed box, nothing more.

    The spline kernels clamp to the nearest knot, so without an explicit mask
    every point outside the box reports the wall's density (and corners report
    a spike), which makes grid diagnostics and support checks silently wrong.
    """
    inside = np.array([[0.0, 0.0], [-1.0, -1.0], [1.0, 1.0], [-1.0, 1.0],
                       [0.5, -1.0]])
    outside = np.array([[1.0 + 1e-9, 0.0], [-1.0 - 1e-9, 0.0], [1.5, 1.5],
                        [50.0, 50.0], [0.0, -3.0]])
    assert np.all(np.isfinite(fleet.log_prob(inside, source_index=0)))
    assert np.all(fleet.log_prob(outside, source_index=0) == -np.inf)
    # the batched path masks the same rows
    both = np.concatenate([inside, outside])
    lp = fleet.log_prob(np.broadcast_to(both, (N,) + both.shape))
    assert np.all(np.isfinite(lp[:, :len(inside)]))
    assert np.all(lp[:, len(inside):] == -np.inf)


def test_log_prob_rejects_a_misshaped_batch(fleet):
    """A (b, dim) array with no source_index is the commonest fleet mistake.
    Unchecked it reached jax.vmap, which reported it as inconsistent mapped
    axis sizes and named an internal parameter leaf."""
    with pytest.raises(ValueError, match=r"\(6, b, 2\) when source_index is "
                                         r"None; got \(4, 2\)") as exc:
        fleet.log_prob(np.zeros((4, DIM)))
    assert "source_index=i" in str(exc.value)          # names the remedy
    assert "vmap" not in str(exc.value)

    for bad, kw in [(np.zeros((2, 4, DIM)), {}),        # wrong n_sources
                    (np.zeros((4, DIM)), {}),
                    (np.zeros(DIM), {"source_index": 0}),      # 1-D
                    (np.zeros((4, DIM + 3)), {"source_index": 0})]:
        with pytest.raises(ValueError, match="x must be") as exc:
            fleet.log_prob(bad, **kw)
        assert "vmap" not in str(exc.value)

    # the shapes that are right stay right
    assert fleet.log_prob(np.zeros((4, DIM)), source_index=0).shape == (4,)
    assert fleet.log_prob(np.zeros((N, 4, DIM))).shape == (N, 4)


def test_log_prob_reports_nan_rows_as_unsupported(fleet):
    """NaN compares False against both bounds, so it slipped past the
    out-of-box mask and returned NaN -- which poisons a downstream
    Metropolis-Hastings ratio instead of being rejected."""
    x = np.zeros((3, DIM))
    x[1, 0] = np.nan
    x[2, :] = np.nan
    lp = fleet.log_prob(x, source_index=0)
    assert np.isfinite(lp[0])
    assert np.all(lp[1:] == -np.inf)
    batched = fleet.log_prob(np.broadcast_to(x, (N,) + x.shape))
    assert np.all(batched[:, 1:] == -np.inf)


def test_source_index_out_of_range_raises(fleet):
    """JAX clamps out-of-bounds indices, so the fleet has to reject them."""
    g = np.zeros((1, DIM))
    assert np.array_equal(fleet.log_prob(g, source_index=-1),
                          fleet.log_prob(g, source_index=N - 1))
    for bad in (N, N + 5, -N - 1, -99):
        with pytest.raises(IndexError, match="out of range"):
            fleet.log_prob(g, source_index=bad)
        with pytest.raises(IndexError, match="out of range"):
            fleet.sample(4, rng_seed=0, source_index=bad)


def test_save_load_roundtrip_exact(fleet):
    """Bitwise round trip, and the file carries the format and kind stamps."""
    import cloudpickle

    from coppuccino.model_io import FLEET_KIND, FORMAT_VERSION

    x = fleet.sample(300, rng_seed=5)
    lp = fleet.log_prob(x)
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "fleet.pkl")
        fleet.save(p)
        with open(p, "rb") as f:
            payload = cloudpickle.load(f)
        f2 = FleetFlow.load(p)
    assert payload["format_version"] == FORMAT_VERSION
    assert payload["kind"] == FLEET_KIND
    assert f2.arch == fleet.arch
    assert np.max(np.abs(lp - f2.log_prob(x))) == 0.0
    s1 = fleet.sample(64, rng_seed=6)
    s2 = f2.sample(64, rng_seed=6)
    assert np.max(np.abs(s1 - s2)) == 0.0


def test_warm_start(chains, fleet):
    """A warm start begins from the previous fleet's parameters.

    At a learning rate of 1e-12 one epoch cannot move them measurably, so the
    warm-started fleet must reproduce the previous one's density. The arch
    tuple alone cannot show that: a fit that silently cold-started from the
    same seed has an identical one.
    """
    f2 = normalizing_flows_fit_fleet(chains, BOUNDS, max_epochs=1,
                                     learning_rate=1e-12, rng_seed=1,
                                     warm_start=fleet)
    assert f2.arch[:5] == fleet.arch[:5]
    x = fleet.sample(500, rng_seed=7)
    assert np.max(np.abs(f2.log_prob(x) - fleet.log_prob(x))) < 1e-6
    # arch mismatch is rejected
    with pytest.raises(ValueError, match="warm_start"):
        normalizing_flows_fit_fleet(chains, BOUNDS, max_epochs=2,
                                    knots=2, warm_start=fleet)


def test_arch_kwargs_respected(chains, fleet):
    """Non-default ``knots`` / ``flow_layers`` reach both the parameters and
    the arch tuple ``load`` rebuilds from.

    The parameter count is compared with a template built directly at those
    settings, so dropping either keyword on its way to the initializer changes
    it; the shared fleet is the default-architecture contrast.
    """
    def nparams(tree):
        leaves = jax.tree_util.tree_leaves(eqx.filter(tree, eqx.is_inexact_array))
        return sum(int(np.prod(leaf.shape)) for leaf in leaves)

    small = normalizing_flows_fit_fleet(chains, BOUNDS, max_epochs=2,
                                        knots=2, flow_layers=3, rng_seed=1)
    assert nparams(small.params) == N * nparams(fleet_mod._template(DIM, 2, 3))
    assert nparams(small.params) < nparams(fleet.params)
    assert small.arch[2:4] == (2, 3) and fleet.arch[2:4] == (4, 6)


def test_determinism(chains):
    f1 = normalizing_flows_fit_fleet(chains, BOUNDS, max_epochs=8, rng_seed=9)
    f2 = normalizing_flows_fit_fleet(chains, BOUNDS, max_epochs=8, rng_seed=9)
    x = f1.sample(100, rng_seed=1)
    assert np.max(np.abs(f1.log_prob(x) - f2.log_prob(x))) == 0.0


def _count_template_builds(monkeypatch):
    """Patch ``_template`` with a counter; returns the (mutable) call list."""
    calls = []
    real = fleet_mod._template

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(fleet_mod, "_template", counting)
    return calls


def test_stacked_build_matches_per_source_loop(monkeypatch):
    """The vmapped initializer must equal the loop it replaced, bit for bit.

    Bitwise equality is what makes the size threshold safe: whichever branch
    `_stack_params` picks, the same seeds must still produce the same fit.
    """
    dim, knots, layers, n_src, seed = 2, 4, 6, 8, 7
    per_source = [eqx.partition(fleet_mod._template(dim, knots, layers,
                                                    rng_seed=seed + i),
                                eqx.is_inexact_array)[0] for i in range(n_src)]
    want = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *per_source)

    for threshold in (0, 10_000):        # force the vmap and the loop branch
        monkeypatch.setattr(fleet_mod, "_VMAP_STACK_MIN_SOURCES", threshold)
        got, _ = fleet_mod._stack_params(dim, knots, layers, n_src, seed)
        for a, b in zip(jax.tree_util.tree_leaves(got),
                        jax.tree_util.tree_leaves(want)):
            assert a.shape == b.shape and a.dtype == b.dtype
            assert np.array_equal(np.asarray(a), np.asarray(b))


def test_small_fleets_skip_the_traced_initializer(monkeypatch):
    """The traced build is flat in n_src but carries a fixed trace cost of
    seconds, which makes it a regression for a fleet of tens of sources -- the
    size in this module's own Example block. Above the threshold the number of
    template builds must be flat; below it, one per source is the cheaper deal.
    """
    dim, knots, layers, seed = 2, 4, 6, 7
    assert fleet_mod._VMAP_STACK_MIN_SOURCES > 64

    calls = _count_template_builds(monkeypatch)
    fleet_mod._stack_params(dim, knots, layers, 8, seed)
    assert len(calls) == 9          # 8 sources + the shared static template

    monkeypatch.setattr(fleet_mod, "_VMAP_STACK_MIN_SOURCES", 0)
    calls.clear()
    fleet_mod._stack_params(dim, knots, layers, 8, seed)
    flat = len(calls)
    calls.clear()
    fleet_mod._stack_params(dim, knots, layers, 32, seed)
    assert len(calls) == flat < 9


def test_load_builds_one_template_and_no_skeleton_storage(fleet, monkeypatch):
    """Deserialisation reads only shape and dtype from `like` and overwrites
    every value, so load needs one template and no device buffers for the
    skeleton. Building one template per source made load scale with the
    fleet, and materialising the skeleton allocated a second full copy of the
    fleet's parameters purely to be discarded -- the dominant avoidable term in
    load's memory, which at fleet scale is GBs.
    """
    x = fleet.sample(32, rng_seed=21)
    lp = fleet.log_prob(x)
    seen = {}
    real = eqx.tree_deserialise_leaves

    def spy(f, like, *a, **kw):
        seen["bytes"] = sum(int(leaf.size) * leaf.dtype.itemsize
                            for leaf in jax.tree_util.tree_leaves(like)
                            if isinstance(leaf, jax.Array))
        seen["leaves"] = len(jax.tree_util.tree_leaves(like))
        return real(f, like, *a, **kw)

    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "fleet.pkl")
        fleet.save(p)
        calls = _count_template_builds(monkeypatch)
        monkeypatch.setattr(fleet_mod.eqx, "tree_deserialise_leaves", spy)
        f2 = FleetFlow.load(p)
    assert len(calls) == 1
    assert seen["leaves"] > 0            # the spy really did see the skeleton
    assert seen["bytes"] == 0
    assert np.max(np.abs(lp - f2.log_prob(x))) == 0.0


class _StopBeforeTraining(Exception):
    """Raised by a spy once it has seen what it needs; nothing is trained."""


def test_training_split_does_not_pin_the_whole_u_tensor(monkeypatch):
    """Only the two halves may stay resident, not a third whole-chain copy.

    The spy stops the fit at the point it inspects, so no training step is
    ever compiled.
    """
    n_src, n_samp, dim = 3, 199, 2
    rng = np.random.default_rng(0)
    ch = rng.uniform(-1.0, 1.0, (n_src, n_samp, dim))
    n_val = max(1, int(round(n_samp * 0.1)))
    seen = {}

    def spy(*args, **kwargs):
        # reached immediately after the split and before any training step
        seen["shapes"] = {tuple(a.shape) for a in jax.live_arrays()}
        raise _StopBeforeTraining

    monkeypatch.setattr(fleet_mod, "_stack_params", spy)
    with pytest.raises(_StopBeforeTraining):
        normalizing_flows_fit_fleet(ch, BOUNDS, max_epochs=1, rng_seed=1)
    assert (n_src, n_samp - n_val, dim) in seen["shapes"]
    assert (n_src, n_val, dim) in seen["shapes"]
    assert (n_src, n_samp, dim) not in seen["shapes"]


def _reference_marginal_knots(x, lo, hi, num_points):
    """Scalar, one-source-dimension-at-a-time knot builder, kept as an oracle.

    This is the pre-vectorization implementation verbatim; the fleet builder
    must stay bitwise equal to it, ties and all.
    """
    from scipy.interpolate import PchipInterpolator

    n = len(x)
    span = max(hi - lo, fleet_mod._POS_FLOOR)
    # the inset and the tie bump follow the *smaller* of the prior and data
    # spans, so that a box far wider than the posterior cannot swallow the
    # chain; the ULP floor keeps ``bound + margin`` representable
    data_span = max(float(x.max()) - float(x.min()), fleet_mod._POS_FLOOR)
    scale = min(span, data_span)
    ulp = 4.0 * float(np.spacing(max(abs(lo), abs(hi))))
    margin = max(1e-9 * scale, ulp)
    xs = np.sort(np.clip(x.astype(float), lo + margin, hi - margin))
    xq = np.quantile(xs, np.linspace(0.0, 1.0, num_points))
    eps_x = max(1e-12 * scale, ulp)
    for i in range(1, num_points):
        if xq[i] <= xq[i - 1]:
            xq[i] = xq[i - 1] + eps_x
    counts = np.searchsorted(xs, xq, side="right")
    yq = np.clip((counts - 0.5) / n, 1e-7, 1.0 - 1e-7)
    for i in range(1, num_points):
        if yq[i] <= yq[i - 1]:
            yq[i] = yq[i - 1] + 1e-9
    xk = np.concatenate([[lo], xq, [hi]])
    yk = np.concatenate([[0.0], yq, [1.0]])
    if xk[1] <= xk[0]:
        xk[1] = xk[0] + eps_x
    if xk[-2] >= xk[-1]:
        xk[-2] = xk[-1] - eps_x
    dk = np.maximum(PchipInterpolator(xk, yk).derivative()(xk),
                    fleet_mod._POS_FLOOR)
    return xk, yk, dk


@pytest.mark.parametrize("kind", ["continuous", "tied"])
def test_fleet_knots_match_scalar_reference(kind):
    """Vectorizing the knot build must not move a single knot."""
    rng = np.random.default_rng(3)
    raw = rng.uniform(-1.0, 1.0, (4, 500, 3))
    # heavy ties (quantized samples, some sitting exactly on a wall) are the
    # case where the tie bumps actually fire
    ch = raw if kind == "continuous" else np.round(np.clip(raw * 1.4, -1, 1), 2)
    lo, hi = np.full((4, 3), -1.0), np.full((4, 3), 1.0)
    got = [np.asarray(a) for a in fleet_mod._build_fleet_knots(ch, lo, hi, 60)]
    for s in range(4):
        for j in range(3):
            want = _reference_marginal_knots(ch[s, :, j], -1.0, 1.0, 60)
            for a, b in zip(got, want):
                assert np.array_equal(a[s, j], b)


def _knot_families(rng):
    """(name, chains, lo, hi) covering the shapes the slope kernel must handle."""
    n, d, ns = 400, 3, 4
    raw = rng.uniform(-1.0, 1.0, (ns, n, d))
    ones = np.ones((ns, d))
    return [
        ("continuous", raw, -ones, ones),
        # quantized samples, many exactly on a wall: ties and clipping
        ("tied-and-clipped", np.round(np.clip(raw * 1.4, -1, 1), 2), -ones, ones),
        # only two distinct values per column: every interior secant is 0, the
        # branch scipy special-cases to a zero slope
        ("two-valued", np.where(raw > 0, 0.5, -0.5) + 1e-9 * raw, -ones, ones),
        # a box orders of magnitude wider than the chain
        ("ragged-bounds", rng.normal(500.0, 60.0, (ns, n, d)),
         np.zeros((ns, d)), np.full((ns, d), 1e14)),
        # one-sided support, mass piled against the lower bound
        ("exponential", rng.exponential(1.0, (ns, n, d)),
         np.zeros((ns, d)), np.full((ns, d), 20.0)),
        # separated modes: long flat stretches of CDF between them
        ("bimodal", np.clip(np.concatenate(
            [rng.normal(-0.9, 0.15, (ns, n // 2, d)),
             rng.normal(0.9, 0.15, (ns, n // 2, d))], axis=1), -1, 1),
         -ones, ones),
        # per-column bounds spanning three orders of magnitude
        ("mixed-scales", rng.uniform(0.0, 1.0, (ns, n, d)) * [1.0, 10.0, 1e3],
         np.zeros((ns, d)), np.tile([1.0, 10.0, 1e3], (ns, 1))),
    ]


def test_vectorized_pchip_slopes_are_bitwise_the_scipy_loop():
    """`_build_fleet_knots` used to call scipy's PchipInterpolator once per
    (source, dimension) -- 80,000 constructions at the documented 10^4-source
    scale, and ~74% of the builder's time. The array form replacing it has to
    be bit-identical, ties, clipped walls and ragged boxes included, or the
    same seeds stop reproducing the same fit. 200 grid points over 400
    samples puts several quantiles between adjacent samples, so the CDF tie
    bumps fire on every family.
    """
    num_points = 200
    from coppuccino.bijections import _rqs_knot_derivatives

    rng = np.random.default_rng(11)
    for name, ch, lo, hi in _knot_families(rng):
        xk, yk, dk = (np.asarray(a) for a in
                      fleet_mod._build_fleet_knots(ch, lo, hi, num_points))
        for s in range(ch.shape[0]):
            for j in range(ch.shape[2]):
                want = _rqs_knot_derivatives(xk[s, j], yk[s, j])
                assert np.array_equal(dk[s, j], want), f"{name} [{s},{j}]"


def test_slope_kernel_matches_scipy_on_degenerate_knot_rows():
    """The knot rows `_build_fleet_knots` produces are always strictly
    increasing in both axes, so PCHIP's flat/sign-change branch and its two
    endpoint corrections never fire there. Exercise them directly, since the
    kernel is written to reproduce scipy's whole rule and not just the part a
    CDF happens to reach.
    """
    from coppuccino.bijections import _rqs_knot_derivatives

    xk = np.linspace(-1.0, 1.0, 9)
    rows = {
        "plateau": [0.0, 0.2, 0.5, 0.5, 0.5, 0.5, 0.7, 0.9, 1.0],
        "non-monotone": [0.0, 0.6, 0.2, 0.8, 0.3, 0.9, 0.4, 1.0, 0.5],
        "flat-start": [0.0, 0.0, 0.0, 0.1, 0.3, 0.6, 0.8, 0.9, 1.0],
        "flat-end": [0.0, 0.1, 0.3, 0.6, 0.8, 1.0, 1.0, 1.0, 1.0],
        "steep-endpoint": [0.0, 0.99, 0.991, 0.992, 0.993, 0.994, 0.995,
                           0.996, 1.0],
        "descending": [1.0, 0.9, 0.7, 0.4, 0.4, 0.3, 0.2, 0.1, 0.0],
        "single-spike": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    }
    # uneven spacing changes the harmonic weights and the endpoint estimate
    xk_uneven = np.array([-1.0, -0.99, -0.5, -0.4, 0.0, 0.05, 0.5, 0.95, 1.0])
    for grid in (xk, xk_uneven):
        stack_x = np.broadcast_to(grid, (len(rows), len(grid))).copy()
        stack_y = np.array(list(rows.values()))
        got = fleet_mod._pchip_knot_derivatives(stack_x, stack_y)
        for i, name in enumerate(rows):
            want = _rqs_knot_derivatives(grid, stack_y[i])
            assert np.array_equal(got[i], want), name

    with pytest.raises(ValueError, match="at least 3 knots"):
        fleet_mod._pchip_knot_derivatives(np.zeros((2, 2)), np.zeros((2, 2)))


def test_slope_kernel_applies_molers_steep_endpoint_correction():
    """PCHIP's endpoint rule has two shape-preserving corrections, and only one
    of them is reachable from a CDF: an endpoint slope that merely disagrees in
    sign with its secant (zeroed). The second -- the one-sided estimate capped
    at three times the end secant when the two end secants disagree in sign --
    needs a genuinely *steep* endpoint, which no monotone knot row produces.
    Dropping the cap therefore leaves the rest of the suite green, so pin it
    here on rows built to fire it, at both ends and on uneven spacing (the
    estimate is h-weighted, so even spacing exercises only one weighting).
    """
    from coppuccino.bijections import _rqs_knot_derivatives

    # each row: end secant m0 = +0.1 against a next secant m1 = -1.0, which is
    # more than steep enough that the raw estimate d overshoots 3 * m0
    rows = {
        "left-uniform": (np.arange(5.0),
                         np.array([0.0, 0.1, -0.9, -0.8, -0.7]), 0),
        "right-uniform": (np.arange(5.0),
                          np.array([0.8, 0.9, 1.0, 0.0, 0.1]), -1),
        "left-uneven": (np.array([0.0, 2.0, 2.25, 3.0, 4.0]),
                        np.array([0.0, 0.2, -0.05, 0.5, 1.0]), 0),
        "right-uneven": (np.array([0.0, 1.0, 1.75, 2.0, 4.0]),
                         np.array([0.0, 0.5, 1.0, 0.75, 0.95]), -1),
    }
    for name, (xk, yk, end) in rows.items():
        hk = np.diff(xk)
        mk = np.diff(yk) / hk
        h0, h1, m0, m1 = ((hk[0], hk[1], mk[0], mk[1]) if end == 0
                          else (hk[-1], hk[-2], mk[-1], mk[-2]))
        raw = ((2 * h0 + h1) * m0 - h0 * m1) / (h0 + h1)
        # the row is only a test of the cap if the cap is what fires: not the
        # sign-disagreement branch, and the raw estimate really is too steep
        assert np.sign(raw) == np.sign(m0), name
        assert np.sign(m0) != np.sign(m1) and abs(raw) > 3.0 * abs(m0), name

        got = fleet_mod._pchip_knot_derivatives(xk[None], yk[None])[0]
        assert np.array_equal(got, _rqs_knot_derivatives(xk, yk)), name
        # the corrected value, and not the uncapped estimate it replaces. Both
        # are well above _POS_FLOOR, so the final clamp cannot hide the swap.
        assert got[end] == 3.0 * m0, name
        assert got[end] != raw, name


def test_slope_kernel_rejects_non_increasing_knots():
    """scipy's PchipInterpolator refused a non-increasing ``x``; the array
    kernel that replaced it must too. A reversed or repeated knot only flips or
    blows up a secant, which every branch below reads as an ordinary sign
    change or a flat, so the kernel would otherwise hand back a plausible slope
    vector for knots that are not a function of x.
    """
    from scipy.interpolate import PchipInterpolator

    xk = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    yk = np.array([0.0, 0.2, 0.5, 0.7, 1.0])
    reversed_step = np.array([0.0, 1.0, 2.0, 1.5, 4.0])
    repeated_knot = np.array([0.0, 1.0, 2.0, 2.0, 4.0])
    for row in (reversed_step, repeated_knot):
        with pytest.raises(ValueError, match="strictly increasing"):
            PchipInterpolator(row, yk)
        with pytest.raises(ValueError, match="strictly increasing"):
            fleet_mod._pchip_knot_derivatives(row[None], yk[None])

    # one bad row in a stack must fail the whole call, and name itself
    stack_x = np.broadcast_to(xk, (3, 5)).copy()
    stack_x[1] = reversed_step
    with pytest.raises(ValueError, match=r"1 of 3 rows.*Row \(1,\)"):
        fleet_mod._pchip_knot_derivatives(stack_x,
                                          np.broadcast_to(yk, (3, 5)).copy())


def test_knot_build_fails_loudly_on_a_box_narrower_than_its_own_resolution():
    """A finite, ordered, per-source-varying box that both validators accept
    can still be too narrow to lay a knot grid in: [1e10, 1e10 + 1e-3] spans
    about 524 float64 ULP, so the 4-ULP tie bumps walk the quantile grid past
    the upper bound and the endpoint guard pulls back only the last interior
    knot. That used to return a slope vector for a knot row that decreases by
    5e-4 in the middle -- a silently meaningless fit. It must raise.
    """
    rng = np.random.default_rng(0)
    lo_v, hi_v = 1e10, 1e10 + 1e-3
    lo = np.full((2, 1), lo_v)
    hi = np.full((2, 1), hi_v)
    ch = rng.uniform(lo_v, hi_v, (2, 700, 1))

    # the box itself is legitimate: neither validator has anything to say
    got_lo, got_hi = fleet_mod._validate_fleet_bounds(
        np.array([[lo_v, hi_v]]), 2, 1)
    assert np.array_equal(got_lo, lo) and np.array_equal(got_hi, hi)
    fleet_mod._validate_fleet_chains(ch, lo, hi)

    with pytest.raises(ValueError, match="strictly increasing knots"):
        fleet_mod._build_fleet_knots(ch, lo, hi, 200)

    # a coarse enough grid still fits inside the same box, so the guard is
    # rejecting the degenerate grid and not the box's magnitude
    xk, _, _ = (np.asarray(a)
                for a in fleet_mod._build_fleet_knots(ch, lo, hi, 20))
    assert (np.diff(xk, axis=-1) > 0).all()


def test_chunked_knot_build_matches_the_unchunked_one(monkeypatch):
    """The block loops are memory management only, so they must be invisible.

    The shipped block size is ~5000 sources wide for a fleet this shape, so at
    test sizes the loops take exactly one iteration and a block-indexing bug
    would go unseen. Shrink the budget until they genuinely iterate.
    """
    rng = np.random.default_rng(17)
    # per-source, per-column bounds: with one shared box a block loop that
    # indexed lo/hi/eps from 0 instead of s0 would be invisible
    lo = np.array([[-1.0 - s, -10.0 * (s + 1)] for s in range(7)])
    hi = np.array([[1.0 + 2 * s, 10.0 * (s + 1)] for s in range(7)])
    ch = lo[:, None, :] + (hi - lo)[:, None, :] * rng.uniform(0, 1, (7, 300, 2))
    ch = np.round(ch, 1)                       # ties, so the bumps fire too

    monkeypatch.setattr(fleet_mod, "_SORT_BLOCK_ELEMENTS", 10**9)
    whole = [np.asarray(a) for a in fleet_mod._build_fleet_knots(ch, lo, hi, 40)]

    small = 1500
    monkeypatch.setattr(fleet_mod, "_SORT_BLOCK_ELEMENTS", small)
    # both loops must take more than one pass at this budget
    assert max(1, small // (ch.shape[2] * ch.shape[1])) < ch.shape[0]
    assert max(1, small // (8 * ch.shape[2] * 42)) < ch.shape[0]
    blocked = [np.asarray(a) for a in fleet_mod._build_fleet_knots(ch, lo, hi, 40)]

    for a, b in zip(whole, blocked):
        assert np.array_equal(a, b)


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


@pytest.mark.parametrize("kwargs, name", [
    ({"batch_size": 0}, "batch_size"),
    ({"val_prop": 0.0}, "val_prop"),
    ({"val_prop": 1.0}, "val_prop"),
    # rounds to every row, leaving nothing to train on
    ({"val_prop": 0.9999}, "val_prop"),
    ({"learning_rate": 0.0}, "learning_rate"),
    ({"max_epochs": 0}, "max_epochs"),
    ({"patience": 0}, "patience"),
    ({"knots": 1}, "knots"),
    ({"flow_layers": 0}, "flow_layers"),
    ({"num_points": 1}, "num_points"),
    ({"rng_seed": -1}, "rng_seed"),
])
def test_training_settings_are_checked_by_name(chains, kwargs, name):
    """The fleet's training knobs get the single-flow fit's range checks.

    Unchecked, a negative batch_size, a non-positive learning_rate or
    max_epochs=0 returned a fleet that never trained, and batch_size=0,
    val_prop=1.0 or knots=0 died on a bare ZeroDivisionError. One boundary
    value per knob shows it is wired to the check; the shared range helpers'
    other branches are tested with ``normalizing_flows_fit``. ``val_prop`` has
    three cases because three different checks guard it.
    """
    with pytest.raises(ValueError, match=name):
        normalizing_flows_fit_fleet(chains, BOUNDS, **kwargs)


def test_fit_options_are_keyword_only(chains):
    with pytest.raises(TypeError, match="positional"):
        normalizing_flows_fit_fleet(chains, BOUNDS, 4)


def test_a_source_that_never_trains_is_named(chains, monkeypatch):
    """A source left at its initialization is reported, and the rest are kept
    at their own best epoch.

    Per-source best parameters only move when that source's validation loss
    improves, so a source that never improves comes back untrained. Source 2
    is started from NaN parameters here, so its loss is NaN throughout, which
    no comparison ever counts as an improvement.

    The same fit pins the per-source bookkeeping. From the second epoch on
    the optimizer turns every update NaN for the odd-numbered sources while
    the even ones keep training, so the odd sources' last epochs are NaN, and
    in every later epoch some sources improve and others do not. They come
    back finite only if the fit returns each source's best parameters rather
    than its last, and the improvement mask keeps a source that did not
    improve at its earlier best. Nothing else distinguishes either, since a
    healthy fit's last epoch is about as good as its best.
    """
    real = fleet_mod._stack_params
    real_adam = fleet_mod.optax.adam
    # the fit's defaults: 200 of the 2000 rows held out, batches of 128
    steps_per_epoch = (NSAMP - 200) // 128
    odd = jnp.arange(N) % 2 == 1

    def poisoned(*args, **kwargs):
        params, static = real(*args, **kwargs)
        params = jax.tree_util.tree_map(
            lambda a: a.at[2].set(jnp.nan) if eqx.is_inexact_array(a) else a, params)
        return params, static

    def adam_then_nan_for_odd_sources(learning_rate):
        inner = real_adam(learning_rate)

        def init(params):
            return inner.init(params), jnp.zeros((), jnp.int32)

        def update(grads, state, params=None):
            inner_state, step = state
            updates, inner_state = inner.update(grads, inner_state, params)
            spoil = odd & (step >= steps_per_epoch)
            updates = jax.tree_util.tree_map(
                lambda u: jnp.where(spoil.reshape((-1,) + (1,) * (u.ndim - 1)),
                                    jnp.nan, u), updates)
            return updates, (inner_state, step + 1)

        return fleet_mod.optax.GradientTransformation(init, update)

    monkeypatch.setattr(fleet_mod, "_stack_params", poisoned)
    monkeypatch.setattr(fleet_mod.optax, "adam", adam_then_nan_for_odd_sources)
    with pytest.warns(UserWarning, match=r"1 of 6 sources .*\[2\]"):
        fl = normalizing_flows_fit_fleet(chains, BOUNDS, max_epochs=3, rng_seed=1)
    lp = fl.log_prob(chains[:, :200])
    assert np.all(np.isfinite(np.delete(lp, 2, axis=0)))


def test_a_fleet_that_never_trains_raises(chains):
    """At an absurd learning_rate no source improves on its initialization."""
    with pytest.raises(RuntimeError, match="learning_rate"):
        normalizing_flows_fit_fleet(chains, BOUNDS, max_epochs=3, learning_rate=10.0,
                                    rng_seed=1)


def test_draw_counts_are_checked(fleet):
    """Both draw methods are wired to the shared ``n_samples`` check, whose
    other branches are tested with the package-level ``sample``."""
    with pytest.raises(ValueError, match="n_samples"):
        fleet.sample(0, rng_seed=0)
    with pytest.raises(ValueError, match="n_samples"):
        fleet.sample_and_log_prob(0, rng_seed=0)


def test_a_damaged_fleet_file_is_named(fleet):
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "fleet.pkl")
        fleet.save(p)
        with open(p, "rb") as f:
            head = f.read(200)
        with open(p, "wb") as f:
            f.write(head)
        with pytest.raises(ValueError, match="fleet.pkl"):
            FleetFlow.load(p)


def test_prior_bounds_get_the_single_flow_path_diagnosis(chains):
    """Reversed, degenerate and non-finite boxes each used to be reported as
    something else: the first two as 'chains contain samples outside
    prior_bounds' (which blames the data), the last two as interpax/scipy's
    '`x` must contain only finite values' (which names no coppuccino symbol).
    copula_flows._validate_prior_bounds diagnoses all of them; so must this.
    """
    def bounds(row0):
        return np.array([row0, [-1.0, 1.0]])

    for row0, pattern in [([1.0, -1.0], r"row 0 must be \[low, high\]"),
                          ([0.0, 0.0], r"row 0 must be \[low, high\]"),
                          ([np.nan, 1.0], "row 0 must be finite"),
                          ([-np.inf, 1.0], "row 0 must be finite"),
                          ([-1.0, np.inf], "row 0 must be finite")]:
        with pytest.raises(ValueError, match=pattern) as exc:
            normalizing_flows_fit_fleet(chains, bounds(row0), max_epochs=1)
        assert "outside prior_bounds" not in str(exc.value)

    # per-source bounds name the source as well as the row
    pb = np.broadcast_to(BOUNDS, (N, DIM, 2)).copy()
    pb[3, 1] = [2.0, -2.0]
    with pytest.raises(ValueError, match=r"prior_bounds\[3\] row 1"):
        normalizing_flows_fit_fleet(chains, pb, max_epochs=1)


def test_infinite_chain_entries_are_rejected(chains):
    """+inf used to be reported as a containment failure, pointing the user at
    prior_bounds rather than at the row that is not a number."""
    for value in (np.inf, -np.inf):
        bad = chains.copy()
        bad[2, 7, 1] = value
        with pytest.raises(ValueError, match="non-finite") as exc:
            normalizing_flows_fit_fleet(bad, BOUNDS, max_epochs=1)
        assert "source 2, parameter (column) 1" in str(exc.value)


def test_constant_column_is_rejected():
    """A column that never moves has no empirical CDF. The fleet used to fit it
    silently and then sample it across the whole prior box, while
    normalizing_flows_fit raised on the identical input."""
    rng = np.random.default_rng(5)
    ch = rng.uniform(0.2, 1.8, (2, 400, 2))
    ch[:, :, 1] = 1.0
    bounds = np.array([[0.0, 2.0], [0.0, 2.0]])
    with pytest.raises(ValueError, match="two distinct sample values") as exc:
        normalizing_flows_fit_fleet(ch, bounds, max_epochs=2)
    assert "parameter (column) 1" in str(exc.value)


def test_a_box_far_wider_than_the_chain_does_not_swallow_it():
    """The quantile grid's inset is 1e-9 of a span. Taken from the *prior*
    span alone, prior_bounds=[[0, 1e14]] insets by 1e5 and clips a chain living
    around 500 onto a single point -- the fit then draws from the inset, not
    from the posterior. The inset has to follow the smaller of the two spans.
    """
    rng = np.random.default_rng(2)
    ch = rng.normal(500.0, 60.0, (2, 400, 1))
    fl = normalizing_flows_fit_fleet(ch, np.array([[0.0, 1e14]]),
                                     max_epochs=20, rng_seed=1)
    s = fl.sample(2000, rng_seed=0, source_index=0)
    lo_d, hi_d = float(ch[0].min()), float(ch[0].max())
    assert lo_d < float(np.median(s)) < hi_d
    # the bulk of the fleet's mass has to sit on the data, not on the box
    assert np.mean((s >= lo_d - 60.0) & (s <= hi_d + 60.0)) > 0.98
    # and the density has to fall away from the data, not toward the far wall
    lp = fl.log_prob(np.array([[500.0], [1100.0], [1e6]]), source_index=0)
    assert lp[0] > lp[1] > lp[2]


def test_sample_and_log_prob_matches_sample_then_log_prob(fleet):
    """Parity with the package-level verb, and bitwise-consistent densities.

    For a fixed seed a single source's draws are the corresponding slice of
    the whole-fleet draws, and its densities the slice of the batched ones --
    at the first, a middle, the last and a negative index, since a falsy
    ``source_index=0`` and an end-of-range clamp are the likely ways to get
    that wrong.
    """
    x, lp = fleet.sample_and_log_prob(128, rng_seed=12)
    assert x.shape == (N, 128, DIM)
    assert lp.shape == (N, 128)
    assert np.max(np.abs(x - fleet.sample(128, rng_seed=12))) == 0.0
    assert np.max(np.abs(lp - fleet.log_prob(x))) == 0.0

    for s in (0, 1, N - 1, -1):
        x1, lp1 = fleet.sample_and_log_prob(128, rng_seed=12, source_index=s)
        assert np.max(np.abs(x1 - x[s])) == 0.0
        assert np.max(np.abs(lp1 - lp[s])) == 0.0


def test_package_level_verbs_reject_a_fleet(fleet):
    """A fleet used to fail deep inside jax/equinox with an internal detail
    (`int()`, a vmap shape dump, a missing `.bijection`) instead of a pointer
    at its own batched API."""
    import coppuccino as cp

    x = fleet.sample(8, rng_seed=0)
    cases = [
        (lambda: cp.sample(fleet, 8), "fleet.sample"),
        (lambda: cp.log_prob(fleet, x[0]), "fleet.log_prob"),
        (lambda: cp.log_prob(fleet, x), "fleet.log_prob"),
        (lambda: cp.sample_and_log_prob(fleet, 8), "fleet.sample_and_log_prob"),
    ]
    for call, hint in cases:
        with pytest.raises(TypeError, match="FleetFlow") as exc:
            call()
        assert hint in str(exc.value)

    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "flow.pkl")
        with pytest.raises(TypeError, match="fleet.save"):
            cp.save_flow(fleet, p)
        assert not os.path.exists(p)


def test_unstamped_legacy_fleet_file_still_loads(fleet):
    """Every fleet pickle written before the stamps existed lacks both keys, so
    a missing version must read as 1 and the kind be inferred from the keys."""
    import cloudpickle

    x = fleet.sample(64, rng_seed=13)
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "fleet.pkl")
        fleet.save(p)
        with open(p, "rb") as f:
            payload = cloudpickle.load(f)
        legacy = {k: v for k, v in payload.items()
                  if k not in ("format_version", "kind")}
        with open(p, "wb") as f:
            cloudpickle.dump(legacy, f)
        reloaded = FleetFlow.load(p)
    assert np.max(np.abs(reloaded.log_prob(x) - fleet.log_prob(x))) == 0.0


def test_load_rejects_a_single_flow_file(fleet):
    """A save_flow payload shares no keys with a fleet's, so this used to die
    on `KeyError: 'arch'`."""
    import cloudpickle

    from coppuccino.model_io import FLOW_KIND, FORMAT_VERSION

    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "flow.pkl")
        with open(p, "wb") as f:
            cloudpickle.dump({"format_version": FORMAT_VERSION,
                              "kind": FLOW_KIND,
                              "spline_data": {}, "inner_flow_bytes": b""}, f)
        with pytest.raises(ValueError, match="coppuccino.load_flow"):
            FleetFlow.load(p)

        with open(p, "wb") as f:
            cloudpickle.dump({"format_version": FORMAT_VERSION + 1,
                              "kind": "fleet"}, f)
        with pytest.raises(ValueError, match="save format version"):
            FleetFlow.load(p)


class _DiskFullHalfway:
    """A writable file whose first write stores half the bytes, then fails
    the way a full disk does."""

    def __init__(self, f):
        self._f = f

    def write(self, data):
        self._f.write(bytes(data)[: len(data) // 2])
        self._f.flush()
        raise OSError(errno.ENOSPC, "No space left on device")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._f.close()
        return False

    def __getattr__(self, name):
        return getattr(self._f, name)


def test_failed_save_leaves_the_previous_fleet_intact(fleet, monkeypatch):
    """The old writer opened the destination 'wb', which truncates on entry.

    The fault is injected in the write itself, after half the payload is on
    disk, at both ways of getting a file to write to: ``os.fdopen`` (how the
    atomic writer opens its temporary sibling) and ``open`` on a path in the
    destination directory (how a direct writer would open the destination).
    So a writer that truncates the destination fails here whichever it uses,
    and the atomic one has a real half-written temporary file to clean up.
    Failing earlier, inside ``cloudpickle.dumps``, could tell neither apart:
    no write ever starts.
    """
    x = fleet.sample(64, rng_seed=14)
    real_fdopen, real_open = os.fdopen, builtins.open
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "fleet.pkl")
        fleet.save(p)
        good = os.path.getsize(p)

        def open_into_a_full_disk(file, mode="r", *args, **kwargs):
            f = real_open(file, mode, *args, **kwargs)
            in_td = (isinstance(file, (str, os.PathLike))
                     and os.path.dirname(os.path.abspath(file)) == td)
            return _DiskFullHalfway(f) if in_td and "r" not in mode else f

        monkeypatch.setattr(os, "fdopen", lambda fd, *args, **kwargs:
                            _DiskFullHalfway(real_fdopen(fd, *args, **kwargs)))
        monkeypatch.setattr(builtins, "open", open_into_a_full_disk)
        with pytest.raises(OSError, match="No space left on device"):
            fleet.save(p)
        monkeypatch.undo()

        assert os.path.getsize(p) == good
        assert os.listdir(td) == ["fleet.pkl"]
        assert np.max(np.abs(FleetFlow.load(p).log_prob(x)
                             - fleet.log_prob(x))) == 0.0


def test_load_documents_the_pickle_trust_boundary():
    """cloudpickle executes code from the file during load; say so, in the same
    terms as `load_flow`."""
    doc = FleetFlow.load.__doc__
    assert ".. warning::" in doc
    lowered = doc.lower()
    for phrase in ("pickle", "arbitrary", "trust"):
        assert phrase in lowered
