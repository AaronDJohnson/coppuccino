"""Tests for normalizing_flows_fit_auto (held-out hyperparameter selection)."""

from types import SimpleNamespace

import numpy as np
import pytest

from coppuccino import (
    log_prob,
    normalizing_flows_fit_auto,
    sample,
)
from coppuccino.copula_flows import (
    _n_trainable_params,
    normalizing_flows_fit,
)


@pytest.fixture(scope="module")
def data():
    """Correlated samples inside BOUNDS, with a wall atom on each bound.

    ``np.clip`` puts a mass of samples *exactly* on the bound, which is both
    the documented use of ``prior_bounds`` and the case that used to poison
    every held-out mean here: ``z_of_theta`` did not apply the clamp that
    ``log_dtheta_dz`` did, so an at-bound row scored around -5e5 nats and
    swamped the comparison the selection rules are built on. Keeping the
    clipped data means these selection tests would notice that regression.
    """
    rng = np.random.default_rng(0)
    n = 1500
    a = rng.uniform(-1.0, 1.0, n)
    b = np.clip(0.6 * a + 0.3 * rng.standard_normal(n), -1.0, 1.0)
    return np.stack([a, b], 1)


BOUNDS = np.array([[-1.0, 1.0], [-1.0, 1.0]])


def test_selects_and_returns_working_flow(data):
    flow, table = normalizing_flows_fit_auto(
        data, prior_bounds=BOUNDS, max_epochs=40,
        candidates=[{"knots": 4, "flow_layers": 3}, {"knots": 8, "flow_layers": 3}],
        return_table=True, comparable_sems=0.0, comparable_nats=0.0,
    )
    assert len(table) == 2
    assert sum(r["selected"] for r in table) == 1
    # scores are finite; with zero comparability tolerance the argmax is selected
    scores = [r["val_logp"] for r in table]
    assert np.all(np.isfinite(scores))
    assert table[int(np.argmax(scores))]["selected"]
    assert all(r["n_params"] > 0 for r in table)
    # returned flow works end to end
    d = sample(flow, 500, rng_seed=1)
    assert d.shape == (500, 2)
    assert np.all((d >= -1) & (d <= 1))
    assert np.all(np.isfinite(log_prob(flow, d)))


def test_smallest_comparable_prefers_parsimony(data):
    """With a generous absolute tolerance, the smaller model must win even if
    the bigger one scores marginally higher."""
    _, table = normalizing_flows_fit_auto(
        data, prior_bounds=BOUNDS, max_epochs=40,
        candidates=[{"knots": 2, "flow_layers": 3}, {"knots": 8, "flow_layers": 6}],
        return_table=True, refit_best=False,
        comparable_nats=1.0,   # everything within 1 nat counts as comparable
    )
    sizes = [r["n_params"] for r in table]
    assert all(r["comparable"] for r in table)
    assert table[int(np.argmin(sizes))]["selected"]


def test_zero_tolerance_degenerates_to_best(data):
    """With tolerance 0, smallest_comparable must select the argmax."""
    _, table = normalizing_flows_fit_auto(
        data, prior_bounds=BOUNDS, max_epochs=40,
        candidates=[{"knots": 2, "flow_layers": 3}, {"knots": 4, "flow_layers": 3}],
        return_table=True, refit_best=False,
        comparable_sems=0.0, comparable_nats=0.0,
    )
    scores = [r["val_logp"] for r in table]
    assert table[int(np.argmax(scores))]["selected"]


def test_reproducible_selection(data):
    _, t1 = normalizing_flows_fit_auto(
        data, prior_bounds=BOUNDS, max_epochs=30,
        candidates=[{"knots": 4, "flow_layers": 3}, {"knots": 6, "flow_layers": 3}],
        return_table=True, refit_best=False, rng_seed=5,
    )
    _, t2 = normalizing_flows_fit_auto(
        data, prior_bounds=BOUNDS, max_epochs=30,
        candidates=[{"knots": 4, "flow_layers": 3}, {"knots": 6, "flow_layers": 3}],
        return_table=True, refit_best=False, rng_seed=5,
    )
    assert [r["selected"] for r in t1] == [r["selected"] for r in t2]
    assert np.allclose([r["val_logp"] for r in t1], [r["val_logp"] for r in t2])


def test_too_few_samples_raises():
    with pytest.raises(ValueError, match="at least"):
        normalizing_flows_fit_auto(np.random.randn(20, 2))


@pytest.mark.parametrize("val_frac", [0.0, -0.5, 1.0, 1.5, np.nan])
def test_val_frac_must_be_a_proper_fraction(data, val_frac):
    """``val_frac <= 0`` used to be clamped to a *one-row* held-out set, which
    reports ``sem: 0.0`` beside a normal-looking ``val_logp`` -- a table a
    caller cannot tell apart from a real split. ``val_frac >= 1`` surfaced as
    'every candidate fit raised ... Insufficient samples for empirical
    transform', naming neither ``val_frac`` nor the real cause."""
    with pytest.raises(ValueError, match="val_frac") as exc:
        normalizing_flows_fit_auto(
            data, val_frac=val_frac, candidates=[{"knots": 2, "flow_layers": 2}],
            max_epochs=2, refit_best=False, n_workers=1)
    assert repr(val_frac) in str(exc.value)


def test_val_frac_that_starves_either_side_names_both_counts(data):
    """A fraction inside (0, 1) can still leave a part too small to use."""
    with pytest.raises(ValueError, match="held-out rows and fitting needs"):
        normalizing_flows_fit_auto(
            data, val_frac=0.999, candidates=[{"knots": 2, "flow_layers": 2}],
            max_epochs=2, refit_best=False, n_workers=1)


def test_auto_reports_dropped_nonfinite_rows_like_the_single_fitter(data):
    """The same dirty chain used to be reported through one entry point and
    silently cleaned by the other."""
    dirty = data.copy()
    dirty[3, 1] = np.nan
    dirty[7, 0] = np.inf
    with pytest.warns(UserWarning, match=r"dropped 2 of 1500 rows"):
        normalizing_flows_fit_auto(
            dirty, candidates=[{"knots": 2, "flow_layers": 2}], max_epochs=2,
            refit_best=False, n_workers=1)


def test_parallel_matches_sequential(data):
    """Concurrent candidate fits must reproduce sequential results exactly
    (fixed per-candidate seeds and split => execution order cannot matter)."""
    kw = {"prior_bounds": BOUNDS, "max_epochs": 30,
          "candidates": [{"knots": 2, "flow_layers": 3},
                         {"knots": 4, "flow_layers": 3}],
          "return_table": True, "refit_best": False, "rng_seed": 11}
    _, t_seq = normalizing_flows_fit_auto(data, n_workers=1, **kw)
    _, t_par = normalizing_flows_fit_auto(data, n_workers=2, **kw)
    assert [r["selected"] for r in t_seq] == [r["selected"] for r in t_par]
    assert np.allclose([r["val_logp"] for r in t_seq],
                       [r["val_logp"] for r in t_par])



def _values_from_edge_masses(masses, n=1000):
    """Values in underflow, first, middle, last and overflow cells."""
    locations = np.array([-3.0, -1.5, 0.0, 1.5, 3.0])
    counts = np.asarray(np.round(np.asarray(masses) * n), dtype=int)
    counts[2] += n - counts.sum()
    return np.repeat(locations, counts)[:, None]


def test_all_candidate_failures_are_aggregated(monkeypatch, data):
    import coppuccino.copula_flows as module

    def fail_fit(values, **kwargs):
        del values, kwargs
        raise RuntimeError("synthetic candidate failure")

    monkeypatch.setattr(module, "normalizing_flows_fit", fail_fit)
    # When *every* candidate raised, nothing diverged: the error leads with
    # the actual exceptions rather than the misleading "try a smaller
    # learning_rate" advice, and still names each failing candidate.
    with pytest.warns(UserWarning, match="excluded failed fits"), \
            pytest.raises(RuntimeError, match="every candidate fit raised") as exc:
            normalizing_flows_fit_auto(
                data, candidates=[{"knots": 2}, {"knots": 4}],
                refit_best=False, n_workers=1)
    assert "candidate 0" in str(exc.value) and "candidate 1" in str(exc.value)
    assert "synthetic candidate failure" in str(exc.value)
    assert "learning_rate" not in str(exc.value)


def test_partial_failure_still_reports_divergence_advice(monkeypatch, data):
    """A mix of raised and non-finite candidates keeps the diagnostic path."""
    import coppuccino.copula_flows as module

    real_fit = module.normalizing_flows_fit
    calls = {"n": 0}

    def flaky_fit(values, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("synthetic candidate failure")
        return real_fit(values, **kwargs)

    monkeypatch.setattr(module, "normalizing_flows_fit", flaky_fit)
    monkeypatch.setattr(module, "log_prob",
                        lambda flow, x: np.full(len(x), np.nan))
    with pytest.warns(UserWarning, match="excluded failed fits"), \
            pytest.raises(RuntimeError, match="likely diverged"):
        normalizing_flows_fit_auto(
            data, candidates=[{"knots": 2}, {"knots": 4}],
            max_epochs=3, refit_best=False, n_workers=1)


def _refit_probe(monkeypatch, refit_log_prob):
    """Monkeypatch the module so the full-chain refit has a chosen held-out score.

    Returns the chain to fit; the produced flows expose ``full_refit``.
    """
    import coppuccino.copula_flows as module

    chain = _values_from_edge_masses([0.10, 0.20, 0.40, 0.20, 0.10])

    def fake_fit(values, **kwargs):
        del kwargs
        return SimpleNamespace(full_refit=len(values) == len(chain))

    monkeypatch.setattr(module, "normalizing_flows_fit", fake_fit)
    monkeypatch.setattr(
        module, "log_prob",
        lambda flow, values: np.full(
            len(values), refit_log_prob if flow.full_refit else 0.0))
    monkeypatch.setattr(module, "_n_trainable_params", lambda flow: 10)
    return chain


def test_a_nonfinite_refit_is_rejected(monkeypatch):
    """The full-chain refit used to be swapped in unchecked."""
    chain = _refit_probe(monkeypatch, np.nan)

    with pytest.warns(UserWarning, match="finite-density check"):
        flow, table = normalizing_flows_fit_auto(
            chain, candidates=[{"knots": 2, "flow_layers": 2}],
            refit_best=True, n_workers=1, return_table=True)
    assert not flow.full_refit
    assert table[0]["refit_used"] is False
    assert table[0]["refit_density_finite"] is False


def test_a_finite_refit_is_kept(monkeypatch):
    """A refit with a finite held-out density is still adopted, and says so."""
    chain = _refit_probe(monkeypatch, -0.5)

    flow, table = normalizing_flows_fit_auto(
        chain, candidates=[{"knots": 2, "flow_layers": 2}],
        refit_best=True, n_workers=1, return_table=True)
    assert flow.full_refit
    assert table[0]["refit_used"] is True
    assert table[0]["refit_density_finite"] is True


def test_selected_row_reports_refit_used_without_a_refit(data):
    """``refit_used`` is on the selected row even when refit_best=False, so the
    caller never has to infer which fit they were handed."""
    _, table = normalizing_flows_fit_auto(
        data, prior_bounds=BOUNDS, max_epochs=5,
        candidates=[{"knots": 2, "flow_layers": 2}],
        refit_best=False, n_workers=1, return_table=True)
    assert table[0]["refit_used"] is False


@pytest.mark.parametrize("refit_best", [True, False])
def test_every_table_row_carries_every_key(monkeypatch, refit_best):
    """The row schema is uniform: same keys on every row.

    Presence used to depend on whether the row was the selected one, so
    ``row["refit_used"]`` was a KeyError on non-selected rows and
    ``pandas.DataFrame(table)`` came out ragged.
    """
    import coppuccino.copula_flows as module

    chain = _values_from_edge_masses([0.10, 0.20, 0.40, 0.20, 0.10])
    candidates = [{"candidate_id": 0}, {"candidate_id": 1}]

    monkeypatch.setattr(
        module, "normalizing_flows_fit",
        lambda values, **kw: SimpleNamespace(
            candidate_id=kw.get("candidate_id", 0)))
    monkeypatch.setattr(module, "log_prob",
                        lambda flow, values: np.full(
                            len(values), -1.0 * flow.candidate_id))
    monkeypatch.setattr(module, "_n_trainable_params",
                        lambda flow: 10 + flow.candidate_id)

    _, table = normalizing_flows_fit_auto(
        chain, candidates=candidates, refit_best=refit_best, n_workers=1,
        return_table=True)

    key_sets = [frozenset(row) for row in table]
    assert len(set(key_sets)) == 1, "rows disagree about their key set"
    for row in table:
        # the keys a caller indexes directly, on every row rather than one
        assert row["refit_used"] in (True, False)
        for key in ("refit_error", "refit_density_finite", "n_params",
                    "fit_status"):
            assert key in row
    assert sum(row["selected"] for row in table) == 1
    # a non-selected row was never refit, and says so rather than omitting it
    unselected = next(row for row in table if not row["selected"])
    assert unselected["refit_used"] is False
    assert unselected["refit_density_finite"] is None


def test_auto_validates_prior_bounds_before_fitting(data):
    """A malformed box used to be swallowed per candidate and resurface as
    'the fits likely diverged. Try a smaller learning_rate'."""
    with pytest.raises(ValueError, match="outside prior_bounds"):
        normalizing_flows_fit_auto(
            data, prior_bounds=np.array([[-0.1, 0.1], [-0.1, 0.1]]),
            candidates=[{"knots": 2, "flow_layers": 2}], max_epochs=2,
            refit_best=False, n_workers=1)
    with pytest.raises(ValueError, match=r"prior_bounds must be \(n_params, 2\)"):
        normalizing_flows_fit_auto(
            data, prior_bounds=np.array([[-1.0, 1.0]]),
            candidates=[{"knots": 2, "flow_layers": 2}], max_epochs=2,
            refit_best=False, n_workers=1)


def test_auto_rejects_a_non_two_dimensional_chain(data):
    with pytest.raises(ValueError, match=r"chain must be \(n_samples, n_params\)"):
        normalizing_flows_fit_auto(data[:, 0], refit_best=False)


def test_parameter_count_is_independent_of_the_marginal_options(data):
    """The selection rule ranks by trainable parameter count, so that count
    must measure the copula architecture only. The marginal transforms are
    frozen, so their options must not move it -- otherwise a candidate list
    varying ``num_points`` would be ranked by marginal resolution rather than
    by capacity."""
    kw = {"max_epochs": 1, "show_progress": False, "knots": 4, "flow_layers": 6}
    marginal_options = {
        "default": {"prior_bounds": BOUNDS},
        "coarse": {"prior_bounds": BOUNDS, "num_points": 40},
        "unbounded heavy": {"tail_alpha": 0.5},
    }
    sizes = {
        name: _n_trainable_params(normalizing_flows_fit(data, **kw, **extra))
        for name, extra in marginal_options.items()
    }
    assert len(set(sizes.values())) == 1, sizes
    assert all(size > 0 for size in sizes.values())
    # and capacity, which is what the rule is meant to measure, does move it
    smaller = _n_trainable_params(normalizing_flows_fit(
        data, prior_bounds=BOUNDS, max_epochs=1, show_progress=False,
        knots=2, flow_layers=2))
    assert smaller < sizes["default"]


def test_a_diverged_candidate_is_reported_as_failed_not_ok(data):
    """A finite-parameter divergence must not occupy a scored table row.

    ``learning_rate=60`` returns finite parameters, so a parameter-only
    convergence check let it through: the candidate arrived here as an
    ordinary flow and the table carried ``fit_status: "ok"`` beside a
    ``val_logp`` hundreds of nats below every healthy rung -- a number a
    caller has no way to read as "this fit blew up". It now raises inside the
    candidate fit, which is the failure path this table already models.
    """
    with pytest.warns(UserWarning, match="excluded failed fits"):
        flow, table = normalizing_flows_fit_auto(
            data, prior_bounds=BOUNDS, max_epochs=20,
            candidates=[{"learning_rate": 60.0}, {"learning_rate": 1e-3}],
            return_table=True, refit_best=False, n_workers=1)
    diverged, healthy = table
    assert diverged["fit_status"] == "failed"
    assert "optimizer diverged" in diverged["fit_error"]
    assert np.isnan(diverged["val_logp"])
    assert diverged["n_params"] is None
    assert not diverged["selected"]
    # ... and the surviving rung is what comes back, still usable
    assert healthy["fit_status"] == "ok" and healthy["selected"]
    assert np.all(np.isfinite(log_prob(flow, data[:50])))


def test_options_are_keyword_only(data):
    """Every option after ``chain`` is passed by name, so a positional call
    fails before any fitting starts instead of binding to whichever option
    occupies that slot."""
    with pytest.raises(TypeError, match="positional"):
        normalizing_flows_fit_auto(data, [{"knots": 2, "flow_layers": 1}],
                                   refit_best=False, n_workers=1,
                                   max_epochs=1, show_progress=False)
