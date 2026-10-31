"""Tests for normalizing_flows_fit_auto (held-out hyperparameter selection)."""

from types import SimpleNamespace

import numpy as np
import pytest

from coppuccino import (
    log_prob,
    normalizing_flows_fit_auto,
    sample,
)


@pytest.fixture(scope="module")
def data():
    """Correlated samples inside BOUNDS, with a wall atom on each bound.

    ``np.clip`` puts a mass of samples *exactly* on the bound, which is both
    the documented use of ``prior_bounds`` and the case that used to poison
    every held-out mean here: ``z_of_theta`` did not apply the clamp that
    ``log_dtheta_dz`` did, so an at-bound row scored around -5e5 nats and
    swamped the comparison the selection rules are built on. Keeping the
    clipped data means ``test_selects_and_returns_working_flow``'s floor on the
    selected held-out score notices that regression.
    """
    rng = np.random.default_rng(0)
    n = 1500
    a = rng.uniform(-1.0, 1.0, n)
    b = np.clip(0.6 * a + 0.3 * rng.standard_normal(n), -1.0, 1.0)
    return np.stack([a, b], 1)


BOUNDS = np.array([[-1.0, 1.0], [-1.0, 1.0]])


def _mock_chain(n=1000):
    """A chain for tests whose fits are mocked out: only its length is read."""
    return np.random.default_rng(0).standard_normal((n, 1))


def test_selects_and_returns_working_flow(data):
    """The one end-to-end run on real fits, including the full-chain refit.

    The selection rule itself is pinned on scripted scores below; this checks
    that real candidates come back with a coherent table and that the flow
    handed back works. The floor on the selected held-out score is what the
    ``data`` fixture's wall atoms are for: the split fit scores -1.05 nats
    here (both rungs, on the current and the oldest supported dependencies,
    and -1.01 to -1.06 at other seeds; about -0.91 for the continuous part of
    the true law), removing the wall clamp from ``z_of_theta`` in a scratch copy drops
    it to -3.14, and the original defect -- about -5e5 nats per at-bound row --
    puts it thousands of nats down.
    """
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
    assert table[int(np.argmax(scores))]["val_logp"] > -2.0
    # the parameter count measures copula capacity: more knots, more parameters
    assert 0 < table[0]["n_params"] < table[1]["n_params"]
    # returned flow works end to end
    d = sample(flow, 500, rng_seed=1)
    assert d.shape == (500, 2)
    assert np.all((d >= -1) & (d <= 1))
    assert np.all(np.isfinite(log_prob(flow, d)))


# ---------------------------------------------------------------------------
# the selection rule, on scripted held-out scores (no fits)
# ---------------------------------------------------------------------------

def _scripted_selection(monkeypatch, per_point, sizes, refit_error=None):
    """Replace every candidate fit with scripted held-out scores.

    Candidate ``i`` scores ``per_point[i](n)`` on the ``n`` held-out rows and
    has ``sizes[i]`` trainable parameters, so the selection rule runs on
    numbers the test chose rather than on whatever a real fit happens to
    score. If ``refit_error`` is given, the full-chain refit raises it.

    Returns the chain and the candidate list to pass to
    ``normalizing_flows_fit_auto``; the produced flows expose ``cid`` (the
    candidate index) and ``full_refit``.
    """
    import coppuccino.copula_flows as module

    chain = _mock_chain()

    def fake_fit(values, **kwargs):
        full_refit = len(values) == len(chain)
        if full_refit and refit_error is not None:
            raise refit_error
        return SimpleNamespace(cid=kwargs["candidate_id"], full_refit=full_refit)

    monkeypatch.setattr(module, "normalizing_flows_fit", fake_fit)
    monkeypatch.setattr(module, "log_prob", lambda flow, values: np.asarray(
        per_point[flow.cid](len(values)), dtype=float))
    monkeypatch.setattr(module, "_n_trainable_params",
                        lambda flow: sizes[flow.cid])
    return chain, [{"candidate_id": i} for i in range(len(sizes))]


def _shared(n):
    """Per-row variation every candidate shares, so it cancels in a paired test.

    Its spread is large on purpose: an unpaired comparison against the
    candidates' individual standard errors would call almost anything a tie.
    """
    return np.linspace(-30.0, 30.0, n)


def _offset(nats):
    """Held-out scores ``nats`` below the shared baseline on every row."""
    return lambda n: _shared(n) - nats


def _select(monkeypatch, per_point, sizes, **options):
    chain, candidates = _scripted_selection(monkeypatch, per_point, sizes)
    _, table = normalizing_flows_fit_auto(
        chain, candidates=candidates, refit_best=False, n_workers=1,
        return_table=True, **options)
    return table


def test_smallest_comparable_prefers_parsimony(monkeypatch):
    """A generous tolerance picks the smaller model over a better bigger one,
    and zero tolerance picks the bigger one back.

    The bigger candidate is listed first and scores 0.4 nats higher on every
    held-out row, so it is the argmax by construction: a rule that ignored
    parsimony, one that took the first comparable row, and one that always
    took the smallest model each fail one of the two halves.
    """
    per_point = [_offset(0.0), _offset(0.4)]
    sizes = [50, 10]

    table = _select(monkeypatch, per_point, sizes,
                    comparable_sems=0.0, comparable_nats=1.0)
    assert table[0]["val_logp"] > table[1]["val_logp"]
    assert all(r["comparable"] for r in table)
    assert [r["selected"] for r in table] == [False, True]

    table = _select(monkeypatch, per_point, sizes,
                    comparable_sems=0.0, comparable_nats=0.0)
    assert [r["comparable"] for r in table] == [True, False]
    assert [r["selected"] for r in table] == [True, False]


def _trailing_by_paired_sems(k):
    """Scores that trail the shared baseline by ``k`` paired standard errors.

    The row-by-row difference from the baseline alternates around its mean,
    so the paired standard error is known exactly; the shared component cancels
    in the difference, so the candidates' individual standard errors are far
    larger than it.
    """
    def per_point(n):
        alt = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
        alt -= alt.mean()
        paired_sem = np.std(alt) / np.sqrt(n)
        return _shared(n) - (k * paired_sem + alt)
    return per_point


@pytest.mark.parametrize("k, small_is_comparable", [(0.5, True), (2.0, False)])
def test_one_paired_standard_error_decides_comparability(monkeypatch, k,
                                                         small_is_comparable):
    """The default rule, on its own: comparable within one *paired* SEM.

    With ``comparable_nats=0`` only ``comparable_sems`` (default 1.0) is in
    play. The smaller candidate trails by half a paired standard error in one
    case and by two in the other. In the second case it is still far inside
    either candidate's own error bar (the shared spread puts that near 1.4
    nats, against a 0.16-nat gap), so a rule built on the individual standard
    errors instead of the paired one would call it comparable and fail here.
    """
    table = _select(monkeypatch, [_offset(0.0), _trailing_by_paired_sems(k)],
                    [50, 10], comparable_nats=0.0)
    gap = table[0]["val_logp"] - table[1]["val_logp"]
    assert 0.0 < gap < table[1]["sem"]
    assert table[1]["comparable"] is small_is_comparable
    assert table[1]["selected"] is small_is_comparable
    assert table[0]["selected"] is not small_is_comparable


def test_equal_sizes_break_the_tie_toward_the_higher_score(monkeypatch):
    """Among comparable candidates of the same size, the better one wins.

    The two small candidates share a parameter count, and the better of them
    is listed second, so a tie broken by list order (or toward the lower
    score) picks the wrong one.
    """
    table = _select(monkeypatch, [_offset(0.3), _offset(0.1), _offset(0.0)],
                    [10, 10, 50], comparable_sems=0.0, comparable_nats=1.0)
    assert all(r["comparable"] for r in table)
    assert [r["selected"] for r in table] == [False, True, False]


def test_a_nonfinite_score_is_excluded_beside_a_finite_one(monkeypatch):
    """A fit that succeeded but scored a non-finite density can never win.

    The candidate is smaller and scores higher on every other row, so with the
    generous tolerance here it would be chosen if its NaN row were ignored.
    It has to come back as its own status, out of the comparison, with the
    finite candidate selected -- and without the failed-fit warning, since
    nothing raised.
    """
    def nan_row(n):
        lp = _shared(n) + 1.0
        lp[0] = np.nan
        return lp

    table = _select(monkeypatch, [nan_row, _offset(0.0)], [5, 50],
                    comparable_sems=0.0, comparable_nats=1.0)
    bad, good = table
    assert bad["fit_status"] == "nonfinite_score"
    assert bad["fit_error"] is None
    assert np.isnan(bad["val_logp"])
    assert bad["n_params"] == 5
    assert not bad["comparable"] and not bad["selected"]
    assert good["fit_status"] == "ok" and good["selected"]


def test_a_refit_that_raises_falls_back_to_the_split_fit(monkeypatch):
    """The refit's own exception is recorded and the validated fit returned."""
    chain, candidates = _scripted_selection(
        monkeypatch, [_offset(0.0)], [10],
        refit_error=RuntimeError("synthetic refit failure"))
    with pytest.warns(UserWarning, match="full-data refit failed"):
        flow, table = normalizing_flows_fit_auto(
            chain, candidates=candidates, refit_best=True, n_workers=1,
            return_table=True)
    assert flow.cid == 0 and not flow.full_refit
    (row,) = table
    assert row["selected"]
    assert row["refit_used"] is False
    assert row["refit_error"] == "RuntimeError: synthetic refit failure"
    assert row["refit_density_finite"] is None


# ---------------------------------------------------------------------------
# argument checks
# ---------------------------------------------------------------------------

def test_too_few_samples_raises():
    with pytest.raises(ValueError, match="at least"):
        normalizing_flows_fit_auto(np.random.randn(20, 2))


@pytest.mark.parametrize("val_frac", [0.0, 1.0, np.nan])
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


@pytest.mark.parametrize("val_frac, counts", [
    (0.999, "1498 held-out and 2 fitting rows"),     # too little to fit on
    (0.0005, "1 held-out and 1499 fitting rows"),    # one row has no sem
])
def test_val_frac_that_starves_either_side_names_both_counts(data, val_frac,
                                                             counts):
    """A fraction inside (0, 1) can still leave a part too small to use.

    Both arms of the check are exercised: a split that leaves 2 rows to fit
    on, and one that holds out a single row, whose ``sem`` would be 0. Either
    raises before any candidate is fitted.
    """
    with pytest.raises(ValueError, match="held-out rows and fitting needs") as exc:
        normalizing_flows_fit_auto(
            data, val_frac=val_frac, candidates=[{"knots": 2, "flow_layers": 2}],
            max_epochs=2, refit_best=False, n_workers=1)
    assert counts in str(exc.value)


def test_auto_reports_dropped_nonfinite_rows_like_the_single_fitter(
        data, monkeypatch):
    """The same dirty chain used to be reported through one entry point and
    silently cleaned by the other. The fits are mocked: the warning is issued
    before any of them, and what they receive is what matters."""
    import coppuccino.copula_flows as module

    seen = []

    def fake_fit(values, **kwargs):
        seen.append(values)
        return SimpleNamespace()

    monkeypatch.setattr(module, "normalizing_flows_fit", fake_fit)
    monkeypatch.setattr(module, "log_prob",
                        lambda flow, values: np.zeros(len(values)))
    monkeypatch.setattr(module, "_n_trainable_params", lambda flow: 10)

    dirty = data.copy()
    dirty[3, 1] = np.nan
    dirty[7, 0] = np.inf
    with pytest.warns(UserWarning, match=r"dropped 2 of 1500 rows"):
        normalizing_flows_fit_auto(
            dirty, candidates=[{"knots": 2, "flow_layers": 2}], max_epochs=2,
            refit_best=False, n_workers=1)
    (fit_part,) = seen
    assert np.all(np.isfinite(fit_part))


def test_parallel_matches_sequential(data):
    """Concurrent candidate fits must reproduce sequential results exactly
    (fixed per-candidate seeds and split => execution order cannot matter).
    Two independent runs with the same seed, so this is also the
    reproducibility check."""
    kw = {"prior_bounds": BOUNDS, "max_epochs": 30,
          "candidates": [{"knots": 2, "flow_layers": 3},
                         {"knots": 4, "flow_layers": 3}],
          "return_table": True, "refit_best": False, "rng_seed": 11}
    _, t_seq = normalizing_flows_fit_auto(data, n_workers=1, **kw)
    _, t_par = normalizing_flows_fit_auto(data, n_workers=2, **kw)
    assert [r["selected"] for r in t_seq] == [r["selected"] for r in t_par]
    assert np.allclose([r["val_logp"] for r in t_seq],
                       [r["val_logp"] for r in t_par])


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

    calls = {"n": 0}

    def flaky_fit(values, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("synthetic candidate failure")
        return SimpleNamespace()

    monkeypatch.setattr(module, "normalizing_flows_fit", flaky_fit)
    monkeypatch.setattr(module, "log_prob",
                        lambda flow, x: np.full(len(x), np.nan))
    monkeypatch.setattr(module, "_n_trainable_params", lambda flow: 10)
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

    chain = _mock_chain()

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


@pytest.mark.parametrize("refit_best", [True, False])
def test_every_table_row_carries_every_key(monkeypatch, refit_best):
    """The row schema is uniform: same keys on every row.

    Presence used to depend on whether the row was the selected one, so
    ``row["refit_used"]`` was a KeyError on non-selected rows and
    ``pandas.DataFrame(table)`` came out ragged. ``refit_used`` is on the
    selected row even when ``refit_best=False``, so the caller never has to
    infer which fit they were handed.
    """
    import coppuccino.copula_flows as module

    chain = _mock_chain()
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
    # and without a refit, neither was the selected one
    selected = next(row for row in table if row["selected"])
    assert selected["refit_used"] is refit_best


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


def test_a_diverged_candidate_is_reported_as_failed_not_ok(data):
    """A finite-parameter divergence must not occupy a scored table row.

    ``learning_rate=60`` returns finite parameters, so a parameter-only
    convergence check let it through: the candidate arrived here as an
    ordinary flow and the table carried ``fit_status: "ok"`` beside a
    ``val_logp`` hundreds of nats below every healthy rung -- a number a
    caller has no way to read as "this fit blew up". It now raises inside the
    candidate fit, which is the failure path this table already models.

    Eight epochs are enough: flowjax keeps the first epoch's parameters, so the
    diverged rung's training log-density falls 2330 nats below its own
    initialization (2335 on the oldest supported dependencies, and at least
    1767 at other seeds), at 8 epochs exactly as at 20, against a 6-nat
    budget.
    """
    with pytest.warns(UserWarning, match="excluded failed fits"):
        flow, table = normalizing_flows_fit_auto(
            data, prior_bounds=BOUNDS, max_epochs=8,
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
