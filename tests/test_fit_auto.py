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
    _edge_fidelity_features,
    _n_trainable_params,
    _score_edge_fidelity,
    _tie_band,
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
        return_table=True, selection="best",
    )
    assert len(table) == 2
    assert sum(r["selected"] for r in table) == 1
    # scores are finite; with selection="best" the argmax is selected
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


def test_literal_edge_metric_detects_cancelling_underflow_error():
    """Equal tail-inclusive CDFs must not hide a wrong visible edge bin."""
    edges = [np.array([-2.0, -1.0, 1.0, 2.0])]
    reference = _values_from_edge_masses([0.10, 0.20, 0.40, 0.20, 0.10])
    misleading = _values_from_edge_masses([0.15, 0.15, 0.40, 0.20, 0.10])
    features, unresolved = _edge_fidelity_features(
        reference, edges, np.full((1, 4), len(reference)), min_count=20)
    assert unresolved == []
    bad_max, _, bad_selection, _ = _score_edge_fidelity(
        misleading, edges, features, confidence=0.95)
    good_max, _, good_selection, _ = _score_edge_fidelity(
        reference, edges, features, confidence=0.95)
    assert bad_max == pytest.approx(0.05)
    assert bad_selection == pytest.approx(0.05)
    assert good_max == good_selection == 0.0


def test_sparse_zero_cell_still_penalises_large_overflow():
    edges = [np.array([-2.0, -1.0, 1.0, 2.0])]
    reference = _values_from_edge_masses([0.0, 0.005, 0.985, 0.01, 0.0])
    bad = _values_from_edge_masses([0.0, 0.005, 0.885, 0.01, 0.10])
    features, unresolved = _edge_fidelity_features(
        reference, edges, np.full((1, 4), len(reference)), min_count=20)
    assert unresolved
    raw, _, selection_error, _ = _score_edge_fidelity(
        bad, edges, features, confidence=0.95)
    assert raw == pytest.approx(0.10)
    assert selection_error > 0.07


def test_edge_endpoint_partition_matches_histogram_convention():
    edges = [np.array([-2.0, -1.0, 1.0, 2.0])]
    reference = np.array([[-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0]])
    features, _ = _edge_fidelity_features(
        reference, edges, np.full((1, 4), 600.0), min_count=1)
    targets = {kind: target for _, kind, _, target, _, _ in features}
    assert targets == {
        "underflow": pytest.approx(1 / 6),
        "lower_bin": pytest.approx(1 / 6),
        "upper_bin": pytest.approx(2 / 6),
        "overflow": pytest.approx(1 / 6),
    }


def test_edge_resolution_is_cell_specific():
    """Dense visible bins must not mark empty adjacent tails as resolved."""
    edges = [np.array([-2.0, -1.0, 1.0, 2.0])]
    reference = _values_from_edge_masses([0.0, 0.20, 0.60, 0.20, 0.0])
    features, unresolved = _edge_fidelity_features(
        reference, edges, np.full((1, 4), 1000.0), min_count=20)
    resolution = {kind: resolved for _, kind, _, _, resolved, _ in features}
    assert resolution == {
        "underflow": False,
        "lower_bin": True,
        "upper_bin": True,
        "overflow": False,
    }
    assert "dimension 0 underflow" in unresolved[0]
    assert "dimension 0 overflow" in unresolved[1]


def test_sparse_score_uses_directional_wilson_interval_gap():
    """Disjoint asymmetric intervals must have a positive selection score."""
    edges = [np.array([-2.0, -1.0, 1.0, 2.0])]
    reference = _values_from_edge_masses([0.005, 0.20, 0.595, 0.20, 0.0])
    candidate = _values_from_edge_masses([0.020, 0.20, 0.580, 0.20, 0.0])
    features, _ = _edge_fidelity_features(
        reference, edges, np.full((1, 4), 1000.0), min_count=20)
    raw, _, selection_error, details = _score_edge_fidelity(
        candidate, edges, features, confidence=0.95)
    underflow = next(item for item in details
                     if item["component"] == "underflow")
    assert raw == pytest.approx(0.015)
    assert selection_error == pytest.approx(0.0013327, rel=1e-4)
    assert underflow["interval_gap"] == pytest.approx(selection_error)


def test_marginal_selection_respects_likelihood_gate(monkeypatch):
    """A perfect marginal cannot beat a materially worse joint fit."""
    import coppuccino.copula_flows as module

    chain = _values_from_edge_masses([0.10, 0.20, 0.40, 0.20, 0.10])
    edges = [np.array([-2.0, -1.0, 1.0, 2.0])]
    candidates = [{"candidate_id": 0}, {"candidate_id": 1}, {"candidate_id": 2}]
    draws = {
        0: _values_from_edge_masses([0.15, 0.15, 0.40, 0.15, 0.15]),
        1: chain.copy(),
        2: chain.copy(),
    }
    logps = {0: 0.0, 1: -0.001, 2: -1.0}
    sizes = {0: 100, 1: 200, 2: 50}

    def fake_fit(_chain, **kwargs):
        return SimpleNamespace(candidate_id=kwargs["candidate_id"])

    def fake_log_prob(flow, values):
        return np.full(len(values), logps[flow.candidate_id])

    def fake_sample(flow, n_samples, rng_seed=None):
        del n_samples, rng_seed
        return draws[flow.candidate_id]

    monkeypatch.setattr(module, "normalizing_flows_fit", fake_fit)
    monkeypatch.setattr(module, "log_prob", fake_log_prob)
    monkeypatch.setattr(module, "sample", fake_sample)
    monkeypatch.setattr(
        module, "_n_trainable_params", lambda flow: sizes[flow.candidate_id])

    flow, table = normalizing_flows_fit_auto(
        chain, candidates=candidates, selection="marginal_fidelity",
        marginal_bin_edges=edges, marginal_samples=1000,
        min_marginal_count=1, marginal_tolerance=0.0,
        max_marginal_error=0.01,
        comparable_nats=0.01, refit_best=False, n_workers=1,
        return_table=True)
    assert flow.candidate_id == 1
    assert table[1]["selected"]
    assert not table[2]["comparable"]
    assert table[2]["marginal_error"] == 0.0
    assert table[0]["marginal_metric_source"] == "literal_edge_bins"
    assert table[0]["marginal_components"] == 4
    assert table[0]["marginal_within_budget"] is False
    assert table[1]["marginal_within_budget"] is True
    assert table[2]["marginal_within_budget"] is True
    assert table[0]["marginal_worst_dimension"] == 0
    assert table[0]["marginal_worst_component"] in {"underflow", "lower_bin"}
    assert len(table[0]["marginal_details"]) == 4


def test_marginal_selection_rejects_bad_full_refit(monkeypatch):
    import coppuccino.copula_flows as module

    chain = _values_from_edge_masses([0.10, 0.20, 0.40, 0.20, 0.10])
    edges = [np.array([-2.0, -1.0, 1.0, 2.0])]
    regressed = _values_from_edge_masses([0.20, 0.10, 0.40, 0.10, 0.20])

    def fake_fit(values, **kwargs):
        del kwargs
        return SimpleNamespace(full_refit=len(values) == len(chain))

    monkeypatch.setattr(module, "normalizing_flows_fit", fake_fit)
    monkeypatch.setattr(module, "log_prob",
                        lambda flow, values: np.zeros(len(values)))
    monkeypatch.setattr(
        module, "sample",
        lambda flow, n_samples, rng_seed=None: regressed if flow.full_refit else chain)
    monkeypatch.setattr(module, "_n_trainable_params", lambda flow: 10)

    with pytest.warns(UserWarning, match="full-data refit worsened"):
        flow, table = normalizing_flows_fit_auto(
            chain, candidates=[{"knots": 2, "flow_layers": 2}],
            selection="marginal_fidelity", marginal_bin_edges=edges,
            marginal_samples=1000, min_marginal_count=1,
            marginal_tolerance=0.0, refit_best=True, n_workers=1,
            return_table=True)
    assert not flow.full_refit
    assert table[0]["refit_used"] is False
    assert table[0]["refit_marginal_error"] == pytest.approx(0.10)


def test_marginal_selection_excludes_nonfinite_candidate(monkeypatch):
    """A NaN-likelihood candidate cannot win through a perfect marginal score."""
    import coppuccino.copula_flows as module

    chain = _values_from_edge_masses([0.10, 0.20, 0.40, 0.20, 0.10])
    edges = [np.array([-2.0, -1.0, 1.0, 2.0])]
    candidates = [{"candidate_id": 0}, {"candidate_id": 1}]
    def module_fit(_chain, **kw):
        return SimpleNamespace(candidate_id=kw["candidate_id"])

    monkeypatch.setattr(module, "normalizing_flows_fit", module_fit)
    monkeypatch.setattr(
        module, "log_prob",
        lambda flow, values: np.full(
            len(values), 0.0 if flow.candidate_id == 0 else np.nan))
    monkeypatch.setattr(module, "sample",
                        lambda flow, n_samples, rng_seed=None: chain)
    monkeypatch.setattr(module, "_n_trainable_params", lambda flow: 10)

    flow, table = normalizing_flows_fit_auto(
        chain, candidates=candidates, selection="marginal_fidelity",
        marginal_bin_edges=edges, marginal_samples=1000,
        min_marginal_count=1, refit_best=False, n_workers=1,
        return_table=True)
    assert flow.candidate_id == 0
    assert table[0]["selected"]
    assert not table[1]["comparable"]


def test_marginal_selection_rejects_nonfinite_joint_refit(monkeypatch):
    """Matching edges do not rescue a full refit with a broken density."""
    import coppuccino.copula_flows as module

    chain = _values_from_edge_masses([0.10, 0.20, 0.40, 0.20, 0.10])
    edges = [np.array([-2.0, -1.0, 1.0, 2.0])]

    def fake_fit(values, **kwargs):
        del kwargs
        return SimpleNamespace(full_refit=len(values) == len(chain))

    monkeypatch.setattr(module, "normalizing_flows_fit", fake_fit)
    monkeypatch.setattr(
        module, "log_prob",
        lambda flow, values: np.full(len(values), np.nan if flow.full_refit else 0.0))
    monkeypatch.setattr(module, "sample",
                        lambda flow, n_samples, rng_seed=None: chain)
    monkeypatch.setattr(module, "_n_trainable_params", lambda flow: 10)

    with pytest.warns(UserWarning, match="finite-density check"):
        flow, table = normalizing_flows_fit_auto(
            chain, candidates=[{"knots": 2, "flow_layers": 2}],
            selection="marginal_fidelity", marginal_bin_edges=edges,
            marginal_samples=1000, min_marginal_count=1,
            refit_best=True, n_workers=1, return_table=True)
    assert not flow.full_refit
    assert table[0]["refit_density_finite"] is False
    assert table[0]["refit_used"] is False


def test_marginal_selection_rejects_finite_joint_refit_regression(monkeypatch):
    """The refit sanity gate catches gross finite joint-density regression."""
    import coppuccino.copula_flows as module

    chain = _values_from_edge_masses([0.10, 0.20, 0.40, 0.20, 0.10])
    edges = [np.array([-2.0, -1.0, 1.0, 2.0])]

    def fake_fit(values, **kwargs):
        del kwargs
        return SimpleNamespace(full_refit=len(values) == len(chain))

    monkeypatch.setattr(module, "normalizing_flows_fit", fake_fit)
    monkeypatch.setattr(
        module, "log_prob",
        lambda flow, values: np.full(len(values), -100.0 if flow.full_refit else 0.0))
    monkeypatch.setattr(module, "sample",
                        lambda flow, n_samples, rng_seed=None: chain)
    monkeypatch.setattr(module, "_n_trainable_params", lambda flow: 10)

    with pytest.warns(UserWarning, match="joint-fit sanity"):
        flow, table = normalizing_flows_fit_auto(
            chain, candidates=[{"knots": 2, "flow_layers": 2}],
            selection="marginal_fidelity", marginal_bin_edges=edges,
            marginal_samples=1000, min_marginal_count=1,
            refit_best=True, n_workers=1, return_table=True)
    assert not flow.full_refit
    assert table[0]["refit_density_finite"] is True
    assert table[0]["refit_joint_sanity_passed"] is False
    assert table[0]["refit_used"] is False


def test_marginal_selection_rejects_refit_error_transfer(monkeypatch):
    """Equal aggregate error cannot hide regression in a different edge cell."""
    import coppuccino.copula_flows as module

    chain = _values_from_edge_masses([0.10, 0.20, 0.40, 0.20, 0.10])
    split_draws = _values_from_edge_masses([0.20, 0.10, 0.40, 0.20, 0.10])
    refit_draws = _values_from_edge_masses([0.10, 0.20, 0.40, 0.10, 0.20])
    edges = [np.array([-2.0, -1.0, 1.0, 2.0])]

    def fake_fit(values, **kwargs):
        del kwargs
        return SimpleNamespace(full_refit=len(values) == len(chain))

    monkeypatch.setattr(module, "normalizing_flows_fit", fake_fit)
    monkeypatch.setattr(module, "log_prob",
                        lambda flow, values: np.zeros(len(values)))
    monkeypatch.setattr(
        module, "sample",
        lambda flow, n_samples, rng_seed=None: (
            refit_draws if flow.full_refit else split_draws))
    monkeypatch.setattr(module, "_n_trainable_params", lambda flow: 10)

    with pytest.warns(UserWarning, match="worsened one or more"):
        flow, table = normalizing_flows_fit_auto(
            chain, candidates=[{"knots": 2, "flow_layers": 2}],
            selection="marginal_fidelity", marginal_bin_edges=edges,
            marginal_samples=1000, min_marginal_count=1,
            marginal_tolerance=0.0, refit_best=True, n_workers=1,
            return_table=True)
    assert not flow.full_refit
    assert table[0]["audit_marginal_error"] == pytest.approx(0.10)
    assert table[0]["refit_marginal_error"] == pytest.approx(0.10)
    assert "dimension 0 upper_bin" in table[0]["refit_regressed_components"]


def test_marginal_selection_isolates_candidate_and_refit_failures(monkeypatch):
    import coppuccino.copula_flows as module

    chain = _values_from_edge_masses([0.10, 0.20, 0.40, 0.20, 0.10])
    edges = [np.array([-2.0, -1.0, 1.0, 2.0])]

    def fake_fit(values, **kwargs):
        candidate_id = kwargs["candidate_id"]
        if candidate_id == 0 or len(values) == len(chain):
            raise RuntimeError("synthetic fit failure")
        return SimpleNamespace(candidate_id=candidate_id)

    monkeypatch.setattr(module, "normalizing_flows_fit", fake_fit)
    monkeypatch.setattr(module, "log_prob",
                        lambda flow, values: np.zeros(len(values)))
    monkeypatch.setattr(module, "sample",
                        lambda flow, n_samples, rng_seed=None: chain)
    monkeypatch.setattr(module, "_n_trainable_params", lambda flow: 10)

    with pytest.warns(UserWarning) as caught:
        flow, table = normalizing_flows_fit_auto(
            chain, candidates=[{"candidate_id": 0}, {"candidate_id": 1}],
            selection="marginal_fidelity", marginal_bin_edges=edges,
            marginal_samples=1000, min_marginal_count=1,
            refit_best=True, n_workers=1, return_table=True)
    assert any("excluded failed fits" in str(item.message) for item in caught)
    assert any("full-data refit failed" in str(item.message) for item in caught)
    assert flow.candidate_id == 1
    assert table[0]["fit_status"] == "failed"
    assert "synthetic fit failure" in table[0]["fit_error"]
    assert table[1]["refit_used"] is False


def test_marginal_fidelity_gaussian_smoke(data):
    flow, table = normalizing_flows_fit_auto(
        data, max_epochs=8,
        candidates=[{"knots": 2, "flow_layers": 2},
                    {"knots": 4, "flow_layers": 3}],
        selection="marginal_fidelity", marginal_samples=2000,
        marginal_tolerance=0.0, refit_best=False, n_workers=1,
        return_table=True)
    assert np.all(np.isfinite([row["marginal_error"] for row in table]))
    assert all(row["marginal_metric_source"] == "latent_quantiles" for row in table)
    assert sum(row["selected"] for row in table) == 1
    assert sample(flow, 50, rng_seed=8).shape == (50, 2)


def test_marginal_fidelity_uniform_smoke(data):
    _, table = normalizing_flows_fit_auto(
        data, prior_bounds=BOUNDS, max_epochs=5,
        candidates=[{"knots": 2, "flow_layers": 2}],
        selection="marginal_fidelity", marginal_samples=1000,
        refit_best=False, n_workers=1, return_table=True)
    assert np.isfinite(table[0]["marginal_error"])
    assert table[0]["marginal_metric_source"] == "latent_quantiles"


def test_marginal_edge_validation_and_all_none(data):
    candidates = [{"knots": 2, "flow_layers": 2}]
    with pytest.raises(ValueError, match="only None"):
        normalizing_flows_fit_auto(
            data, candidates=candidates, selection="marginal_fidelity",
            marginal_bin_edges=[None, None], refit_best=False)
    with pytest.raises(ValueError, match="strictly increasing"):
        normalizing_flows_fit_auto(
            data, candidates=candidates, selection="marginal_fidelity",
            marginal_bin_edges=[np.array([0.0, 1.0, 0.5]), None],
            refit_best=False)
    with pytest.raises(ValueError, match="marginal_samples"):
        normalizing_flows_fit_auto(
            data, candidates=candidates, selection="marginal_fidelity",
            marginal_samples=1000.0, refit_best=False)
    with pytest.raises(ValueError, match="marginal_tolerance"):
        normalizing_flows_fit_auto(
            data, candidates=candidates, selection="marginal_fidelity",
            marginal_tolerance=np.nan, refit_best=False)
    with pytest.raises(ValueError, match="reference size"):
        normalizing_flows_fit_auto(
            data, candidates=candidates, selection="marginal_fidelity",
            marginal_bin_edges=[np.linspace(-1.0, 1.0, 5), None],
            marginal_ess=len(data) + 1, refit_best=False)
    with pytest.raises(ValueError, match="at least one"):
        normalizing_flows_fit_auto(data, candidates=[], refit_best=False)


def test_new_marginal_options_preserve_old_positional_api():
    import inspect

    parameters = list(inspect.signature(normalizing_flows_fit_auto).parameters.values())
    names = [parameter.name for parameter in parameters]
    assert names[names.index("comparable_nats") + 1:names.index("verbose") + 2] == [
        "verbose", "return_table"]
    assert all(
        parameters[names.index(name)].kind == inspect.Parameter.KEYWORD_ONLY
        for name in ("marginal_samples", "marginal_bin_edges",
                     "max_marginal_error")
    )


def test_default_marginal_ladder_keeps_existing_fit_as_baseline(monkeypatch):
    import coppuccino.copula_flows as module

    chain = _values_from_edge_masses([0.10, 0.20, 0.40, 0.20, 0.10])
    edges = [np.array([-2.0, -1.0, 1.0, 2.0])]

    def fake_fit(values, **kwargs):
        del values
        return SimpleNamespace(config={
            key: kwargs[key] for key in ("knots", "flow_layers", "tanh_max_val")
            if key in kwargs
        })

    monkeypatch.setattr(module, "normalizing_flows_fit", fake_fit)
    monkeypatch.setattr(module, "log_prob",
                        lambda flow, values: np.zeros(len(values)))
    monkeypatch.setattr(module, "sample",
                        lambda flow, n_samples, rng_seed=None: chain)
    monkeypatch.setattr(
        module, "_n_trainable_params",
        lambda flow: flow.config["knots"] * flow.config["flow_layers"])

    _, table = normalizing_flows_fit_auto(
        chain, selection="marginal_fidelity", marginal_bin_edges=edges,
        marginal_samples=1000, min_marginal_count=1,
        refit_best=False, n_workers=1, return_table=True)
    assert [row["config"] for row in table] == [
        {"knots": 2, "flow_layers": 2, "tanh_max_val": 2.0},
        {"knots": 4, "flow_layers": 4, "tanh_max_val": 2.0},
        {"knots": 4, "flow_layers": 6},
    ]


def test_marginal_budget_uses_fresh_refit_audit(monkeypatch):
    import coppuccino.copula_flows as module

    chain = _values_from_edge_masses([0.10, 0.20, 0.40, 0.20, 0.10])
    split_audit = _values_from_edge_masses([0.12, 0.18, 0.40, 0.20, 0.10])
    refit_audit = _values_from_edge_masses([0.115, 0.185, 0.40, 0.20, 0.10])
    edges = [np.array([-2.0, -1.0, 1.0, 2.0])]
    selection_seed = (999 + 104_729) % (2**31 - 1)

    def fake_fit(values, **kwargs):
        del kwargs
        return SimpleNamespace(full_refit=len(values) == len(chain))

    def fake_sample(flow, n_samples, rng_seed=None):
        del n_samples
        if rng_seed == selection_seed and not flow.full_refit:
            return chain
        return refit_audit if flow.full_refit else split_audit

    monkeypatch.setattr(module, "normalizing_flows_fit", fake_fit)
    monkeypatch.setattr(module, "log_prob",
                        lambda flow, values: np.zeros(len(values)))
    monkeypatch.setattr(module, "sample", fake_sample)
    monkeypatch.setattr(module, "_n_trainable_params", lambda flow: 10)

    with pytest.warns(UserWarning, match="error budget"):
        flow, table = normalizing_flows_fit_auto(
            chain, candidates=[{"knots": 2, "flow_layers": 2}],
            selection="marginal_fidelity", marginal_bin_edges=edges,
            marginal_samples=1000, min_marginal_count=1,
            marginal_tolerance=0.0, max_marginal_error=0.01,
            refit_best=True, n_workers=1, return_table=True)
    assert not flow.full_refit
    assert table[0]["marginal_within_budget"] is True
    assert table[0]["audit_marginal_within_budget"] is False
    assert table[0]["refit_marginal_within_budget"] is False
    assert table[0]["refit_used"] is False


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


def test_marginal_options_require_marginal_selection(data):
    with pytest.raises(ValueError, match="require selection='marginal_fidelity'"):
        normalizing_flows_fit_auto(
            data, marginal_bin_edges=[None, None], refit_best=False)
    with pytest.raises(ValueError, match="require selection='marginal_fidelity'"):
        normalizing_flows_fit_auto(
            data, marginal_samples=1000, refit_best=False)


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
    monkeypatch.setattr(module, "sample",
                        lambda flow, n_samples, rng_seed=None: chain)
    monkeypatch.setattr(module, "_n_trainable_params", lambda flow: 10)
    return chain


@pytest.mark.parametrize("selection", ["smallest_comparable", "best"])
def test_default_selections_reject_a_nonfinite_refit(monkeypatch, selection):
    """The finite-density check now guards every selection mode, not just
    marginal_fidelity: the default paths used to swap in the refit unchecked."""
    chain = _refit_probe(monkeypatch, np.nan)

    with pytest.warns(UserWarning, match="finite-density check"):
        flow, table = normalizing_flows_fit_auto(
            chain, candidates=[{"knots": 2, "flow_layers": 2}],
            selection=selection, refit_best=True, n_workers=1,
            return_table=True)
    assert not flow.full_refit
    assert table[0]["refit_used"] is False
    assert table[0]["refit_density_finite"] is False


@pytest.mark.parametrize("selection", ["smallest_comparable", "best"])
def test_default_selections_keep_a_finite_refit(monkeypatch, selection):
    """A refit with a finite held-out density is still adopted, and says so."""
    chain = _refit_probe(monkeypatch, -0.5)

    flow, table = normalizing_flows_fit_auto(
        chain, candidates=[{"knots": 2, "flow_layers": 2}],
        selection=selection, refit_best=True, n_workers=1, return_table=True)
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


@pytest.mark.parametrize(
    "selection, refit_best",
    [("best", True), ("smallest_comparable", False),
     ("marginal_fidelity", True)],
)
def test_every_table_row_carries_every_key(monkeypatch, selection, refit_best):
    """The row schema is uniform: same keys on every row, in every mode.

    Presence used to depend on the selection rule *and* on whether the row was
    the selected one, so ``row["refit_used"]`` was a KeyError on non-selected
    rows and ``pandas.DataFrame(table)`` came out ragged.
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
    monkeypatch.setattr(module, "sample",
                        lambda flow, n_samples, rng_seed=None: chain)
    monkeypatch.setattr(module, "_n_trainable_params",
                        lambda flow: 10 + flow.candidate_id)

    _, table = normalizing_flows_fit_auto(
        chain, candidates=candidates, selection=selection,
        refit_best=refit_best, n_workers=1, return_table=True,
        **({"marginal_samples": 1000, "min_marginal_count": 1,
            "marginal_bin_edges": [np.array([-2.0, -1.0, 1.0, 2.0])]}
           if selection == "marginal_fidelity" else {}))

    key_sets = [frozenset(row) for row in table]
    assert len(set(key_sets)) == 1, "rows disagree about their key set"
    for row in table:
        # the keys a caller indexes directly, on every row rather than one
        assert row["refit_used"] in (True, False)
        for key in ("refit_error", "refit_density_finite", "marginal_error",
                    "marginal_details", "refit_regressed_components",
                    "audit_marginal_error", "n_params", "fit_status"):
            assert key in row
    assert sum(row["selected"] for row in table) == 1
    # a non-selected row was never refit, and says so rather than omitting it
    unselected = next(row for row in table if not row["selected"])
    assert unselected["refit_used"] is False
    assert unselected["refit_density_finite"] is None


def test_tie_band_default_and_override():
    """Both the selection ranking and the refit audit read the same band."""
    assert _tie_band(10_000, None) == pytest.approx(0.005)
    assert _tie_band(40_000, None) == pytest.approx(0.0025)
    assert _tie_band(10_000, 0.0) == 0.0
    assert _tie_band(10_000, 0.02) == pytest.approx(0.02)


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


def test_parameter_count_is_independent_of_the_marginal_family(data):
    """``smallest_comparable`` ranks by trainable parameter count, so that count
    must measure the copula architecture only. The marginal transforms are
    frozen, so the family must not move it -- otherwise a candidate list mixing
    families would be ranked by family rather than by capacity."""
    kw = {"prior_bounds": BOUNDS, "max_epochs": 1, "show_progress": False,
          "knots": 4, "flow_layers": 6}
    sizes = {
        marginal: _n_trainable_params(
            normalizing_flows_fit(data, marginal=marginal, **kw))
        for marginal in ("smooth", "rqs", "pchip")
    }
    assert len(set(sizes.values())) == 1, sizes
    assert all(size > 0 for size in sizes.values())
    # and capacity, which is what the rule is meant to measure, does move it
    smaller = _n_trainable_params(normalizing_flows_fit(
        data, prior_bounds=BOUNDS, max_epochs=1, show_progress=False,
        knots=2, flow_layers=2))
    assert smaller < sizes["smooth"]


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
            data, prior_bounds=BOUNDS, max_epochs=20, selection="best",
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
