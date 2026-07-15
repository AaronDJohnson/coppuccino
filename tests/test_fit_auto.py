"""Tests for normalizing_flows_fit_auto (held-out hyperparameter selection)."""

from types import SimpleNamespace

import numpy as np
import pytest

from coppuccino import (
    log_prob,
    normalizing_flows_fit_auto,
    sample,
)
from coppuccino.copula_flows import _edge_fidelity_features, _score_edge_fidelity


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    n = 1500
    a = rng.uniform(-1.0, 1.0, n)
    b = np.clip(0.6 * a + 0.3 * rng.standard_normal(n), -1, 1)
    return np.stack([a, b], 1)


BOUNDS = np.array([[-1.0, 1.0], [-1.0, 1.0]])


def test_selects_and_returns_working_flow(data):
    flow, table = normalizing_flows_fit_auto(
        data, prior_bounds=BOUNDS, copula_space="uniform", max_epochs=40,
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
        data, prior_bounds=BOUNDS, copula_space="uniform", max_epochs=40,
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
        data, prior_bounds=BOUNDS, copula_space="uniform", max_epochs=40,
        candidates=[{"knots": 2, "flow_layers": 3}, {"knots": 4, "flow_layers": 3}],
        return_table=True, refit_best=False,
        comparable_sems=0.0, comparable_nats=0.0,
    )
    scores = [r["val_logp"] for r in table]
    assert table[int(np.argmax(scores))]["selected"]


def test_cross_mode_candidates(data):
    flow, table = normalizing_flows_fit_auto(
        data, prior_bounds=BOUNDS, max_epochs=40,
        candidates=[{"copula_space": "uniform", "knots": 4, "flow_layers": 3},
                    {"copula_space": "gaussian", "knots": 4, "flow_layers": 3}],
        return_table=True,
    )
    # scores comparable in theta space across modes; one candidate selected
    assert sum(r["selected"] for r in table) == 1
    assert np.all(np.isfinite([r["val_logp"] for r in table]))


def test_reproducible_selection(data):
    _, t1 = normalizing_flows_fit_auto(
        data, prior_bounds=BOUNDS, copula_space="uniform", max_epochs=30,
        candidates=[{"knots": 4, "flow_layers": 3}, {"knots": 6, "flow_layers": 3}],
        return_table=True, refit_best=False, rng_seed=5,
    )
    _, t2 = normalizing_flows_fit_auto(
        data, prior_bounds=BOUNDS, copula_space="uniform", max_epochs=30,
        candidates=[{"knots": 4, "flow_layers": 3}, {"knots": 6, "flow_layers": 3}],
        return_table=True, refit_best=False, rng_seed=5,
    )
    assert [r["selected"] for r in t1] == [r["selected"] for r in t2]
    assert np.allclose([r["val_logp"] for r in t1], [r["val_logp"] for r in t2])


def test_too_few_samples_raises():
    with pytest.raises(ValueError, match="at least"):
        normalizing_flows_fit_auto(np.random.randn(20, 2))


def test_parallel_matches_sequential(data):
    """Concurrent candidate fits must reproduce sequential results exactly
    (fixed per-candidate seeds and split => execution order cannot matter)."""
    kw = dict(prior_bounds=BOUNDS, copula_space="uniform", max_epochs=30,
              candidates=[{"knots": 2, "flow_layers": 3},
                          {"knots": 4, "flow_layers": 3}],
              return_table=True, refit_best=False, rng_seed=11)
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
        data, prior_bounds=BOUNDS, copula_space="uniform", max_epochs=5,
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
        inspect.Parameter.KEYWORD_ONLY == parameters[names.index(name)].kind
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
    with pytest.warns(UserWarning, match="excluded failed fits"):
        with pytest.raises(RuntimeError, match="Fit errors"):
            normalizing_flows_fit_auto(
                data, candidates=[{"knots": 2}, {"knots": 4}],
                refit_best=False, n_workers=1)


def test_marginal_options_require_marginal_selection(data):
    with pytest.raises(ValueError, match="require selection='marginal_fidelity'"):
        normalizing_flows_fit_auto(
            data, marginal_bin_edges=[None, None], refit_best=False)
    with pytest.raises(ValueError, match="require selection='marginal_fidelity'"):
        normalizing_flows_fit_auto(
            data, marginal_samples=1000, refit_best=False)
