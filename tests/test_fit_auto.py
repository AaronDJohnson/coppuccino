"""Tests for normalizing_flows_fit_auto (held-out hyperparameter selection)."""

import numpy as np
import pytest

from coppuccino import (
    log_prob,
    normalizing_flows_fit_auto,
    sample,
)


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
