# Changelog

All notable changes to **coppuccino** are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] — 2026-07-11

### Changed

- **BREAKING: the Gaussian pipeline's default marginal is the new smooth-tail
  diffeomorphism (`marginal="smooth"`)**, replacing the clipped PCHIP/RQS
  spline pair. Each marginal is a single clip-free monotone map of all of R
  onto the support (spline bulk at data quantiles + analytic tails), so the
  fitted flow produces no exact-bound sample atoms, no wall density cliffs,
  and sample/`log_prob` agreement to float precision — the properties that
  make it safe as an independence-MCMC proposal at hard prior bounds. Bounded
  sides approach walls exponentially (C1-matched to the bulk); unbounded
  sides expose `tail_alpha` (0 = Gaussian tails, larger = heavier) for
  outlier tolerance. Draws that would saturate the float64 wall-resolution
  limit are redrawn by `sample`/`sample_and_log_prob` (a constant that
  cancels in Metropolis-Hastings ratios). The legacy marginals remain
  available as `marginal="rqs"`/`"pchip"` and are auto-selected when the
  legacy tail models (`tail_extension`, `tail_model="gpd"`) are requested.

### Added

- `copula_space="uniform"`: cube-native pipeline — exact unit-range PIT
  marginals plus an RQS coupling flow over a uniform base on the unit box;
  boundary-exact by construction and the natural choice when
  weakly-constrained, flat-to-the-wall marginals dominate.
- `normalizing_flows_fit_auto`: held-out-likelihood model selection with
  parallel candidate fits and smallest-comparable selection.
- `normalizing_flows_fit_fleet` / `FleetFlow`: vmapped simultaneous
  per-source fits for large catalogs.
- `weight_decay` and `show_progress` fit options; save/load support for all
  new model types.

## [1.1.0] — 2026-07-05

### Changed

- **`sample()` and `sample_and_log_prob()` now default to a fresh random seed.**
  `rng_seed` defaults to `None`, drawing fresh system entropy on each call, so
  repeated calls return *independent* draws. Previously both defaulted to a fixed
  `999`, so back-to-back calls silently returned identical samples. Pass an
  explicit integer for reproducible output. This is a behavior change for code
  that relied on the old fixed default.
- **No-tail marginal CDF now clips continuously at the data edges.** In the
  default (no `tail_extension`, no `prior_bounds`) configuration the empirical
  CDF previously jumped to `min_eps` exactly at each dimension's min/max sample,
  mapping those extreme samples to a spurious `z ≈ ±5.2` outlier in Gaussian
  space. It now clips flat to the boundary CDF value, which is continuous with
  the spline. Fitted models in this configuration change slightly (for the
  better). The `prior_bounds` path — recommended for MCMC chains — is unchanged.

### Added

- `compute_injection_hdr()` gained an explicit `rng_seed` argument that seeds
  *both* the flow fit and the Monte Carlo HDR sampling, so the HDR estimate and
  its Monte Carlo error are reproducible and controllable.
- A `warnings.warn` when `normalizing_flows_fit()` drops NaN-containing rows,
  reporting how many of how many rows were removed.
- Tests: end-to-end `log_prob` accuracy vs. an analytic Gaussian density, a
  finite-difference validation of the bijection Jacobian log-det, a GPD-tail
  fit → save → load round-trip, and a real HDR mode-vs-tail ordering check.
- Packaging/CI: `ruff` and `mypy` configuration and a lint/type-check CI job,
  Dependabot for GitHub Actions, and a `CONTRIBUTING.md`.

### Fixed

- The experimental GPD tail density (`_gpd_pdf`) now returns `0` beyond the
  bounded support when `ξ < -1`, instead of blowing up to ~1e10 from a clipped
  denominator raised to a negative power.
- The `sdist` no longer ships local editor config (`.claude/`), the 1.8 MB logo,
  or CI internals; it is scoped to source, tests, example notebooks, and docs.
- `check_in_support()` now returns a Python `bool` rather than a `numpy.bool_`,
  matching its `-> bool` annotation.
- Type hints: `Optional[...]` on `prior_bounds` parameters, full annotations on
  `make_empirical_cdf_spline`, and a reformatted `_fit_chain_entry` signature.
- Actionable error message when a parameter has too few samples for an empirical
  transform (now names the column, its count, and the required minimum).

## [1.0.0] — 2026-06-16

Initial public release.

- Copula normalizing flows in JAX: nonparametric empirical-CDF marginal
  transforms (Gaussianization) composed with a triangular spline copula flow
  ([FlowJAX](https://github.com/danielward27/flowjax)).
- Public API: `normalizing_flows_fit`, `sample`, `log_prob`,
  `sample_and_log_prob`, `save_flow` / `load_flow`, `compute_injection_hdr`,
  `check_in_support`.
- Marginal interpolants: rational-quadratic spline (`"rqs"`, default — exact
  closed-form inverse) and PCHIP (`"pchip"`, retained for backward
  compatibility).
- Optional `prior_bounds` extension, Gaussian `tail_extension`, and an
  experimental peaks-over-threshold Generalized Pareto tail (`tail_model="gpd"`).

[1.1.0]: https://github.com/AaronDJohnson/coppuccino/releases/tag/v1.1.0
[1.0.0]: https://github.com/AaronDJohnson/coppuccino/releases/tag/v1.0.0
