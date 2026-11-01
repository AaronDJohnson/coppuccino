# Changelog

All notable changes to **coppuccino** are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.0.0] — 2026-11-01

This is the first release since 1.0.0, and every change below is measured
against 1.0.0. Earlier drafts of this file had entries for 1.1.0, 2.0.0 and
3.0.0, but none of those versions was published. Their surviving changes are
folded in here. Features they added and later dropped are not listed: the
unit-cube copula pipeline, `whiten`, `flow_layers=0`, `calibrate_edge_bins`,
`selection="marginal_fidelity"` and `reference="flow"`.

### Upgrading from 1.0.0

Most 1.0.0 code needs at least one change, and every refit gives a different
density.

- **There is one marginal family.** Each marginal is now the smooth-tail map
  described under Changed. The `marginal`, `tail_extension`, `tail_model` and
  `tail_quantile` options are gone, and passing any of them raises `TypeError`
  naming it. For tails heavier than Gaussian on an unbounded side, use
  `tail_alpha`. No setting reproduces a 1.0.0 fit.
- **Options are keyword-only.** Every `normalizing_flows_fit` argument after
  `chain` must be passed by name, as must `compute_injection_hdr`'s `rng_seed`
  and `return_flow`. The 1.0.0 positional order no longer exists, so a
  positional call raises `TypeError` instead of quietly assigning values to the
  wrong options.
- **`load_flow` refuses 1.0.0 model files.** A saved flow stores its training
  columns, and `load_flow` rebuilds the marginals from them. The 1.0.0 builder
  is gone, so a reload would give a different density without saying so.
  Instead it raises `ValueError` naming the file. Refit the chain, or read the
  file under `coppuccino<2`.
- **HDR credibility is ranked against a held-out half.**
  `compute_injection_hdr` fits on a random half of `samples` and ranks each
  injection against the other half. It no longer ranks against fresh draws
  from the fitted flow, so `num_samples` is gone (see Changed for why). The
  same inputs give different values, and `samples` needs at least 42 rows.
- **`sample` and `sample_and_log_prob` draw fresh randomness by default.**
  `rng_seed` now defaults to `None`, meaning new system entropy on each call,
  instead of a fixed `999`. Back-to-back calls used to return identical draws
  and are now independent. Pass an integer for reproducible output.
- **Input that 1.0.0 accepted can now raise.** See Fixed.

### Added

- `normalizing_flows_fit_auto` fits a list of candidate configurations in
  parallel and scores each one by its mean log-density on a held-out split.
  It returns the smallest candidate, by trainable parameter count, whose
  score is statistically comparable to the best. The test is paired on the
  shared validation rows and is set by `comparable_sems` and
  `comparable_nats`. By default the winner is refit on the full chain, falling
  back to the validated split fit with a warning if the refit raises or its
  held-out density is not finite. The default ladder is (`knots`, `flow_layers`) =
  (2, 4), (4, 6), (8, 8). `return_table=True` adds a per-candidate table
  whose rows all carry the same keys. `val_frac` must lie in (0, 1) and leave
  enough rows on both sides of the split. Every option after `chain` is
  keyword-only.
- `normalizing_flows_fit_fleet` and `FleetFlow` fit one flow per source for a
  catalog of chains that share a prior box, training all of them together
  under one `vmap`. A `FleetFlow` samples and scores every source in one call,
  or a single `source_index`, and `FleetFlow.save` / `FleetFlow.load` persist
  it. It is a different model from `normalizing_flows_fit`: it has
  clipped-RQS marginals onto the unit cube and learns the copula over a
  uniform base. Its input contract is stricter: `prior_bounds` is required,
  finite and ordered, chains must be finite, and a column that is constant
  within a source is an error. `FleetFlow.sample(rng_seed=s, source_index=i)`
  reproduces `FleetFlow.sample(rng_seed=s)[i]`. The options after
  `prior_bounds` are keyword-only and range-checked by name. A source whose
  validation loss never improves on its untrained initialization, which is
  what a diverged fit looks like, is returned untrained and named in a
  warning; if no source improves, the fit raises `RuntimeError`.
- New `normalizing_flows_fit` options: `tail_alpha` (tail heaviness on
  unbounded sides, 0 to 10, where 0 gives Gaussian tails), `weight_decay`
  (AdamW decay on the trainable copula parameters), `batch_size` (rows per
  optimizer step, default 100 as before) and `show_progress`.
- `normalizing_flows_fit` and `normalizing_flows_fit_auto` warn, with a count,
  when they drop rows that contain NaN or ±inf. 1.0.0 dropped NaN rows without
  saying so and let infinities through.
- Saved files carry `format_version` (now 3), `kind`, and, for `save_flow`,
  `writer_version`. A load error can then name the file, what kind of model
  it holds, and the right loader.

### Changed

- **A repeat fit compiles nothing.** A fit with the same chain shape,
  `prior_bounds` and options as an earlier fit in the same process reuses its
  compiled training step, density and marginal programs. 1.0.0 compiled four
  programs again on every fit, the training step twice. One exception: tied
  values in a column, such as rows sitting on a bound, merge spline knots.
  The size of the marginal map then depends on the chain, and the map from
  data to copula space compiles again when that size changes. On a 10,000×8
  chain at default settings without `prior_bounds`, on CPU, the first fit in
  a process takes about 13 s in both versions. Refitting the same chain, which
  trains for 60 epochs in both, takes 7.2 s against 9.6 s.
- **The marginal map.** Each marginal is a single clip-free monotone map of all
  of R onto the support: a rational-quadratic-spline bulk at the data
  quantiles, joined to analytic tails. Its forward map, inverse and derivative
  share one parameterization, so the inverse is exact. The 1.0.0 spline pair
  inverted only up to interpolation error. Unbounded sides get Gaussian tails
  by default, or heavier ones through `tail_alpha`. Bounded sides get a wall
  tail, and no draw lands exactly on a bound.
- **At a bound that the training chain touches**, the density is finite but
  much higher than in the interior. The marginal spreads the mass of rows
  sitting on a wall over a short interval next to it. On 1500×2 chains of
  `clip(N(0.77, 0.25), -1, 1)` with `prior_bounds=[[-1, 1]] * 2`,
  `max_epochs=80` and `rng_seed=s` for s = 0–3, the median `log_prob` of the
  rows on a bound (460–500 of them) is +8.46 to +13.35, against +0.29 to
  +0.49 for the other rows. This
  reflects real mass at the wall, not a defect, but a pointwise density read
  there will be large.
- **`sample` truncates at such a wall.** Draws whose bounded tail saturates at
  the wall are redrawn, so `sample` and `sample_and_log_prob` emit the density
  truncated to the region away from the wall and renormalized. `log_prob` is
  not renormalized. Between two points the sampler can reach, the ratio of the
  two densities is the same, so it cancels in Metropolis–Hastings ratios and
  in self-normalized importance weights. But the truncated sliver is never
  proposed. It is empty for a chain strictly inside its box and grows with the
  share of rows on a wall. Chains with 52–54% of their rows clipped onto a wall
  redrew 18–21% of raw draws; the protocol is in `sample`'s docstring. The
  sliver can hold much of the wall's mass, so draws come back with too little
  of it. On 1500×2 chains of `clip(N(0.77, 0.25), -1, 1)` in a ±1 box, at
  default settings and seeds 0–2, 16–18% of the first parameter's rows sit
  exactly on +1, and 8.1–14.6% of draws land within 0.01 of it, against
  17.3–18.7% of the chain.
- **At a bound the training chain does not reach**, the density is too high
  between the outermost data point and the wall, most of all on the wall. On
  3000×2 `U(0, 1)` data in its own box, where the true log-density is 0
  everywhere, `log_prob` is +2.0 to +4.3 at the edge midpoints and +3.5 to
  +12.1 at the corners, against −0.08 in the interior, and back to −0.3 to
  −0.1 by 0.001 inside the wall (default settings, seeds 0–2). That gap is
  under 0.001 wide there and held 0.24–0.46% of draws, where the truth puts
  0.09–0.14%.
- **Why `compute_injection_hdr` changed its reference.** The reference points
  are now posterior rows the fit never saw. An injection drawn from the
  posterior is exchangeable with them, so its rank is uniform for any fitted
  density, and a bad fit costs resolution without biasing a P-P plot. 1.0.0
  ranked against draws from the fitted flow, and any error in the fit biased
  credibility upward. On a 1-D standard normal with 1500 samples, that method
  gave a mean HDR of 0.569–0.576 over seeds 0–3 where 0.5 is unbiased. The
  held-out method gives 0.499–0.526. The cost is a noisier single call,
  because both the fit and the reference use half the rows. `rng_seed` seeds
  both the split and the fit. Injections outside the samples' coordinate-wise
  hull still return the sentinel 1.0. A diverged fit now raises
  `RuntimeError`, where it used to return a plausible-looking 1.0 or 0.0.
  The hull is computed once, and `log_prob` runs in one batched call instead
  of once per injection.
- `log_prob` returns `-inf` strictly outside `prior_bounds`, from both the
  package-level `log_prob` and `FleetFlow.log_prob`. Testing
  `np.isfinite(log_prob(flow, x))` is therefore a support test; the bounds
  themselves count as inside. The flowjax method `flow.log_prob(x)` on a
  fitted flow does not apply this mask, and scores such a point as if it sat
  on the nearest wall. `log_prob` accepts nested sequences and rejects a
  mis-shaped array with a message naming the expected shape.
- `save_flow` and `FleetFlow.save` write atomically, to a temporary file in
  the destination directory that is then moved into place with `os.replace`.
  An interrupted save no longer leaves a truncated file where a readable
  model used to be. The destination's existing permissions are kept.
- Clearer errors:
  - An error from a marginal builder names the column (`parameter (column)
    N: ...`).
  - `load_flow` raises `ValueError` naming the file for a truncated, garbage
    or incomplete payload, and points a fleet file at `FleetFlow.load`.
  - The package-level `sample`, `log_prob`, `sample_and_log_prob` and
    `save_flow` reject a `FleetFlow` with a `TypeError` naming the method to
    use instead.
  - `save_flow` rejects an object that is not a fitted flow with a
    `TypeError`.
  - `sample` and `sample_and_log_prob` reject an `n_samples` that is not an
    integer `>= 1`, instead of failing with a JAX static-shape `TypeError`.
  - A column with too few samples for a marginal fails with a message giving
    the column, its count and the required minimum.
- `check_in_support` returns a Python `bool`, accepts lists and tuples, and
  rejects a 1-D `samples` array with a message naming the expected shape.
- The README now says that importing coppuccino turns on JAX 64-bit mode for
  the whole process. 1.0.0 already did this, but only its package docstring
  said so.

### Fixed

These changes make code raise on input that 1.0.0 accepted, often with no
sign that anything was wrong:

- `normalizing_flows_fit` validates `prior_bounds`: it must have shape
  `(n_params, 2)` with `low < high` on every row, and the cleaned chain must
  lie inside the box. A non-finite side means unbounded, and NaN is rejected.
  It also validates `chain`, which must be a non-empty 2-D array. 1.0.0
  accepted a box narrower than the data.
- `normalizing_flows_fit` range-checks every training option. Integers are
  required for `rng_seed >= 0`, `knots >= 2`, `patience >= 1`,
  `max_epochs >= 1`, `num_points >= 2`, `flow_layers >= 1` and
  `batch_size >= 1`. Finite values are required for `learning_rate > 0`,
  `tanh_max_val > 0` and `weight_decay >= 0`. Each raises `ValueError` naming the option and the
  value. `max_epochs=0` and a negative `learning_rate` used to return a flow
  that looked fitted.
- `normalizing_flows_fit` raises `RuntimeError` when training diverges. That
  covers non-finite parameters, and also finite parameters whose mean
  log-density on the training rows is non-finite or more than 3 nats per
  dimension below the untrained starting point. On 800 rows of a correlated
  2-D Gaussian (ρ = 0.9) at `max_epochs=30`, learning rates from 1 to 100 all
  produced finite parameters and meaningless densities, and all now raise.
  `normalizing_flows_fit_auto` records this as a failed candidate.
- `compute_injection_hdr` rejects NaN or ±inf in `samples` or
  `injection_params`, and rejects a `samples` array that is not 2-D. A NaN
  injection used to be scored as out of support and given the 1.0 sentinel.

### Removed

- The `marginal`, `tail_extension`, `tail_model` and `tail_quantile` options of
  `normalizing_flows_fit`, with the module-level `make_empirical_cdf_spline`
  and `EmpiricalMarginalToGaussian` that implemented them.
- `compute_injection_hdr(num_samples=...)`.
- Loading 1.0.0 model files (see Upgrading).
- The `interpax` dependency, which only the 1.0.0 PCHIP marginal used.

### Dependencies and packaging

- `optax` (`>=0.2.2,<0.3.0`) is a direct dependency.
- `paramax` is capped at `<0.1.0` and `scipy` at `<2.0.0`. flowjax 17.2.1
  leaves paramax unbounded, so on the day paramax releases 0.1.0 this cap is
  the only bound on it. An environment that also needs a newer flowjax built
  for 0.1.0 then cannot resolve until coppuccino widens the cap. We accept
  that: a resolver error is loud, while a silent API break under the pickle
  format is not.
- The sdist contains only the source, tests, example notebooks, `README.md`,
  `CHANGELOG.md`, `LICENSE` and `pyproject.toml`, plus the `.gitignore` that
  hatchling always adds. It no longer includes editor config, the 1.8 MB logo
  or CI internals.
- CI pins every action to a full commit SHA. The publish workflow runs the
  test suite on the release commit, then installs and imports the built wheel
  in a clean environment before uploading. Dependabot updates both GitHub
  Actions and `uv.lock`.
- `ruff` (with `B`, `C4`, `I`, `SIM` and `C901`), `mypy` (with
  `check_untyped_defs` and no module-wide overrides) and a strict pytest
  configuration (`--strict-markers`, `--strict-config`,
  `filterwarnings = ["error"]`) gate every change.

### Security

- The README documents the pickle trust boundary. `load_flow` and
  `FleetFlow.load` run any code contained in the file, so load only model
  files you made yourself or got from a source you trust.

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

[Unreleased]: https://github.com/AaronDJohnson/coppuccino/compare/v2.0.0...HEAD
[2.0.0]: https://github.com/AaronDJohnson/coppuccino/compare/v1.0.0...v2.0.0
[1.0.0]: https://github.com/AaronDJohnson/coppuccino/releases/tag/v1.0.0
