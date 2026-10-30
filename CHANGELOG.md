# Changelog

All notable changes to **coppuccino** are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

_Slated for release as **3.0.0** — the version `pyproject.toml` already carries._

### Removed

- **BREAKING:** the unit-cube copula pipeline. `copula_space="uniform"`, the
  `CopulaFlow` class, `EmpiricalMarginalToUniform` and the `unit_range` mode of
  `make_empirical_cdf_spline` are gone. The Gaussian pipeline with the default
  `marginal="smooth"` replaces it: samples land strictly inside the prior, the
  reported density matches the sample stream, and one pipeline is maintained
  instead of two.
  - **It does not replace it at a wall.** On `U[0,1]^2` data with
    `prior_bounds=[[0,1],[0,1]]` (1500 rows from `default_rng(s)`,
    `max_epochs=80`, `rng_seed=s`, s = 0–3, true log-density 0 everywhere,
    probing `[x, 0.5]` at both walls) the Gaussian pipeline reports +0.53 to
    +4.52 nats *on* the bound, where the cube marginals that live on inside
    `normalizing_flows_fit_fleet` report −2.36 to +1.47 on the same data and the
    same seeds. Away from the wall the two are comparable: from the centre out
    to 1e-2 from either bound the Gaussian pipeline stays within −1.11 to +1.32
    nats, and −2.11 to +1.85 at 1e-4. If a weakly-constrained, flat-to-the-wall
    marginal is the use case the 2.0.0 notes named for the cube pipeline, fit it
    with `normalizing_flows_fit_fleet` — a one-source fleet is a supported way
    to do that.
  - `tests/test_uniform_copula.py` (169 lines, 8 tests) is deleted with the
    pipeline it covered. Seven of its eight test names exist nowhere else in
    `tests/` — they assert cube-specific properties (`CopulaFlow` arch, shell
    occupancy, the cube's wall behaviour, the `uniform` mode guards) that no
    longer have a subject. Only `test_save_load_roundtrip_exact` has a
    same-named counterpart elsewhere in the suite.
  - `normalizing_flows_fit(copula_space=...)` still accepts the argument and
    raises `ValueError` naming the removal, so v2 call sites and
    `normalizing_flows_fit_auto` candidate dicts fail with an actionable
    message rather than a bare `TypeError`.
  - `load_flow` raises `ValueError` on a cube-mode model file (detected by its
    `inner_arch` key) instead of failing obscurely. Pin `coppuccino<3` to read
    those files, or refit.
  - `normalizing_flows_fit_fleet` / `FleetFlow` are **unaffected**: the fleet
    models the cube through its own marginal code and now carries its own
    copula-flow builder. Existing fleet files still load and warm-start.

### Changed

- **BREAKING:** `make_empirical_cdf_spline`'s `marginal` default is now
  `"rqs"` (was `"pchip"`). Both entry-point defaults are unchanged —
  `normalizing_flows_fit` still resolves to `"smooth"`. RQS is the
  self-consistent family: its forward map and inverse agree to machine
  precision. Measured as `cdf_fn(quantile_fn(u))` over 999 points spanning
  `u ∈ [0.001, 0.999]`, 8 seeds of 2000-row `np.random.default_rng(s)`
  standard normals: worst round-trip error over all 7992 probes is 5.6e-17 for
  `"rqs"` against 8.3e-4 for the PCHIP pair, whose per-seed worst case has a
  median of 6.6e-4 (its median over all probes is far smaller, 4.9e-6 — PCHIP
  is accurate on most of the range and loses precision in the tails). Pass
  `marginal="pchip"` explicitly to reproduce older direct calls.
- `_n_trainable_params`, and therefore
  `normalizing_flows_fit_auto(selection="smallest_comparable")` and the
  `n_params` column, no longer counts `paramax.NonTrainable` leaves. Frozen
  components are excluded from the optimizer by flowjax, so they are not
  fitted degrees of freedom and must not count toward model size.
- `paramax` is now capped at `<0.1.0`. It is the only pre-1.0 runtime
  dependency, coppuccino imports `non_trainable`/`unwrap` from it directly, and
  flowjax 17.2.1 requires it as a bare `paramax`, so nothing else in the
  resolution graph bounds it. Precautionary — no paramax release has broken us.
  **The cost is real and worth stating:** because flowjax leaves paramax
  unbounded, the day paramax cuts 0.1.0 this ceiling is the only thing pinning
  it, and an environment that also wants a newer flowjax requiring 0.1.0 becomes
  unsatisfiable until coppuccino ships a release that widens the line. We take
  that trade — a resolver error is loud and a two-function API break in the
  layer the pickle format depends on is not — but a maintainer who disagrees
  should drop the ceiling rather than pin paramax elsewhere.
- README: the install section now states that importing coppuccino enables JAX
  64-bit mode process-wide (previously only the package docstring said so), and
  the Requirements list gains the previously omitted `optax` entry.
- `save_flow` and `FleetFlow.save` publish atomically: the payload is written
  to a temporary file in the destination's own directory and moved into place
  with `os.replace`. An interrupted or failing save no longer leaves a
  truncated file where a readable model used to be, and a concurrent reader
  sees either the old file or the new one. The destination's existing
  permissions are preserved; if it did not exist, the process umask applies
  rather than `mkstemp`'s owner-only mode.
- `normalizing_flows_fit_auto(return_table=True)` rows now carry the **same
  keys in every mode and in every row**, selected or not. Previously
  `refit_used`, `refit_density_finite`, `refit_error` and the `audit_*` keys
  were present only on some rows and only on some selection paths, so
  `pandas.DataFrame(table)` produced ragged columns and `row["refit_used"]`
  could raise `KeyError`. Rows gained keys; no pre-existing key changed value.
  The full schema is documented key by key in the function's `Returns` section.
- Error messages that pointed at removed or wrong API were corrected.
  `calibrate_edge_bins` no longer tells the user to pass `copula_space=...`
  (removed in this release), and handed a `FleetFlow` it now explains that the
  fleet carries per-source cube marginals with no Gaussian stack to calibrate
  and names the refit to perform, rather than saying only that it "requires a
  fitted Transformed flow". Message text only; the exception types and the
  conditions that raise are unchanged.

Packaging and quality gates (no runtime behavior change unless noted):

- `mypy` now runs with `check_untyped_defs = true` and **no** module-wide
  `disable_error_code` overrides. Previously the numerical core was exempt:
  planting `_typo_check: int = "not an int"` inside `_rqs_bulk` and
  `_bad: int = "nope"` inside `_resolve_seed`, in the pre-change source,
  produced `Success: no issues found in 7 source files` — the first hidden by
  `check_untyped_defs = false`, the second by the module-wide
  `disable_error_code` on `copula_flows`. Under the new settings both are
  errors. Applying the new settings to the pre-change source reports 63
  diagnostics in 5 files where the old settings reported none. Those were fixed
  rather than silenced, except for 15 per-line `# type: ignore[...]`:
  - 12 are the equinox `AbstractVar`/`ClassVar` shape/cond_shape fields and the
    `__init__` signature equinox synthesises from them — 6 on the field
    declarations themselves (`bijections.py`, `calibration.py`) and 6 across the
    four marginal-bijection construction sites in `copula_flows.py` /
    `model_io.py`, where mypy reads the synthesised `shape`/`cond_shape`
    parameters as the first two positional arguments and so reports both missing
    arguments and mistyped ones.
  - 3 are the tail-constant pre-declaration at the top of
    `make_empirical_cdf_spline` (`bijections.py`), where twelve constants are
    declared `: float`, assigned `None`, and bound only inside their guarding
    GPD/Gaussian-tail branch. This is the load-bearing suppression in the tree:
    writing the truthful `Optional[float]` and dropping the three ignores takes
    `bijections.py` from clean to 54 errors, because mypy cannot tie
    "`use_gpd` is true" to "`F_lo` is not None" at the reads downstream.
    The reasoning is spelled out in a comment at the declaration site.
- `src/coppuccino/bijections.py` is fully annotated: counting every parameter of
  every function in the module including nested closures, 93 of 93 now carry a
  type against 15 of 57 before this change set, so the RQS/GPD/tail kernels are
  checked rather than typed `Any`.
- `ruff` selects `B`, `C4`, `I`, `SIM` and `C901` in addition to the previous
  `E4/E7/E9/F/W`, with `max-complexity = 15`. A probe function with a mutable
  default argument previously produced zero diagnostics and now produces a
  `B006`. One legacy function, `make_empirical_cdf_spline`, carries a
  per-function `# noqa: C901` naming the reason.
  `B905` (`zip(..., strict=)`) is disabled project-wide with a comment.
- `[tool.pytest.ini_options]` exists for the first time: `--strict-markers`,
  `--strict-config` and `filterwarnings = ["error"]`. One real defect surfaced
  (below). One allowance is carried: `ignore:IProgress not found:UserWarning`,
  for tqdm's notebook advisory, which flowjax's progress bar and equinox's
  `filter_jit` both re-emit when the suite runs without ipywidgets. Warnings the
  library is *meant* to raise are asserted with `pytest.warns` at the call site
  rather than added here.
- Both workflows pin every action to a full commit SHA. `publish.yml` — which
  holds the PyPI trusted-publishing grant — previously used
  `pypa/gh-action-pypi-publish@release/v1`, a mutable *branch* ref. It also now
  declares `permissions: {}` at workflow level, runs the test suite on the
  released commit before building, and installs and imports the built wheel in
  a clean environment (the check that would have caught an untracked module).
- `test.yml` runs `pytest tests/ -q` instead of `pytest tests/ -x -q`: the
  matrix already sets `fail-fast: false`, so each leg runs to completion anyway,
  and a numerical regression usually fans out across many tests. Seeing all of
  them in one run says more about the root cause than seeing the first.
- `.github/dependabot.yml` gains a `uv` ecosystem entry, so `uv.lock` and the
  pyproject bounds receive updates. Previously only GitHub Actions did, and
  every CI leg resolved from the committed lock or the declared floors.
- `optax` and both `scipy` markers gained upper bounds (`<0.3.0`, `<2.0.0`),
  matching the other nine runtime dependencies.
- `scripts/` and `docs/*.tex` are tracked; the benchmark drivers' `pandas` /
  `pyarrow` requirements are declared in a new optional `bench` dependency
  group. Their generated output (`docs/*.pdf`, `scripts/*_results.json`) is
  gitignored.
- The sdist `exclude` list no longer carries `/.gitignore` and
  `/.gitattributes`, because neither entry ever did anything. Built from this
  configuration the sdist contains `.gitignore` — hatchling force-includes it,
  since it needs the file for VCS-based file selection, and no `exclude` pattern
  removes it — and does not contain `.gitattributes` at all, with or without the
  entry. The comment that accompanied them claimed both files ship; only one
  does.

### Added

- `normalizing_flows_fit(flow_layers=0)` builds no trainable copula. Combined
  with `whiten=True` this is a zero-parameter Gaussian-copula baseline, useful
  as the smallest rung of a `normalizing_flows_fit_auto` ladder to quantify
  what the flow is buying over plain correlation.
- `normalizing_flows_fit(whiten=True)` (**experimental**) freezes the
  copula-space Cholesky factor into the flow as a non-trainable
  `TriangularAffine`. Measured on the galactic-binary example it does not help
  at converged training and degrades with capacity; treat it as a research
  knob, not a recommended setting.

- `normalizing_flows_fit_auto(selection="marginal_fidelity")`, a non-calibration
  model-selection remedy for Gaussian marginal drift. It ranks marginal fits
  only among candidates that remain statistically comparable in held-out joint
  likelihood, with an optional absolute probability-mass error budget. Shared
  physical bin edges score literal first/last bins together with under/overflow;
  each cell supports its own ESS and directional Wilson-interval treatment.
  Component diagnostics are returned, failed candidates are isolated, and a
  full-data refit that regresses any audited component or the joint-fit sanity
  check automatically falls back to the validated split fit.

- `compute_injection_hdr(reference="held_out")`. The default,
  `reference="flow"`, ranks an injection against fresh draws from the *fitted*
  density, so error in the fit shows up as a bias in the credibility — upward,
  and largest exactly where posteriors are thinnest. On 1-D standard normal —
  `samples` of n rows and then 8000 injections drawn in turn from
  `default_rng(s)`, `num_samples=40000`, `rng_seed=s`, s = 0–3 — the mean
  credibility (0.5 if unbiased) is 0.569–0.576 at n=1500 and 0.518–0.530 at
  n=6000.
  `reference="held_out"` fits on half the rows and ranks against the other
  half, so both sides of the comparison are posterior draws and the ranks are
  uniform for any fitted density: the same measurement gives 0.499–0.526 and
  0.495–0.502. It is opt-in because flipping the default would silently move
  every existing caller's numbers and make `num_samples` inert; prefer it for
  P-P plots. The full protocol and a KS table are in the function's Notes.
- `save_flow` payloads carry a `writer_version` string naming the coppuccino
  release that wrote them. Additive: older installs ignore unknown keys, so
  files written by this release still load on `coppuccino<3`, and no
  `format_version` bump was needed. (`FleetFlow.save` does not yet stamp it.)

### Fixed

Marginals and density:

- **Samples sitting exactly on a prior bound.** `make_smooth_marginal` clamped
  `z` inside `log_dtheta_dz` but not inside `z_of_theta`, and the bounded tail
  could not carry a finite positive density at the wall at all, so the at-bound
  log-density was not a meaningful quantity: it varied by orders of magnitude
  from seed to seed on the same configuration. It is now stable, finite and
  consistent with the sampler. Measured on 1500×2 chains of
  `clip(N(0.77, 0.25), -1, 1)` with `prior_bounds=[[-1,1],[-1,1]]` (460–500 rows
  touching a bound), `max_epochs=80`, `rng_seed=s`, s = 0–3: the median at-bound
  `log_prob` is +8.46 to +13.35 across the four seeds.
  **It is not close to the interior density.**
  The interior median on the same fits is +0.29 to +0.50, so an at-bound row
  scores +8.17 to +12.87 nats *above* it. That is what the model is: the
  wall density is the atom's mass spread over the outermost inset interval, and
  a chain with a third of its rows clipped onto a wall really does have that
  mass there. It is a better model of such data by held-out likelihood, and the
  density integrates to 1, but a reader who takes a pointwise density off a
  wall will see a large number. Putting mass on a bound is the documented use
  of `prior_bounds`, so this is reachable from the recommended workflow; see
  `SmoothMarginalToGaussian` for the mechanism.
- `marginal="rqs"` no longer collapses the density at the outermost knot.
  `make_empirical_cdf_spline` built its bulk spline without an
  `edge_slope_floor`, and PCHIP's one-sided endpoint rule returns exactly zero
  on empirical-quantile grids in most cases, so the most extreme sample of each
  dimension — a real training row — was scored at the `min_eps` floor of 1e-7
  regardless of the true density there — an error of whatever separates the true
  density from 1e-7, which on ordinary data is several nats and grows the denser
  the sampling gets. Measured as
  `make_empirical_cdf_spline(x, marginal="rqs").pdf_fn` at `x.min()` and
  `x.max()`, 8 seeds × 2 tails of 2000-row `N(0,1)`, against the analytic
  density, the absolute log-density error is now **median 1.29 / max 2.08**.
  The residual is the sampling error of the outermost order statistic, not a
  systematic offset, so it moves with the seed family.
- `tail_extension=True` is monotone again past ~5.2 fitted sigmas. Both `ndtri`
  arguments were floored at `min_eps=1e-7`, whose inverse is only −5.199, so
  the tail branch returned values *inside* the data range and the marginal
  stopped being invertible; the right tail also lost every digit of small
  survival probabilities to `1 - x` cancellation. That region is routinely
  reached: on 5000-row Student-t(3), lognormal and exponential draws, all 8
  seeds tested put part of the tail-extended map past |z| = 5.199, and the map
  is monotone across the whole swept range on every one of them.
- `tail_model="gpd"` now joins the bulk with a continuous density, not just a
  continuous CDF. The GPD scale was fitted by MLE with nothing tying it to the
  spline slope at the threshold. Measured as `|log pdf(u−h) − log pdf(u+h)|` at
  each threshold `u = np.quantile(x, q)`, `h = 1e-9·max(|u|,1)`, over 40 seeds
  × 2 tails of 4000-row `np.random.default_rng(s).standard_t(3, 4000)` with
  `tail_quantile=0.05`: for the default `marginal="rqs"` the residual is
  **median 2.2e-8 / max 1.8e-7 nats**, and it scales linearly with the probe
  offset — the signature of a continuous density sampled either side of a
  point, leaving only the density's own slope. Tail parameters and total mass
  are unchanged; `marginal="pchip"` still steps at the join, median 0.28 / max
  1.1 nats on the same measurement (documented in `tail_model`). These figures
  move by up to an order of magnitude under a different generator, so the
  generator is named above rather than left to the reader.
- `EmpiricalMarginalToGaussian` records `marginal="rqs"` by default, matching
  the builder. A hand-constructed bijection previously recorded a family that
  contradicted the closures it held, and `save_flow`/`load_flow` serialised
  that field verbatim, silently swapping the interpolant on reload.

Input validation — these raise on input that previously "worked":

- `normalizing_flows_fit` validates `prior_bounds` (shape, ordering, and that
  the chain lies inside the box) and `chain` (2-D, coerced, non-finite rows
  dropped with a warning). Previously, bounds narrower than the data were
  folded destructively by `marginal="smooth"` and ignored outright by
  `"rqs"`/`"pchip"`, with no warning and finite-looking densities; a malformed
  bounds array surfaced as an unrelated complaint about the *data*; and a 3-D
  chain with equal trailing axes was silently flattened and fitted as 2-D.
- Errors raised inside a marginal builder now carry a `parameter (column) N: `
  prefix, so a single degenerate column in a wide posterior no longer has to be
  found by bisection.
- `FleetFlow` raises `IndexError` for an out-of-range `source_index`. JAX
  clamps out-of-bounds indices, so this previously returned a neighbouring
  source's flow — a wrong-but-plausible proposal density with no diagnostic.
  Negative indices in `[-n_sources, -1]` keep working and are now documented.
- `load_flow` and `FleetFlow.load` raise `ValueError` naming the file, the kind
  found and the correct loader, instead of a bare `KeyError` on an internal
  dict key. Payloads now carry `format_version` and `kind`; a missing version
  is treated as version 1, so every file already written still loads.
- The package-level `sample`, `log_prob`, `sample_and_log_prob` and `save_flow`
  reject a `FleetFlow` with a `TypeError` naming the method to use instead of
  failing on an internal attribute or dumping a JAX trace.
- `compute_injection_hdr` validates that `samples` is 2-D.
- `normalizing_flows_fit` range-checks every training knob, not only
  `flow_layers`. `rng_seed`, `knots >= 2`, `patience >= 1`, `max_epochs >= 1`,
  `num_points >= 2` and `flow_layers >= 0` must be integers; `learning_rate`
  and `tanh_max_val` must be finite and positive, `weight_decay` finite and
  non-negative. Each raises `ValueError` naming the parameter and the value
  seen. Previously `knots=0` reached a `ZeroDivisionError` inside flowjax,
  `rng_seed=1.5` a jax "PRNG key seed must be an integer", and — worst of the
  set — `max_epochs<=0`, `learning_rate=0.0` and a sign-typo
  `learning_rate=-1e-3` were all accepted and returned a flow indistinguishable
  from a fitted one. A negative learning rate trains to convergence in the
  wrong direction and reports finite densities throughout, so no post-fit
  finiteness check can catch it.
- `normalizing_flows_fit` raises `RuntimeError` when the optimizer diverges.
  The check has two arms. The first rejects a trained copula with any non-finite
  parameter — that flow's every draw is NaN and every log-density is NaN or
  `-inf`. The second, which does the real work, rejects a flow whose parameters
  are *finite* but whose mean copula-space log-density on the training rows is
  non-finite, or more than 3 nats per dimension below the flow's own untrained
  initialization. flowjax keeps the best-validation-loss parameters and accepts
  epoch 1 unconditionally, so a rate that blows up on the first step is frozen
  there with finite parameters and a nonsense density; a parameter-only guard
  never sees it. Measured on 800 rows of a correlated 2-D Gaussian (ρ = 0.9,
  `default_rng(0)`), `max_epochs=30`, `rng_seed=0`: `learning_rate` 1e-3, 1e-2
  and 0.5 return normally, and 1, 3, 10, 30, 60 and 100 all raise — every one of
  them through the density arm, with finite parameters.
  `normalizing_flows_fit_auto` catches this per candidate, so it surfaces as a
  candidate failure with the message attached, and `compute_injection_hdr` now
  propagates it rather than returning `1.0` for every in-support injection. The
  cost is two extra copula-space log-density evaluations per fit. The threshold
  and the grids it was chosen from are documented in `_assert_fit_converged`.
- `normalizing_flows_fit_auto` validates `val_frac`: outside `(0, 1)`, or a
  fraction that leaves too few rows on either side of the split, now raises
  `ValueError` naming the value and the row counts. `val_frac <= 0` used to be
  clamped to a one-row held-out set and scored against it.
- The package-level `sample` and `sample_and_log_prob` require `n_samples` to
  be an integer `>= 1`; `sample(flow, -5)` and `sample(flow, 10.5)` used to
  surface as jax static-shape `TypeError`s. (`FleetFlow.sample` is not yet
  covered and still raises the jax error.) `log_prob` accepts nested sequences
  (`log_prob(flow, [[0.1, 0.2]])`) rather than raising flowjax's "Expected x to
  be arraylike", matching what `FleetFlow.log_prob` already did, and rejects a
  mis-shaped or non-numeric array with a message naming the expected shape.
- `normalizing_flows_fit_auto` warns when it drops non-finite rows, with a
  count, as `normalizing_flows_fit` already did. The same dirty chain used to
  be reported through one entry point and silently cleaned by the other.
- `normalizing_flows_fit_fleet` validates its inputs to the same standard as
  the single-flow path. `prior_bounds` must be finite and `low < high` on every
  row (reversed, degenerate and NaN boxes were all misreported as "chains
  contain samples outside prior_bounds", and infinite bounds crashed inside
  scipy); `chains` must be finite (an `inf` was reported as a containment
  failure); a chain with no sources or no parameters raises instead of dying
  with an `IndexError` on an empty `Stack`; and a column that is constant
  within a source is an error, as it already was in `normalizing_flows_fit`,
  rather than a silently meaningless fit that samples across the whole box.
  **This can turn a previously-succeeding fleet fit into a failure** — that is
  the intended parity. Note the fleet's contract is deliberately *stricter*
  than `normalizing_flows_fit`'s, which documents that either side of a
  `prior_bounds` row may be non-finite, meaning unbounded there; the fleet has
  no unbounded mode.
- `FleetFlow.log_prob` validates the shape of its input, raising `ValueError`
  instead of a jax vmap dump naming `p.base_dist.bijection.loc.tree` when
  handed a `(b, dim)` array without `source_index`. A row containing NaN now
  scores `-inf` rather than NaN, so finiteness of `log_prob` is a support test
  for those rows too.
- `compute_injection_hdr` validates `injection_params` for NaN/Inf and
  `num_samples` for being a positive integer. A NaN injection used to be
  scored out-of-support by the hull test and short-circuited to the `1.0`
  sentinel — a plausible-looking answer for input that has none.
- `check_in_support` coerces list and tuple input instead of raising
  `AttributeError: 'list' object has no attribute 'ndim'`, and a 1-D `samples`
  array raises a `ValueError` naming the expected shape instead of an
  `IndexError`.
- `save_flow` raises `TypeError` for an object that is not a fitted flow,
  instead of `AttributeError` on an internal attribute. `load_flow` raises
  `ValueError` naming the file for a truncated, garbage or incomplete payload
  (with the original exception attached as `__cause__`) instead of a bare
  `KeyError`/`UnpicklingError`/`EOFError`; `FileNotFoundError` for a missing
  path is unchanged. The atomic writer names the destination in every failure
  path and rejects a destination that names a directory — including the empty
  string, which `pathlib` resolves to `.` — with a `ValueError` before doing
  any I/O, rather than an `OSError(EINVAL)` from `os.replace` after leaving a
  temp file behind in that directory.
- `make_smooth_marginal` raises when the data or the prior bounds exceed the
  spline kernel's float64 headroom (`|value| > 1e75`, where the kernel's
  quadratic overflows and the map stops being finite and monotone) instead of
  returning a broken map, and `tail_alpha` is capped at 10 — it is a growth
  rate per unit `z`, so the map spans `exp(tail_alpha·|z|)` decades of `theta`
  over the `z` range the base actually visits, and an unbounded `tail_alpha`
  therefore ran the sampler straight into the top of float64. Both messages name
  the parameter and the observed value.

Other correctness:

- `log_prob` returns `-inf` strictly outside `prior_bounds`, on both the
  single-flow and the fleet path, instead of a finite wall-density plateau.
  Each marginal evaluates an out-of-range query at its nearest bound, so an
  unmasked point outside the box scored as if it sat on the wall: on a
  `U[0,1]^2` fit with `prior_bounds=[[0,1],[0,1]]`, `log_prob([5, 0.5])`,
  `log_prob([1e9, 0.5])` and `log_prob([inf, 0.5])` all returned the same value
  as `log_prob([1, 0.5])`, however far outside the box the query was. Since the
  at-bound density can exceed the interior density (see the entry above), that
  plateau could rank a point outside the support above real draws.
  `np.isfinite(log_prob(x))` is now a correct support test; the bounds
  themselves remain inside.
- `FleetFlow.sample(rng_seed=s, source_index=i)` now reproduces
  `sample(rng_seed=s)[i]`, matching what `log_prob` already guaranteed. **The
  draws for a given seed differ from previous versions**; the distribution was
  correct before and after.
- `normalizing_flows_fit_auto` checks the full-chain refit for a finite
  held-out density on *every* selection path, not only
  `selection="marginal_fidelity"`, falling back to the validated split fit with
  a warning. The selected table row now always reports `refit_used`.
- `calibrate_edge_bins`'s `min_calibration_tail_count` guard measures the gaps
  between *all* consecutive knots rather than only the active anchors.
  Inactive anchors remain knots of the spline, so the guard could report an
  interval comfortably above the threshold while the real gaps were both below
  it.
- The rational-quadratic spline kernel, previously implemented three times with
  three different out-of-range policies, is now one implementation. Two of the
  copies returned non-monotone values and negative densities outside the knot
  range; the shared kernel saturates at the end knot. Unreachable through any
  public entry point today — every call site pre-clips — so in-range results
  are bit-identical.
- The float-resolution floors of the bounded tail are scaled by the smaller of
  the prior and data spans, not by the prior span alone, in both the smooth
  marginal and the fleet's knot builder. Scaling by the prior span alone let an
  uninformative prior eat the low end of the chain: the floor that keeps knots
  numerically distinct was sized against a box many orders of magnitude wider
  than the data, so the bottom of the chain collapsed into a fraction of the
  Gaussian range it should occupy while the top stayed fully resolved, and the
  fitted marginal therefore depended on how wide the prior was. It no longer
  does. Measured on a 1500×2 log-uniform chain spanning ~[5e2, 4e3]: under
  `prior_bounds` of `[[0, 1e12]] * 2` and of `[[0, 1e4]] * 2` alike, the data
  range now maps onto the symmetric `z ∈ [−3.403, 3.403]` in both dimensions.
  For boxes comparable to the data span — the normal case — the knots are
  bit-identical to before.
- Docstring figures that did not reproduce. `compute_injection_hdr`'s
  out-of-hull deficit and `sample`'s tail-clamp redraw rate were both stated as
  single numbers for quantities that have no single value; both now state their
  measurement protocol and give a range, and the redraw-rate note says outright
  that the rate is driven by how much of the training chain sits on a bound and
  varies by more than a factor of two between seeds of one configuration.
- `make_smooth_marginal` no longer raises `ZeroDivisionError` when the prior
  bound's magnitude dwarfs the data span and the knot-to-bound gap rounds to
  zero, and the tail inversion stays finite when JAX x64 is disabled.
- `marginal="pchip"` and `marginal="rqs"` with `prior_bounds` no longer emit
  draws sitting exactly on a bound, and the smooth marginal no longer has a
  bitwise-constant plateau just inside one, so two distinct near-wall points
  no longer receive an identical `z` and an identical density.
- `normalizing_flows_fit` warns when `tail_alpha` is silently discarded because
  `tail_extension`/`tail_model="gpd"` resolved the default marginal to `"rqs"`.
  The same request written explicitly already raised.

Performance (all bit-identical in output):

- Fitting a fleet no longer builds N complete flowjax models to stack six
  arrays; `FleetFlow.load` no longer pays that cost purely to build a skeleton
  that is immediately overwritten. Both are now flat in source count rather
  than linear in it. The crossover where the traced initializer starts paying
  for itself shifts with architecture, so `fleet.py` documents the figures it
  was tuned against rather than quoting one here.
- Fleet training no longer pins a full duplicate u-space tensor for the whole
  fit, and the knot sort runs in bounded blocks rather than allocating a copy
  of the entire fleet.
- The fleet's per-(source, dimension) knot build is vectorized instead of
  looping scipy's PCHIP over every pair, and small fleets skip the traced
  parameter initializer that only pays for itself at larger source counts.
  `FleetFlow.load` and `.save` hold fewer full copies of the parameter stack.
- `compute_injection_hdr` computes the support hull once and evaluates
  `log_prob` in one batched call rather than once per injection.
- Marginal-fidelity scoring is wrapped in `filter_jit`, like every other
  sampling site.

### Security

- Documented the pickle trust boundary. `save_flow`/`load_flow` and
  `FleetFlow.save`/`.load` use Python pickle, so loading a model file executes
  arbitrary code contained in it; the README's API section now says so and
  tells readers to load only files from a trusted source. Documentation only —
  the file format is unchanged.

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

  > **Corrected in 3.0.0:** "no wall density cliffs" and "bounded sides
  > approach walls exponentially" overstated what this family does *at* a
  > bound, and the 3.0.0 rebuild of the bounded tail changed the behaviour
  > again. The at-bound density is finite, positive and consistent with the
  > sampler, but it is well above the interior density rather than a smooth
  > continuation of it — see the "Samples sitting exactly on a prior bound"
  > entry above and `SmoothMarginalToGaussian`. The no-exact-bound-atoms and
  > redraw statements are accurate as written.

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
- Experimental `calibrate_edge_bins()` post-fit calibration for smooth
  Gaussian-space flows. It corrects selected marginal edge-bin CDF boundaries
  with compact monotone splines while leaving a central region and the fitted
  rank copula unchanged. A Wilson/ESS-aware minimum-change gate leaves compatible
  tails untouched and rejects under-resolved calibration unless explicitly
  forced. Calibrated models retain consistent sampling, `log_prob`, and save/load
  behavior.

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
  or CI internals; it is scoped to source, tests, example notebooks, `README.md`,
  `CHANGELOG.md`, `LICENSE` and `pyproject.toml`. (Corrected in 3.0.0: this
  originally read "and docs", but `docs/` has never been in the include list and
  has never appeared in a built sdist.)
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

[Unreleased]: https://github.com/AaronDJohnson/coppuccino/compare/v2.0.0...HEAD
[2.0.0]: https://github.com/AaronDJohnson/coppuccino/releases/tag/v2.0.0
[1.1.0]: https://github.com/AaronDJohnson/coppuccino/releases/tag/v1.1.0
[1.0.0]: https://github.com/AaronDJohnson/coppuccino/releases/tag/v1.0.0
