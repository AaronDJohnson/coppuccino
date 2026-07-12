<p align="center">
  <img src="https://raw.githubusercontent.com/AaronDJohnson/coppuccino/main/image.png" alt="coppuccino" width="320">
</p>

<h1 align="center">coppuccino</h1>

<p align="center">
  <em>Density estimation for multivariate data with copula normalizing flows, in JAX.</em>
</p>

<p align="center">
  <a href="https://github.com/AaronDJohnson/coppuccino/actions/workflows/test.yml"><img src="https://github.com/AaronDJohnson/coppuccino/actions/workflows/test.yml/badge.svg" alt="Tests"></a>
  <img src="https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue.svg" alt="Python 3.11 | 3.12 | 3.13 | 3.14">
  <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License">
</p>

---

**coppuccino** fits and samples from complex multivariate probability
distributions by combining two classical ideas: *copulas*, which separate a
joint distribution into its one-dimensional marginals and a dependency
structure, and *normalizing flows*, which learn that dependency structure
flexibly. It is designed for — though not limited to — density estimation on
**MCMC posterior samples**, where it enables fast resampling, density
evaluation, importance weighting, and calibration diagnostics.

## The idea

Any continuous joint distribution can be decomposed into its marginals and a
*copula* that couples them. This is **Sklar's theorem**: for a joint density
$p(\mathbf{x})$ on $\mathbb{R}^d$ with marginal CDFs $F_i$ and marginal
densities $p_i$,

$$
p(\mathbf{x}) = \underbrace{c\big(F_1(x_1), \dots, F_d(x_d)\big)}_{\text{copula density}} \prod_{i=1}^{d} \underbrace{p_i(x_i)}_{\text{marginals}}.
$$

The copula density $c$ captures *all* of the dependence between dimensions; the
$p_i$ capture the shape of each dimension on its own. Estimating these two
pieces separately is often much easier than estimating $p(\mathbf{x})$
directly — especially when the marginals are sharp, bounded, skewed, or
multimodal, which tends to defeat a plain normalizing flow.

coppuccino estimates the marginals **nonparametrically** and the copula with a
**normalizing flow**.

## How it works

Fitting proceeds in two stages, and two pipelines are available. The default
(`copula_space="gaussian"`) maps each marginal to Gaussian space through a
clip-free smooth-tail transform and learns the copula with a triangular
spline flow over a normal base — the strongest choice for correlated
posteriors. The cube-native alternative (`copula_space="uniform"`), shown in
the diagram below, stops the marginals at the probability integral transform
and learns the copula directly on the unit cube — the natural choice when
weakly-constrained, flat-to-the-wall marginals dominate. Both are
boundary-safe; they differ in where the copula lives and how the tails are
modeled.

```mermaid
flowchart LR
  X["data x in R^d<br/>(e.g. MCMC samples)"]
  U["unit cube u in [0,1]^d<br/>uniform marginals,<br/>dependency retained"]
  E["base distribution<br/>Uniform([0,1]^d)"]
  X -->|"empirical CDF u_i = F_i(x_i)<br/>(per dimension, frozen)"| U
  U -->|"RQS coupling flow on the cube<br/>(learns the copula)"| E
```

**1. Empirical marginal transforms (probability integral transform).** Each
dimension is mapped onto the unit interval through its empirical CDF,

$$
u_i = F_i(x_i), \qquad i = 1, \dots, d,
$$

where each $F_i$ is built from the data quantiles as a **monotone
rational-quadratic spline** — an exact bijection onto $[0,1]$ whose forward
map, inverse, and derivative share a single parameterization (see
[Marginal transforms](#marginal-transforms)). After this step $\mathbf{u}$ has
exactly uniform marginals while retaining the full dependency structure — by
Sklar's theorem, the density of $\mathbf{u}$ *is* the copula. These transforms
are estimated once and then frozen.

**2. The copula flow.** A rational-quadratic-spline coupling flow
([flowjax](https://github.com/danielward27/flowjax)) models the density of
$\mathbf{u}$ against a uniform base on the unit cube. Every layer maps the
cube to itself with a strictly positive derivative, so the model's support is
*exactly* the cube — no clipping anywhere in the pipeline. Only this stage is
trained.

The log-density of a point is available in closed form,

$$
\log p(\mathbf{x}) = \log c(\mathbf{u}) + \sum_{i=1}^{d} \log p_i(x_i),
$$

where $\log c$ is supplied by the flow and $p_i = F_i'$ are the marginal
densities. **Sampling** runs the pipeline in reverse: draw $\mathbf{u}$ from
the copula flow and invert the marginals with $x_i = F_i^{-1}(u_i)$.

### The Gaussianized variant (`copula_space="gaussian"`)

The original pipeline inserts an inverse-normal step,
$z_i = \Phi^{-1}(F_i(x_i))$, and models the copula with a triangular spline
flow over a standard-normal base on unbounded space. It remains available —
and is required — for the **tail models** (`tail_extension=True`,
`tail_model="gpd"`), i.e. for extrapolating beyond the observed data range on
unbounded parameters.

The historical trade-off was at hard support boundaries: the legacy
$\Phi^{-1}$ endpoint handling could leave the sample stream and the reported
density inconsistent in thin slivers at the walls (sample atoms sitting on a
density cliff). That matters whenever boundary behavior matters — most acutely
when the fitted flow is used as an independence MCMC proposal, where a
proposal drawing from true density $s$ but reporting density $q$ has
stationary law $\pi\, s/q$, so boundary mismatches become exponentially
amplified biases. As of v2.0 the Gaussian-space pipeline's default
`marginal="smooth"` removes this at the root: each marginal is a single
clip-free monotone map of all of $\mathbb{R}$ onto the support (spline bulk +
analytic tails), so samples land strictly inside the prior, densities decay
smoothly to the walls, and out-of-data queries get honest tail values
(`tail_alpha` selects Gaussian or heavier tails for outlier tolerance).

> **Changed in v2.0:** the Gaussian pipeline's default marginals are the new
> clip-free smooth-tail maps, and the cube-native `copula_space="uniform"`
> pipeline is available as the box-native alternative — the natural choice
> when flat-to-the-wall marginals dominate.

### Why this design

- **Separation of concerns.** The flow spends all of its capacity on
  dependency, because the marginals are already uniformized. The empirical
  CDFs, in turn, reproduce difficult marginal shapes (bounded, spiky,
  heavy-tailed, multimodal) that flows struggle to fit on their own.
- **Honest density and importance weights.** The marginal transforms and the
  cube flow are closed-form bijections whose forward map, inverse, and
  derivative are mutually exact to machine precision, so `sample` and
  `log_prob` are self-consistent everywhere — including at support
  boundaries. `sample_and_log_prob` weights are honest by construction.
- **Built for bounded posteriors.** The fitted support is exactly the
  observed data range, or the full prior box when `prior_bounds` is given —
  the common situation for MCMC chains — with a strictly positive density all
  the way to the walls.

## Installation

```bash
pip install coppuccino
```

From source:

```bash
git clone https://github.com/AaronDJohnson/coppuccino.git
cd coppuccino
pip install .
```

### Development

This project is managed with [uv](https://docs.astral.sh/uv/):

```bash
uv sync           # create the environment and install coppuccino + dev deps
uv run pytest     # run the test suite
```

## Quick start

```python
import numpy as np
from coppuccino import normalizing_flows_fit, sample, log_prob, save_flow, load_flow

# Fit a copula flow to multivariate data (e.g. MCMC posterior samples)
data = np.random.randn(5000, 3)
flow = normalizing_flows_fit(data, max_epochs=200)

# Draw new samples from the fitted distribution
new_samples = sample(flow, n_samples=1000, rng_seed=42)

# Evaluate the log probability density
log_probs = log_prob(flow, new_samples)

# Persist and reload the model
save_flow(flow, "my_flow.pkl")
loaded_flow = load_flow("my_flow.pkl")
```

The [`examples/`](examples/) directory contains a worked, end-to-end notebook —
[`galactic_binary_example.ipynb`](examples/galactic_binary_example.ipynb) — that
fits a flow to a real gravitational-wave posterior and compares the samples to
the original chain, plus an HDR calibration walkthrough in
[`hdr_credibility.ipynb`](examples/hdr_credibility.ipynb).

## API

| Function | Purpose |
| --- | --- |
| `normalizing_flows_fit(chain, ...)` | Fit a copula flow to data; returns the fitted model. |
| `normalizing_flows_fit_auto(chain, ...)` | Fit a ladder of candidate architectures concurrently; return the smallest statistically-comparable model. |
| `normalizing_flows_fit_fleet(chains, prior_bounds, ...)` | Fit many per-source flows simultaneously (vmapped); returns a `FleetFlow` with batched `sample`/`log_prob`. |
| `sample(flow, n_samples, rng_seed=...)` | Draw samples from a fitted flow. |
| `log_prob(flow, samples)` | Evaluate the log density at given points. |
| `sample_and_log_prob(flow, n_samples, ...)` | Draw samples and their log densities together (for importance sampling). |
| `save_flow(flow, path)` / `load_flow(path)` | Serialize / deserialize a fitted flow. |
| `compute_injection_hdr(samples, injection_params, ...)` | HDR credibility of injected/true parameters (calibration). |
| `check_in_support(samples, injection_params)` | Whether a point lies within the sample support. |

Key options to `normalizing_flows_fit` include `copula_space` (the pipeline:
`"gaussian"` default with smooth-tail marginals / `"uniform"` cube-native,
above), `flow_layers` and `knots`
(flow capacity), `max_epochs`, `learning_rate`, `patience` (training),
`prior_bounds`, `tail_model`, and `marginal` (the marginal model, below). See
the docstrings for the full list. `normalizing_flows_fit_auto` fits a small
ladder of capacities concurrently and returns the smallest model whose
held-out fit is statistically comparable to the best.

### HDR credibility (inference validation)

The highest-density-region (HDR) credibility of a known/injected parameter is
the fraction of the fitted distribution that is *more probable* than that point
— equivalently, the smallest HDR credible level whose region contains it. A
point at the mode scores near 0; a point far in the tails scores near 1. For
well-calibrated Bayesian inference these values are uniform on $[0, 1]$ across
many events — a standard probability–probability check.

```python
from coppuccino import compute_injection_hdr

posterior_samples = ...          # shape (n_samples, n_params)
true_params = np.array([1.0, 2.0, 3.0])

hdr = compute_injection_hdr(posterior_samples, true_params)
# `hdr` is an array: hdr[0] for a single injection, or one value per row for a
# 2D batch of injections. Injections outside the sample support return 1.0.
```

## Marginal transforms

The empirical marginals are the heart of the method, and a few options control
how they behave at and beyond the edges of the data.

### Interpolant

```python
flow = normalizing_flows_fit(data)                    # marginal="smooth" (default)
flow = normalizing_flows_fit(data, copula_space="uniform")   # marginal="rqs"
flow = normalizing_flows_fit(data, marginal="pchip",
                             copula_space="gaussian")  # original PCHIP interpolant
```

`marginal=None` resolves per pipeline. The gaussian default `"smooth"` is a
single clip-free monotone map of all of R onto the support — a
rational-quadratic-spline bulk at the data quantiles joined to analytic
tails — whose forward map, inverse, and derivative share one
parameterization, so samples land strictly inside the support and densities
match the sample stream to machine precision. The uniform pipeline uses
`"rqs"` (a monotone rational-quadratic spline with the same exact-inverse
property, as an exact bijection onto [0, 1]). The `"pchip"` path builds the
CDF and its inverse as two independent splines that are only approximate
inverses of each other; it is retained for reproducing older fits and
requires the Gaussianized pipeline (`copula_space="gaussian"`).

### Prior bounds (recommended for MCMC chains)

```python
# Extend the empirical CDF out to a known prior support
bounds = np.array([[-10, 10], [-5, 5], [0, 100]])
flow = normalizing_flows_fit(data, prior_bounds=bounds)
```

Without `prior_bounds`, the default smooth marginals treat each side as
unbounded and extend analytic tails beyond the data (Gaussian tails by
default; see `tail_alpha` for heavier ones), while the uniform pipeline's
support ends at the observed data range. Supplying `prior_bounds` anchors
each marginal to the known support instead, so the flow covers the full
prior box — recommended whenever the true bounds are known.

### Heavy-tailed marginals (experimental)

```python
# EXPERIMENTAL: peaks-over-threshold Generalized Pareto tails
flow = normalizing_flows_fit(data, tail_model="gpd", tail_quantile=0.05)
```

For heavy-tailed marginals, `tail_model="gpd"` fits a Generalized Pareto
Distribution to each tail via peaks-over-threshold, modeling the extremes more
faithfully than the default Gaussian tail. `tail_quantile` sets the fraction of
samples in each tail (default 0.05). This feature is **experimental** — its API
may change — and it ignores `tail_extension` and `prior_bounds`.

## Requirements

- Python 3.11–3.14
- JAX / jaxlib >=0.4.38,<0.8
- NumPy >=1.26,<3
- SciPy >=1.10 (>=1.11.3 on Python 3.12, the oldest with cp312 wheels)
- Equinox >=0.13.2,<0.14
- jaxtyping >=0.3.6,<0.4
- interpax >=0.3.11,<0.4
- FlowJAX >=17.2.1,<18
- paramax >=0.0.3
- cloudpickle >=2.2.1,<4 (used by `save_flow` / `load_flow`)

Python **3.11 through 3.14** are supported and tested. The lower bounds are the
oldest versions that pass the test suite on 3.11/3.12; on 3.13 and 3.14 a normal
install resolves newer versions (the oldest floors predate cp313/cp314 wheels),
which is what the CI "highest" jobs exercise there. The JAX upper bound is set by
the flowjax 17.2.1 pairing, not by choice — jax 0.7.x is the newest series that
runs cleanly with it while still shipping cp314 wheels. The exact ranges
(including per-Python markers) live in `pyproject.toml` and `pip`/`uv` resolve
them automatically. The example notebooks need a few extra packages — see
[`examples/requirements.txt`](examples/requirements.txt).

## Citation

If you use coppuccino in your research, please cite it:

```bibtex
@software{coppuccino,
  author  = {Johnson, Aaron D.},
  title   = {coppuccino: copula normalizing flows in JAX},
  year    = {2026},
  version = {2.0.0},
  url     = {https://github.com/AaronDJohnson/coppuccino}
}
```

## References

- A. Sklar (1959). *Fonctions de répartition à n dimensions et leurs marges.*
  Publications de l'Institut de Statistique de l'Université de Paris. — Sklar's
  theorem.
- C. Durkan, A. Bekasov, I. Murray, G. Papamakarios (2019).
  *[Neural Spline Flows](https://arxiv.org/abs/1906.04032).* NeurIPS. — The
  rational-quadratic spline transforms used here.
- G. Papamakarios, E. Nalisnick, D. J. Rezende, S. Mohamed, B. Lakshminarayanan
  (2021). *[Normalizing Flows for Probabilistic Modeling and Inference](https://arxiv.org/abs/1912.02762).*
  JMLR. — Review of normalizing flows.
- The flow implementation builds on
  [FlowJAX](https://github.com/danielward27/flowjax).

## License

[MIT](LICENSE) © Aaron D. Johnson
