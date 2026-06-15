# coppuccino

![Coppuccino Logo](image.png)

Fit distributions with normalizing flows + copulas using JAX.

## What is coppuccino?

Coppuccino is a JAX-based library for fitting and sampling from complex multivariate probability distributions using copula normalizing flows. It works in two stages:

1. **Empirical marginal transforms** — each dimension is mapped to a Gaussian via a spline-based empirical CDF. By default this uses a monotone rational-quadratic spline (`marginal="rqs"`), whose forward, inverse and derivative are closed-form and mutually exact; pass `marginal="pchip"` for the original PCHIP interpolant.
2. **Normalizing flow** — a triangular spline flow models the dependency structure (copula) in Gaussian space

This approach is particularly well-suited for density estimation on MCMC posterior samples, enabling resampling, density evaluation, and calibration checks (HDR credibility).

## Installation

### Using pip

```bash
pip install coppuccino
```

### From source

```bash
git clone https://github.com/AaronDJohnson/coppuccino.git
cd coppuccino
pip install .
```

### Development

This project uses [uv](https://docs.astral.sh/uv/). After cloning:

```bash
uv sync           # create the env and install coppuccino + dev deps
uv run pytest     # run the test suite
```

### Requirements

- Python >=3.11
- JAX / jaxlib >=0.4.38,<0.8
- NumPy >=1.26,<3
- SciPy >=1.10
- Equinox >=0.13.2,<0.14
- jaxtyping >=0.3.4,<0.3.6
- interpax >=0.3.11,<0.4
- FlowJAX >=17.2.1,<18
- paramax >=0.0.3
- cloudpickle >=2.2.1,<4 (used by `save_flow` / `load_flow`)

These are floored at the oldest versions that pass the test suite. The exact
supported ranges live in `pyproject.toml`; installing with `pip` or `uv`
resolves them automatically. The example notebooks need a few extra packages —
see `examples/requirements.txt`.

## Quick Start

```python
import numpy as np
from coppuccino import normalizing_flows_fit, sample, log_prob, save_flow, load_flow

# Fit a copula flow to multivariate data
data = np.random.randn(5000, 3)
flow = normalizing_flows_fit(data, max_epochs=200)

# Generate new samples from the fitted distribution
new_samples = sample(flow, n_samples=1000, rng_seed=42)

# Evaluate log probability density
log_probs = log_prob(flow, new_samples)

# Save and load models
save_flow(flow, "my_flow.pkl")
loaded_flow = load_flow("my_flow.pkl")
```

### HDR Credibility (Bayesian inference validation)

```python
from coppuccino import compute_injection_hdr

# Check if true parameters are well-recovered by the posterior
posterior_samples = ...  # shape (n_samples, n_params)
true_params = np.array([1.0, 2.0, 3.0])

hdr = compute_injection_hdr(posterior_samples, true_params)
# `hdr` is an array: hdr[0] for a single injection, or one value per row when
# passing a 2D batch of injections. For well-calibrated inference, the HDR
# values should be uniform on [0, 1] across many events.
```

### Prior bounds (recommended for MCMC chains)

```python
# Extend the empirical CDF to the full prior support
bounds = np.array([[-10, 10], [-5, 5], [0, 100]])
flow = normalizing_flows_fit(data, prior_bounds=bounds)
```

### Marginal interpolant

```python
# Default: rational-quadratic spline marginals (exact closed-form inverse)
flow = normalizing_flows_fit(data)                  # marginal="rqs"

# Original PCHIP marginals (for reproducing older fits)
flow = normalizing_flows_fit(data, marginal="pchip")
```

Both families use the same empirical-quantile knots and the same tail and
prior-bound handling, so they are statistically equivalent. `"rqs"` is preferred
because its forward transform, inverse, and density derivative all come from a
single parameterization and are mutually exact to machine precision, which keeps
`sample_and_log_prob` importance weights honest; the `"pchip"` path builds the
CDF and its inverse as two independent splines that are only approximate
inverses of each other.

### Heavy-tailed marginals (experimental)

For marginals with heavy tails, `tail_model="gpd"` fits a Generalized Pareto
Distribution to each tail via peaks-over-threshold, which models the extremes
more faithfully than the default Gaussian tail:

```python
# EXPERIMENTAL: peaks-over-threshold GPD tails for heavy-tailed marginals
flow = normalizing_flows_fit(data, tail_model="gpd", tail_quantile=0.05)
```

`tail_quantile` (default 0.05) sets the fraction of samples in each tail. This
feature is **experimental** — its API may change in a future release — and it
ignores `tail_extension` and `prior_bounds`.
