# coppuccino

![Coppuccino Logo](image.png)

Fit distributions with normalizing flows + copulas using JAX.

## What is coppuccino?

Coppuccino is a JAX-based library for fitting and sampling from complex multivariate probability distributions using copula normalizing flows. It works in two stages:

1. **Empirical marginal transforms** — each dimension is mapped to a Gaussian via a spline-based empirical CDF
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

### Requirements

- Python >=3.11
- JAX >=0.7.2
- NumPy >=2.3.3
- SciPy >=1.11.0
- Equinox >=0.13.2
- FlowJAX >=17.2.1

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
# For well-calibrated inference, HDR values should be uniform on [0, 1]
```

### Prior bounds (recommended for MCMC chains)

```python
# Extend the empirical CDF to the full prior support
bounds = np.array([[-10, 10], [-5, 5], [0, 100]])
flow = normalizing_flows_fit(data, prior_bounds=bounds)
```
