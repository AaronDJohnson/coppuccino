from typing import Callable
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
from flowjax.bijections import Stack
from flowjax.flows import masked_autoregressive_flow
from flowjax.train import fit_to_data
from flowjax.bijections import RationalQuadraticSpline
from flowjax.distributions import Normal, Transformed
from paramax import non_trainable
from equinox import filter_jit

from coppuccino.bijections import EmpiricalMarginalToGaussian
from coppuccino.bijections import make_empirical_cdf_spline


def _create_empirical_transforms(samples: np.ndarray, min_eps: float = 1e-7):
    """
    Create empirical marginal transforms for a single source's data.

    Parameters
    ----------
    samples : ndarray
        Samples with shape (n_samples, n_params)

    Returns
    -------
    transform : Stack
        Stacked empirical transforms for each parameter
    inverse_log_det : callable
        JIT-compiled inverse log determinant function
    """
    x = samples[~np.isnan(samples).any(axis=1)]
    empirical_transforms = []

    for j in range(x.shape[1]):
        param_samples = x[:, j]
        if len(param_samples) > 20:  # Need sufficient samples for empirical CDF
            _, cdf_fn, quantile_fn, pdf_fn = make_empirical_cdf_spline(param_samples, num_points=200, min_eps=min_eps)
            empirical_transforms.append(non_trainable(EmpiricalMarginalToGaussian(param_samples, cdf_fn, quantile_fn, pdf_fn, min_eps=min_eps)))
        else:
            raise ValueError("Insufficient samples for empirical transform")

    transform = Stack(empirical_transforms)
    inverse_log_det = jax.jit(jax.vmap(transform.inverse_and_log_det))
    # inverse_log_det = jax.vmap(transform.inverse_and_log_det)

    return transform, inverse_log_det


def fit_chain_entry(input_flow, transform, inverse_log_det, chain_entry: np.ndarray, patience=10, learning_rate=1e-3, rng_seed: int = 999, max_epochs: int = 100):
    key = jr.key(rng_seed)
    key, subkey_2 = jr.split(key)
    x_train, __ = inverse_log_det(chain_entry)

    learning_rate = learning_rate
    patience = patience
    epochs = max_epochs

    # Train with adaptive parameters and optional compactness penalty
    kwargs = {
        'max_epochs': epochs,
        'max_patience': patience,
        'learning_rate': learning_rate
    }

    # Standard training
    updated_flow, losses = fit_to_data(subkey_2, input_flow, x_train, **kwargs)

    final_flow = Transformed(updated_flow, transform)  # unstandardize and back to uniform distribution
    return final_flow


def normalizing_flows_fit(chain:np.ndarray, rng_seed: int = 999,
                          knots: int = 16, interval:float = 4, nn_depth=2,
                          patience=10, learning_rate=1e-3, max_epochs=100) -> Transformed:
    """
    Fit a flow to the chain.

    Parameters
    ----------
    chain : ndarray, shape (n_samples, n_sources, n_params_per_source)
        Posterior samples, possibly containing NaNs for missing sources.
    rng_seed : int, optional
        Random seed for reproducibility (default is 999).

    Returns
    -------
    list of Transformed
        List of fitted flows for each chain entry.
    """
    key = jr.key(rng_seed)
    # Create initial transforms using the helper function
    transform, inverse_log_det = _create_empirical_transforms(chain)

    flow = masked_autoregressive_flow(
        key,
        base_dist=Normal(jnp.zeros(chain.shape[1])),
        transformer=RationalQuadraticSpline(knots=knots, interval=interval),
        invert=True,
        nn_depth=nn_depth,
    )
    flow = fit_chain_entry(flow, transform, inverse_log_det, chain, rng_seed=rng_seed,
                           patience=patience, learning_rate=learning_rate, max_epochs=max_epochs)

    return flow


def sample(flow: Callable, n_samples: int, rng_seed: int = 999) -> np.ndarray:
    """
    Generate samples from the fitted flow.

    Parameters
    ----------
    flow : Callable
        The fitted flow.
    n_samples : int
        Number of samples to generate.
    rng_seed : int, optional
        Random seed for reproducibility (default is 999).

    Returns
    -------
    ndarray
        Generated samples with shape (n_samples, n_params).

    Notes
    -----
    This function uses JIT compilation for improved performance. The first call
    will be slower due to compilation overhead (~0.5s), but subsequent calls
    will be much faster (~2-5x speedup).
    """
    sample_shape = (n_samples,)
    key = jr.key(rng_seed)
    # Use filter_jit for improved performance
    samples = filter_jit(flow.sample)(key, sample_shape)
    return np.array(samples)


def log_prob(flow: Transformed, samples: np.ndarray) -> np.ndarray:
    """
    Compute log probabilities of given samples under the fitted flow.

    Parameters
    ----------
    flow : Callable
        The fitted flow.
    samples : ndarray
        Samples for which to compute log probabilities, shape (n_samples, n_params).

    Returns
    -------
    ndarray
        Log probabilities of the samples, shape (n_samples,).
    """
    log_probs = filter_jit(flow.log_prob)(samples)
    return np.array(log_probs)


def sample_and_log_prob(flow: Transformed, n_samples: int, rng_seed: int = 999) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate samples and compute their log probabilities under the fitted flow.

    Parameters
    ----------
    flow : Callable
        The fitted flow.
    n_samples : int
        Number of samples to generate.
    rng_seed : int, optional
        Random seed for reproducibility (default is 999).

    Returns
    -------
    tuple of ndarray
        Tuple containing:
        - Generated samples with shape (n_samples, n_params).
        - Log probabilities of the samples with shape (n_samples,).
    """
    sample_shape = (n_samples,)
    key = jr.key(rng_seed)
    samples, log_probs = filter_jit(flow.sample_and_log_prob)(key, sample_shape)
    return np.array(samples), np.array(log_probs)
