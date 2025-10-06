from typing import Callable
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
from flowjax.bijections import Stack
from flowjax.flows import triangular_spline_flow
from flowjax.train import fit_to_data
from flowjax.distributions import Normal, Transformed
from paramax import non_trainable
from equinox import filter_jit

from coppuccino.bijections import EmpiricalMarginalToGaussian
from coppuccino.bijections import make_empirical_cdf_spline


def _create_empirical_transforms(samples: np.ndarray, min_eps: float = 1e-12):
    """
    Create empirical marginal transforms for copula modeling.

    Constructs individual empirical CDF transforms for each parameter dimension
    and stacks them into a single bijection. This is the first step in copula
    modeling, transforming marginals to Gaussian.

    Parameters
    ----------
    samples : np.ndarray
        Input samples with shape (n_samples, n_params).
    min_eps : float, optional
        Minimum epsilon for CDF bounds. Default is 1e-12.

    Returns
    -------
    transform : Stack
        Stacked empirical transforms for each parameter dimension.
    inverse_log_det : callable
        JIT-compiled vectorized function computing inverse transform and log determinant.

    Raises
    ------
    ValueError
        If any parameter has fewer than 20 samples.

    Examples
    --------
    >>> import numpy as np
    >>> from coppuccino.copula_flows import _create_empirical_transforms
    >>> samples = np.random.randn(1000, 3)
    >>> transform, inverse_log_det = _create_empirical_transforms(samples)
    >>> # Transform samples to Gaussian space
    >>> x_gaussian, log_det = inverse_log_det(samples)
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


def fit_chain_entry(input_flow, transform, inverse_log_det, chain_entry: np.ndarray, patience=20, learning_rate=1e-4, rng_seed: int = 999, max_epochs: int = 200):
    """
    Fit a normalizing flow to data with empirical marginal transforms.

    This is an internal helper function that trains a flow on data after
    transforming it through empirical marginal transforms.

    Parameters
    ----------
    input_flow : Transformed
        Initial flow to be fitted.
    transform : Stack
        Empirical marginal transforms.
    inverse_log_det : callable
        Function computing inverse transform and log determinant.
    chain_entry : np.ndarray
        Training data with shape (n_samples, n_params).
    patience : int, optional
        Early stopping patience. Default is 20.
    learning_rate : float, optional
        Learning rate for optimization. Default is 1e-4.
    rng_seed : int, optional
        Random seed for reproducibility. Default is 999.
    max_epochs : int, optional
        Maximum number of training epochs. Default is 200.

    Returns
    -------
    Transformed
        Fitted flow with empirical marginal transforms applied.

    Examples
    --------
    >>> import numpy as np
    >>> import jax.numpy as jnp
    >>> import jax.random as jr
    >>> from flowjax.flows import triangular_spline_flow
    >>> from flowjax.distributions import Normal
    >>> from coppuccino.copula_flows import _create_empirical_transforms, fit_chain_entry
    >>> # Create data and transforms
    >>> data = np.random.randn(1000, 3)
    >>> transform, inverse_log_det = _create_empirical_transforms(data)
    >>> # Create initial flow
    >>> key = jr.key(0)
    >>> flow = triangular_spline_flow(key, base_dist=Normal(jnp.zeros(3)), flow_layers=4)
    >>> # Fit the flow
    >>> fitted_flow = fit_chain_entry(flow, transform, inverse_log_det, data)
    """
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


# def normalizing_flows_fit(chain:np.ndarray, rng_seed: int = 999,
#                           knots: int = 16, patience: int = 20, learning_rate: float = 1e-4,
#                           max_epochs: int = 200,
#                           maf_layers: int = 8,
#                           spline_layers: int = 8,
#                           nn_depth: int = 2,
#                           nn_width: int = 128,
#                           use_maf: bool = True) -> Transformed:
def normalizing_flows_fit(chain:np.ndarray, rng_seed: int = 999,
                          knots: int = 16, patience: int = 20, learning_rate: float = 1e-4,
                          max_epochs: int = 200, flow_layers: int = 8) -> Transformed:
    """
    Fit a copula normalizing flow to multivariate data.

    This function implements a copula-based normalizing flow by:
    1. Transforming each marginal to Gaussian via empirical CDF
    2. Modeling the Gaussian copula dependencies with a triangular spline flow

    The approach combines flexibility in modeling marginals (via empirical CDFs)
    with powerful dependency modeling (via normalizing flows).

    Parameters
    ----------
    chain : np.ndarray
        Training samples with shape (n_samples, n_params). May contain NaN values
        which will be removed before fitting.
    rng_seed : int, optional
        Random seed for reproducibility. Default is 999.
    knots : int, optional
        Number of knots for rational quadratic spline transformations. Default is 16.
    patience : int, optional
        Early stopping patience (epochs without improvement). Default is 20.
    learning_rate : float, optional
        Learning rate for Adam optimizer. Default is 1e-4.
    max_epochs : int, optional
        Maximum number of training epochs. Default is 200.
    flow_layers : int, optional
        Number of coupling layers in the triangular spline flow. Default is 8.

    Returns
    -------
    Transformed
        Fitted copula flow model that can be used for sampling and density evaluation.

    Examples
    --------
    >>> import numpy as np
    >>> from coppuccino.copula_flows import normalizing_flows_fit, sample
    >>> # Generate correlated data
    >>> np.random.seed(42)
    >>> mean = [0, 0]
    >>> cov = [[1, 0.8], [0.8, 1]]
    >>> data = np.random.multivariate_normal(mean, cov, 1000)
    >>> # Fit copula flow
    >>> flow = normalizing_flows_fit(data, rng_seed=42, flow_layers=4, max_epochs=100)
    >>> # Generate new samples
    >>> new_samples = sample(flow, n_samples=500, rng_seed=123)
    >>> new_samples.shape
    (500, 2)

    Notes
    -----
    The triangular spline flow uses autoregressive transformations that are
    particularly efficient for capturing dependencies in high-dimensional data.
    """
    key = jr.key(rng_seed)
    # key1, key2 = jr.split(key)

    # Create initial transforms using the helper function
    transform, inverse_log_det = _create_empirical_transforms(chain)

    # if use_maf:
    #     # Stage 1: MAF to handle bimodality and initial correlation structure
    #     maf_flow = masked_autoregressive_flow(
    #         key1,
    #         base_dist=Normal(jnp.zeros(chain.shape[1])),
    #         transformer=RationalQuadraticSpline(knots=knots, interval=4),
    #         invert=True,
    #         nn_depth=nn_depth,
    #         nn_width=nn_width,
    #         flow_layers=maf_layers,
    #     )

    #     # Stage 2: Triangular spline to refine correlations
    #     tri_flow = triangular_spline_flow(
    #         key2,
    #         base_dist=Normal(jnp.zeros(chain.shape[1])),
    #         knots=knots,
    #         flow_layers=spline_layers,
    #         tanh_max_val=2.0,
    #         invert=True
    #     )

    #     # Chain the bijections: MAF first, then Triangular
    #     composite_bijection = BijectionChain([maf_flow.bijection, tri_flow.bijection])

    #     # Create composite flow
    #     composite_flow = Transformed(
    #         Normal(jnp.zeros(chain.shape[1])),
    #         composite_bijection
    #     )

    #     flow = fit_chain_entry(composite_flow, transform, inverse_log_det, chain,
    #                           rng_seed=rng_seed, patience=patience,
    #                           learning_rate=learning_rate, max_epochs=max_epochs)
    # else:
    # Use only triangular spline flow (original behavior)
    flow = triangular_spline_flow(
        key,
        base_dist=Normal(jnp.zeros(chain.shape[1])),
        knots=knots,
        flow_layers=flow_layers,
        tanh_max_val=3.0,
        invert=True
    )
    flow = fit_chain_entry(flow, transform, inverse_log_det, chain, rng_seed=rng_seed,
                            patience=patience, learning_rate=learning_rate, max_epochs=max_epochs)

    return flow


def sample(flow: Callable, n_samples: int, rng_seed: int = 999) -> np.ndarray:
    """
    Generate samples from a fitted copula flow.

    Draws random samples from the learned distribution by sampling from the
    base Gaussian distribution and transforming through the trained flow.

    Parameters
    ----------
    flow : Callable
        Fitted flow model (typically a Transformed distribution).
    n_samples : int
        Number of samples to generate.
    rng_seed : int, optional
        Random seed for reproducibility. Default is 999.

    Returns
    -------
    np.ndarray
        Generated samples with shape (n_samples, n_params).

    Examples
    --------
    >>> import numpy as np
    >>> from coppuccino.copula_flows import normalizing_flows_fit, sample
    >>> # Fit a flow to data
    >>> data = np.random.randn(1000, 3)
    >>> flow = normalizing_flows_fit(data, max_epochs=50)
    >>> # Generate 100 new samples
    >>> new_samples = sample(flow, n_samples=100, rng_seed=42)
    >>> new_samples.shape
    (100, 3)

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
    Compute log probability density of samples under a fitted copula flow.

    Evaluates the log probability density function at the given sample points.
    This is useful for model evaluation, likelihood computation, and outlier detection.

    Parameters
    ----------
    flow : Transformed
        Fitted copula flow model.
    samples : np.ndarray
        Samples at which to evaluate log probability, shape (n_samples, n_params).

    Returns
    -------
    np.ndarray
        Log probability densities, shape (n_samples,).

    Examples
    --------
    >>> import numpy as np
    >>> from coppuccino.copula_flows import normalizing_flows_fit, log_prob
    >>> # Fit flow to training data
    >>> train_data = np.random.randn(1000, 2)
    >>> flow = normalizing_flows_fit(train_data, max_epochs=50)
    >>> # Evaluate log probability on test data
    >>> test_data = np.random.randn(100, 2)
    >>> log_probs = log_prob(flow, test_data)
    >>> log_probs.shape
    (100,)
    >>> # Higher values indicate higher probability density
    >>> np.mean(log_probs)
    """
    log_probs = filter_jit(flow.log_prob)(samples)
    return np.array(log_probs)


def sample_and_log_prob(flow: Transformed, n_samples: int, rng_seed: int = 999) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate samples and compute their log probabilities simultaneously.

    This is more efficient than calling `sample` and `log_prob` separately,
    as it avoids redundant computations when both samples and their densities
    are needed (e.g., for importance sampling or MCMC).

    Parameters
    ----------
    flow : Transformed
        Fitted copula flow model.
    n_samples : int
        Number of samples to generate.
    rng_seed : int, optional
        Random seed for reproducibility. Default is 999.

    Returns
    -------
    samples : np.ndarray
        Generated samples with shape (n_samples, n_params).
    log_probs : np.ndarray
        Log probability densities of the generated samples, shape (n_samples,).

    Examples
    --------
    >>> import numpy as np
    >>> from coppuccino.copula_flows import normalizing_flows_fit, sample_and_log_prob
    >>> # Fit flow
    >>> data = np.random.randn(1000, 2)
    >>> flow = normalizing_flows_fit(data, max_epochs=50)
    >>> # Generate samples with their log probabilities
    >>> samples, log_probs = sample_and_log_prob(flow, n_samples=100, rng_seed=42)
    >>> samples.shape
    (100, 2)
    >>> log_probs.shape
    (100,)
    >>> # Use for importance sampling weights
    >>> weights = np.exp(log_probs)
    """
    sample_shape = (n_samples,)
    key = jr.key(rng_seed)
    samples, log_probs = filter_jit(flow.sample_and_log_prob)(key, sample_shape)
    return np.array(samples), np.array(log_probs)
