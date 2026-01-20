"""
Model I/O utilities for saving and loading copula flows.

This module provides functions to serialize and deserialize copula flow models,
including the spline-based empirical CDF transforms that cannot be directly
serialized with standard equinox serialization.
"""

import io
import pickle
from pathlib import Path
from typing import Union

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from flowjax.bijections import Stack
from flowjax.distributions import Normal, Transformed
from flowjax.flows import triangular_spline_flow
import jax.random as jr
from paramax import non_trainable, unwrap

from coppuccino.bijections import EmpiricalMarginalToGaussian, make_empirical_cdf_spline


def _extract_spline_data(flow: Transformed) -> dict:
    """
    Extract spline reconstruction data from a fitted copula flow.

    Parameters
    ----------
    flow : Transformed
        A fitted copula flow with empirical marginal transforms.

    Returns
    -------
    dict
        Dictionary containing all data needed to reconstruct the splines:
        - 'samples': list of sample arrays for each dimension
        - 'min_eps': list of min_eps values for each dimension
        - 'num_dims': number of dimensions
    """
    # The flow structure is: Transformed(inner_flow, Stack([EmpiricalMarginalToGaussian, ...]))
    transform = flow.bijection

    spline_data = {
        'samples': [],
        'min_eps': [],
        'num_dims': 0
    }

    if isinstance(transform, Stack):
        spline_data['num_dims'] = len(transform.bijections)
        for bij in transform.bijections:
            if isinstance(bij, EmpiricalMarginalToGaussian):
                # Unwrap the NonTrainable wrapper to get the actual numpy array
                samples = unwrap(bij.samples)
                spline_data['samples'].append(np.array(samples))
                # min_eps may also be wrapped
                min_eps = unwrap(bij.min_eps)
                spline_data['min_eps'].append(float(min_eps))
            else:
                raise ValueError(f"Unexpected bijection type: {type(bij)}")
    else:
        raise ValueError(f"Expected Stack transform, got: {type(transform)}")

    return spline_data


def _reconstruct_empirical_transforms(spline_data: dict) -> Stack:
    """
    Reconstruct empirical transforms from saved spline data.

    Parameters
    ----------
    spline_data : dict
        Dictionary containing spline reconstruction data.

    Returns
    -------
    Stack
        Stacked empirical transforms for each parameter dimension.
    """
    empirical_transforms = []

    for i in range(spline_data['num_dims']):
        samples = spline_data['samples'][i]
        min_eps = spline_data['min_eps'][i]

        _, cdf_fn, quantile_fn, pdf_fn = make_empirical_cdf_spline(
            samples, num_points=200, min_eps=min_eps
        )

        transform = non_trainable(EmpiricalMarginalToGaussian(
            samples, cdf_fn, quantile_fn, pdf_fn, min_eps=min_eps
        ))
        empirical_transforms.append(transform)

    return Stack(empirical_transforms)


def _get_flow_config(flow: Transformed) -> dict:
    """
    Extract flow configuration from a fitted flow.

    Parameters
    ----------
    flow : Transformed
        A fitted copula flow.

    Returns
    -------
    dict
        Dictionary containing flow configuration parameters.
    """
    # Get the inner flow (before the empirical transform)
    inner_flow = flow.base_dist

    # Extract dimensionality from the base distribution
    if hasattr(inner_flow, 'base_dist'):
        base_dist = inner_flow.base_dist
        if hasattr(base_dist, 'loc'):
            num_dims = len(base_dist.loc)
        else:
            num_dims = base_dist.shape[0] if hasattr(base_dist, 'shape') else 1
    else:
        num_dims = 1

    return {
        'num_dims': num_dims,
    }


def save_flow(flow: Transformed, path: Union[str, Path]) -> None:
    """
    Save a copula flow model to disk.

    This function saves both the flow parameters and the spline data needed
    to reconstruct the empirical CDF transforms. The saved file can be loaded
    with `load_flow`.

    Parameters
    ----------
    flow : Transformed
        A fitted copula flow model from `normalizing_flows_fit`.
    path : str or Path
        Path to save the model. Recommended extension is '.pkl'.

    Examples
    --------
    >>> import numpy as np
    >>> from coppuccino.copula_flows import normalizing_flows_fit
    >>> from coppuccino.model_io import save_flow, load_flow
    >>> # Fit a flow
    >>> data = np.random.randn(1000, 3)
    >>> flow = normalizing_flows_fit(data, max_epochs=50)
    >>> # Save the flow
    >>> save_flow(flow, "my_flow.pkl")
    >>> # Load it back
    >>> loaded_flow = load_flow("my_flow.pkl")
    """
    path = Path(path)

    # Extract spline data for reconstruction
    spline_data = _extract_spline_data(flow)

    # Extract flow configuration
    flow_config = _get_flow_config(flow)

    # Get the inner flow (the actual normalizing flow without the empirical transform)
    inner_flow = flow.base_dist

    # Serialize the inner flow using equinox to a bytes buffer
    inner_flow_buffer = io.BytesIO()
    eqx.tree_serialise_leaves(inner_flow_buffer, inner_flow)
    inner_flow_bytes = inner_flow_buffer.getvalue()

    save_dict = {
        'spline_data': spline_data,
        'flow_config': flow_config,
        'inner_flow_bytes': inner_flow_bytes,
        'inner_flow_like': inner_flow,  # Keep structure for deserialization
    }

    # Use cloudpickle for better compatibility with JAX objects
    try:
        import cloudpickle
        with open(path, 'wb') as f:
            cloudpickle.dump(save_dict, f)
    except ImportError:
        # Fall back to dill if cloudpickle not available
        try:
            import dill
            with open(path, 'wb') as f:
                dill.dump(save_dict, f)
        except ImportError:
            raise ImportError(
                "Either 'cloudpickle' or 'dill' is required for saving flows. "
                "Install with: pip install cloudpickle"
            )


def load_flow(path: Union[str, Path]) -> Transformed:
    """
    Load a copula flow model from disk.

    This function loads a flow that was saved with `save_flow`, reconstructing
    both the flow parameters and the spline-based empirical CDF transforms.

    Parameters
    ----------
    path : str or Path
        Path to the saved model file.

    Returns
    -------
    Transformed
        The loaded copula flow model, ready for sampling and density evaluation.

    Examples
    --------
    >>> from coppuccino.model_io import load_flow
    >>> from coppuccino.copula_flows import sample
    >>> # Load a previously saved flow
    >>> flow = load_flow("my_flow.pkl")
    >>> # Generate samples
    >>> samples = sample(flow, n_samples=100, rng_seed=42)
    """
    path = Path(path)

    # Try cloudpickle first, then dill, then standard pickle
    try:
        import cloudpickle
        with open(path, 'rb') as f:
            save_dict = cloudpickle.load(f)
    except ImportError:
        try:
            import dill
            with open(path, 'rb') as f:
                save_dict = dill.load(f)
        except ImportError:
            with open(path, 'rb') as f:
                save_dict = pickle.load(f)

    spline_data = save_dict['spline_data']

    # Deserialize the inner flow
    inner_flow_bytes = save_dict['inner_flow_bytes']
    inner_flow_like = save_dict['inner_flow_like']

    inner_flow_buffer = io.BytesIO(inner_flow_bytes)
    inner_flow = eqx.tree_deserialise_leaves(inner_flow_buffer, inner_flow_like)

    # Reconstruct the empirical transforms with working splines
    transform = _reconstruct_empirical_transforms(spline_data)

    # Reconstruct the full flow
    loaded_flow = Transformed(inner_flow, transform)

    return loaded_flow
