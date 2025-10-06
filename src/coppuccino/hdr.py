import numpy as np
from coppuccino.copula_flows import normalizing_flows_fit
from coppuccino.copula_flows import sample_and_log_prob, log_prob


def compute_injection_hdr(samples: np.ndarray, injection_params: np.ndarray, num_samples: int = 100_000, return_flow=False, **nf_kwargs):
    """
    Compute Highest Density Region (HDR) credibility for injection parameters.

    The HDR credibility is the fraction of samples from the fitted distribution
    that have equal or lower probability density than the injection parameters.
    This metric is used to validate Bayesian inference: if the inference is
    well-calibrated, HDR values should be uniformly distributed between 0 and 1.

    Parameters
    ----------
    samples : np.ndarray
        Posterior samples from inference, shape (n_samples, n_params).
    injection_params : np.ndarray
        True parameter values to evaluate. Can be:
        - 1D array of shape (n_params,) for a single injection
        - 2D array of shape (n_injections, n_params) for multiple injections
    num_samples : int, optional
        Number of samples to draw from fitted flow for HDR computation.
        Default is 100,000.
    return_flow : bool, optional
        If True, return the fitted flow along with HDR values. Default is False.
    **nf_kwargs : dict, optional
        Additional keyword arguments passed to `normalizing_flows_fit`.
        Common options include:
        - knots : int, default 4
        - patience : int, default 30
        - learning_rate : float, default 1e-3
        - max_epochs : int, default 400
        - flow_layers : int, default 6

    Returns
    -------
    hdrs : np.ndarray
        HDR credibility values. Shape (1,) for single injection or
        (n_injections,) for multiple injections. Values range from 0 to 1.
    flow : Transformed, optional
        Fitted copula flow (only returned if return_flow=True).

    Raises
    ------
    ValueError
        If injection_params is a scalar (must be at least 1D).

    Examples
    --------
    >>> import numpy as np
    >>> from coppuccino.hdr import compute_injection_hdr
    >>> # Generate posterior samples from inference
    >>> true_params = np.array([1.0, 2.0])
    >>> posterior = np.random.randn(1000, 2) + true_params
    >>> # Compute HDR for true parameters
    >>> hdr = compute_injection_hdr(posterior, true_params, num_samples=50000)
    >>> # For well-calibrated inference, HDR should be ~0.5 on average
    >>> # (true params are at median of posterior)
    >>> hdr[0]

    >>> # Multiple injections
    >>> injections = np.array([[1.0, 2.0], [1.5, 2.5], [0.5, 1.5]])
    >>> hdrs = compute_injection_hdr(posterior, injections, num_samples=50000)
    >>> hdrs.shape
    (3,)

    >>> # Return fitted flow for reuse
    >>> hdrs, flow = compute_injection_hdr(
    ...     posterior, true_params, num_samples=50000, return_flow=True
    ... )

    Notes
    -----
    The HDR credibility is computed as:

        HDR = (# of generated samples with log_prob >= log_prob(injection)) / num_samples

    For well-calibrated Bayesian inference, HDR values should follow a uniform
    distribution U(0, 1). Deviations indicate miscalibration or model misspecification.

    The default normalizing flow parameters are tuned for typical inference problems.
    Adjust them based on your data characteristics:
    - Increase `flow_layers` for more complex dependencies
    - Increase `max_epochs` if training hasn't converged
    - Increase `knots` for more flexible marginal transformations
    """
    if np.any(np.isnan(samples)):
        raise ValueError("samples contain NaNs")
    if np.any(np.isinf(samples)):
        raise ValueError("samples contain Infs")
    if injection_params.ndim == 0:
        raise ValueError("injection_params must be at least 1D")
    default_kwargs = {'knots':4, 'patience':30, 'learning_rate':1e-3, 'max_epochs':400, 'flow_layers':6}
    # default_kwargs = {'knots': 32,
    #                   'patience': 20,
    #                   'learning_rate': 1e-4,
    #                   'max_epochs': 200,
    #                   'maf_layers': 8,
    #                   'spline_layers': 8,
    #                   'nn_depth': 2,
    #                   'nn_width': 128,
    #                   'use_maf': True}
    kwargs = nf_kwargs if nf_kwargs else default_kwargs
    # fit NF to samples
    flow = normalizing_flows_fit(samples, **kwargs)  # TODO: document kwargs in docstring
    # sample from flow and compute log probability of those samples
    _, gen_log_probs = sample_and_log_prob(flow, n_samples=num_samples)
    injection_probs = log_prob(flow, injection_params)

    # Sort once for all searchsorted operations
    sorted_gen_log_probs = np.sort(gen_log_probs)
    count = num_samples - np.searchsorted(sorted_gen_log_probs, injection_probs, side='right')
    hdrs = count / num_samples

    if return_flow:
        return hdrs, flow

    return hdrs
