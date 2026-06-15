import numpy as np
from coppuccino.copula_flows import normalizing_flows_fit
from coppuccino.copula_flows import sample_and_log_prob, log_prob

__all__ = ["compute_injection_hdr", "check_in_support"]


def check_in_support(samples: np.ndarray, injection_params: np.ndarray) -> bool:
    """
    Check if injection parameters are within the support of the samples.

    Parameters
    ----------
    samples : np.ndarray
        Posterior samples from inference, shape (n_samples, n_params).
    injection_params : np.ndarray
        True parameter values to evaluate, shape (n_params,).

    Returns
    -------
    in_support : bool
        True if all injection parameters are within the min/max range of the samples.
        False otherwise.

    Raises
    ------
    ValueError
        If injection_params is not 1D or its length does not match number of parameters in samples.

    Examples
    --------
    >>> import numpy as np
    >>> from coppuccino.hdr import check_in_support
    >>> # Generate posterior samples from inference
    >>> true_params = np.array([1.0, 2.0])
    >>> posterior = np.random.randn(1000, 2) + true_params
    >>> # Check if true parameters are within support of posterior samples
    >>> in_support = check_in_support(posterior, true_params)
    >>> print(in_support)
    True

    >>> # Example where injection is outside support
    >>> out_of_bounds = np.array([10.0, 20.0])
    >>> in_support = check_in_support(posterior, out_of_bounds)
    >>> print(in_support)
    False

    Notes
    -----
    This function checks if each component of the injection parameters lies within
    the minimum and maximum values observed in the corresponding dimension of the samples.
    It is a simple heuristic to identify injections that are clearly outside the range
    of the inferred posterior distribution.
    """
    if injection_params.ndim != 1:
        raise ValueError("injection_params must be a 1D array")
    if injection_params.shape[0] != samples.shape[1]:
        raise ValueError("injection_params length must match number of parameters in samples")

    mins = np.min(samples, axis=0)
    maxs = np.max(samples, axis=0)

    in_support = np.all((injection_params >= mins) & (injection_params <= maxs))
    return in_support


def compute_injection_hdr(samples: np.ndarray, injection_params: np.ndarray, num_samples: int = 100_000, return_flow=False, **nf_kwargs):
    """
    Compute Highest Density Region (HDR) credibility for injection parameters.

    The HDR credibility is the fraction of samples from the fitted distribution
    that have equal or higher probability density than the injection parameters
    (so a point at the mode scores near 0 and a point in the tails near 1).
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
        Any injection lying outside the coordinate-wise support of ``samples``
        is assigned exactly ``1.0`` without evaluating the flow (see Notes).
    flow : Transformed, optional
        Fitted copula flow (only returned if return_flow=True).

    Raises
    ------
    ValueError
        If ``samples`` contain NaN or Inf values, or if ``injection_params``
        is a scalar (it must be at least 1D).

    Examples
    --------
    >>> import numpy as np
    >>> from coppuccino.hdr import compute_injection_hdr
    >>> # Generate posterior samples from inference
    >>> true_params = np.array([1.0, 2.0])
    >>> posterior = np.random.randn(1000, 2) + true_params
    >>> # Compute HDR for true parameters
    >>> hdr = compute_injection_hdr(posterior, true_params, num_samples=50000)
    >>> # hdr[0] is a single value in [0, 1]; for an injection near the
    >>> # posterior peak it is small (close to 0), approaching 1 in the tails.

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

    Injections outside the coordinate-wise min/max support of ``samples`` are a
    special case: ``check_in_support`` short-circuits them to HDR = 1.0 without
    fitting or evaluating the flow, since such points fall in the extreme tail /
    clipped region where the flow density is effectively minimal. When building
    a calibration (P-P) plot, be aware that these 1.0 values are sentinels rather
    than density-based results.

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

    # Ensure injection_params is 2D: (n_injections, n_params)
    if injection_params.ndim == 1:
        injection_params = injection_params[np.newaxis, :]

    flow = normalizing_flows_fit(samples, **nf_kwargs)
    # sample from flow and compute log probability of those samples
    _, gen_log_probs = sample_and_log_prob(flow, n_samples=num_samples)
    hdrs = []

    sorted_gen_log_probs = np.sort(gen_log_probs)

    for injection_param in injection_params:
        # skip if injection is outside support of samples
        if not check_in_support(samples, injection_param):
            hdrs.append(1.0)
            continue
        injection_prob = float(log_prob(flow, injection_param[np.newaxis, :])[0])

        count = num_samples - np.searchsorted(sorted_gen_log_probs, injection_prob, side='right')
        hdrs.append(count / num_samples)

    if return_flow:
        return np.array(hdrs), flow

    return np.array(hdrs)
