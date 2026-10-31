import numpy as np
import numpy.typing as npt
from flowjax.distributions import Transformed

from coppuccino.copula_flows import (
    _MIN_SAMPLES_PER_DIM,
    _validate_prior_bounds,
    log_prob,
    normalizing_flows_fit,
)

__all__ = ["compute_injection_hdr", "check_in_support"]


# Each half of a cross-fit split has to clear normalizing_flows_fit's own row
# floor, so the split path refuses inputs the fit would only reject later under
# a message that names neither the split nor the caller's row count.
_MIN_CROSS_FIT_HALF = _MIN_SAMPLES_PER_DIM + 1


def _as_sample_matrix(samples: npt.ArrayLike) -> np.ndarray:
    """Coerce ``samples`` to a float (n_samples, n_params) array."""
    array = np.asarray(samples, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"samples must be (n_samples, n_params); got shape {array.shape}.")
    return array


def _require_finite(array: np.ndarray, name: str) -> None:
    """Raise if ``array`` holds any NaN or Inf, naming the first offender.

    Both arguments are user data that reaches the flow unmodified, and a
    non-finite row is scored as out-of-support rather than rejected, which
    launders it into the ``1.0`` sentinel. Reporting the position and the value
    is what makes the failure traceable back to a row of the caller's table.
    """
    bad = ~np.isfinite(array)
    if not bad.any():
        return
    first = tuple(int(i) for i in np.argwhere(bad)[0])
    location = f"row {first[0]}, column {first[1]}" if len(first) == 2 else f"index {first[0]}"
    raise ValueError(
        f"{name} contains {int(bad.sum())} non-finite value(s); first at "
        f"{location} with value {float(array[first])}."
    )


def _in_support_mask(samples: np.ndarray, injection_params: np.ndarray) -> np.ndarray:
    """
    Batched support test against the coordinate-wise hull of ``samples``.

    Parameters
    ----------
    samples : np.ndarray
        Posterior samples, shape (n_samples, n_params).
    injection_params : np.ndarray
        Injections to test, shape (n_injections, n_params).

    Returns
    -------
    mask : np.ndarray
        Boolean array of shape (n_injections,), True where the row lies within
        the coordinate-wise min/max hull of ``samples``.

    Notes
    -----
    This is the one implementation of the hull test; ``check_in_support`` is a
    single-row wrapper around it. The hull is computed once for the whole batch
    instead of rescanning ``samples`` per injection.
    """
    mins = np.min(samples, axis=0)
    maxs = np.max(samples, axis=0)
    return np.all((injection_params >= mins) & (injection_params <= maxs), axis=1)


def check_in_support(samples: npt.ArrayLike, injection_params: npt.ArrayLike) -> bool:
    """
    Check if injection parameters are within the support of the samples.

    Parameters
    ----------
    samples : array_like
        Posterior samples from inference, shape (n_samples, n_params). Nested
        sequences are accepted and converted.
    injection_params : array_like
        True parameter values to evaluate, shape (n_params,). Nested sequences
        are accepted and converted.

    Returns
    -------
    in_support : bool
        True if all injection parameters are within the min/max range of the samples.
        False otherwise.

    Raises
    ------
    ValueError
        If ``samples`` is not 2D, if ``injection_params`` is not 1D, or if its
        length does not match the number of parameters in ``samples``.

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

    A non-finite injection coordinate is outside every hull, so this returns
    False for it rather than raising; ``compute_injection_hdr`` is the stricter
    of the two and rejects non-finite input outright.
    """
    samples = _as_sample_matrix(samples)
    injection_params = np.asarray(injection_params, dtype=float)
    if injection_params.ndim != 1:
        raise ValueError("injection_params must be a 1D array")
    if injection_params.shape[0] != samples.shape[1]:
        raise ValueError("injection_params length must match number of parameters in samples")

    return bool(_in_support_mask(samples, injection_params[np.newaxis, :])[0])


def _reference_log_probs(
    samples: np.ndarray, rng_seed: int, nf_kwargs: dict
) -> tuple[Transformed, np.ndarray]:
    """Fit the flow on half the rows; score and sort the other half.

    Ranking against the half the fit never saw is what makes the credibility
    unbiased (see the ``compute_injection_hdr`` Notes).
    """
    if len(samples) < 2 * _MIN_CROSS_FIT_HALF:
        raise ValueError(
            "compute_injection_hdr splits samples into a fit half and a reference "
            f"half, so it needs at least {2 * _MIN_CROSS_FIT_HALF} rows; samples "
            f"has {len(samples)}."
        )
    order = np.random.default_rng(rng_seed).permutation(len(samples))
    fit_rows, held_out_rows = np.split(samples[order], [len(samples) // 2])
    flow = normalizing_flows_fit(fit_rows, rng_seed=rng_seed, **nf_kwargs)
    return flow, np.sort(np.asarray(log_prob(flow, held_out_rows), dtype=float))


def compute_injection_hdr(samples: npt.ArrayLike, injection_params: npt.ArrayLike, *, rng_seed: int = 999, return_flow: bool = False, **nf_kwargs):
    """
    Compute Highest Density Region (HDR) credibility for injection parameters.

    The flow is fitted on a random half of ``samples``, and the HDR
    credibility is the fraction of the *other* half that has strictly higher
    probability density, under that flow, than the injection parameters (so a
    point at the mode scores near 0 and a point in the tails near 1). This
    metric is used to validate Bayesian inference: if the inference is
    well-calibrated, HDR values should be uniformly distributed between 0 and
    1.

    Parameters
    ----------
    samples : array_like
        Posterior samples from inference, shape (n_samples, n_params), with
        at least 42 rows.
    injection_params : array_like
        True parameter values to evaluate. Can be:
        - 1D array of shape (n_params,) for a single injection
        - 2D array of shape (n_injections, n_params) for multiple injections
    rng_seed : int, optional
        Keyword-only. Seed for *both* the split into fit and reference halves
        and the flow fit. Default is 999. Vary it to assess the Monte Carlo
        error of the HDR estimate, or to decouple a calibration study from a
        single fixed seed.
    return_flow : bool, optional
        Keyword-only. If True, return the fitted flow along with HDR values.
        Default is False.
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
        (n_injections,) for multiple injections. Values range from 0 to 1 in
        steps of ``1 / (n_samples - n_samples // 2)``, the size of the
        reference half. Any injection lying outside the coordinate-wise
        support of ``samples`` is assigned exactly ``1.0`` without evaluating
        the flow (see Notes).
    flow : Transformed, optional
        The flow fitted on the fit half of ``samples`` (only returned if
        return_flow=True).

    Raises
    ------
    ValueError
        If ``samples`` or ``injection_params`` contain NaN or Inf values, if
        ``samples`` is not 2D or has fewer than 42 rows, if
        ``injection_params`` is a scalar or has more than two dimensions, if
        its parameter count does not match ``samples``, or if a
        ``prior_bounds`` passed through ``nf_kwargs`` is malformed or does not
        contain every row of ``samples``.
    RuntimeError
        If the fitted flow scores the reference half, or any in-support
        injection, non-finitely, which is what a diverged fit looks like (see
        Notes).

    Examples
    --------
    >>> import numpy as np
    >>> from coppuccino.hdr import compute_injection_hdr
    >>> # Generate posterior samples from inference
    >>> true_params = np.array([1.0, 2.0])
    >>> posterior = np.random.randn(1000, 2) + true_params
    >>> # Compute HDR for true parameters
    >>> hdr = compute_injection_hdr(posterior, true_params)
    >>> # hdr[0] is a single value in [0, 1]; for an injection near the
    >>> # posterior peak it is small (close to 0), approaching 1 in the tails.

    >>> # Multiple injections
    >>> injections = np.array([[1.0, 2.0], [1.5, 2.5], [0.5, 1.5]])
    >>> hdrs = compute_injection_hdr(posterior, injections)
    >>> hdrs.shape
    (3,)

    >>> # Return fitted flow for reuse
    >>> hdrs, flow = compute_injection_hdr(posterior, true_params, return_flow=True)

    Notes
    -----
    The HDR credibility is computed as:

        HDR = (# of reference points with log_prob > log_prob(injection))
              / (# of reference points)

    For well-calibrated Bayesian inference, HDR values should follow a uniform
    distribution U(0, 1), and deviations indicate miscalibration. Because the
    reference points are posterior draws the fit never saw, an injection drawn
    from the posterior is exchangeable with them, so its rank is uniform for
    *any* fitted density: error in the fit costs resolution but does not bend
    a P-P plot. Ranking against fresh draws from the fitted flow instead, as
    coppuccino 1.0.0 did, compares draws from two different laws whenever the
    fit is imperfect, and biases HDR upward. On a 1-D standard normal with
    n = 1500 samples that bias gave a mean HDR of 0.569-0.576 over seeds 0-3,
    against 0.499-0.526 for the held-out reference.

    The price is a noisier single call: the flow is fitted on only half the
    rows, and the reference half holds only ``n_samples // 2`` points (one more
    for odd ``n_samples``). That noise averages away over the events of a
    calibration study, and a fit bias does not.

    Injections outside the coordinate-wise min/max hull of ``samples`` (all of
    it, not only the fit half) are a special case: they are short-circuited to
    HDR = 1.0 by convention, using the same test as ``check_in_support`` and
    without evaluating the flow at that point (the flow itself is still
    fitted). This is a sentinel, not an approximation of what the flow would
    have said. The default smooth-tail marginals assign finite, smoothly
    decaying density beyond the data hull, and where a coordinate's largest
    order statistics nearly coincide the fitted marginal packs their mass into
    a very short interval, so the density just past the sample maximum can stay
    comparable to the bulk and such a point would rank near the middle. When
    building a calibration (P-P) plot, treat these exact-1.0 values as
    conservative sentinels rather than density-based results, and count how
    many of them you have.

    A diverged fit (typically an over-large ``learning_rate``) produces NaN
    log-densities, and both sides of the ranking are guarded because NaN fails
    in opposite directions on each. An all-NaN *reference* set leaves every
    injection above nothing at all, returning 1.0 for each -- indistinguishable
    from a column of out-of-support events. A NaN *injection* score does the
    reverse: ``searchsorted`` orders NaN after every finite reference point, so
    the count of more-probable points is zero and the injection is reported at
    HDR 0.0, the value a point at the mode gets. That direction is the more
    dangerous of the two, and neither guard covers the other, so both raise.

    The default normalizing flow parameters are tuned for typical inference problems.
    Adjust them based on your data characteristics:
    - Increase `flow_layers` for more complex dependencies
    - Increase `max_epochs` if training hasn't converged
    - Increase `knots` for a more flexible copula
    """
    samples = _as_sample_matrix(samples)
    injection_params = np.asarray(injection_params, dtype=float)
    _require_finite(samples, "samples")
    if nf_kwargs.get("prior_bounds") is not None:
        # The fit only validates its own half; a row outside the box that the
        # split sends to the reference half would be scored -inf and reported
        # as a diverged fit instead.
        nf_kwargs["prior_bounds"] = _validate_prior_bounds(nf_kwargs["prior_bounds"], samples)
    if injection_params.ndim == 0:
        raise ValueError("injection_params must be at least 1D")
    if injection_params.ndim > 2:
        raise ValueError(f"injection_params must be 1D or 2D; got shape {injection_params.shape}")

    # Ensure injection_params is 2D: (n_injections, n_params)
    if injection_params.ndim == 1:
        injection_params = injection_params[np.newaxis, :]
    if injection_params.shape[1] != samples.shape[1]:
        raise ValueError("injection_params length must match number of parameters in samples")
    # A NaN or Inf injection is outside every hull, so without this it would be
    # laundered into the out-of-support sentinel and read as a valid 1.0.
    _require_finite(injection_params, "injection_params")

    flow, sorted_reference_log_probs = _reference_log_probs(samples, rng_seed, nf_kwargs)
    n_reference = len(sorted_reference_log_probs)
    non_finite_reference = int((~np.isfinite(sorted_reference_log_probs)).sum())
    if non_finite_reference:
        raise RuntimeError(
            f"the fitted flow scored {non_finite_reference} of {n_reference} "
            "reference points non-finitely, so no ranking is possible; the fit "
            "likely diverged. Try a smaller learning_rate or more training "
            "samples."
        )

    # `samples` is fixed for the whole batch, so the support hull is computed
    # once and every in-support injection is scored in a single flow call.
    in_support = _in_support_mask(samples, injection_params)
    hdrs = np.ones(len(injection_params), dtype=float)
    if in_support.any():
        injection_log_probs = np.asarray(log_prob(flow, injection_params[in_support]), dtype=float)
        if np.isnan(injection_log_probs).any():
            raise RuntimeError(
                f"the fitted flow returned NaN log-density for "
                f"{int(np.isnan(injection_log_probs).sum())} of "
                f"{int(in_support.sum())} in-support injections; the fit likely "
                "diverged. Try a smaller learning_rate or more training samples."
            )
        counts = n_reference - np.searchsorted(
            sorted_reference_log_probs, injection_log_probs, side='right'
        )
        hdrs[in_support] = counts / n_reference

    if return_flow:
        return hdrs, flow

    return hdrs
