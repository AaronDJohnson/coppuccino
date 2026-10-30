import warnings
from collections.abc import Sequence
from typing import Any, Callable, NamedTuple, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from equinox import filter_jit
from flowjax.bijections import Stack
from flowjax.distributions import Normal, Transformed
from flowjax.flows import triangular_spline_flow
from flowjax.train import fit_to_data
from paramax import non_trainable, unwrap
from paramax.wrappers import NonTrainable

from coppuccino._stats import wilson_interval
from coppuccino.bijections import (
    EmpiricalMarginalToGaussian,
    SmoothMarginalToGaussian,
    make_empirical_cdf_spline,
    make_smooth_marginal,
)

__all__ = [
    "normalizing_flows_fit", "normalizing_flows_fit_auto",
    "sample", "log_prob", "sample_and_log_prob",
]


# Minimum samples per dimension needed to build a usable empirical CDF.
_MIN_SAMPLES_PER_DIM = 20


def _validate_chain(chain, name: str = "chain") -> np.ndarray:
    """Coerce a training chain to a 2-D float array.

    Parameters
    ----------
    chain : array_like
        Training samples, expected shape ``(n_samples, n_params)``.
    name : str, optional
        Argument name used in the error message.

    Returns
    -------
    np.ndarray
        ``chain`` as a 2-D float array.

    Raises
    ------
    ValueError
        If ``chain`` is not two-dimensional, or has no columns.

    Notes
    -----
    The shape check has to come before anything indexes ``chain``: a 1-D input
    otherwise dies in an ``axis=1`` reduction, and a 3-D input whose trailing
    axes happen to be equal in size is silently flattened and fitted as if it
    were 2-D. A zero-column chain has to be caught here too -- it survives
    every per-column loop untouched and only fails much later, inside
    ``Stack([])``, as a bare ``IndexError: list index out of range``.
    """
    chain = np.asarray(chain, dtype=float)
    if chain.ndim != 2:
        raise ValueError(
            f"{name} must be (n_samples, n_params); got shape {chain.shape}.")
    if chain.shape[1] == 0:
        raise ValueError(
            f"{name} must have at least one parameter (column); got shape "
            f"{chain.shape}.")
    return chain


def _validate_prior_bounds(prior_bounds, chain: np.ndarray) -> np.ndarray:
    """Coerce ``prior_bounds`` to ``(n_params, 2)`` and check the chain fits inside.

    Parameters
    ----------
    prior_bounds : array_like
        Prior support bounds, one ``[low, high]`` row per parameter. Either
        side may be ``-inf``/``+inf``, meaning unbounded there. ``nan`` is
        *not* accepted on either side (see Notes): a missing bound has to be
        written as an infinity.
    chain : np.ndarray
        The cleaned training chain, shape ``(n_samples, n_params)``.

    Returns
    -------
    np.ndarray
        ``prior_bounds`` as an ``(n_params, 2)`` float array.

    Raises
    ------
    ValueError
        If the shape is wrong, a row is not strictly increasing, or any column
        of ``chain`` leaves its row's interval.

    Notes
    -----
    Containment is the check that catches the common
    ``[[low0, low1], [high0, high1]]`` transposition: laid out that way the
    rows usually still satisfy ``low < high``, so an ordering test alone lets
    it through. It also has to be an error rather than a repair, because the
    marginal families disagree about what a too-narrow box means -- the
    default ``"smooth"`` marginal folds every out-of-range row onto the wall
    while the legacy spline families ignore the bound entirely.

    The containment message prints the offending numbers at full ``repr``
    precision rather than ``%g``: the violation that a float round-tripped
    through a config file actually produces is sub-ULP, and six significant
    digits render the two intervals identically.
    """
    pb = np.asarray(prior_bounds, dtype=float)
    n_params = chain.shape[1]
    if pb.shape != (n_params, 2):
        raise ValueError(
            f"prior_bounds must be (n_params, 2) = {(n_params, 2)}; "
            f"got {pb.shape}.")
    # ``~(low < high)`` rather than ``low >= high`` so a NaN bound is rejected
    # too (it would otherwise be demoted to "unbounded on that side").
    misordered = ~(pb[:, 0] < pb[:, 1])
    if misordered.any():
        j = int(np.flatnonzero(misordered)[0])
        raise ValueError(
            f"prior_bounds row {j} must be [low, high] with low < high; "
            f"got [{pb[j, 0]:g}, {pb[j, 1]:g}].")
    if len(chain):
        lows, highs = chain.min(axis=0), chain.max(axis=0)
        outside = (lows < pb[:, 0]) | (highs > pb[:, 1])
        if outside.any():
            j = int(np.flatnonzero(outside)[0])
            overshoot = max(float(pb[j, 0] - lows[j]), float(highs[j] - pb[j, 1]))
            raise ValueError(
                f"chain contains samples outside prior_bounds: parameter "
                f"(column) {j} spans [{float(lows[j])!r}, {float(highs[j])!r}] "
                f"but prior_bounds[{j}] is "
                f"[{float(pb[j, 0])!r}, {float(pb[j, 1])!r}] "
                f"(worst overshoot {overshoot!r}). Widen prior_bounds to the "
                f"chain's own range, or clip the chain into the box before "
                f"fitting.")
    return pb


def _require_positive_int(name: str, value, minimum: int) -> None:
    """Reject a training knob that is not an integer ``>= minimum``."""
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) < minimum):
        raise ValueError(
            f"{name} must be an integer >= {minimum}; got {value!r}.")


def _require_finite_float(name: str, value, minimum: float,
                          strict: bool = True) -> None:
    """Reject a training knob that is not a finite float above ``minimum``."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"{name} must be a finite float; got {value!r}.") from None
    bad = not np.isfinite(number) or (
        number <= minimum if strict else number < minimum)
    if bad:
        comparison = ">" if strict else ">="
        raise ValueError(
            f"{name} must be a finite float {comparison} {minimum:g}; "
            f"got {value!r}.")


def _validate_fit_hyperparameters(
    rng_seed, knots: int, patience: int, learning_rate: float,
    max_epochs: int, flow_layers: int, num_points: int, tanh_max_val: float,
    weight_decay: float,
) -> None:
    """Range-check every training knob of :func:`normalizing_flows_fit`.

    Raises
    ------
    ValueError
        Naming the offending parameter and the value observed.

    Notes
    -----
    Only ``flow_layers`` used to be checked, so the rest failed in three
    different ways: ``knots=0`` reached a ``ZeroDivisionError`` inside flowjax,
    ``rng_seed=1.5`` a jax "PRNG key seed must be an integer", while
    ``max_epochs<=0`` and ``learning_rate=0.0`` were accepted and returned an
    untrained flow that is indistinguishable from a fitted one. A negative
    ``learning_rate`` is the worst of the set: it trains to convergence in the
    wrong direction and every density it reports is finite, so no post-fit
    finiteness check can catch it.

    ``knots >= 2`` is flowjax's own requirement for a rational-quadratic
    spline; ``num_points >= 2`` is the smallest grid that spans an interval
    (the builders then clamp it up to their documented floor).
    """
    _require_positive_int("rng_seed", rng_seed, 0)
    _require_positive_int("knots", knots, 2)
    _require_positive_int("patience", patience, 1)
    _require_positive_int("max_epochs", max_epochs, 1)
    _require_positive_int("num_points", num_points, 2)
    _require_positive_int("flow_layers", flow_layers, 1)
    _require_finite_float("learning_rate", learning_rate, 0.0)
    _require_finite_float("tanh_max_val", tanh_max_val, 0.0)
    _require_finite_float("weight_decay", weight_decay, 0.0, strict=False)


# How far, in nats per dimension, the trained copula's mean log-density on the
# training rows may fall below the *untrained* copula the optimizer started
# from before the fit is called diverged. See _assert_fit_converged for the
# measurement behind the value.
_MAX_INIT_LOGP_DEFICIT_PER_DIM = 3.0


def _copula_mean_log_prob(copula, z_train: jnp.ndarray) -> float:
    """Mean log-density of a copula-space distribution at the training points.

    Parameters
    ----------
    copula : flowjax distribution
        A distribution over copula (Gaussianized) space -- either the trained
        ``flow.base_dist`` or the untrained flow it started from.
    z_train : jnp.ndarray
        Training rows already mapped through the frozen marginal transforms.

    Returns
    -------
    float
        The mean, or NaN if any row scored non-finitely.
    """
    values = np.asarray(filter_jit(copula.log_prob)(z_train), dtype=float)
    if not np.all(np.isfinite(values)):
        return float("nan")
    return float(np.mean(values))


def _assert_fit_converged(flow: Transformed, initial_copula,
                          z_train: jnp.ndarray, learning_rate: float,
                          max_epochs: int) -> None:
    """Fail loudly for a flow whose optimizer diverged.

    Parameters
    ----------
    flow : Transformed
        The just-fitted flow.
    initial_copula : flowjax distribution
        The *untrained* copula ``_fit_chain_entry`` was handed. It is the
        baseline the density check measures against.
    z_train : jnp.ndarray
        Training rows in copula space, i.e. after the frozen marginal maps.
    learning_rate, max_epochs :
        Quoted in the message, because they are what the caller has to change.

    Raises
    ------
    RuntimeError
        If any trained parameter is non-finite, or if the trained copula's
        mean log-density on ``z_train`` is non-finite or falls more than
        ``_MAX_INIT_LOGP_DEFICIT_PER_DIM`` nats per dimension below the
        untrained copula's.

    Notes
    -----
    A diverged fit is silent everywhere downstream: ``sample`` returns all-NaN
    rows, ``log_prob`` returns ``-inf`` on every training row (NaN fails the
    ``<=``/``>=`` clamp comparisons, so the redraw loop does not see it
    either), ``save_flow`` persists it and ``compute_injection_hdr`` scores
    every injection at the out-of-support sentinel.

    Finite parameters are not enough, which is why the density check exists.
    flowjax keeps the best-*validation*-loss parameters and epoch 1 is
    unconditionally accepted (it is trivially the minimum of a one-element
    loss history), so a learning rate that blows up on the very first step is
    frozen there. The parameters stay finite over a wide band of such rates
    while the density is nonsense: measured on 800 correlated Gaussian rows,
    ``max_epochs=30``, ``learning_rate`` from 1 to 100 all returned normally
    under the old parameter-only guard, with mean training log-densities of
    -49 to -2043 against -2.2 for a healthy fit, and 100%-NaN draws from
    ``learning_rate=60`` upward.

    The baseline is the flow's *own untrained initialization* rather than a
    fixed reference density: it costs one extra log-density evaluation, needs
    no architecture-dependent tuning, and asks the one question with an
    unambiguous answer -- did the optimizer make this flow worse than the
    point it started from?

    Measured, all at ``max_epochs=40`` and all in nats per dimension. A
    legitimate grid of 842 fits (9 laws x ``learning_rate`` in 1e-4, 1e-3,
    1e-2, 5e-2 x shapes (400, 2) and (1200, 3) x ``marginal`` "smooth" and
    "rqs" x seeds 0 and 7 x three architectures) produced no positive deficit
    at all: every fit beat its own initialization, the closest by -0.0395. A
    second envelope of 48 fits covering the axes that grid omits
    (``prior_bounds`` including a Beta(0.5, 3) spike on [0, 1], the since-removed
    ``whiten`` option, ``weight_decay``, ``tail_alpha``, ``patience=1``,
    ``marginal="pchip"``, dimensions up to 8, ``n`` from 200 to 8,000,
    ``max_epochs=200``) ran from -2.21 to 0.0 -- the 0.0 being a since-removed
    zero-layer rung, which had nothing to train and so was exactly its own
    baseline.

    Against that, a 252-fit divergence grid over the same 9 laws with
    ``learning_rate`` in 0.5, 1, 3, 10, 30, 60, 100: 18 (all at rate 100)
    fail on parameters alone, and of the 198 remaining fits at a rate of 1 or
    above, the *smallest* deficit is +5.17, with a median of +11.6 at rate 1.
    Rate 0.5 straddles the boundary at -3.27 to +2.61 and is deliberately let
    through: a fit a nat or two worse than its initialization is a
    model-selection question for ``normalizing_flows_fit_auto``, not a
    divergence.

    Three nats per dimension sits in that gap -- 1.7x below the smallest
    divergence measured and 76x above the closest legitimate fit. It is
    chosen from these grids, not derived, and the narrow side is the
    divergence side, so the failure mode to expect is a diverged fit slipping
    through rather than a good one being refused.
    """
    # ``base_dist`` is the trained copula (the outer bijection is the frozen
    # marginal Stack, which is built from data and cannot diverge).
    leaves = [leaf for leaf in jax.tree_util.tree_leaves(flow.base_dist)
              if eqx.is_inexact_array(leaf)]
    if not all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves):
        raise RuntimeError(
            "normalizing_flows_fit: the optimizer diverged -- the trained flow "
            f"has non-finite parameters after max_epochs={max_epochs} at "
            f"learning_rate={learning_rate!r}. Every draw from it is NaN and "
            "every density is -inf. Retry with a smaller learning_rate."
        )

    fitted = _copula_mean_log_prob(flow.base_dist, z_train)
    baseline = _copula_mean_log_prob(initial_copula, z_train)
    n_params = int(z_train.shape[1])
    budget = _MAX_INIT_LOGP_DEFICIT_PER_DIM * n_params
    if np.isfinite(fitted) and fitted >= baseline - budget:
        return
    reported = "non-finite" if not np.isfinite(fitted) else f"{fitted:.4g}"
    raise RuntimeError(
        "normalizing_flows_fit: the optimizer diverged -- the parameters are "
        "finite, but the trained copula's mean log-density on the training "
        f"rows is {reported}, against {baseline:.4g} for the untrained copula "
        f"it started from, after max_epochs={max_epochs} at "
        f"learning_rate={learning_rate!r}. A fit that is worse than its own "
        f"initialization by more than {_MAX_INIT_LOGP_DEFICIT_PER_DIM} nats "
        f"per dimension ({budget:.4g} nats here) has blown up rather than "
        "trained; every density it reports is meaningless. Retry with a "
        "smaller learning_rate."
    )


def _resolve_seed(rng_seed: Optional[int]) -> int:
    """Return ``rng_seed`` if given, else a fresh seed drawn from system entropy.

    Sampling helpers default ``rng_seed=None`` so that repeated calls draw
    independent samples. Pass an explicit integer for reproducible output.
    """
    if rng_seed is None:
        return int(np.random.SeedSequence().generate_state(1)[0])
    return int(rng_seed)


def _create_empirical_transforms(samples: np.ndarray, min_eps: float = 1e-7,
                                  num_points: int = 200, tail_extension: bool = False,
                                  prior_bounds: Optional[np.ndarray] = None,
                                  tail_model: str = "gaussian",
                                  tail_quantile: float = 0.05,
                                  marginal: str = "rqs",
                                  tail_alpha: float = 0.0):
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
        Minimum epsilon for CDF bounds. Default is 1e-7.
    num_points : int, optional
        Number of grid points for each marginal CDF spline. Default is 200.
        Automatically clamped to a sensible range based on sample size.
    tail_extension : bool, optional
        If True, extend the CDF/quantile beyond the observed data range using
        a Gaussian tail model. Default is False (clip to data bounds), which
        is appropriate for MCMC chains bounded by prior support.
    prior_bounds : np.ndarray, optional
        Prior support bounds with shape (n_params, 2), where each row is
        [low, high] for that parameter. When provided, the CDF grid extends
        to the prior edges so the flow can sample the full prior support.
    tail_model : str, optional
        ``"gaussian"`` (default) or the experimental ``"gpd"`` peaks-over-
        threshold model.
    tail_quantile : float, optional
        GPD tail fraction, only used when ``tail_model="gpd"`` (default 0.05).
        See :func:`coppuccino.bijections.make_empirical_cdf_spline` for both.
    marginal : str, optional
        Marginal family for every dimension: ``"smooth"`` (the clip-free
        diffeomorphism built by
        :func:`coppuccino.bijections.make_smooth_marginal`, and what
        ``normalizing_flows_fit`` resolves to by default), or the legacy spline
        families ``"rqs"`` (default here) and ``"pchip"``. See
        :func:`coppuccino.bijections.make_empirical_cdf_spline` for the latter
        two. ``min_eps``, ``tail_extension``, ``tail_model`` and
        ``tail_quantile`` apply only to the legacy families.
    tail_alpha : float, optional
        Tail heaviness on unbounded sides, used only by ``marginal="smooth"``.
        Default 0.0 (Gaussian tails).

    Returns
    -------
    transform : Stack
        Stacked empirical transforms for each parameter dimension.
    inverse_log_det : callable
        JIT-compiled vectorized function computing inverse transform and log determinant.

    Raises
    ------
    ValueError
        If ``samples`` has no columns, if any parameter has 20 or fewer
        non-NaN samples, or if building its marginal fails (the message is
        prefixed with the column index).

    Examples
    --------
    >>> import numpy as np
    >>> from coppuccino.copula_flows import _create_empirical_transforms
    >>> samples = np.random.randn(1000, 3)
    >>> transform, inverse_log_det = _create_empirical_transforms(samples)
    >>> # Transform samples to Gaussian space
    >>> x_gaussian, log_det = inverse_log_det(samples)
    """
    if samples.ndim != 2 or samples.shape[1] == 0:
        # Every per-column loop below is a no-op on a zero-column input, so the
        # first thing that notices is Stack([]) with a bare IndexError.
        raise ValueError(
            f"samples must be (n_samples, n_params) with n_params >= 1; got "
            f"shape {samples.shape}.")
    x = samples[~np.isnan(samples).any(axis=1)]
    empirical_transforms = []

    for j in range(x.shape[1]):
        param_samples = x[:, j]
        if len(param_samples) > _MIN_SAMPLES_PER_DIM:
            # The builders (and scipy underneath them) know nothing about the
            # column they were handed, so a degenerate parameter otherwise
            # surfaces as a bare "needs at least two distinct sample values" /
            # "`x` must contain at least 2 elements" with nothing to bisect on.
            try:
                p_low = float(prior_bounds[j, 0]) if prior_bounds is not None else None
                p_high = float(prior_bounds[j, 1]) if prior_bounds is not None else None
                # A non-finite bound means unbounded on that side (the legacy
                # spline builders would otherwise fail inside scipy with an
                # unrelated-looking error)
                if p_low is not None and not np.isfinite(p_low):
                    p_low = None
                if p_high is not None and not np.isfinite(p_high):
                    p_high = None
                if marginal == "smooth":
                    # clip-free diffeomorphism z in R <-> theta in (lo, hi):
                    # no CDF clip, no quantile clamp (no boundary sample atoms),
                    # no floored wall density (no cliffs); honest analytic tails.
                    theta_of_z, z_of_theta, log_dtheta_dz, x_clamps = \
                        make_smooth_marginal(
                            param_samples, num_points=num_points,
                            prior_low=p_low, prior_high=p_high,
                            tail_alpha=tail_alpha)
                    # mypy folds flowjax's AbstractVar shape/cond_shape into the
                    # synthesised __init__, so it reads these as the first two
                    # positional arguments. See the ClassVar note on the class.
                    empirical_transforms.append(non_trainable(SmoothMarginalToGaussian(  # type: ignore[call-arg]
                        param_samples, theta_of_z, z_of_theta, log_dtheta_dz,
                        num_points=num_points, prior_low=p_low, prior_high=p_high,
                        tail_alpha=tail_alpha,
                        x_clamp_lo=x_clamps[0], x_clamp_hi=x_clamps[1])))
                    continue
                _, cdf_fn, quantile_fn, pdf_fn = make_empirical_cdf_spline(
                    param_samples, num_points=num_points, min_eps=min_eps,
                    tail_extension=tail_extension, prior_low=p_low, prior_high=p_high,
                    tail_model=tail_model, tail_quantile=tail_quantile,
                    marginal=marginal)
                empirical_transforms.append(non_trainable(EmpiricalMarginalToGaussian(  # type: ignore[call-arg]
                    param_samples, cdf_fn, quantile_fn, pdf_fn, min_eps=min_eps,  # type: ignore[arg-type]
                    num_points=num_points, tail_extension=tail_extension,
                    prior_low=p_low, prior_high=p_high,
                    tail_model=tail_model, tail_quantile=tail_quantile,
                    marginal=marginal)))
            except ValueError as exc:
                raise ValueError(f"parameter (column) {j}: {exc}") from exc
        else:
            raise ValueError(
                f"Insufficient samples for empirical transform: parameter "
                f"(column) {j} has {len(param_samples)} non-NaN samples, but at "
                f"least {_MIN_SAMPLES_PER_DIM + 1} are required."
            )

    transform = Stack(empirical_transforms)
    inverse_log_det = jax.jit(jax.vmap(transform.inverse_and_log_det))

    return transform, inverse_log_det


def _fit_chain_entry(
    input_flow: Transformed,
    transform: Stack,
    inverse_log_det: Callable,
    chain_entry: np.ndarray,
    patience: int = 20,
    learning_rate: float = 1e-4,
    rng_seed: int = 999,
    max_epochs: int = 200,
    weight_decay: float = 0.0,
    show_progress: bool = True,
) -> Transformed:
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
    >>> from coppuccino.copula_flows import _create_empirical_transforms, _fit_chain_entry
    >>> # Create data and transforms
    >>> data = np.random.randn(1000, 3)
    >>> transform, inverse_log_det = _create_empirical_transforms(data)
    >>> # Create initial flow
    >>> key = jr.key(0)
    >>> flow = triangular_spline_flow(key, base_dist=Normal(jnp.zeros(3)), flow_layers=4)
    >>> # Fit the flow
    >>> fitted_flow = _fit_chain_entry(flow, transform, inverse_log_det, data)
    """
    key = jr.key(rng_seed)
    x_train, _ = inverse_log_det(chain_entry)

    kwargs: dict[str, Any] = {
        'max_epochs': max_epochs,
        'max_patience': patience,
        'learning_rate': learning_rate,
        'show_progress': show_progress,
    }
    if weight_decay and weight_decay > 0.0:
        # Continuous capacity control: AdamW decays the trainable (conditioner)
        # weights toward zero. In the coupling layers the spline parameters are
        # (conditioner output + identity initialization), so the decay pulls
        # every layer toward the identity map -- a generous architecture then
        # self-limits its effective capacity instead of fitting sampling noise.
        import optax
        kwargs['optimizer'] = optax.adamw(learning_rate, weight_decay=weight_decay)

    updated_flow, _losses = fit_to_data(key, input_flow, x_train, **kwargs)

    # apply the empirical marginal transforms on top of the trained copula
    return Transformed(updated_flow, transform)


def normalizing_flows_fit(chain:np.ndarray, rng_seed: int = 999,
                          knots: int = 4, patience: int = 30, learning_rate: float = 1e-3,
                          max_epochs: int = 400, flow_layers: int = 6,
                          num_points: int = 200, tail_extension: bool = False,
                          tanh_max_val: float = 3.0,
                          prior_bounds: Optional[np.ndarray] = None,
                          tail_model: str = "gaussian",
                          tail_quantile: float = 0.05,
                          marginal: Optional[str] = None,
                          tail_alpha: float = 0.0,
                          weight_decay: float = 0.0,
                          show_progress: bool = True) -> Transformed:
    """
    Fit a copula normalizing flow to multivariate data.

    This function implements a copula-based normalizing flow by:
    1. Transforming each marginal via its empirical CDF
    2. Modeling the remaining dependence structure (the copula) with a flow

    Marginals map to Gaussian space and the copula is a triangular spline flow
    over a normal base -- the strongest choice for heavily correlated
    posteriors (the affine layer captures near-linear dependence). With the
    default ``marginal="smooth"`` the marginal map is a clip-free
    diffeomorphism of all of R onto the support (RQS bulk + analytic tails):
    no boundary sample atoms, no wall density cliffs (though the density *at*
    a bound is not quantitatively faithful -- see ``SmoothMarginalToGaussian``),
    and honest smoothly-decaying densities for outliers / beyond-data queries
    (``tail_alpha`` sets Gaussian vs heavier tails on unbounded sides). The
    legacy clipped marginals remain available as ``marginal="rqs"`` /
    ``"pchip"`` (required for ``tail_extension`` / ``tail_model="gpd"``);
    they can disagree between sample stream and reported density in slivers
    at hard support boundaries, which can absorb independence-MCMC
    walkers.
    The approach combines flexibility in modeling marginals (via empirical CDFs)
    with powerful dependency modeling (via normalizing flows).

    Parameters
    ----------
    chain : np.ndarray
        Training samples with shape (n_samples, n_params). Rows containing
        non-finite values (NaN or +-inf) are dropped, with a warning, before
        fitting. Any other shape raises.
    rng_seed : int, optional
        Random seed for reproducibility. Must be a non-negative integer.
        Default is 999.
    knots : int, optional
        Number of knots for rational quadratic spline transformations. Must be
        an integer >= 2. Default is 4.
    patience : int, optional
        Early stopping patience (epochs without improvement). Must be an
        integer >= 1. Default is 30.
    learning_rate : float, optional
        Learning rate for Adam optimizer. Must be finite and strictly
        positive. Default is 1e-3.
    max_epochs : int, optional
        Maximum number of training epochs. Must be an integer >= 1. Default is
        400.
    flow_layers : int, optional
        Number of coupling layers in the triangular spline flow. Must be an
        integer >= 1. Default is 6.
    num_points : int, optional
        Number of grid points for each marginal CDF spline. Must be an integer
        >= 2. Default is 200. Automatically clamped based on sample size.
        Increase for distributions with fine structure (e.g. many narrow modes).
    tail_extension : bool, optional
        If True, extend the marginal CDF beyond the observed data range using
        a Gaussian tail model. This allows the flow to generate samples beyond
        the training range. Default is False, which is appropriate for MCMC
        chains bounded by prior support.
    tanh_max_val : float, optional
        Maximum value for the tanh clipping in the spline flow base space.
        Must be finite and strictly positive.
        Default is 3.0. Increase (e.g. 4.0–5.0) for heavy-tailed data where
        tail dependencies matter. The copula transform is *linear* beyond
        this many sigmas in latent space, so it is also the ceiling on
        representable tail dependence.
    prior_bounds : np.ndarray, optional
        Prior support bounds with shape (n_params, 2), where each row is
        [low, high] for that parameter. When provided, the empirical CDF grid
        extends to the prior edges, allowing the flow to sample the full prior
        support rather than being limited to the most extreme training sample.
        Recommended for MCMC chains. Either side may be ``-inf``/``+inf``,
        meaning unbounded there; ``nan`` is rejected rather than read as
        "unbounded". Validated: the shape must be exactly (n_params, 2),
        every row must satisfy ``low < high``, and the (cleaned) chain must lie
        inside the box -- a box narrower than the data raises rather than
        being silently folded onto the wall or ignored.

        .. versionchanged:: 3.0
            Malformed or too-narrow ``prior_bounds`` now raise ValueError.
            Previously a narrower-than-data box was accepted and handled
            differently by each marginal family.
    tail_model : str, optional
        Marginal tail model: ``"gaussian"`` (default) or ``"gpd"``.
        ``"gpd"`` is **experimental**: it fits a Generalized Pareto Distribution
        via peaks-over-threshold to each marginal's tails, which is appropriate
        for heavy-tailed data but ignores ``tail_extension`` and ``prior_bounds``.
        See :func:`coppuccino.bijections.make_empirical_cdf_spline`.
    tail_quantile : float, optional
        Fraction of samples in each GPD tail. Only used when
        ``tail_model="gpd"``. Default is 0.05 (5%).
    marginal : str, optional
        Marginal transform family: ``"smooth"``, ``"rqs"``, or ``"pchip"``.
        ``"smooth"`` (the gaussian pipeline's default) is a single clip-free
        monotone map of R onto the support with analytic tails. ``"rqs"``
        (the legacy gaussian bulk) is a monotone rational-quadratic spline
        with a closed-form,
        machine-precision inverse from a single parameterization.
        ``"pchip"`` builds the CDF and its inverse as two independent splines
        that are only approximate inverses of each other; it is retained for
        backward compatibility and reproducing older fits.

        Default ``None`` resolves to ``"smooth"`` (clip-free smooth-tail
        diffeomorphism; see
        :func:`coppuccino.bijections.make_smooth_marginal`), except when a
        legacy tail model is requested (``tail_extension=True`` or
        ``tail_model="gpd"``), which pre-date the smooth marginal and resolve
        to ``"rqs"``. ``tail_alpha`` applies only to ``"smooth"``.
    tail_alpha : float, optional
        Tail heaviness for ``marginal="smooth"`` on *unbounded* sides (no
        prior bound given): 0.0 (default) gives Gaussian tails; larger values
        give progressively heavier tails. Bounded sides always use the
        exponential wall-approach tails. Ignored by other marginals.
    weight_decay : float, optional
        AdamW weight decay on the trainable flow parameters. Must be finite
        and non-negative. Default 0.0
        (plain Adam; original behavior). In the coupling layers the spline
        parameters are (conditioner output + identity initialization), so the
        decay pulls layers toward the identity map. Note: measured on a
        typical MCMC posterior, this is a *weak* regularizer -- it does not
        make an over-parameterized architecture match a well-chosen small one
        (choosing few ``knots`` imposes smoothness structurally and works
        better; decay >= ~10 over-regularizes). Treat it as a supplementary
        knob, not a substitute for architecture choice.
    show_progress : bool, optional
        Show the training progress bar. Default True. Set False for quiet
        fits (e.g. candidate fits inside ``normalizing_flows_fit_auto``).

    Returns
    -------
    Transformed
        Fitted copula flow model that can be used for sampling and density evaluation.

    Raises
    ------
    ValueError
        If ``chain`` is not a non-empty 2-D array, if ``prior_bounds`` is
        malformed or narrower than the chain, if ``marginal``/``tail_model``
        name an unknown family, or if any training knob is out of range
        (``rng_seed >= 0``, ``knots >= 2``, ``patience >= 1``,
        ``max_epochs >= 1``, ``num_points >= 2``, ``flow_layers >= 1``,
        ``learning_rate > 0``, ``tanh_max_val > 0``, ``weight_decay >= 0``).
    RuntimeError
        If the optimizer diverged. Two forms: the trained flow ends up with
        non-finite parameters (it then samples nothing but NaN and scores
        every point at ``-inf``), or its parameters are finite while its mean
        log-density on the training rows collapsed relative to the untrained
        flow it started from. Nothing downstream notices either, so both are
        refused here rather than returned. Raising is not a verdict on fit
        *quality*: the density budget is deliberately loose enough that a
        merely mediocre fit passes and is left to
        :func:`normalizing_flows_fit_auto` to rank.

        .. versionadded:: 3.0
            Out-of-range training knobs and diverged fits used to be accepted
            silently.

        .. versionchanged:: 3.0
            The check was parameter-finiteness only, which left a wide band of
            learning rates -- 1 to 100 on 800 rows of correlated Gaussian data
            at ``max_epochs=30`` -- returning a finite-parameter flow whose
            every density was nonsense.

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
    >>> # With prior bounds (recommended for MCMC chains)
    >>> bounds = np.array([[-5, 5], [-5, 5]])
    >>> flow = normalizing_flows_fit(data, prior_bounds=bounds, max_epochs=100)

    Notes
    -----
    The triangular spline flow uses autoregressive transformations that are
    particularly efficient for capturing dependencies in high-dimensional data.
    """
    if tail_model not in ("gaussian", "gpd"):
        raise ValueError(
            f"tail_model must be 'gaussian' or 'gpd', got {tail_model!r}")
    if marginal is None:
        # The clip-free smooth-tail marginal is the default. The legacy tail
        # models pre-date it and still need the legacy spline family, so
        # requesting them resolves to it (old code keeps working).
        marginal = "rqs" if (tail_extension or tail_model == "gpd") else "smooth"
        if marginal != "smooth" and tail_alpha != 0.0:
            # Requesting the same combination explicitly raises below; the
            # default path silently resolved away the only consumer of
            # tail_alpha, so say so rather than dropping it.
            warnings.warn(
                f"tail_alpha={tail_alpha!r} is ignored: tail_extension=True / "
                "tail_model='gpd' resolved the default marginal to 'rqs', and "
                "tail_alpha applies only to marginal='smooth'.",
                stacklevel=2,
            )
    if marginal not in ("pchip", "rqs", "smooth"):
        raise ValueError(
            f"marginal must be 'pchip', 'rqs' or 'smooth', got {marginal!r}")

    _validate_fit_hyperparameters(
        rng_seed, knots, patience, learning_rate, max_epochs, flow_layers,
        num_points, tanh_max_val, weight_decay)

    if marginal == "smooth" and (tail_extension or tail_model == "gpd"):
        raise ValueError(
            "marginal='smooth' models tails natively (see tail_alpha) and "
            "is incompatible with tail_extension=True / tail_model='gpd'. "
            "Use the legacy marginal='rqs' or 'pchip' for those code paths."
        )

    key = jr.key(rng_seed)

    chain = _validate_chain(chain)

    # Filter non-finite rows once so both transform creation and training use
    # clean data. Infinities have to go too: they reach scipy as a bare "`y`
    # must contain only finite values", and under marginal="pchip" they do not
    # raise at all -- the fit returns a flow that samples nothing but NaN.
    bad_rows = ~np.isfinite(chain).all(axis=1)
    n_dropped = int(bad_rows.sum())
    if n_dropped:
        warnings.warn(
            f"normalizing_flows_fit dropped {n_dropped} of {len(chain)} rows "
            f"containing non-finite values; fitting on the remaining "
            f"{len(chain) - n_dropped}.",
            stacklevel=2,
        )
    clean_chain = chain[~bad_rows]

    if prior_bounds is not None:
        prior_bounds = _validate_prior_bounds(prior_bounds, clean_chain)

    # Create initial transforms using the helper function
    transform, inverse_log_det = _create_empirical_transforms(
        clean_chain, num_points=num_points, tail_extension=tail_extension,
        prior_bounds=prior_bounds,
        tail_model=tail_model, tail_quantile=tail_quantile,
        marginal=marginal, tail_alpha=tail_alpha)

    # The training chain in copula space. inverse_log_det is the jitted
    # marginal map; _fit_chain_entry calls the same compiled function, so this
    # is a cache hit rather than a second trace. The post-fit divergence check
    # needs it.
    z_train, _ = inverse_log_det(clean_chain)
    z_train = jnp.asarray(z_train)

    n_params = clean_chain.shape[1]
    # Gaussianized copula: triangular spline flow on unbounded space.
    flow = triangular_spline_flow(
        key,
        base_dist=Normal(jnp.zeros(n_params)),
        knots=knots,
        flow_layers=flow_layers,
        tanh_max_val=tanh_max_val,
        invert=True
    )

    # Keep the untrained copula: it is the baseline the divergence check
    # compares the fitted density against.
    initial_copula = flow

    flow = _fit_chain_entry(flow, transform, inverse_log_det, clean_chain, rng_seed=rng_seed,
                            patience=patience, learning_rate=learning_rate, max_epochs=max_epochs,
                            weight_decay=weight_decay, show_progress=show_progress)

    _assert_fit_converged(flow, initial_copula, z_train, learning_rate, max_epochs)
    return flow


def _is_frozen(leaf) -> bool:
    """True for a ``paramax.NonTrainable`` wrapper (an optimizer-excluded leaf)."""
    return isinstance(leaf, NonTrainable)


def _n_trainable_params(flow: Transformed) -> int:
    """Total scalar parameter count of the trained (inner) flow.

    The frozen marginal transforms are excluded (identical role across
    candidates); this is the model-size measure used by the
    ``"smallest_comparable"`` selection rule.

    ``paramax.NonTrainable`` subtrees are excluded too: flowjax's training loop
    partitions them out of the optimizer, so they carry no fitted degrees of
    freedom and must not count toward "model size".
    """
    leaves = jax.tree_util.tree_leaves(flow.base_dist, is_leaf=_is_frozen)
    return int(sum(leaf.size for leaf in leaves
                   if not _is_frozen(leaf) and eqx.is_inexact_array(leaf)))


def _normalise_marginal_bin_edges(
    bin_edges: Sequence[Optional[np.ndarray]], n_dims: int,
) -> list[Optional[np.ndarray]]:
    """Validate shared plotting edges for marginal-fidelity scoring."""
    if len(bin_edges) != n_dims:
        raise ValueError(
            f"marginal_bin_edges has {len(bin_edges)} dimensions, but the "
            f"chain has {n_dims}."
        )
    normalised: list[Optional[np.ndarray]] = []
    for dim, edge in enumerate(bin_edges):
        if edge is None:
            normalised.append(None)
            continue
        edge = np.asarray(edge, dtype=float)
        if edge.ndim != 1 or len(edge) < 3:
            raise ValueError(
                f"marginal_bin_edges[{dim}] must be a 1D array with at "
                "least 3 edges, or None."
            )
        if not np.all(np.isfinite(edge)) or not np.all(np.diff(edge) > 0.0):
            raise ValueError(
                f"marginal_bin_edges[{dim}] must be finite and strictly increasing."
            )
        normalised.append(edge)
    return normalised


def _normalise_marginal_ess(
    marginal_ess: Optional[
        float | Sequence[float] | Sequence[Sequence[float]]
    ],
    n_dims: int,
    n_reference: int,
) -> np.ndarray:
    """Return an effective size for each dimension and literal edge cell."""
    if marginal_ess is None:
        return np.full((n_dims, 4), float(n_reference))
    ess = np.asarray(marginal_ess, dtype=float)
    if ess.ndim == 0:
        ess = np.full((n_dims, 4), float(ess))
    elif ess.shape == (n_dims,):
        ess = np.repeat(ess[:, None], 4, axis=1)
    if (ess.shape != (n_dims, 4) or not np.all(np.isfinite(ess))
            or np.any(ess <= 0.0) or np.any(ess > n_reference)):
        raise ValueError(
            "marginal_ess must be a positive scalar, one value per dimension, "
            "or a (n_dimensions, 4) array in underflow/lower-bin/upper-bin/"
            f"overflow order; every value must be <= the reference size ({n_reference})."
        )
    return ess


def _edge_fidelity_features(
    reference: np.ndarray,
    bin_edges: list[Optional[np.ndarray]],
    effective_sizes: np.ndarray,
    min_count: int,
) -> tuple[list[tuple[int, str, float, float, bool, float]], list[str]]:
    """Build literal edge-bin and under/overflow probability targets."""
    features: list[tuple[int, str, float, float, bool, float]] = []
    unresolved: list[str] = []
    for dim, edge in enumerate(bin_edges):
        if edge is None:
            continue
        values = reference[:, dim]
        specifications = [
            ("underflow", float(edge[0]), float(np.mean(values < edge[0]))),
            ("lower_bin", float(edge[1]),
             float(np.mean((values >= edge[0]) & (values < edge[1])))),
            ("upper_bin", float(edge[-2]),
             float(np.mean((values >= edge[-2]) & (values <= edge[-1])))),
            ("overflow", float(edge[-1]), float(np.mean(values > edge[-1]))),
        ]
        for cell, (kind, boundary, probability) in enumerate(specifications):
            reference_size = float(effective_sizes[dim, cell])
            effective_count = probability * reference_size
            resolved = effective_count >= min_count
            features.append((dim, kind, boundary, probability, resolved,
                             reference_size))
            if not resolved:
                unresolved.append(
                    f"dimension {dim} {kind} ({effective_count:.1f} effective)"
                )
    if not features:
        raise ValueError(
            "marginal_bin_edges contains only None entries; select at least one dimension."
        )
    return features, unresolved


def _interval_gap(
    target: float,
    target_size: float,
    estimate: float,
    estimate_size: float,
    confidence: float,
) -> float:
    """Directional separation between two Wilson intervals."""
    target_low, target_high = wilson_interval(target, target_size, confidence)
    estimate_low, estimate_high = wilson_interval(
        estimate, estimate_size, confidence)
    return float(max(0.0, float(estimate_low) - float(target_high),
                     float(target_low) - float(estimate_high)))


def _score_edge_fidelity(
    draws: np.ndarray,
    bin_edges: list[Optional[np.ndarray]],
    features: list[tuple[int, str, float, float, bool, float]],
    confidence: float,
) -> tuple[float, float, float, list[dict]]:
    """Return raw and resolution-aware literal edge-mass errors.

    Resolved literal bins use raw probability error. For a sparse cell, the
    selection score counts only discrepancy beyond the reference and candidate
    Wilson intervals. Thus a zero-count target cannot hide substantial
    candidate overflow, while 1--7-count noise does not choose an architecture.
    """
    if not np.all(np.isfinite(draws)):
        return np.inf, np.inf, np.inf, []
    errors, selection_errors = [], []
    details = []
    for dim, kind, boundary, target, resolved, reference_size in features:
        edge = bin_edges[dim]
        assert edge is not None
        values = draws[:, dim]
        if kind == "underflow":
            estimate = np.mean(values < boundary)
        elif kind == "lower_bin":
            estimate = np.mean((values >= edge[0]) & (values < boundary))
        elif kind == "upper_bin":
            estimate = np.mean((values >= boundary) & (values <= edge[-1]))
        else:
            estimate = np.mean(values > boundary)
        error = abs(float(estimate) - target)
        errors.append(error)
        interval_gap = _interval_gap(
            target, reference_size, float(estimate), len(draws), confidence)
        selection_error = error if resolved else interval_gap
        selection_errors.append(selection_error)
        details.append({
            "dimension": dim,
            "component": kind,
            "target": target,
            "estimate": float(estimate),
            "error": error,
            "selection_error": selection_error,
            "interval_gap": interval_gap,
            "resolved": resolved,
            "reference_ess": reference_size,
            "reference_effective_count": target * reference_size,
        })
    return (float(np.max(errors)), float(np.mean(errors)),
            float(np.max(selection_errors)), details)


def _score_latent_marginal_fidelity(
    flow: Transformed,
    quantiles: np.ndarray,
    n_samples: int,
    rng_seed: int,
) -> tuple[float, float, float, list[dict]]:
    """Score drift from the marginal law expected by the outer PIT map."""
    from scipy.special import ndtri

    if not isinstance(flow.bijection, Stack):
        raise ValueError("Expected a stacked empirical marginal transform.")
    marginal_types = [type(unwrap(b)) for b in flow.bijection.bijections]
    gaussian_types = (SmoothMarginalToGaussian, EmpiricalMarginalToGaussian)
    if not all(issubclass(kind, gaussian_types) for kind in marginal_types):
        raise ValueError(
            "Marginal-fidelity selection requires a Gaussian-space marginal "
            "transform in every dimension."
        )
    thresholds = ndtri(quantiles)
    # jitted like every other sampling site: filter_jit caches per flow object,
    # so each candidate pays one compile and the audit calls reuse it.
    draws = np.asarray(
        filter_jit(flow.base_dist.sample)(jr.key(rng_seed), (n_samples,)))
    if not np.all(np.isfinite(draws)):
        return np.inf, np.inf, np.inf, []
    estimates = np.mean(draws[:, :, None] < thresholds[None, None, :], axis=0)
    errors = np.abs(estimates - quantiles[None, :])
    maximum = float(np.max(errors))
    details = [
        {
            "dimension": dim,
            "component": f"cdf_{quantile:g}",
            "target": float(quantile),
            "estimate": float(estimates[dim, quantile_index]),
            "error": float(errors[dim, quantile_index]),
            "selection_error": float(errors[dim, quantile_index]),
            "interval_gap": float(errors[dim, quantile_index]),
            "resolved": True,
            "reference_ess": np.inf,
            "reference_effective_count": np.inf,
        }
        for dim in range(draws.shape[1])
        for quantile_index, quantile in enumerate(quantiles)
    ]
    return maximum, float(np.mean(errors)), maximum, details


class _CandidateResult(NamedTuple):
    """One candidate configuration's fitted flow and scores.

    ``flow`` is None and ``fit_error`` is set when the candidate raised; the
    remaining fields then carry NaN (or +inf under marginal-fidelity
    selection, so a failed candidate never wins the minimum).
    """

    flow: Optional[Transformed]
    val_log_prob: np.ndarray
    marginal_error: float
    marginal_mean_error: float
    marginal_selection_error: float
    marginal_details: list[dict]
    fit_error: Optional[str]


def _tie_band(marginal_samples: int, marginal_tolerance: Optional[float]) -> float:
    """Monte Carlo tie width for marginal-fidelity comparisons.

    Parameters
    ----------
    marginal_samples : int
        Draws used to estimate each candidate's marginal probability masses.
    marginal_tolerance : float or None
        Explicit override; ``None`` selects the default heuristic.

    Returns
    -------
    float
        The tie half-width, ``0.5 / sqrt(marginal_samples)`` by default.

    Notes
    -----
    Roughly half the standard error of a probability-mass estimate at
    ``p = 0.5``. It is a heuristic, not a simultaneous confidence interval
    over all candidates and components -- hence the override. Selection and
    the refit audit must use the same band, which is why this is a function
    rather than two copies of the expression.
    """
    if marginal_tolerance is None:
        return float(0.5 / np.sqrt(marginal_samples))
    return float(marginal_tolerance)


def _validate_marginal_options(
    marginal_samples: int,
    min_marginal_count: int,
    marginal_confidence: float,
    marginal_tolerance: Optional[float],
    max_marginal_error: Optional[float],
) -> None:
    """Range-check the scalar ``selection="marginal_fidelity"`` options.

    Raises
    ------
    ValueError
        With a message naming the offending option.
    """
    if (not isinstance(marginal_samples, (int, np.integer))
            or isinstance(marginal_samples, (bool, np.bool_))
            or marginal_samples < 1000):
        raise ValueError("marginal_samples must be at least 1000.")
    if (not isinstance(min_marginal_count, (int, np.integer))
            or isinstance(min_marginal_count, (bool, np.bool_))
            or min_marginal_count < 1):
        raise ValueError("min_marginal_count must be at least 1.")
    if not 0.0 < marginal_confidence < 1.0:
        raise ValueError("marginal_confidence must be strictly between 0 and 1.")
    if (marginal_tolerance is not None
            and (not np.isfinite(marginal_tolerance)
                 or marginal_tolerance < 0.0)):
        raise ValueError(
            "marginal_tolerance must be finite and non-negative or None.")
    if (max_marginal_error is not None
            and (not np.isfinite(max_marginal_error)
                 or not 0.0 <= max_marginal_error <= 1.0)):
        raise ValueError(
            "max_marginal_error must be finite and between 0 and 1, or None.")


#: Every ``return_table`` row carries every one of these keys, in this order.
#: Presence used to depend on the selection mode and on whether the row was the
#: selected one, so ``row["refit_used"]`` was a KeyError on most rows and a
#: DataFrame built from the table had ragged columns.
_SELECTION_TABLE_KEYS: tuple[str, ...] = (
    "config", "val_logp", "sem", "n_params", "comparable", "selected",
    "fit_status", "fit_error",
    "marginal_error", "marginal_mean_error", "marginal_selection_error",
    "marginal_comparable", "marginal_within_budget", "marginal_components",
    "marginal_unresolved_components", "marginal_metric_source",
    "marginal_worst_dimension", "marginal_worst_component", "marginal_details",
    "refit_used", "refit_density_finite", "refit_error",
    "refit_joint_sanity_passed", "refit_regressed_components",
    "audit_marginal_error", "audit_marginal_mean_error",
    "audit_marginal_selection_error", "audit_marginal_within_budget",
    "audit_marginal_details",
    "refit_marginal_error", "refit_marginal_mean_error",
    "refit_marginal_selection_error", "refit_marginal_within_budget",
    "refit_marginal_details",
)


def _blank_selection_row() -> dict:
    """A table row with every documented key at its not-applicable default.

    ``None`` means "not measured in this configuration"; the collection-valued
    keys default to an empty list so a caller can iterate them unconditionally.
    """
    row: dict = dict.fromkeys(_SELECTION_TABLE_KEYS)
    row["refit_used"] = False
    for key in ("marginal_details", "audit_marginal_details",
                "refit_marginal_details", "refit_regressed_components"):
        row[key] = []
    return row


class _ScoredCandidates(NamedTuple):
    """Per-candidate arrays derived from the candidate fits, one entry each."""

    flows: list[Optional[Transformed]]
    val_log_probs: list[np.ndarray]
    scores: np.ndarray
    sems: np.ndarray
    sizes: np.ndarray
    finite: np.ndarray
    fit_errors: list[Optional[str]]
    marginal_errors: np.ndarray
    marginal_mean_errors: np.ndarray
    marginal_selection_errors: np.ndarray
    marginal_details: list[list[dict]]


class _Selection(NamedTuple):
    """The chosen candidate index and the comparability masks behind it."""

    # Named ``chosen`` rather than ``index``: a NamedTuple field called
    # ``index`` shadows ``tuple.index`` on every instance.
    chosen: int
    comparable: np.ndarray
    marginal_comparable: np.ndarray
    marginal_within_budget: np.ndarray


def _score_candidates(results: list[_CandidateResult]) -> _ScoredCandidates:
    """Transpose the per-candidate results into the arrays selection reads."""
    flows = [r.flow for r in results]
    lps = [r.val_log_prob for r in results]
    failed_size = np.iinfo(np.int64).max
    return _ScoredCandidates(
        flows=flows,
        val_log_probs=lps,
        scores=np.array([float(np.mean(lp)) for lp in lps]),
        sems=np.array([float(np.std(lp) / np.sqrt(len(lp))) for lp in lps]),
        sizes=np.array([
            _n_trainable_params(fl) if fl is not None else failed_size
            for fl in flows
        ]),
        finite=np.asarray([
            fl is not None and bool(np.all(np.isfinite(lp)))
            for fl, lp in zip(flows, lps)
        ]),
        fit_errors=[r.fit_error for r in results],
        marginal_errors=np.asarray([r.marginal_error for r in results]),
        marginal_mean_errors=np.asarray(
            [r.marginal_mean_error for r in results]),
        marginal_selection_errors=np.asarray(
            [r.marginal_selection_error for r in results]),
        marginal_details=[r.marginal_details for r in results],
    )


def _paired_comparability(scored: _ScoredCandidates, best_idx: int,
                          comparable_sems: float,
                          comparable_nats: float) -> np.ndarray:
    """Which candidates are statistically comparable to the best one.

    A *paired* test on the shared validation points: the per-point score
    differences are highly correlated across candidates, so this is far
    sharper than comparing the individual ``+-sem`` error bars.
    """
    comparable = np.zeros(len(scored.flows), dtype=bool)
    for i, lp in enumerate(scored.val_log_probs):
        if not scored.finite[i]:
            continue
        diff = scored.val_log_probs[best_idx] - lp
        sem_diff = float(np.std(diff) / np.sqrt(len(diff)))
        comparable[i] = float(np.mean(diff)) <= (
            comparable_sems * sem_diff + comparable_nats
        )
    return comparable


def _select_candidate(scored: _ScoredCandidates, selection: str,
                      comparable_sems: float, comparable_nats: float,
                      max_marginal_error: Optional[float],
                      tie_band: float) -> _Selection:
    """Apply the selection rule and report the masks it used.

    Parameters
    ----------
    scored : _ScoredCandidates
        The per-candidate arrays from :func:`_score_candidates`.
    selection : str
        ``"best"``, ``"smallest_comparable"`` or ``"marginal_fidelity"``.
    comparable_sems, comparable_nats : float
        Held-out comparability tolerance (see ``normalizing_flows_fit_auto``).
    max_marginal_error : float or None
        Absolute raw edge-mass budget, marginal-fidelity selection only.
    tie_band : float
        Monte Carlo tie half-width from :func:`_tie_band`.

    Returns
    -------
    _Selection
        The winning index plus the ``comparable`` / ``marginal_comparable`` /
        ``marginal_within_budget`` masks, which the table reports verbatim.

    Raises
    ------
    RuntimeError
        If no held-out-comparable candidate has a finite marginal score.
    """
    n = len(scored.flows)
    best_idx = int(np.argmax(np.where(scored.finite, scored.scores, -np.inf)))
    comparable = _paired_comparability(scored, best_idx, comparable_sems,
                                       comparable_nats)
    marginal_comparable = np.zeros(n, dtype=bool)
    marginal_within_budget = np.zeros(n, dtype=bool)
    if selection == "best":
        return _Selection(best_idx, comparable, marginal_comparable,
                          marginal_within_budget)

    pool = np.flatnonzero(comparable)
    if len(pool) == 0:
        # Defensive fallback for a degenerate paired comparison. The best
        # index was selected only from wholly finite held-out arrays.
        pool = np.array([best_idx])
    if selection == "smallest_comparable":
        # smallest parameter count; break size ties by score
        sel_idx = int(pool[np.lexsort((-scored.scores[pool],
                                       scored.sizes[pool]))[0]])
        return _Selection(sel_idx, comparable, marginal_comparable,
                          marginal_within_budget)

    finite_pool = pool[np.isfinite(scored.marginal_selection_errors[pool])]
    if len(finite_pool) == 0:
        raise RuntimeError(
            "Every held-out-comparable candidate produced a non-finite "
            "marginal-fidelity score."
        )
    ranking_pool = finite_pool
    if max_marginal_error is not None:
        finite_marginals = np.isfinite(scored.marginal_errors)
        marginal_within_budget[finite_marginals] = (
            scored.marginal_errors[finite_marginals] <= max_marginal_error)
        budget_pool = finite_pool[marginal_within_budget[finite_pool]]
        if len(budget_pool):
            ranking_pool = budget_pool
        else:
            warnings.warn(
                "No held-out-likelihood-comparable candidate met "
                f"max_marginal_error={max_marginal_error:g}; returning "
                "the least-bad fit. Inspect marginal_details before use.",
                stacklevel=3,
            )
    best_marginal = float(np.min(
        scored.marginal_selection_errors[ranking_pool]))
    marginal_comparable[ranking_pool] = (
        scored.marginal_selection_errors[ranking_pool]
        <= best_marginal + tie_band)
    marginal_pool = np.flatnonzero(marginal_comparable)
    order = np.lexsort((
        -scored.scores[marginal_pool],
        scored.marginal_selection_errors[marginal_pool],
        scored.sizes[marginal_pool],
    ))
    return _Selection(int(marginal_pool[order[0]]), comparable,
                      marginal_comparable, marginal_within_budget)


def _build_selection_table(candidates: list, scored: _ScoredCandidates,
                           sel: _Selection, selection: str,
                           max_marginal_error: Optional[float],
                           marginal_components: Optional[int],
                           marginal_metric_source: Optional[str],
                           verbose: bool) -> list[dict]:
    """Build the uniform ``return_table``, one row per candidate.

    Every row carries every key in :data:`_SELECTION_TABLE_KEYS`; the
    marginal-fidelity and refit fields stay at their ``None``/``[]`` defaults
    when the configuration did not produce them. The refit fields are filled in
    afterwards by :func:`_refit_and_audit`, on the selected row only.
    """
    table = []
    for i, cand in enumerate(candidates):
        row = _blank_selection_row()
        row.update({
            "config": dict(cand),
            "val_logp": float(scored.scores[i]),
            "sem": float(scored.sems[i]),
            "n_params": (None if scored.flows[i] is None
                         else int(scored.sizes[i])),
            "comparable": bool(sel.comparable[i]),
            "selected": i == sel.chosen,
            "fit_status": ("failed" if scored.flows[i] is None else
                           "nonfinite_score" if not scored.finite[i] else "ok"),
            "fit_error": scored.fit_errors[i],
        })
        if selection == "marginal_fidelity":
            details = scored.marginal_details[i]
            worst = (max(details, key=lambda item: item["error"])
                     if details else None)
            row.update({
                "marginal_error": float(scored.marginal_errors[i]),
                "marginal_mean_error": float(scored.marginal_mean_errors[i]),
                "marginal_selection_error": float(
                    scored.marginal_selection_errors[i]),
                "marginal_comparable": bool(sel.marginal_comparable[i]),
                "marginal_within_budget": (
                    None if max_marginal_error is None
                    else bool(sel.marginal_within_budget[i])),
                "marginal_components": marginal_components,
                "marginal_unresolved_components": sum(
                    not item["resolved"] for item in details),
                "marginal_metric_source": marginal_metric_source,
                "marginal_worst_dimension": (
                    None if worst is None else worst["dimension"]),
                "marginal_worst_component": (
                    None if worst is None else worst["component"]),
                "marginal_details": details,
            })
        table.append(row)
        if verbose:
            params_text = ("failed" if scored.flows[i] is None
                           else str(scored.sizes[i]))
            marginal_text = (
                f"  marginal {scored.marginal_errors[i]:.5f} "
                f"(selection {scored.marginal_selection_errors[i]:.5f})"
                if selection == "marginal_fidelity" else "")
            print(f"[fit_auto] {i + 1}/{len(candidates)} {cand}: "
                  f"held-out logp {scored.scores[i]:.3f} +- {scored.sems[i]:.3f}  "
                  f"params {params_text}  comparable {bool(sel.comparable[i])}"
                  f"{marginal_text}")
    return table


def _refit_and_audit(refit_flow: Transformed, split_flow: Transformed,
                     row: dict, val: np.ndarray,
                     split_val_log_prob: np.ndarray, selection: str,
                     score_marginals: Callable[[Transformed, int], tuple],
                     audit_seed: int, tie_band: float,
                     comparable_sems: float, comparable_nats: float,
                     max_marginal_error: Optional[float]) -> Transformed:
    """Decide whether the full-chain refit may replace the split fit.

    Parameters
    ----------
    refit_flow : Transformed
        The flow refitted on the whole (clean) chain.
    split_flow : Transformed
        The held-out-validated fit that is returned if the refit is rejected.
    row : dict
        The selected table row; the refit/audit diagnostics are written here.
    val : np.ndarray
        The held-out rows, used for the finite-density check.
    split_val_log_prob : np.ndarray
        ``split_flow``'s per-row held-out log-density, for the paired sanity
        comparison.
    selection : str
        The active selection rule; only ``"marginal_fidelity"`` runs the
        componentwise audit.
    score_marginals : callable
        ``(flow, seed) -> (max_error, mean_error, selection_error, details)``.
    audit_seed : int
        Draw seed for the audit, distinct from the selection seed so the
        comparison is not against the sample the winner was chosen on.
    tie_band, comparable_sems, comparable_nats, max_marginal_error :
        Tolerances, as documented on ``normalizing_flows_fit_auto``.

    Returns
    -------
    Transformed
        ``refit_flow`` if every applicable check passed, else ``split_flow``
        (with a warning naming which check failed).

    Notes
    -----
    Only the *finite-density* check generalises across selection rules:
    comparing the refit's held-out score against the split fit's does not,
    because the refit trained on those rows and its score is optimistically
    biased. The marginal audit is applied to ``marginal_fidelity`` alone for
    the same reason -- it is a conservative regression check, not a second
    held-out guarantee.
    """
    refit_lp = np.asarray(log_prob(refit_flow, val))
    refit_density_finite = (
        refit_lp.shape == (len(val),)
        and bool(np.all(np.isfinite(refit_lp)))
    )
    row["refit_density_finite"] = bool(refit_density_finite)

    if selection != "marginal_fidelity":
        if refit_density_finite:
            row["refit_used"] = True
            return refit_flow
        warnings.warn(
            "The full-data refit failed the finite-density check on "
            "the held-out rows; returning the validated split fit "
            "instead.",
            stacklevel=3,
        )
        return split_flow

    (audit_error, audit_mean_error, audit_selection_error,
     audit_details) = score_marginals(split_flow, audit_seed)
    (refit_error, refit_mean_error, refit_selection_error,
     refit_details) = score_marginals(refit_flow, audit_seed)
    refit_joint_sanity_passed = False
    if refit_density_finite:
        refit_diff = split_val_log_prob - refit_lp
        refit_sem_diff = float(np.std(refit_diff) / np.sqrt(len(refit_diff)))
        refit_joint_sanity_passed = float(np.mean(refit_diff)) <= (
            comparable_sems * refit_sem_diff + comparable_nats)
    details_align = (
        len(audit_details) == len(refit_details)
        and all(
            (before["dimension"], before["component"])
            == (after["dimension"], after["component"])
            for before, after in zip(audit_details, refit_details)
        )
    )
    regressions = []
    if details_align:
        regressions = [
            f"dimension {before['dimension']} {before['component']}"
            for before, after in zip(audit_details, refit_details)
            if after["selection_error"] > before["selection_error"] + tie_band
        ]
    audit_met_budget = (max_marginal_error is None
                        or audit_error <= max_marginal_error)
    refit_met_budget = (max_marginal_error is None
                        or refit_error <= max_marginal_error)
    refit_used = bool(
        np.isfinite(audit_selection_error)
        and np.isfinite(refit_selection_error)
        and refit_density_finite
        and refit_joint_sanity_passed
        and details_align
        and not regressions
        and refit_met_budget
    )
    row.update({
        "audit_marginal_error": float(audit_error),
        "audit_marginal_mean_error": float(audit_mean_error),
        "audit_marginal_selection_error": float(audit_selection_error),
        "audit_marginal_details": audit_details,
        "audit_marginal_within_budget": (
            None if max_marginal_error is None else bool(audit_met_budget)),
        "refit_marginal_error": float(refit_error),
        "refit_marginal_mean_error": float(refit_mean_error),
        "refit_marginal_selection_error": float(refit_selection_error),
        "refit_marginal_details": refit_details,
        "refit_marginal_within_budget": (
            None if max_marginal_error is None else bool(refit_met_budget)),
        "refit_joint_sanity_passed": bool(refit_joint_sanity_passed),
        "refit_regressed_components": regressions,
        "refit_used": refit_used,
    })
    if refit_used:
        return refit_flow
    budget_note = (
        " The returned split fit also exceeded the error budget "
        "on the fresh audit."
        if max_marginal_error is not None and not audit_met_budget else ""
    )
    warnings.warn(
        "The full-data refit worsened one or more marginal "
        "components, exceeded the requested error budget, or "
        "failed the joint-fit sanity or finite-density check; "
        "returning the validated split fit instead. The selected "
        "table row reports the refit diagnostics." + budget_note,
        stacklevel=3,
    )
    return split_flow


def _fit_one_candidate(fit_part: np.ndarray, val: np.ndarray, fit_kwargs: dict,
                       selection: str, score_marginals: Callable,
                       marginal_seed: int) -> _CandidateResult:
    """Fit one candidate and score it on the shared held-out rows.

    Parameters
    ----------
    fit_part, val : np.ndarray
        The fitting and held-out halves of the split, identical across
        candidates so the comparison is paired.
    fit_kwargs : dict
        Fully merged keyword arguments for :func:`normalizing_flows_fit`.
    selection : str
        Only ``"marginal_fidelity"`` scores marginals.
    score_marginals : callable
        ``(flow, seed) -> (max_error, mean_error, selection_error, details)``.
    marginal_seed : int
        Draw seed for that scorer, shared by every candidate.

    Returns
    -------
    _CandidateResult
        With ``fit_error`` set and ``flow=None`` if anything raised.

    Notes
    -----
    Every exception is captured rather than propagated: one candidate is one
    rung of a ladder, and a rung that will not fit is information about that
    rung, not a reason to abandon the search. ``normalizing_flows_fit_auto``
    re-raises only when *no* candidate survives. The marginal scores of a
    failed candidate are ``+inf`` under marginal-fidelity selection (which
    minimises them) and NaN otherwise, so a failure can never win.
    """
    try:
        flow = normalizing_flows_fit(fit_part, **fit_kwargs)
        lp = np.asarray(log_prob(flow, val))
        if lp.shape != (len(val),):
            raise ValueError(
                f"log_prob returned shape {lp.shape}, expected {(len(val),)}."
            )
        marginal_error = np.nan
        marginal_mean_error = np.nan
        marginal_selection_error = np.nan
        marginal_details: list[dict] = []
        if selection == "marginal_fidelity":
            (marginal_error, marginal_mean_error,
             marginal_selection_error, marginal_details) = (
                score_marginals(flow, marginal_seed))
        return _CandidateResult(flow, lp, marginal_error, marginal_mean_error,
                                marginal_selection_error, marginal_details,
                                None)
    except Exception as exc:
        marginal_failure = (np.inf if selection == "marginal_fidelity"
                            else np.nan)
        error = f"{type(exc).__name__}: {exc}"
        return _CandidateResult(None, np.full(len(val), np.nan),
                                marginal_failure, marginal_failure,
                                marginal_failure, [], error)


def _run_candidates(fit_and_score: Callable, candidates: list,
                    n_workers: Optional[int]) -> list[_CandidateResult]:
    """Evaluate every candidate, concurrently when asked.

    ``None`` workers uses ``min(len(candidates), os.cpu_count())``. Threads,
    not processes: JAX releases the GIL during compiled compute, and these
    fits individually underuse the machine. Results are order-preserving and
    identical to the sequential path, because each candidate's seed and data
    split are fixed before any of them starts.
    """
    if n_workers is None:
        import os
        n_workers = max(1, min(len(candidates), os.cpu_count() or 1))
    if n_workers > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            return list(ex.map(fit_and_score, candidates))
    return [fit_and_score(c) for c in candidates]


_DEFAULT_MARGINAL_QUANTILES = (0.01, 0.025, 0.05, 0.95, 0.975, 0.99)


class _MarginalMetricPlan(NamedTuple):
    """The candidate ladder and the marginal metric ``fit_auto`` will score with.

    ``edge_features`` / ``normalised_edges`` are set for the literal edge-bin
    metric and ``quantiles`` for the copula-space one; exactly one of the two
    is populated, and neither under the non-marginal selection rules.
    """

    candidates: list
    edge_features: Optional[list]
    normalised_edges: Optional[list[Optional[np.ndarray]]]
    quantiles: np.ndarray
    components: Optional[int]
    metric_source: Optional[str]


def _reject_unusable_selection_options(
    selection: str,
    marginal_samples: int,
    marginal_quantiles: Sequence[float],
    marginal_bin_edges: Optional[Sequence[Optional[np.ndarray]]],
    marginal_ess,
    min_marginal_count: int,
    marginal_confidence: float,
    marginal_tolerance: Optional[float],
    max_marginal_error: Optional[float],
) -> None:
    """Reject an unknown selection rule, and marginal options it would ignore.

    Raises
    ------
    ValueError
        If ``selection`` is not one of the three rules, or if any
        ``marginal_*`` option was moved off its default under a rule that does
        not read it -- silently ignoring those is how a caller ends up
        believing an edge-mass budget was applied when it was not.
    """
    if selection not in ("smallest_comparable", "best", "marginal_fidelity"):
        raise ValueError(
            "selection must be 'smallest_comparable', 'best', or "
            f"'marginal_fidelity', got {selection!r}."
        )
    if selection == "marginal_fidelity":
        return
    try:
        quantiles_are_default = (
            tuple(marginal_quantiles) == _DEFAULT_MARGINAL_QUANTILES)
    except (TypeError, ValueError):
        quantiles_are_default = False
    if any((
        marginal_samples != 20_000,
        not quantiles_are_default,
        marginal_bin_edges is not None,
        marginal_ess is not None,
        min_marginal_count != 20,
        marginal_confidence != 0.95,
        marginal_tolerance is not None,
        max_marginal_error is not None,
    )):
        raise ValueError(
            "Non-default marginal_* options require "
            "selection='marginal_fidelity'."
        )


def _resolve_candidate_ladder(selection: str, candidates: Optional[list],
                              shared_kwargs: dict) -> list:
    """The caller's candidate ladder, or the default one for this rule.

    ``shared_kwargs`` is read only to see whether the caller pinned
    ``tanh_max_val``: the marginal-fidelity default otherwise softens the two
    small rungs to 2.0, and it must not override an explicit shared value.

    Raises
    ------
    ValueError
        If the ladder is empty; every downstream reduction over candidates
        would otherwise produce an empty-sequence error far from the cause.
    """
    if candidates is None:
        if selection == "marginal_fidelity":
            candidates = [{"knots": 2, "flow_layers": 2},
                          {"knots": 4, "flow_layers": 4},
                          {"knots": 4, "flow_layers": 6}]
            if "tanh_max_val" not in shared_kwargs:
                candidates[0]["tanh_max_val"] = 2.0
                candidates[1]["tanh_max_val"] = 2.0
        else:
            candidates = [{"knots": 2, "flow_layers": 4},
                          {"knots": 4, "flow_layers": 6},
                          {"knots": 8, "flow_layers": 8}]
    if len(candidates) == 0:
        raise ValueError("candidates must contain at least one configuration.")
    return candidates


def _plan_marginal_metric(
    selection: str,
    candidates: list,
    clean: np.ndarray,
    marginal_samples: int,
    marginal_quantiles: Sequence[float],
    marginal_bin_edges: Optional[Sequence[Optional[np.ndarray]]],
    marginal_ess,
    min_marginal_count: int,
    marginal_confidence: float,
    marginal_tolerance: Optional[float],
    max_marginal_error: Optional[float],
) -> _MarginalMetricPlan:
    """Resolve which marginal metric ``fit_auto`` will score candidates with.

    Parameters
    ----------
    selection : str
        The active selection rule; only ``"marginal_fidelity"`` builds a
        metric, and the other two return an all-empty plan.
    candidates : list of dict
        The resolved ladder, carried through so the caller has one object.
    clean : np.ndarray
        The finite rows, used for the reference edge masses and for the
        dimension count.
    marginal_samples, marginal_quantiles, marginal_bin_edges, marginal_ess, \
marginal_confidence, min_marginal_count, marginal_tolerance, \
max_marginal_error :
        As documented on :func:`normalizing_flows_fit_auto`.

    Returns
    -------
    _MarginalMetricPlan
        Everything the per-candidate scorer needs.

    Raises
    ------
    ValueError
        For out-of-range marginal options, or quantiles outside ``(0, 1)``.

    Notes
    -----
    Extracted from ``normalizing_flows_fit_auto`` to bring that function's
    branch count down; it has no other caller.
    """
    edge_features = None
    normalised_edges = None
    marginal_q = np.asarray([], dtype=float)
    if selection == "marginal_fidelity":
        _validate_marginal_options(
            marginal_samples, min_marginal_count, marginal_confidence,
            marginal_tolerance, max_marginal_error)
        if marginal_bin_edges is not None:
            normalised_edges = _normalise_marginal_bin_edges(
                marginal_bin_edges, clean.shape[1])
            effective_sizes = _normalise_marginal_ess(
                marginal_ess, clean.shape[1], len(clean))
            edge_features, unresolved = _edge_fidelity_features(
                clean, normalised_edges, effective_sizes, min_marginal_count)
            if unresolved:
                warnings.warn(
                    "Marginal-fidelity selection found statistically under-resolved "
                    "cells. Only discrepancies beyond their Wilson uncertainty "
                    "will affect the relative score (an explicit raw-error budget "
                    "still applies): " + "; ".join(unresolved),
                    stacklevel=3,
                )
        else:
            marginal_q = np.asarray(marginal_quantiles, dtype=float)
            if (marginal_q.ndim != 1 or len(marginal_q) == 0
                    or not np.all(np.isfinite(marginal_q))
                    or np.any((marginal_q <= 0.0) | (marginal_q >= 1.0))):
                raise ValueError(
                    "marginal_quantiles must be a non-empty sequence strictly "
                    "between 0 and 1."
                )
            marginal_q = np.unique(marginal_q)

    if selection != "marginal_fidelity":
        components: Optional[int] = None
        metric_source: Optional[str] = None
    elif edge_features is not None:
        components = len(edge_features)
        metric_source = "literal_edge_bins"
    else:
        components = clean.shape[1] * len(marginal_q)
        metric_source = "latent_quantiles"

    return _MarginalMetricPlan(candidates, edge_features, normalised_edges,
                               marginal_q, components, metric_source)


def _holdout_split(clean: np.ndarray, val_frac: float, rng_seed: int
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Split ``clean`` into (validation, fit) parts, or say why it cannot.

    Raises
    ------
    ValueError
        If ``val_frac`` is outside ``(0, 1)``, or if the split it asks for
        leaves either part too small to build an empirical marginal from.

    Notes
    -----
    ``val_frac <= 0`` used to be clamped to a *one-row* held-out set, which
    reports a normal-looking ``val_logp`` beside ``sem: 0.0`` -- a table a
    caller cannot distinguish from a real split. ``val_frac >= 1`` surfaced as
    "every candidate fit raised ... Insufficient samples for empirical
    transform", naming neither ``val_frac`` nor the real cause.
    """
    if not np.isfinite(val_frac) or not 0.0 < float(val_frac) < 1.0:
        raise ValueError(
            f"val_frac must be strictly between 0 and 1; got {val_frac!r}.")
    n_val = int(round(len(clean) * float(val_frac)))
    n_fit = len(clean) - n_val
    # The fitting part has to support an empirical marginal; the held-out part
    # needs two rows before ``sem`` means anything at all.
    if n_val < 2 or n_fit < _MIN_SAMPLES_PER_DIM + 1:
        raise ValueError(
            f"val_frac={val_frac!r} splits {len(clean)} finite rows into "
            f"{n_val} held-out and {n_fit} fitting rows, but scoring needs at "
            f"least 2 held-out rows and fitting needs at least "
            f"{_MIN_SAMPLES_PER_DIM + 1}. Use more samples or a val_frac "
            f"nearer 0.5."
        )
    rng = np.random.default_rng(rng_seed)
    perm = rng.permutation(len(clean))
    return clean[perm[:n_val]], clean[perm[n_val:]]


def normalizing_flows_fit_auto(chain: np.ndarray,
                               candidates: Optional[list] = None,
                               val_frac: float = 0.15,
                               refit_best: bool = True,
                               rng_seed: int = 999,
                               n_workers: Optional[int] = None,
                               selection: str = "smallest_comparable",
                               comparable_sems: float = 1.0,
                               comparable_nats: float = 0.0,
                               verbose: bool = False,
                               return_table: bool = False,
                               *,
                               marginal_samples: int = 20_000,
                               marginal_quantiles: Sequence[float] = (
                                   0.01, 0.025, 0.05, 0.95, 0.975, 0.99),
                               marginal_bin_edges: Optional[
                                   Sequence[Optional[np.ndarray]]
                               ] = None,
                               marginal_ess: Optional[
                                   float | Sequence[float]
                                   | Sequence[Sequence[float]]
                               ] = None,
                               min_marginal_count: int = 20,
                               marginal_confidence: float = 0.95,
                               marginal_tolerance: Optional[float] = None,
                               max_marginal_error: Optional[float] = None,
                               **shared_kwargs):
    """
    Fit a copula flow with automatic hyperparameter selection.

    Splits the chain into a fit part and a held-out validation part, fits one
    flow per candidate configuration, scores each candidate by its mean
    log-density on the held-out samples, and returns the winning configuration
    (by default refit on the full chain).

    **Selection rule.** With the default ``selection="smallest_comparable"``,
    the winner is the *smallest* model (by trainable parameter count) whose
    held-out score is statistically comparable to the best score: candidate
    ``c`` is comparable when ``mean(lp_best - lp_c) <= comparable_sems *
    sem(lp_best - lp_c) + comparable_nats``, a *paired* test on the shared
    validation points (much sharper than comparing the individual score
    error bars). With a clear winner nothing changes; statistical ties are
    resolved toward parsimony -- fewer parameters, faster fits, less
    overfitting risk on refits. ``selection="best"`` restores the plain
    argmax.

    ``selection="marginal_fidelity"`` is the non-calibration option for edge
    bins. It first forms the same held-out-likelihood-comparable pool, so a
    better-looking marginal cannot buy its way in by damaging the joint fit.
    Within that pool it selects from the lowest-error Monte Carlo tie band,
    preferring the smallest model in a tie. This ranks candidates; set
    ``max_marginal_error`` when an absolute edge-mass budget is required.
    With ``marginal_bin_edges`` the score uses the literal first/last finite
    intervals and their adjacent under/overflow masses, all on shared edges.
    Without edges it measures drift at canonical copula-space tail quantiles.

    The held-out mean log-density estimates ``-H(pi, q)``; maximizing it over
    candidates minimizes ``KL(pi || q)`` up to the (candidate-independent)
    entropy of the target — and ``KL(pi || q)`` is precisely the divergence
    that governs the acceptance rate of an independence MCMC proposal. In
    other words, this selects the flow that will accept best, and it
    automatically penalizes over-parameterized configurations that fit
    sampling noise (which raise train likelihood but lower held-out
    likelihood *and* MCMC acceptance).

    Because every candidate is scored on the same held-out samples in the
    *original* parameter space (marginal transforms included), candidates may
    differ in any ``normalizing_flows_fit`` argument — ``knots``,
    ``flow_layers``, ``learning_rate`` and ``num_points`` — and remain
    directly comparable.

    .. note::
        ``selection="smallest_comparable"`` ranks by trainable parameter
        count, which measures the *copula* architecture and nothing else:
        ``knots`` and ``flow_layers`` move it, while the marginal family and
        its tail options are frozen transforms and contribute zero (measured:
        ``knots=4, flow_layers=6`` is 220 parameters on 2-D data under
        ``marginal="smooth"``, ``"rqs"`` and ``"pchip"`` alike). The rung
        ordering is therefore comparable across every candidate axis, and
        ``selection="best"`` is only needed to opt out of parsimony itself.

    .. note::
        Held-out likelihood measures proposal *efficiency* only. Boundary
        behavior -- no sampling atoms / density cliffs at hard prior bounds --
        lives in a measure-near-zero region that validation samples
        essentially never probe, so this criterion cannot reward it. The
        default smooth-tail marginals are boundary-safe by construction, so
        let the candidates tune capacity freely.

    Parameters
    ----------
    chain : np.ndarray
        Training samples with shape (n_samples, n_params). Rows containing
        non-finite values (NaN or +-inf) are dropped, with a warning naming
        the count, before splitting; any other shape raises.
    candidates : list of dict, optional
        Candidate keyword overrides for :func:`normalizing_flows_fit`. Each
        dict is merged over ``**shared_kwargs``. The default is a small
        capacity ladder (validated on MCMC posteriors: the small rung wins on
        mild copulas, the larger rungs only when the dependence structure
        genuinely needs them)::

            [{"knots": 2, "flow_layers": 4},
             {"knots": 4, "flow_layers": 6},
             {"knots": 8, "flow_layers": 8}]

        Marginal-fidelity selection uses three fits: 2/2 and 4/4 knots/layers
        with ``tanh_max_val=2.0``, plus the existing 4/6,
        ``tanh_max_val=3.0`` default as a protected baseline. An explicit
        shared ``tanh_max_val`` applies to all three. Supply a shorter or
        broader list to trade cost against the search range.

    val_frac : float, optional
        Fraction of the (clean) chain held out for scoring. Default 0.15.
        Must lie strictly between 0 and 1, and must leave at least 2 held-out
        rows and at least 21 fitting rows.

        .. versionchanged:: 3.0
            ``val_frac <= 0`` used to be clamped to a one-row held-out set,
            which reports ``sem: 0.0`` beside a normal-looking ``val_logp``.
    refit_best : bool, optional
        If True (default), refit the winning configuration on the full chain
        before returning it. Every refit must produce a finite held-out
        density; if it does not, the validated split fit is returned with a
        warning. Marginal-fidelity refits additionally get a fresh,
        componentwise marginal audit and a conservative joint-density
        regression sanity check; failures return the validated split fit.
        Because a full-data refit includes the validation rows, that latter
        check is not a second held-out guarantee (which is why it is not
        applied to the other selection modes). Use ``refit_best=False`` to
        return the strictly held-out-validated split fit. Reported held-out
        scores always refer to split fits, and the selected table row always
        carries ``refit_used``.
    rng_seed : int, optional
        Seed for the split and for every candidate fit (all candidates see
        the identical split, so the comparison is paired). Default 999.
    n_workers : int, optional
        Number of candidate fits to run concurrently (in threads; JAX
        releases the GIL during compiled compute, and these small fits
        individually underuse the machine). ``None`` (default) uses
        ``min(len(candidates), os.cpu_count())``; set ``1`` for sequential
        fitting. Results are identical either way: every candidate's seed
        and data split are fixed, so execution order cannot change the
        selection.
    selection : str, optional
        ``"smallest_comparable"`` (default): smallest model whose held-out
        score is within tolerance of the best (see summary above).
        ``"best"``: plain argmax of the held-out score.
        ``"marginal_fidelity"``: best marginal score among held-out-likelihood-
        comparable candidates, with parsimony inside a Monte Carlo tie band.
    comparable_sems : float, optional
        Tolerance in units of the *paired* standard error of the score
        difference against the best candidate. Default 1.0 (the classic
        one-standard-error rule).
    comparable_nats : float, optional
        Additional absolute tolerance in nats added to the comparability
        threshold. Default 0.0. A value like 0.05 declares models within
        0.05 nats of KL comparable regardless of statistical resolution.
    verbose : bool, optional
        Print one line per candidate. Default False.
    return_table : bool, optional
        If True, return ``(flow, table)``; see Returns for the row schema.
    marginal_samples : int, optional
        Candidate draws used for marginal scoring. Default 20,000.
    marginal_quantiles : sequence of float, optional
        Copula-space CDF probabilities used when physical edges are omitted.
    marginal_bin_edges : sequence of array or None, optional
        Shared physical histogram edges, one entry per dimension. ``None``
        skips a dimension. The metric includes literal edge bins and
        under/overflow outside each finite crop.
    marginal_ess : float or sequence of float, optional
        Effective reference size for sparse-bin decisions. Accepts a scalar,
        one value per dimension, or shape ``(n_dimensions, 4)`` in
        underflow/lower-bin/upper-bin/overflow order. Defaults to raw N.
    min_marginal_count : int, optional
        Effective count at which each literal cell is treated as resolved.
        Sparse cells affect selection only by the directional gap between
        reference and candidate Wilson intervals.
    marginal_confidence : float, optional
        Wilson confidence level for sparse cells. Default 0.95.
    marginal_tolerance : float, optional
        Heuristic Monte Carlo tie band, applied componentwise to refit audits.
        Defaults to ``0.5/sqrt(marginal_samples)``; it is not a simultaneous
        confidence interval over all candidates and components.
    max_marginal_error : float, optional
        Maximum acceptable raw probability-mass error in every scored cell.
        Candidates inside this budget are preferred. If no held-out-
        likelihood-comparable candidate meets it, the least-bad candidate is
        returned with a warning. By default selection is relative only.
    **shared_kwargs
        Arguments passed to every candidate fit (e.g. ``prior_bounds``,
        ``num_points``, ``max_epochs``).

    Returns
    -------
    Transformed  (or  (Transformed, list) when ``return_table=True``)
        The selected fitted flow, and one table row per candidate.

        **Table row schema.** Every row carries *every* key below, in this
        order (:data:`_SELECTION_TABLE_KEYS`), so ``pandas.DataFrame(table)``
        has no ragged columns and ``row["refit_used"]`` never raises. ``None``
        means "not measured in this configuration"; the four list-valued keys
        default to ``[]``.

        Always populated:

        ``config``
            The candidate's keyword overrides, as given.
        ``val_logp``, ``sem``
            Mean held-out log-density of the *split* fit and its standard
            error. NaN if the candidate raised.
        ``n_params``
            Trainable parameter count of the copula, or None if it raised.
        ``comparable``
            Held-out-comparable with the best candidate (paired test).
        ``selected``
            Whether this row's configuration produced the returned flow.
        ``fit_status``
            ``"ok"``, ``"nonfinite_score"`` or ``"failed"``.
        ``fit_error``
            ``"TypeName: message"`` when the candidate raised, else None.
        ``refit_used``
            Whether the returned flow is this row's full-chain refit. False on
            every non-selected row and whenever ``refit_best=False``.

        Populated only under ``selection="marginal_fidelity"``, from the
        candidate's own scoring draw:

        ``marginal_error``
            Largest raw probability-mass error over the scored cells; ``inf``
            if the candidate raised or drew non-finite samples.
        ``marginal_mean_error``
            Mean raw probability-mass error over the same cells.
        ``marginal_selection_error``
            Largest *resolution-aware* error -- the raw error on a resolved
            cell, and on a sparse one only the gap between the reference and
            candidate Wilson intervals. This is what the rule ranks on.
        ``marginal_comparable``
            Whether this candidate is inside the marginal tie band of the
            best marginal score. Independent of ``comparable``, which is the
            held-out-likelihood gate the pool is formed from first.
        ``marginal_within_budget``
            Whether ``marginal_error`` met ``max_marginal_error``. ``None``
            when no budget was set.
        ``marginal_components``
            Number of scored cells. The same on every row -- the metric is
            fixed before any candidate is fitted.
        ``marginal_unresolved_components``
            How many of them this candidate scored as statistically
            under-resolved, and therefore ranked by interval gap rather than
            by raw error. Always 0 under ``"latent_quantiles"``.
        ``marginal_metric_source``
            ``"literal_edge_bins"`` when ``marginal_bin_edges`` was given,
            else ``"latent_quantiles"``. Also the same on every row.
        ``marginal_worst_dimension``, ``marginal_worst_component``
            Which cell carried ``marginal_error``: the chain column, and the
            component name (``"underflow"``, ``"lower_bin"``, ``"upper_bin"``,
            ``"overflow"``, or ``"cdf_<q>"`` under latent quantiles). Both
            ``None`` when no cell was scored.
        ``marginal_details``
            One dict per scored cell, in the order the metric produced them;
            see ``audit_marginal_details`` below for the fields.

        Populated on the selected row only, and only when ``refit_best=True``:

        ``refit_density_finite``
            Whether the full-chain refit scored a finite density on every
            held-out row. The only refit check that applies to all three
            selection rules.
        ``refit_error``
            ``"TypeName: message"`` when the refit itself raised; set instead
            of every other ``refit_*`` key in that case.

        Populated on the selected row only, when ``refit_best=True`` *and*
        ``selection="marginal_fidelity"``. ``audit_marginal_*`` scores the
        *split* fit and ``refit_marginal_*`` the *full-chain refit*, both on
        the same fresh draw, taken with a seed distinct from the one selection
        used -- so the pair is a like-for-like before/after comparison rather
        than a re-reading of the sample the winner was chosen on:

        ``refit_joint_sanity_passed``
            Whether the refit's held-out score stayed within the comparability
            tolerance of the split fit's. Not a held-out guarantee: the refit
            trained on those rows, so this is a one-sided regression check.
        ``refit_regressed_components``
            Names (``"dimension <i> <component>"``) of the scored cells whose
            ``selection_error`` the refit worsened by more than the tie band.
            Empty when the refit was accepted; also empty, with
            ``refit_used=False``, when the two audits scored different cells.
        ``audit_marginal_error``, ``refit_marginal_error``
            Largest raw probability-mass error over the scored cells. This is
            the quantity ``max_marginal_error`` budgets.
        ``audit_marginal_mean_error``, ``refit_marginal_mean_error``
            Mean raw probability-mass error over the same cells.
        ``audit_marginal_selection_error``, ``refit_marginal_selection_error``
            Largest *resolution-aware* error: the raw error on a resolved
            cell, and on a sparse one only the gap between the reference and
            candidate Wilson intervals. This is what ranks and what
            ``refit_regressed_components`` compares.
        ``audit_marginal_within_budget``, ``refit_marginal_within_budget``
            Whether that fit's ``*_marginal_error`` met ``max_marginal_error``.
            ``None`` when no budget was set.
        ``audit_marginal_details``, ``refit_marginal_details``
            One dict per scored cell, in the order the metric produced them,
            carrying ``dimension``, ``component``, ``target``, ``estimate``,
            ``error``, ``selection_error`` and the resolution fields
            (``interval_gap``, ``resolved``, ``reference_ess``,
            ``reference_effective_count``). ``[]`` when the scorer could not
            run -- a non-finite draw, for instance.

        .. versionchanged:: 3.0
            Rows used to carry a different key set depending on the selection
            mode and on whether the row was the selected one.

    Examples
    --------
    >>> import numpy as np
    >>> from coppuccino import normalizing_flows_fit_auto
    >>> data = np.random.randn(2000, 2)
    >>> flow = normalizing_flows_fit_auto(data, max_epochs=100)   # doctest: +SKIP
    >>> # an explicit capacity ladder:
    >>> flow = normalizing_flows_fit_auto(                        # doctest: +SKIP
    ...     data, prior_bounds=np.array([[-5, 5], [-5, 5]]),
    ...     candidates=[{"knots": 2, "flow_layers": 4},
    ...                 {"knots": 4, "flow_layers": 6},
    ...                 {"knots": 8, "flow_layers": 8}])
    """
    chain = _validate_chain(chain)
    bad_rows = ~np.isfinite(chain).all(axis=1)
    n_dropped = int(bad_rows.sum())
    if n_dropped:
        # Same wording and the same count as normalizing_flows_fit: the two
        # entry points used to report an identically dirty chain differently.
        warnings.warn(
            f"normalizing_flows_fit_auto dropped {n_dropped} of {len(chain)} "
            f"rows containing non-finite values; fitting on the remaining "
            f"{len(chain) - n_dropped}.",
            stacklevel=2,
        )
    clean = chain[~bad_rows]
    if len(clean) < 2 * (_MIN_SAMPLES_PER_DIM + 1):
        raise ValueError(
            f"normalizing_flows_fit_auto needs at least "
            f"{2 * (_MIN_SAMPLES_PER_DIM + 1)} finite samples to split; "
            f"got {len(clean)}."
        )
    if shared_kwargs.get("prior_bounds") is not None:
        # Validate here as well as inside each candidate fit: a candidate
        # exception is caught and aggregated, so a malformed box would
        # otherwise resurface as "the fits likely diverged".
        shared_kwargs["prior_bounds"] = _validate_prior_bounds(
            shared_kwargs["prior_bounds"], clean)
    val, fit_part = _holdout_split(clean, val_frac, rng_seed)

    _reject_unusable_selection_options(
        selection, marginal_samples, marginal_quantiles, marginal_bin_edges,
        marginal_ess, min_marginal_count, marginal_confidence,
        marginal_tolerance, max_marginal_error)
    candidates = _resolve_candidate_ladder(selection, candidates, shared_kwargs)
    plan = _plan_marginal_metric(
        selection, candidates, clean, marginal_samples, marginal_quantiles,
        marginal_bin_edges, marginal_ess, min_marginal_count,
        marginal_confidence, marginal_tolerance, max_marginal_error)
    edge_features = plan.edge_features
    normalised_edges = plan.normalised_edges
    marginal_q = plan.quantiles

    marginal_seed = (int(rng_seed) + 104_729) % (2**31 - 1)
    marginal_audit_seed = (int(rng_seed) + 130_363) % (2**31 - 1)

    def _score_flow_marginals(fl, seed):
        if normalised_edges is not None:
            assert edge_features is not None
            draws = sample(fl, marginal_samples, rng_seed=seed)
            return _score_edge_fidelity(
                draws, normalised_edges, edge_features, marginal_confidence)
        return _score_latent_marginal_fidelity(
            fl, marginal_q, marginal_samples, seed)

    def _fit_and_score(cand):
        # candidate fits are quiet by default (parallel progress bars garble);
        # an explicit show_progress in shared_kwargs/candidate still wins.
        kw = {"rng_seed": rng_seed, "show_progress": False,
              **shared_kwargs, **cand}
        return _fit_one_candidate(fit_part, val, kw, selection,
                                  _score_flow_marginals, marginal_seed)

    results = _run_candidates(_fit_and_score, candidates, n_workers)

    scored = _score_candidates(results)
    failed = [f"candidate {i}: {error}"
              for i, error in enumerate(scored.fit_errors) if error is not None]
    if failed:
        warnings.warn(
            "normalizing_flows_fit_auto excluded failed fits: "
            + "; ".join(failed),
            stacklevel=2,
        )
    if not scored.finite.any():
        if len(failed) == len(candidates):
            # Nothing diverged -- every candidate raised. Lead with the actual
            # exceptions; divergence advice would be actively misleading here.
            raise RuntimeError(
                "normalizing_flows_fit_auto: every candidate fit raised. "
                + "; ".join(failed)
            )
        failure_text = f" Fit errors: {'; '.join(failed)}." if failed else ""
        raise RuntimeError(
            "normalizing_flows_fit_auto: every candidate produced a non-finite "
            f"held-out score ({scored.scores.tolist()}); the fits likely "
            "diverged. "
            "Try a smaller learning_rate or more training samples."
            + failure_text
        )
    sel = _select_candidate(
        scored, selection, comparable_sems, comparable_nats,
        max_marginal_error, _tie_band(marginal_samples, marginal_tolerance))
    sel_idx = sel.chosen

    table = _build_selection_table(
        candidates, scored, sel, selection, max_marginal_error,
        plan.components, plan.metric_source, verbose)

    best_flow = scored.flows[sel_idx]
    assert best_flow is not None
    used_refit = False
    if refit_best:
        kw = {"rng_seed": rng_seed, "show_progress": False,
              **shared_kwargs, **candidates[sel_idx]}
        try:
            refit_flow = normalizing_flows_fit(clean, **kw)
            chosen = _refit_and_audit(
                refit_flow, best_flow, table[sel_idx], val,
                scored.val_log_probs[sel_idx], selection, _score_flow_marginals,
                marginal_audit_seed,
                _tie_band(marginal_samples, marginal_tolerance),
                comparable_sems, comparable_nats, max_marginal_error)
            used_refit = chosen is refit_flow
            best_flow = chosen
        except Exception as exc:
            refit_error_text = f"{type(exc).__name__}: {exc}"
            table[sel_idx].update({
                "refit_used": False,
                "refit_error": refit_error_text,
            })
            warnings.warn(
                "The full-data refit failed; returning the validated split fit "
                f"instead ({refit_error_text}).",
                stacklevel=2,
            )
    if verbose:
        print(f"[fit_auto] selected {candidates[sel_idx]} "
              f"(held-out logp {scored.scores[sel_idx]:.3f}, "
              f"{scored.sizes[sel_idx]} params"
              f"{', refit on full chain' if used_refit else ''})")

    if return_table:
        return best_flow, table
    return best_flow


def _smooth_clamp_limits(flow) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Per-dimension tail-clamp x limits of a smooth-marginal flow.

    Returns ``(los, his)`` arrays (``-inf``/``+inf`` where not applicable),
    or None when the flow has no smooth marginals (no rejection needed).
    """
    bijection = getattr(flow, "bijection", None)
    if not isinstance(bijection, Stack):
        return None
    los, his, found = [], [], False
    for b in bijection.bijections:
        m = unwrap(b)
        if isinstance(m, SmoothMarginalToGaussian):
            found = True
            los.append(m.x_clamp_lo if m.x_clamp_lo is not None else -np.inf)
            his.append(m.x_clamp_hi if m.x_clamp_hi is not None else np.inf)
        else:
            los.append(-np.inf)
            his.append(np.inf)
    if not found:
        return None
    return np.asarray(los), np.asarray(his)


def _clamped_rows(samples: np.ndarray, limits) -> np.ndarray:
    los, his = limits
    return np.any((samples <= los[None, :]) | (samples >= his[None, :]), axis=1)


def _prior_box(flow) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Per-dimension prior bounds declared by a flow's marginals.

    Returns ``(lows, highs)``, ``-inf``/``+inf`` on a side no marginal bounds,
    or None when the flow declares no bound at all and nothing needs masking.
    """
    bijection = getattr(flow, "bijection", None)
    if not isinstance(bijection, Stack):
        return None
    lows, highs, bounded = [], [], False
    for b in bijection.bijections:
        m = unwrap(b)
        lo = getattr(m, "prior_low", None)
        hi = getattr(m, "prior_high", None)
        bounded = bounded or lo is not None or hi is not None
        lows.append(-np.inf if lo is None else float(lo))
        highs.append(np.inf if hi is None else float(hi))
    if not bounded:
        return None
    return np.asarray(lows), np.asarray(highs)


def _outside_prior_box(samples: np.ndarray, box) -> np.ndarray:
    """Boolean mask of rows lying strictly outside the prior box.

    The bounds themselves are *inside*: a sample sitting exactly on a bound is
    a supported value with a finite density, and the documented use of
    ``prior_bounds`` puts mass there.
    """
    lows, highs = box
    x = np.atleast_2d(np.asarray(samples))
    return np.any((x < lows[None, :]) | (x > highs[None, :]), axis=1)


_MAX_REDRAW_ROUNDS = 50


def _coerce_query_points(samples) -> np.ndarray:
    """Coerce density-query points to a float array, or say why they are not.

    ``FleetFlow.log_prob`` accepts a nested list; flowjax rejects the same
    input with "Expected x to be arraylike; got list", so the single-flow verbs
    coerce here to keep the two siblings interchangeable.
    """
    try:
        x = np.asarray(samples, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "samples must be a numeric array of shape (n_samples, n_params); "
            f"got {type(samples).__name__} ({exc})."
        ) from exc
    if x.ndim not in (1, 2):
        raise ValueError(
            "samples must be (n_samples, n_params), or (n_params,) for a "
            f"single point; got shape {x.shape}.")
    return x


def _validate_n_samples(n_samples, verb: str) -> int:
    """Range-check a draw count before it becomes a jax static shape.

    Unvalidated, ``-5`` and ``10.5`` surface as "Only non-negative indices are
    allowed when broadcasting static shapes" and "Shapes must be 1D sequences
    of concrete values of integer type" -- neither names ``n_samples``.
    """
    if (isinstance(n_samples, (bool, np.bool_))
            or not isinstance(n_samples, (int, np.integer))
            or int(n_samples) < 1):
        raise ValueError(
            f"{verb}() needs n_samples to be an integer >= 1; "
            f"got {n_samples!r}.")
    return int(n_samples)


def _redraw_clamped_rows(draw: Callable[[jax.Array, int], tuple],
                         arrays: tuple[np.ndarray, ...], limits, key,
                         n_samples: int, verb: str) -> None:
    """Redraw tail-clamped rows in place until none are left.

    Parameters
    ----------
    draw : callable
        ``draw(key, block)`` returning one fresh array per entry of ``arrays``,
        each with ``block`` leading rows.
    arrays : tuple of np.ndarray
        The arrays to patch, mutated in place. ``arrays[0]`` must be the
        ``(n, dim)`` sample matrix that decides which rows are clamped.
    limits : tuple of np.ndarray
        Per-dimension clamp limits from :func:`_smooth_clamp_limits`.
    key : jax PRNG key
        Seed for the replacement draws; split on each round.
    n_samples : int
        The requested draw count, which caps the replacement block size.
    verb : str
        Name of the calling function, quoted in the give-up warning.

    Notes
    -----
    Replacements are drawn in power-of-two blocks so that repeated rounds reuse
    a bounded number of compiled shapes instead of tracing one per round. The
    loop is shared by ``sample`` and ``sample_and_log_prob``; it used to be
    copy-pasted, including the warning text.
    """
    samples = arrays[0]
    for _ in range(_MAX_REDRAW_ROUNDS):
        bad = _clamped_rows(samples, limits)
        if not bad.any():
            return
        m = int(bad.sum())
        block = min(1 << (m - 1).bit_length(), n_samples)
        key, sub = jr.split(key)
        for target, fresh in zip(arrays, draw(sub, block)):
            target[bad] = np.asarray(fresh)[:m]
    warnings.warn(
        f"{verb}(): tail-clamped rows persisted after "
        f"{_MAX_REDRAW_ROUNDS} redraw rounds; returning the last "
        "draw. The flow's z-tails are extremely heavy relative to "
        "the bounded support.",
        stacklevel=3,
    )


def _reject_fleet_flow(flow, verb: str, alternative: str) -> None:
    """Reject a ``FleetFlow`` handed to one of the single-flow verbs.

    The package exports the fleet fitter alongside these verbs, but a fleet has
    its own batched API and none of the ``Transformed`` attributes they reach
    for. Without this the mistake surfaces as an ``int()`` TypeError, a raw jax
    vmap shape dump, or an ``AttributeError`` about ``.bijection`` -- each
    naming an internal detail rather than the confusion.

    Parameters
    ----------
    flow : object
        The first argument the caller passed.
    verb : str
        Name of the calling function, quoted in the message.
    alternative : str
        The fleet method to use instead, written as a call.

    Raises
    ------
    TypeError
        If ``flow`` is a :class:`~coppuccino.fleet.FleetFlow`.

    Notes
    -----
    The import is deferred: ``coppuccino.fleet`` depends on this module and on
    ``coppuccino.model_io``, so importing it at module scope from either would
    close an import cycle.
    """
    from coppuccino.fleet import FleetFlow

    if isinstance(flow, FleetFlow):
        raise TypeError(
            f"{verb}() takes a single fitted flow; a FleetFlow of "
            f"{flow.n_sources} sources has its own batched API. "
            f"Use {alternative} instead."
        )


def sample(flow: Transformed, n_samples: int, rng_seed: Optional[int] = None) -> np.ndarray:
    """
    Generate samples from a fitted copula flow.

    Draws random samples from the learned distribution by sampling from the
    standard normal base distribution and transforming through the trained
    flow.

    For smooth-marginal flows (the default ``marginal="smooth"``), rows whose
    bounded tails saturated at the float64 wall-resolution limit are redrawn.
    The emitted stream is then the reported density truncated to the
    clamp-free region and renormalized -- a constant that cancels in MH
    ratios -- so there are no near-boundary sample lumps.

    The redraw rate has no single value: it is the base-normal mass the fitted
    marginal maps into its saturated wall region, so it is driven by how much
    of the *training chain* sits exactly on a prior bound, and it varies by
    more than a factor of two between seeds of one configuration. It is
    exactly zero for a chain that stays strictly inside its box, and reaches
    the 20% range once about half the rows have been clipped onto a wall
    (measured over 50,000 raw draws from 40-epoch fits of 1,500 rows, four
    seeds each: 0/0/0/0 for ``tanh(N(0,1))*3`` in a +-3.5 box, and
    0.180/0.188/0.206/0.211 for ``clip(N(0,1), -1, 1)`` in a +-1 box, whose
    chains put 52-54% of their rows on a wall).
    If the cost of drawing large batches matters, measure it for your own fit
    rather than assuming either end::

        raw = np.asarray(equinox.filter_jit(flow.sample)(key, (n,)))
        rate = _clamped_rows(raw, _smooth_clamp_limits(flow)).mean()

    Parameters
    ----------
    flow : Transformed
        Fitted copula flow model from `normalizing_flows_fit`.
    n_samples : int
        Number of samples to generate.
    rng_seed : int, optional
        Random seed. If None (the default), a fresh seed is drawn from system
        entropy on each call, so repeated calls return *independent* draws.
        Pass an explicit integer for reproducible output.

    Returns
    -------
    np.ndarray
        Generated samples with shape (n_samples, n_params).

    Raises
    ------
    TypeError
        If `flow` is a `FleetFlow`, which draws from every source at once via
        its own `fleet.sample(...)`.
    ValueError
        If `n_samples` is not an integer >= 1.

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
    The draw is JIT-compiled, so the first call for a given flow and
    ``n_samples`` pays the compilation and every later one with the same shape
    is a cache hit -- two orders of magnitude apart, not a small constant:
    measured on a 3-D flow drawing 5,000 rows, 0.47 s for the first call
    against a 0.0035 s median over the next nine (130x). A different
    ``n_samples`` is a different static shape and recompiles; so can the
    redraw loop, which is why its replacement blocks are rounded to powers of
    two.
    """
    _reject_fleet_flow(flow, "sample", "fleet.sample(n_samples, rng_seed=...)")
    n_samples = _validate_n_samples(n_samples, "sample")
    key = jr.key(_resolve_seed(rng_seed))
    # Use filter_jit for improved performance
    samples = np.array(filter_jit(flow.sample)(key, (n_samples,)))
    limits = _smooth_clamp_limits(flow)
    if limits is not None:
        _redraw_clamped_rows(
            lambda sub, block: (filter_jit(flow.sample)(sub, (block,)),),
            (samples,), limits, key, n_samples, "sample")
    return samples


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
        Log probability densities, shape (n_samples,). When the flow was fitted
        with ``prior_bounds``, that box is hard support: a row lying strictly
        outside it scores ``-inf``, so ``np.isfinite(log_prob(flow, x))`` is a
        support test. The bounds themselves are inside and score finitely.

    Raises
    ------
    TypeError
        If `flow` is a `FleetFlow`, which scores every source at once via its
        own `fleet.log_prob(...)`.
    ValueError
        If `samples` cannot be read as a numeric array of shape
        (n_samples, n_params) or (n_params,).

    Notes
    -----
    `samples` is coerced with ``np.asarray``, so a nested list works here just
    as it does on ``FleetFlow.log_prob``; flowjax alone would reject it with
    "Expected x to be arraylike; got list".

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
    >>> # Higher values indicate higher probability density (e.g. log_probs.mean())
    """
    _reject_fleet_flow(flow, "log_prob", "fleet.log_prob(x)")
    samples = _coerce_query_points(samples)
    log_probs = np.array(filter_jit(flow.log_prob)(samples))
    # Each marginal evaluates an out-of-range query at its nearest bound, so an
    # unmasked point outside the prior box is scored as if it sat on the wall --
    # a finite plateau that is bitwise identical at the bound and at infinity,
    # and that can exceed the density of every real draw. The prior box is hard
    # support, so report -inf there and let finiteness be the support test.
    box = _prior_box(flow)
    if box is not None:
        log_probs = np.where(_outside_prior_box(samples, box), -np.inf, log_probs)
    return log_probs


def sample_and_log_prob(flow: Transformed, n_samples: int, rng_seed: Optional[int] = None) -> tuple[np.ndarray, np.ndarray]:
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
        Random seed. If None (the default), a fresh seed is drawn from system
        entropy on each call, so repeated calls return *independent* draws.
        Pass an explicit integer for reproducible output.

    Returns
    -------
    samples : np.ndarray
        Generated samples with shape (n_samples, n_params).
    log_probs : np.ndarray
        Log probability densities of the generated samples, shape (n_samples,).

    Raises
    ------
    TypeError
        If `flow` is a `FleetFlow`, which has its own
        `fleet.sample_and_log_prob(...)`.
    ValueError
        If `n_samples` is not an integer >= 1.

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
    >>> # np.exp(log_probs) is the proposal density q(x); importance
    >>> # weights against a target require it in the denominator:
    >>> # weights = target_pdf(samples) / np.exp(log_probs)
    >>> proposal_density = np.exp(log_probs)

    Notes
    -----
    The returned densities are the *untruncated* model density, while the
    returned draws come from the redraw-truncated stream (see ``sample``). The
    two differ by the constant ``1 / (1 - clamp_rate)``, so it cancels in
    Metropolis-Hastings ratios and in self-normalized importance sampling --
    including the weight expression in the example above, if you normalize the
    weights -- but *not* in an unnormalized importance-sampling estimate of a
    normalizing constant. On a chain with no mass on its bounds the constant
    is exactly 1 because nothing is redrawn.
    """
    _reject_fleet_flow(flow, "sample_and_log_prob",
                       "fleet.sample_and_log_prob(n_samples, rng_seed=...)")
    n_samples = _validate_n_samples(n_samples, "sample_and_log_prob")
    key = jr.key(_resolve_seed(rng_seed))
    jax_samples, jax_log_probs = filter_jit(
        flow.sample_and_log_prob)(key, (n_samples,))
    # Bind the numpy views to their own names: rebinding the jax Arrays in place
    # left the declared `-> tuple[np.ndarray, np.ndarray]` unverifiable.
    samples = np.array(jax_samples)
    log_probs = np.array(jax_log_probs)
    limits = _smooth_clamp_limits(flow)
    if limits is not None:
        _redraw_clamped_rows(
            lambda sub, block: filter_jit(
                flow.sample_and_log_prob)(sub, (block,)),
            (samples, log_probs), limits, key, n_samples,
            "sample_and_log_prob")
        # The forward pass computes the log-det at the generated z, while
        # log_prob reconstructs z from the float-truncated x. Near a hard
        # bound the inverse is steep enough that the ULP-level truncation of
        # x costs nat-scale differences. Recompute those few rows via
        # log_prob so the returned densities are bitwise consistent with any
        # later log_prob evaluation (MH ratios, importance weights).
        los, his = limits
        tol_lo = 1e6 * np.spacing(np.abs(np.where(np.isfinite(los), los, 1.0)))
        tol_hi = 1e6 * np.spacing(np.abs(np.where(np.isfinite(his), his, 1.0)))
        near = np.any(
            (np.isfinite(los)[None, :] & (samples - los[None, :] <= tol_lo[None, :]))
            | (np.isfinite(his)[None, :] & (his[None, :] - samples <= tol_hi[None, :])),
            axis=1,
        )
        if near.any():
            log_probs[near] = np.array(filter_jit(flow.log_prob)(samples[near]))
    return samples, log_probs
