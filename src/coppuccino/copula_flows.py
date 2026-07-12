import warnings
from typing import Callable, Optional

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
from flowjax.bijections import RationalQuadraticSpline, Stack
from flowjax.flows import coupling_flow, triangular_spline_flow
from flowjax.train import fit_to_data
from flowjax.distributions import Normal, Transformed, Uniform
from paramax import non_trainable
import equinox as eqx
from equinox import filter_jit

from coppuccino.bijections import EmpiricalMarginalToGaussian
from coppuccino.bijections import EmpiricalMarginalToUniform
from coppuccino.bijections import SmoothMarginalToGaussian
from coppuccino.bijections import make_empirical_cdf_spline
from coppuccino.bijections import make_smooth_marginal

__all__ = [
    "normalizing_flows_fit", "normalizing_flows_fit_auto",
    "sample", "log_prob", "sample_and_log_prob",
    "CopulaFlow",
]


class CopulaFlow(Transformed):
    """A fitted copula flow carrying its architecture spec as static metadata.

    Behaves exactly like ``flowjax.distributions.Transformed`` (it is one).
    The extra ``arch`` tuple records how the *inner* (trained) flow was
    constructed, so :func:`coppuccino.model_io.load_flow` can rebuild an
    identical skeleton and deserialize the trained arrays into it. This avoids
    pickling flow *structure*: the coupling layers used by the uniform-copula
    mode hold dynamically generated constructor closures that do not survive
    a pickle round-trip.

    ``arch`` layout: ``("uniform", n_params, knots, flow_layers)``.
    """

    arch: Optional[tuple] = eqx.field(static=True, default=None)

    def __init__(self, base_dist, bijection, arch=None):
        super().__init__(base_dist, bijection)
        self.arch = arch


def _build_uniform_inner(arch: tuple, rng_seed: int = 0):
    """Build the (untrained) unit-cube coupling flow described by ``arch``.

    Used both by :func:`normalizing_flows_fit` (uniform mode) and by
    :func:`coppuccino.model_io.load_flow`, which deserializes trained arrays
    into this skeleton — construction must therefore stay deterministic in
    structure (the ``rng_seed`` only sets initial parameter values, which are
    overwritten on load).
    """
    mode, n_params, knots, flow_layers = arch
    if mode != "uniform":
        raise ValueError(f"Unknown copula flow architecture: {arch!r}")
    return coupling_flow(
        jr.key(rng_seed),
        base_dist=Uniform(jnp.zeros(n_params), jnp.ones(n_params)),
        transformer=RationalQuadraticSpline(knots=knots, interval=(0, 1)),
        flow_layers=flow_layers,
        invert=True,
    )

# Minimum samples per dimension needed to build a usable empirical CDF.
_MIN_SAMPLES_PER_DIM = 20


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
                                  copula_space: str = "gaussian",
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
        Bulk interpolant family for every marginal: ``"rqs"`` (default) or
        ``"pchip"``. See :func:`coppuccino.bijections.make_empirical_cdf_spline`.
    copula_space : str, optional
        ``"gaussian"`` (default): marginals map to standard normal
        (:class:`EmpiricalMarginalToGaussian`). ``"uniform"``: marginals map to
        the unit interval as exact bijections
        (:class:`EmpiricalMarginalToUniform`, built with ``unit_range=True``);
        used by the uniform-copula pipeline.

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
    if copula_space not in ("gaussian", "uniform"):
        raise ValueError(
            f"copula_space must be 'gaussian' or 'uniform', got {copula_space!r}."
        )
    unit_range = copula_space == "uniform"
    marginal_cls = EmpiricalMarginalToUniform if unit_range else EmpiricalMarginalToGaussian

    x = samples[~np.isnan(samples).any(axis=1)]
    empirical_transforms = []

    for j in range(x.shape[1]):
        param_samples = x[:, j]
        if len(param_samples) > _MIN_SAMPLES_PER_DIM:
            p_low = float(prior_bounds[j, 0]) if prior_bounds is not None else None
            p_high = float(prior_bounds[j, 1]) if prior_bounds is not None else None
            if marginal == "smooth":
                # clip-free diffeomorphism z in R <-> theta in (lo, hi):
                # no CDF clip, no quantile clamp (no boundary sample atoms),
                # no floored wall density (no cliffs); honest analytic tails.
                # A non-finite bound means unbounded on that side.
                if p_low is not None and not np.isfinite(p_low):
                    p_low = None
                if p_high is not None and not np.isfinite(p_high):
                    p_high = None
                theta_of_z, z_of_theta, log_dtheta_dz, x_clamps = \
                    make_smooth_marginal(
                        param_samples, num_points=num_points,
                        prior_low=p_low, prior_high=p_high,
                        tail_alpha=tail_alpha)
                empirical_transforms.append(non_trainable(SmoothMarginalToGaussian(
                    param_samples, theta_of_z, z_of_theta, log_dtheta_dz,
                    num_points=num_points, prior_low=p_low, prior_high=p_high,
                    tail_alpha=tail_alpha,
                    x_clamp_lo=x_clamps[0], x_clamp_hi=x_clamps[1])))
                continue
            _, cdf_fn, quantile_fn, pdf_fn = make_empirical_cdf_spline(
                param_samples, num_points=num_points, min_eps=min_eps,
                tail_extension=tail_extension, prior_low=p_low, prior_high=p_high,
                tail_model=tail_model, tail_quantile=tail_quantile,
                marginal=marginal, unit_range=unit_range)
            empirical_transforms.append(non_trainable(marginal_cls(
                param_samples, cdf_fn, quantile_fn, pdf_fn, min_eps=min_eps,
                num_points=num_points, tail_extension=tail_extension,
                prior_low=p_low, prior_high=p_high,
                tail_model=tail_model, tail_quantile=tail_quantile,
                marginal=marginal)))
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
    arch: Optional[tuple] = None,
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

    kwargs = {
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

    # apply empirical marginal transforms; carry the architecture spec when
    # given (uniform mode) so save/load can rebuild the inner flow skeleton
    if arch is not None:
        return CopulaFlow(updated_flow, transform, arch=arch)
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
                          copula_space: str = "gaussian",
                          tail_alpha: float = 0.0,
                          weight_decay: float = 0.0,
                          show_progress: bool = True) -> Transformed:
    """
    Fit a copula normalizing flow to multivariate data.

    This function implements a copula-based normalizing flow by:
    1. Transforming each marginal via its empirical CDF
    2. Modeling the remaining dependence structure (the copula) with a flow

    Two pipelines are available via ``copula_space``:

    * ``"gaussian"`` (default): marginals map to Gaussian space and the copula
      is a triangular spline flow over a normal base -- the strongest choice
      for heavily correlated posteriors (the affine layer captures near-linear
      dependence). With the default ``marginal="smooth"`` the marginal map is
      a clip-free diffeomorphism of all of R onto the support (RQS bulk +
      analytic tails): no boundary sample atoms, no wall density cliffs, and
      honest smoothly-decaying densities for outliers / beyond-data queries
      (``tail_alpha`` sets Gaussian vs heavier tails on unbounded sides). The
      legacy clipped marginals remain available as ``marginal="rqs"`` /
      ``"pchip"`` (required for ``tail_extension`` / ``tail_model="gpd"``);
      they can disagree between sample stream and reported density in slivers
      at hard support boundaries, which can absorb independence-MCMC
      walkers.
    * ``"uniform"``: marginals stop at the probability integral
      transform (``u = F(x)``, an exact bijection onto ``[0, 1]``) and the
      copula is a rational-quadratic-spline coupling flow over a *uniform base
      on the unit cube* -- no Gaussianization and no tanh clipping. Because
      sampling and density evaluation share one closed-form bijection with
      strictly positive derivative everywhere (including the walls), the
      flow's sample stream and its ``log_prob`` agree at hard support
      boundaries by construction. The natural choice when weakly-constrained,
      flat-to-the-wall marginals dominate (its box-native tails model those
      shapes best). Requires ``marginal="rqs"`` and no tail model (bounded
      support: the fitted flow's support is the prior box when
      ``prior_bounds`` is given, else the observed data range).

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
        Number of knots for rational quadratic spline transformations. Default is 4.
    patience : int, optional
        Early stopping patience (epochs without improvement). Default is 30.
    learning_rate : float, optional
        Learning rate for Adam optimizer. Default is 1e-3.
    max_epochs : int, optional
        Maximum number of training epochs. Default is 400.
    flow_layers : int, optional
        Number of coupling layers in the triangular spline flow. Default is 6.
    num_points : int, optional
        Number of grid points for each marginal CDF spline. Default is 200.
        Automatically clamped based on sample size. Increase for distributions
        with fine structure (e.g. many narrow modes).
    tail_extension : bool, optional
        If True, extend the marginal CDF beyond the observed data range using
        a Gaussian tail model. This allows the flow to generate samples beyond
        the training range. Default is False, which is appropriate for MCMC
        chains bounded by prior support.
    tanh_max_val : float, optional
        Maximum value for the tanh clipping in the spline flow base space.
        Default is 3.0. Increase (e.g. 4.0–5.0) for heavy-tailed data where
        tail dependencies matter. Only used when ``copula_space="gaussian"``;
        the uniform-copula pipeline has no clipping.
    prior_bounds : np.ndarray, optional
        Prior support bounds with shape (n_params, 2), where each row is
        [low, high] for that parameter. When provided, the empirical CDF grid
        extends to the prior edges, allowing the flow to sample the full prior
        support rather than being limited to the most extreme training sample.
        Recommended for MCMC chains.
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
        Bulk interpolant family for the empirical marginal transforms:
        ``"rqs"`` (default) or ``"pchip"``. Both use the same empirical-quantile
        knots and tail/prior handling and are statistically equivalent; ``"rqs"``
        (monotone rational-quadratic spline) has a closed-form, machine-precision
        inverse and derivative from a single parameterization, whereas ``"pchip"``
        builds two independent splines for the CDF and its inverse that are only
        approximate inverses of each other. Prefer ``"rqs"`` (better self-
        consistency at equal cost and fit quality); ``"pchip"`` is retained for
        backward compatibility and reproducing older fits.

        Default ``None`` resolves per pipeline: ``"rqs"`` (exact unit-range
        spline) for ``copula_space="uniform"``, ``"smooth"`` (clip-free
        smooth-tail diffeomorphism; see
        :func:`coppuccino.bijections.make_smooth_marginal`) for
        ``copula_space="gaussian"``.
    tail_alpha : float, optional
        Tail heaviness for ``marginal="smooth"`` on *unbounded* sides (no
        prior bound given): 0.0 (default) gives Gaussian tails; larger values
        give progressively heavier tails. Bounded sides always use the
        exponential wall-approach tails. Ignored by other marginals.
    copula_space : str, optional
        ``"gaussian"`` (default) or ``"uniform"`` -- see the summary above.
        ``"uniform"`` requires ``marginal="rqs"``, ``tail_extension=False`` and
        ``tail_model="gaussian"`` (bounded support; no tail models); pass
        ``prior_bounds`` so the flow covers the full prior box.

        .. versionchanged:: 2.0
            The Gaussian pipeline's default marginal became the clip-free
            smooth-tail diffeomorphism (``marginal="smooth"``): boundary-safe
            by construction (no exact-bound sample atoms, no wall density
            cliffs, sampler-consistent densities) with honest outlier tails.
            The cube-native ``"uniform"`` pipeline was added as the box-native
            alternative.
    weight_decay : float, optional
        AdamW weight decay on the trainable flow parameters. Default 0.0
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
    if marginal is None:
        # per-pipeline default: the uniform mode's exact unit-range spline, or
        # the gaussian mode's clip-free smooth-tail marginal. The legacy tail
        # models pre-date the smooth marginal and still need the legacy spline
        # family, so requesting them resolves to it (old code keeps working).
        if copula_space == "uniform" or tail_extension or tail_model == "gpd":
            marginal = "rqs"
        else:
            marginal = "smooth"
    if marginal not in ("pchip", "rqs", "smooth"):
        raise ValueError(
            f"marginal must be 'pchip', 'rqs' or 'smooth', got {marginal!r}")

    if copula_space == "uniform":
        # Friendly guards: the low-level spline construction would also raise,
        # but these name the actionable fix at the API the user called.
        if tail_extension or tail_model == "gpd":
            raise ValueError(
                "Tail models (tail_extension=True / tail_model='gpd') describe "
                "unbounded support and require the Gaussianized pipeline: pass "
                "copula_space='gaussian'. The 'uniform' pipeline models "
                "bounded support exactly and has no tails."
            )
        if marginal != "rqs":
            raise ValueError(
                "copula_space='uniform' requires marginal='rqs' (a single "
                "self-consistent unit-range spline). 'smooth' and 'pchip' are "
                "gaussian-space marginals: pass copula_space='gaussian'."
            )
    else:
        if marginal == "smooth" and (tail_extension or tail_model == "gpd"):
            raise ValueError(
                "marginal='smooth' models tails natively (see tail_alpha) and "
                "is incompatible with tail_extension=True / tail_model='gpd'. "
                "Use the legacy marginal='rqs' or 'pchip' for those code paths."
            )

    key = jr.key(rng_seed)

    # Filter NaN rows once so both transform creation and training use clean data
    nan_rows = np.isnan(chain).any(axis=1)
    n_dropped = int(nan_rows.sum())
    if n_dropped:
        warnings.warn(
            f"normalizing_flows_fit dropped {n_dropped} of {len(chain)} rows "
            f"containing NaNs; fitting on the remaining {len(chain) - n_dropped}.",
            stacklevel=2,
        )
    clean_chain = chain[~nan_rows]

    # Create initial transforms using the helper function
    transform, inverse_log_det = _create_empirical_transforms(
        clean_chain, num_points=num_points, tail_extension=tail_extension,
        prior_bounds=prior_bounds,
        tail_model=tail_model, tail_quantile=tail_quantile,
        marginal=marginal, copula_space=copula_space, tail_alpha=tail_alpha)

    n_params = clean_chain.shape[1]
    arch = None
    if copula_space == "uniform":
        # Copula modeled natively on the unit cube: uniform base + RQS coupling
        # layers that map [0, 1]^d to itself with strictly positive derivative
        # everywhere (flowjax RQS enforces min_derivative > 0), so support is
        # exactly the cube and no clipping of any kind is involved.
        arch = ("uniform", n_params, knots, flow_layers)
        flow = _build_uniform_inner(arch, rng_seed=rng_seed)
    else:
        # Gaussianized copula: triangular spline flow on unbounded space
        # (original behavior).
        flow = triangular_spline_flow(
            key,
            base_dist=Normal(jnp.zeros(n_params)),
            knots=knots,
            flow_layers=flow_layers,
            tanh_max_val=tanh_max_val,
            invert=True
        )
    flow = _fit_chain_entry(flow, transform, inverse_log_det, clean_chain, rng_seed=rng_seed,
                            patience=patience, learning_rate=learning_rate, max_epochs=max_epochs,
                            arch=arch, weight_decay=weight_decay, show_progress=show_progress)

    return flow


def _n_trainable_params(flow: Transformed) -> int:
    """Total scalar parameter count of the trained (inner) flow.

    The frozen marginal transforms are excluded (identical role across
    candidates); this is the model-size measure used by the
    ``"smallest_comparable"`` selection rule.
    """
    leaves = jax.tree_util.tree_leaves(
        eqx.filter(flow.base_dist, eqx.is_inexact_array)
    )
    return int(sum(leaf.size for leaf in leaves))


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
    ``flow_layers``, ``learning_rate``, ``num_points``, and even
    ``copula_space`` — and remain directly comparable.

    .. note::
        Held-out likelihood measures proposal *efficiency* only. Boundary
        behavior -- no sampling atoms / density cliffs at hard prior bounds --
        lives in a measure-near-zero region that validation samples
        essentially never probe, so this criterion cannot reward it. Both
        pipelines are boundary-safe as of v2.0 (the default gaussian pipeline
        via its smooth-tail marginals, the uniform pipeline by construction),
        so let the candidates tune capacity; mixing ``copula_space``
        candidates is meaningful when comparing correlation-heavy versus
        flat-to-the-wall regimes.

    Parameters
    ----------
    chain : np.ndarray
        Training samples with shape (n_samples, n_params). NaN rows are
        dropped before splitting.
    candidates : list of dict, optional
        Candidate keyword overrides for :func:`normalizing_flows_fit`. Each
        dict is merged over ``**shared_kwargs``. The default is a small
        capacity ladder (validated on MCMC posteriors: the small rung wins on
        mild copulas, the larger rungs only when the dependence structure
        genuinely needs them)::

            [{"knots": 2, "flow_layers": 4},
             {"knots": 4, "flow_layers": 6},
             {"knots": 8, "flow_layers": 8}]

    val_frac : float, optional
        Fraction of the (clean) chain held out for scoring. Default 0.15.
    refit_best : bool, optional
        If True (default), refit the winning configuration on the full chain
        before returning it, so no samples are wasted. The reported scores
        always refer to the split fits.
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
        If True, return ``(flow, table)`` where ``table`` is a list of dicts
        with keys ``config``, ``val_logp``, ``sem``, ``n_params``,
        ``comparable``, ``selected``.
    **shared_kwargs
        Arguments passed to every candidate fit (e.g. ``prior_bounds``,
        ``copula_space``, ``max_epochs``).

    Returns
    -------
    Transformed  (or  (Transformed, list) when ``return_table=True``)
        The selected fitted flow.

    Examples
    --------
    >>> import numpy as np
    >>> from coppuccino import normalizing_flows_fit_auto
    >>> data = np.random.randn(2000, 2)
    >>> flow = normalizing_flows_fit_auto(data, max_epochs=100)   # doctest: +SKIP
    >>> # selecting across copula spaces as well:
    >>> flow = normalizing_flows_fit_auto(                        # doctest: +SKIP
    ...     data, prior_bounds=np.array([[-5, 5], [-5, 5]]),
    ...     candidates=[{"copula_space": "uniform", "knots": 4},
    ...                 {"copula_space": "uniform", "knots": 8},
    ...                 {"copula_space": "gaussian", "knots": 4}])
    """
    chain = np.asarray(chain)
    clean = chain[~np.isnan(chain).any(axis=1)]
    if len(clean) < 2 * (_MIN_SAMPLES_PER_DIM + 1):
        raise ValueError(
            f"normalizing_flows_fit_auto needs at least "
            f"{2 * (_MIN_SAMPLES_PER_DIM + 1)} non-NaN samples to split; "
            f"got {len(clean)}."
        )
    rng = np.random.default_rng(rng_seed)
    perm = rng.permutation(len(clean))
    n_val = max(1, int(round(len(clean) * val_frac)))
    val, fit_part = clean[perm[:n_val]], clean[perm[n_val:]]

    if candidates is None:
        candidates = [{"knots": 2, "flow_layers": 4},
                      {"knots": 4, "flow_layers": 6},
                      {"knots": 8, "flow_layers": 8}]

    if selection not in ("smallest_comparable", "best"):
        raise ValueError(
            f"selection must be 'smallest_comparable' or 'best', got {selection!r}."
        )

    def _fit_and_score(cand):
        # candidate fits are quiet by default (parallel progress bars garble);
        # an explicit show_progress in shared_kwargs/candidate still wins.
        kw = {"rng_seed": rng_seed, "show_progress": False,
              **shared_kwargs, **cand}
        fl = normalizing_flows_fit(fit_part, **kw)
        lp = np.asarray(log_prob(fl, val))
        return fl, lp

    if n_workers is None:
        import os
        n_workers = max(1, min(len(candidates), os.cpu_count() or 1))
    if n_workers > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            results = list(ex.map(_fit_and_score, candidates))
    else:
        results = [_fit_and_score(c) for c in candidates]

    flows = [fl for fl, _ in results]
    lps = [lp for _, lp in results]
    scores = np.array([float(np.mean(lp)) for lp in lps])
    sems = np.array([float(np.std(lp) / np.sqrt(len(lp))) for lp in lps])
    sizes = np.array([_n_trainable_params(fl) for fl in flows])
    best_idx = int(np.argmax(scores))

    # Comparability against the best, as a PAIRED test on the shared
    # validation points: the per-point score differences are highly
    # correlated across candidates, so this is far sharper than comparing
    # the individual +-sem error bars.
    comparable = np.zeros(len(candidates), dtype=bool)
    for i, lp in enumerate(lps):
        diff = lps[best_idx] - lp
        sem_diff = float(np.std(diff) / np.sqrt(len(diff)))
        comparable[i] = float(np.mean(diff)) <= (
            comparable_sems * sem_diff + comparable_nats
        )

    if selection == "smallest_comparable":
        pool = np.flatnonzero(comparable)
        # smallest parameter count; break size ties by score
        sel_idx = int(pool[np.lexsort((-scores[pool], sizes[pool]))[0]])
    else:
        sel_idx = best_idx

    table = []
    for i, cand in enumerate(candidates):
        table.append({"config": dict(cand), "val_logp": float(scores[i]),
                      "sem": float(sems[i]), "n_params": int(sizes[i]),
                      "comparable": bool(comparable[i]),
                      "selected": i == sel_idx})
        if verbose:
            print(f"[fit_auto] {i + 1}/{len(candidates)} {cand}: "
                  f"held-out logp {scores[i]:.3f} +- {sems[i]:.3f}  "
                  f"params {sizes[i]}  comparable {bool(comparable[i])}")

    best_flow = flows[sel_idx]
    if refit_best:
        kw = {"rng_seed": rng_seed, "show_progress": False,
              **shared_kwargs, **candidates[sel_idx]}
        best_flow = normalizing_flows_fit(clean, **kw)
    if verbose:
        print(f"[fit_auto] selected {candidates[sel_idx]} "
              f"(held-out logp {scores[sel_idx]:.3f}, {sizes[sel_idx]} params"
              f"{', refit on full chain' if refit_best else ''})")

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
    from paramax import unwrap
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


_MAX_REDRAW_ROUNDS = 50


def sample(flow: Transformed, n_samples: int, rng_seed: Optional[int] = None) -> np.ndarray:
    """
    Generate samples from a fitted copula flow.

    Draws random samples from the learned distribution by sampling from the
    base Gaussian distribution and transforming through the trained flow.

    For smooth-marginal flows (``copula_space="gaussian"``, the default
    ``marginal="smooth"``), rows whose bounded tails saturated at the float64
    wall-resolution limit are redrawn (typically well under 1% of draws).
    The emitted stream is then the reported density truncated to the
    clamp-free region and renormalized -- a constant that cancels in MH
    ratios -- so there are no near-boundary sample lumps.

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
    key = jr.key(_resolve_seed(rng_seed))
    # Use filter_jit for improved performance
    samples = np.array(filter_jit(flow.sample)(key, (n_samples,)))
    limits = _smooth_clamp_limits(flow)
    if limits is not None:
        for _ in range(_MAX_REDRAW_ROUNDS):
            bad = _clamped_rows(samples, limits)
            if not bad.any():
                break
            m = int(bad.sum())
            # draw in power-of-two blocks to bound JIT recompilations
            block = min(1 << (m - 1).bit_length(), n_samples)
            key, sub = jr.split(key)
            samples[bad] = np.array(
                filter_jit(flow.sample)(sub, (block,)))[:m]
        else:
            warnings.warn(
                "sample(): tail-clamped rows persisted after "
                f"{_MAX_REDRAW_ROUNDS} redraw rounds; returning the last "
                "draw. The flow's z-tails are extremely heavy relative to "
                "the bounded support.")
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
    >>> # Higher values indicate higher probability density (e.g. log_probs.mean())
    """
    log_probs = filter_jit(flow.log_prob)(samples)
    return np.array(log_probs)


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
    key = jr.key(_resolve_seed(rng_seed))
    samples, log_probs = filter_jit(flow.sample_and_log_prob)(key, (n_samples,))
    samples, log_probs = np.array(samples), np.array(log_probs)
    limits = _smooth_clamp_limits(flow)
    if limits is not None:
        # redraw tail-clamped rows (see `sample`); the returned density is
        # then exactly log_prob(samples) for every emitted row
        for _ in range(_MAX_REDRAW_ROUNDS):
            bad = _clamped_rows(samples, limits)
            if not bad.any():
                break
            m = int(bad.sum())
            block = min(1 << (m - 1).bit_length(), n_samples)
            key, sub = jr.split(key)
            s_new, lp_new = filter_jit(flow.sample_and_log_prob)(sub, (block,))
            samples[bad] = np.array(s_new)[:m]
            log_probs[bad] = np.array(lp_new)[:m]
        else:
            warnings.warn(
                "sample_and_log_prob(): tail-clamped rows persisted after "
                f"{_MAX_REDRAW_ROUNDS} redraw rounds; returning the last "
                "draw. The flow's z-tails are extremely heavy relative to "
                "the bounded support.")
    return samples, log_probs
