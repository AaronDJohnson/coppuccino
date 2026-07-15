import warnings
from collections.abc import Sequence
from typing import Callable, Optional

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
from flowjax.bijections import RationalQuadraticSpline, Stack
from flowjax.flows import coupling_flow, triangular_spline_flow
from flowjax.train import fit_to_data
from flowjax.distributions import Normal, Transformed, Uniform
from paramax import non_trainable, unwrap
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
        Marginal transform family: ``"smooth"``, ``"rqs"``, or ``"pchip"``.
        ``"smooth"`` (the gaussian pipeline's default) is a single clip-free
        monotone map of R onto the support with analytic tails. ``"rqs"``
        (the uniform pipeline's required family, also the legacy gaussian
        bulk) is a monotone rational-quadratic spline with a closed-form,
        machine-precision inverse from a single parameterization.
        ``"pchip"`` builds the CDF and its inverse as two independent splines
        that are only approximate inverses of each other; it is retained for
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
    if tail_model not in ("gaussian", "gpd"):
        raise ValueError(
            f"tail_model must be 'gaussian' or 'gpd', got {tail_model!r}")
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


def _wilson_interval(
    probability: float, n_eff: float, confidence: float,
) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion with effective size."""
    from scipy.special import ndtri

    z = float(ndtri(0.5 + confidence / 2.0))
    denominator = 1.0 + z**2 / n_eff
    centre = (probability + z**2 / (2.0 * n_eff)) / denominator
    radius = z * np.sqrt(
        probability * (1.0 - probability) / n_eff + z**2 / (4.0 * n_eff**2)
    ) / denominator
    return float(max(0.0, centre - radius)), float(min(1.0, centre + radius))


def _interval_gap(
    target: float,
    target_size: float,
    estimate: float,
    estimate_size: float,
    confidence: float,
) -> float:
    """Directional separation between two Wilson intervals."""
    target_low, target_high = _wilson_interval(target, target_size, confidence)
    estimate_low, estimate_high = _wilson_interval(
        estimate, estimate_size, confidence)
    return float(max(0.0, estimate_low - target_high,
                     target_low - estimate_high))


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
    if all(issubclass(kind, gaussian_types) for kind in marginal_types):
        thresholds = ndtri(quantiles)
    elif all(issubclass(kind, EmpiricalMarginalToUniform) for kind in marginal_types):
        thresholds = quantiles
    else:
        raise ValueError(
            "Marginal-fidelity selection requires a consistent Gaussian or "
            "uniform marginal transform across dimensions."
        )
    draws = np.asarray(flow.base_dist.sample(jr.key(rng_seed), (n_samples,)))
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

        Marginal-fidelity selection uses three fits: 2/2 and 4/4 knots/layers
        with ``tanh_max_val=2.0``, plus the existing 4/6,
        ``tanh_max_val=3.0`` default as a protected baseline. An explicit
        shared ``tanh_max_val`` applies to all three. Supply a shorter or
        broader list to trade cost against the search range.

    val_frac : float, optional
        Fraction of the (clean) chain held out for scoring. Default 0.15.
    refit_best : bool, optional
        If True (default), refit the winning configuration on the full chain
        before returning it. Marginal-fidelity refits get a fresh,
        componentwise marginal audit and a conservative joint-density
        regression sanity check; failures return the validated split fit.
        Because a full-data refit includes the validation rows, that latter
        check is not a second held-out guarantee. Use ``refit_best=False`` to
        return the strictly held-out-validated split fit. Reported held-out
        scores always refer to split fits.
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
        If True, return ``(flow, table)`` where ``table`` is a list of dicts
        with keys ``config``, ``val_logp``, ``sem``, ``n_params``,
        ``comparable``, ``selected``. Marginal selection adds component-level
        targets, estimates, errors, resolution status, and refit diagnostics.
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

    if selection not in ("smallest_comparable", "best", "marginal_fidelity"):
        raise ValueError(
            "selection must be 'smallest_comparable', 'best', or "
            f"'marginal_fidelity', got {selection!r}."
        )
    try:
        marginal_quantiles_are_default = tuple(marginal_quantiles) == (
            0.01, 0.025, 0.05, 0.95, 0.975, 0.99)
    except (TypeError, ValueError):
        marginal_quantiles_are_default = False
    if selection != "marginal_fidelity" and any((
        marginal_samples != 20_000,
        not marginal_quantiles_are_default,
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

    edge_features = None
    normalised_edges = None
    marginal_q = np.asarray([], dtype=float)
    if selection == "marginal_fidelity":
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
                    stacklevel=2,
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
        try:
            fl = normalizing_flows_fit(fit_part, **kw)
            lp = np.asarray(log_prob(fl, val))
            if lp.shape != (len(val),):
                raise ValueError(
                    f"log_prob returned shape {lp.shape}, expected {(len(val),)}."
                )
            marginal_error = np.nan
            marginal_mean_error = np.nan
            marginal_selection_error = np.nan
            marginal_details = []
            if selection == "marginal_fidelity":
                (marginal_error, marginal_mean_error,
                 marginal_selection_error, marginal_details) = (
                    _score_flow_marginals(fl, marginal_seed))
            return (fl, lp, marginal_error, marginal_mean_error,
                    marginal_selection_error, marginal_details, None)
        except Exception as exc:
            marginal_failure = (np.inf if selection == "marginal_fidelity"
                                else np.nan)
            error = f"{type(exc).__name__}: {exc}"
            return (None, np.full(len(val), np.nan), marginal_failure,
                    marginal_failure, marginal_failure, [], error)

    if n_workers is None:
        import os
        n_workers = max(1, min(len(candidates), os.cpu_count() or 1))
    if n_workers > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            results = list(ex.map(_fit_and_score, candidates))
    else:
        results = [_fit_and_score(c) for c in candidates]

    flows = [fl for fl, _, _, _, _, _, _ in results]
    lps = [lp for _, lp, _, _, _, _, _ in results]
    marginal_errors = np.asarray([err for _, _, err, _, _, _, _ in results])
    marginal_mean_errors = np.asarray([
        err for _, _, _, err, _, _, _ in results])
    marginal_selection_errors = np.asarray([
        err for _, _, _, _, err, _, _ in results])
    marginal_details = [details for _, _, _, _, _, details, _ in results]
    fit_errors = [error for _, _, _, _, _, _, error in results]
    failed = [f"candidate {i}: {error}" for i, error in enumerate(fit_errors)
              if error is not None]
    if failed:
        warnings.warn(
            "normalizing_flows_fit_auto excluded failed fits: "
            + "; ".join(failed),
            stacklevel=2,
        )
    scores = np.array([float(np.mean(lp)) for lp in lps])
    sems = np.array([float(np.std(lp) / np.sqrt(len(lp))) for lp in lps])
    failed_size = np.iinfo(np.int64).max
    sizes = np.array([
        _n_trainable_params(fl) if fl is not None else failed_size
        for fl in flows
    ])
    finite_candidates = np.asarray([
        fl is not None and bool(np.all(np.isfinite(lp)))
        for fl, lp in zip(flows, lps)
    ])
    if not finite_candidates.any():
        failure_text = f" Fit errors: {'; '.join(failed)}." if failed else ""
        raise RuntimeError(
            "normalizing_flows_fit_auto: every candidate produced a non-finite "
            f"held-out score ({scores.tolist()}); the fits likely diverged. "
            "Try a smaller learning_rate or more training samples."
            + failure_text
        )
    best_idx = int(np.argmax(np.where(finite_candidates, scores, -np.inf)))

    # Comparability against the best, as a PAIRED test on the shared
    # validation points: the per-point score differences are highly
    # correlated across candidates, so this is far sharper than comparing
    # the individual +-sem error bars.
    comparable = np.zeros(len(candidates), dtype=bool)
    for i, lp in enumerate(lps):
        if not finite_candidates[i]:
            continue
        diff = lps[best_idx] - lp
        sem_diff = float(np.std(diff) / np.sqrt(len(diff)))
        comparable[i] = float(np.mean(diff)) <= (
            comparable_sems * sem_diff + comparable_nats
        )

    marginal_comparable = np.zeros(len(candidates), dtype=bool)
    marginal_within_budget = np.zeros(len(candidates), dtype=bool)
    if selection in ("smallest_comparable", "marginal_fidelity"):
        pool = np.flatnonzero(comparable)
        if len(pool) == 0:
            # Defensive fallback for a degenerate paired comparison. The best
            # index was selected only from wholly finite held-out arrays.
            pool = np.array([best_idx])
        if selection == "smallest_comparable":
            # smallest parameter count; break size ties by score
            sel_idx = int(pool[np.lexsort((-scores[pool], sizes[pool]))[0]])
        else:
            finite_pool = pool[np.isfinite(marginal_selection_errors[pool])]
            if len(finite_pool) == 0:
                raise RuntimeError(
                    "Every held-out-comparable candidate produced a non-finite "
                    "marginal-fidelity score."
                )
            ranking_pool = finite_pool
            if max_marginal_error is not None:
                finite_marginals = np.isfinite(marginal_errors)
                marginal_within_budget[finite_marginals] = (
                    marginal_errors[finite_marginals] <= max_marginal_error)
                budget_pool = finite_pool[marginal_within_budget[finite_pool]]
                if len(budget_pool):
                    ranking_pool = budget_pool
                else:
                    warnings.warn(
                        "No held-out-likelihood-comparable candidate met "
                        f"max_marginal_error={max_marginal_error:g}; returning "
                        "the least-bad fit. Inspect marginal_details before use.",
                        stacklevel=2,
                    )
            best_marginal = float(np.min(
                marginal_selection_errors[ranking_pool]))
            tolerance = (0.5 / np.sqrt(marginal_samples)
                         if marginal_tolerance is None else marginal_tolerance)
            marginal_comparable[ranking_pool] = (
                marginal_selection_errors[ranking_pool]
                <= best_marginal + tolerance)
            marginal_pool = np.flatnonzero(marginal_comparable)
            order = np.lexsort((
                -scores[marginal_pool],
                marginal_selection_errors[marginal_pool],
                sizes[marginal_pool],
            ))
            sel_idx = int(marginal_pool[order[0]])
    else:
        sel_idx = best_idx

    table = []
    for i, cand in enumerate(candidates):
        fit_status = ("failed" if flows[i] is None else
                      "nonfinite_score" if not finite_candidates[i] else "ok")
        row = {"config": dict(cand), "val_logp": float(scores[i]),
               "sem": float(sems[i]),
               "n_params": (None if flows[i] is None else int(sizes[i])),
               "comparable": bool(comparable[i]),
               "selected": i == sel_idx,
               "fit_status": fit_status,
               "fit_error": fit_errors[i]}
        if selection == "marginal_fidelity":
            worst = (max(marginal_details[i], key=lambda item: item["error"])
                     if marginal_details[i] else None)
            row.update({
                "marginal_error": float(marginal_errors[i]),
                "marginal_mean_error": float(marginal_mean_errors[i]),
                "marginal_selection_error": float(marginal_selection_errors[i]),
                "marginal_comparable": bool(marginal_comparable[i]),
                "marginal_within_budget": (
                    None if max_marginal_error is None
                    else bool(marginal_within_budget[i])),
                "marginal_components": (len(edge_features)
                    if edge_features is not None
                    else clean.shape[1] * len(marginal_q)),
                "marginal_unresolved_components": sum(
                    not item["resolved"] for item in marginal_details[i]),
                "marginal_metric_source": ("literal_edge_bins"
                    if edge_features is not None else "latent_quantiles"),
                "marginal_worst_dimension": (
                    None if worst is None else worst["dimension"]),
                "marginal_worst_component": (
                    None if worst is None else worst["component"]),
                "marginal_details": marginal_details[i],
            })
        table.append(row)
        if verbose:
            params_text = ("failed" if flows[i] is None else str(sizes[i]))
            marginal_text = (
                f"  marginal {marginal_errors[i]:.5f} "
                f"(selection {marginal_selection_errors[i]:.5f})"
                if selection == "marginal_fidelity" else "")
            print(f"[fit_auto] {i + 1}/{len(candidates)} {cand}: "
                  f"held-out logp {scores[i]:.3f} +- {sems[i]:.3f}  "
                  f"params {params_text}  comparable {bool(comparable[i])}"
                  f"{marginal_text}")

    best_flow = flows[sel_idx]
    assert best_flow is not None
    used_refit = False
    if refit_best:
        kw = {"rng_seed": rng_seed, "show_progress": False,
              **shared_kwargs, **candidates[sel_idx]}
        try:
            refit_flow = normalizing_flows_fit(clean, **kw)
            if selection == "marginal_fidelity":
                (audit_error, audit_mean_error, audit_selection_error,
                 audit_details) = _score_flow_marginals(
                    best_flow, marginal_audit_seed)
                (refit_error, refit_mean_error, refit_selection_error,
                 refit_details) = _score_flow_marginals(
                    refit_flow, marginal_audit_seed)
                refit_lp = np.asarray(log_prob(refit_flow, val))
                refit_density_finite = (
                    refit_lp.shape == (len(val),)
                    and bool(np.all(np.isfinite(refit_lp)))
                )
                refit_joint_sanity_passed = False
                if refit_density_finite:
                    refit_diff = lps[sel_idx] - refit_lp
                    refit_sem_diff = float(
                        np.std(refit_diff) / np.sqrt(len(refit_diff)))
                    refit_joint_sanity_passed = float(np.mean(refit_diff)) <= (
                        comparable_sems * refit_sem_diff + comparable_nats)
                tolerance = (0.5 / np.sqrt(marginal_samples)
                             if marginal_tolerance is None
                             else marginal_tolerance)
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
                        if after["selection_error"]
                        > before["selection_error"] + tolerance
                    ]
                audit_met_budget = (
                    max_marginal_error is None
                    or audit_error <= max_marginal_error
                )
                refit_met_budget = (
                    max_marginal_error is None
                    or refit_error <= max_marginal_error
                )
                refit_used = (
                    np.isfinite(audit_selection_error)
                    and np.isfinite(refit_selection_error)
                    and refit_density_finite
                    and refit_joint_sanity_passed
                    and details_align
                    and not regressions
                    and refit_met_budget
                )
                table[sel_idx].update({
                    "audit_marginal_error": float(audit_error),
                    "audit_marginal_mean_error": float(audit_mean_error),
                    "audit_marginal_selection_error": float(
                        audit_selection_error),
                    "audit_marginal_details": audit_details,
                    "audit_marginal_within_budget": (
                        None if max_marginal_error is None
                        else bool(audit_met_budget)),
                    "refit_marginal_error": float(refit_error),
                    "refit_marginal_mean_error": float(refit_mean_error),
                    "refit_marginal_selection_error": float(
                        refit_selection_error),
                    "refit_marginal_details": refit_details,
                    "refit_density_finite": bool(refit_density_finite),
                    "refit_marginal_within_budget": (
                        None if max_marginal_error is None
                        else bool(refit_met_budget)),
                    "refit_joint_sanity_passed": bool(
                        refit_joint_sanity_passed),
                    "refit_regressed_components": regressions,
                    "refit_used": bool(refit_used),
                })
                if refit_used:
                    best_flow = refit_flow
                    used_refit = True
                else:
                    budget_note = (
                        " The returned split fit also exceeded the error budget "
                        "on the fresh audit."
                        if max_marginal_error is not None
                        and not audit_met_budget else ""
                    )
                    warnings.warn(
                        "The full-data refit worsened one or more marginal "
                        "components, exceeded the requested error budget, or "
                        "failed the joint-fit sanity or finite-density check; "
                        "returning the validated split fit instead. The selected "
                        "table row reports the refit diagnostics." + budget_note,
                        stacklevel=2,
                    )
            else:
                best_flow = refit_flow
                used_refit = True
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
              f"(held-out logp {scores[sel_idx]:.3f}, {sizes[sel_idx]} params"
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


_MAX_REDRAW_ROUNDS = 50


def sample(flow: Transformed, n_samples: int, rng_seed: Optional[int] = None) -> np.ndarray:
    """
    Generate samples from a fitted copula flow.

    Draws random samples from the learned distribution by sampling from the
    base distribution (standard normal for the gaussian pipeline, uniform on
    the unit cube for the uniform pipeline) and transforming through the
    trained flow.

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
    >>> # np.exp(log_probs) is the proposal density q(x); importance
    >>> # weights against a target require it in the denominator:
    >>> # weights = target_pdf(samples) / np.exp(log_probs)
    >>> proposal_density = np.exp(log_probs)
    """
    key = jr.key(_resolve_seed(rng_seed))
    samples, log_probs = filter_jit(flow.sample_and_log_prob)(key, (n_samples,))
    samples, log_probs = np.array(samples), np.array(log_probs)
    limits = _smooth_clamp_limits(flow)
    if limits is not None:
        # redraw tail-clamped rows (see `sample`)
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
