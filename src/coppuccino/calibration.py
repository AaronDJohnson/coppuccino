"""Post-fit marginal calibration for Gaussian-space copula flows.

The trained Gaussian-space flow is flexible enough that its one-dimensional
marginals need not remain exactly standard normal.  Small marginal drift is
most visible in sparse edge bins.  This module corrects selected edge-bin CDF
boundaries after fitting, without retraining the joint flow.

The correction is coordinatewise and strictly increasing, so it preserves the
copula (and therefore every coordinate's ranks) exactly.  Each correction has
compact support: it is explicitly the identity through a central region and in
the deep tails.  Forward, inverse, and Jacobian evaluations share one rational-
quadratic-spline parameterization.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any, ClassVar, NamedTuple, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from flowjax.bijections import AbstractBijection, Stack
from flowjax.distributions import Transformed
from paramax import non_trainable, unwrap

from coppuccino._stats import wilson_interval
from coppuccino.bijections import (
    _POS_FLOOR,
    SmoothMarginalToGaussian,
    _rqs_derivative,
    _rqs_forward,
    _rqs_inverse,
    _rqs_knot_derivatives,
)

__all__ = ["calibrate_edge_bins"]


_KNOT_DERIVATIVE_FLOOR = 1e-8


def _rqs_forward_and_log_det(x, xk, yk, dk):
    """Evaluate a monotone RQS and its log derivative."""
    return (_rqs_forward(x, xk, yk, dk),
            jnp.log(_rqs_derivative(x, xk, yk, dk, floor=_POS_FLOOR)))


def _rqs_inverse_and_log_det(y, xk, yk, dk):
    """Evaluate the closed-form inverse RQS and its log derivative."""
    x = _rqs_inverse(y, xk, yk, dk)
    _, forward_log_det = _rqs_forward_and_log_det(x, xk, yk, dk)
    return x, -forward_log_det


class _EdgeMarginalCalibration(AbstractBijection):
    """Two compact RQS windows separated by an exact identity core.

    The lower and upper windows both begin and end at identity guard knots.
    Outside either active window the implementation returns the input and a
    zero log-Jacobian directly; the core and deep tails are therefore bitwise
    unchanged rather than merely algebraically equivalent to the identity.
    """

    lower_x: jax.Array
    lower_y: jax.Array
    lower_derivatives: jax.Array
    upper_x: jax.Array
    upper_y: jax.Array
    upper_derivatives: jax.Array
    lower_enabled: bool = eqx.field(static=True)
    upper_enabled: bool = eqx.field(static=True)
    # flowjax declares shape/cond_shape as equinox ``AbstractVar``, which mypy
    # reads as instance variables of the base; a concrete ``ClassVar`` is the
    # intended implementation here (both are per-class constants).
    cond_shape: ClassVar[None] = None  # type: ignore[misc]
    shape: ClassVar[tuple[int, ...]] = ()  # type: ignore[misc]

    def __init__(
        self,
        lower_x,
        lower_y,
        lower_derivatives,
        upper_x,
        upper_y,
        upper_derivatives,
        lower_enabled: bool = True,
        upper_enabled: bool = True,
    ):
        self.lower_x = jnp.asarray(lower_x)
        self.lower_y = jnp.asarray(lower_y)
        self.lower_derivatives = jnp.asarray(lower_derivatives)
        self.upper_x = jnp.asarray(upper_x)
        self.upper_y = jnp.asarray(upper_y)
        self.upper_derivatives = jnp.asarray(upper_derivatives)
        self.lower_enabled = bool(lower_enabled)
        self.upper_enabled = bool(upper_enabled)

    def transform_and_log_det(self, x, condition=None):
        x = jnp.asarray(x)
        lower_active = (
            (x > self.lower_x[0]) & (x < self.lower_x[-1])
            if self.lower_enabled
            else jnp.asarray(False)
        )
        upper_active = (
            (x > self.upper_x[0]) & (x < self.upper_x[-1])
            if self.upper_enabled
            else jnp.asarray(False)
        )

        lower_input = jnp.clip(x, self.lower_x[0], self.lower_x[-1])
        lower_value, lower_log_det = _rqs_forward_and_log_det(
            lower_input, self.lower_x, self.lower_y, self.lower_derivatives
        )
        upper_input = jnp.clip(x, self.upper_x[0], self.upper_x[-1])
        upper_value, upper_log_det = _rqs_forward_and_log_det(
            upper_input, self.upper_x, self.upper_y, self.upper_derivatives
        )

        y = jnp.where(
            lower_active,
            lower_value,
            jnp.where(upper_active, upper_value, x),
        )
        log_det = jnp.where(
            lower_active,
            lower_log_det,
            jnp.where(upper_active, upper_log_det, 0.0),
        )
        return y, log_det

    def inverse_and_log_det(self, y, condition=None):
        y = jnp.asarray(y)
        lower_active = (
            (y > self.lower_y[0]) & (y < self.lower_y[-1])
            if self.lower_enabled
            else jnp.asarray(False)
        )
        upper_active = (
            (y > self.upper_y[0]) & (y < self.upper_y[-1])
            if self.upper_enabled
            else jnp.asarray(False)
        )

        lower_input = jnp.clip(y, self.lower_y[0], self.lower_y[-1])
        lower_value, lower_log_det = _rqs_inverse_and_log_det(
            lower_input, self.lower_x, self.lower_y, self.lower_derivatives
        )
        upper_input = jnp.clip(y, self.upper_y[0], self.upper_y[-1])
        upper_value, upper_log_det = _rqs_inverse_and_log_det(
            upper_input, self.upper_x, self.upper_y, self.upper_derivatives
        )

        x = jnp.where(
            lower_active,
            lower_value,
            jnp.where(upper_active, upper_value, y),
        )
        log_det = jnp.where(
            lower_active,
            lower_log_det,
            jnp.where(upper_active, upper_log_det, 0.0),
        )
        return x, log_det


def _window_derivatives(x_knots: np.ndarray, y_knots: np.ndarray) -> np.ndarray:
    """Positive shape-preserving knot derivatives with unit-slope guards.

    The floor is far above the generic ``_POS_FLOOR`` the other RQS knots use:
    a window here must stay invertible at working precision, not merely
    non-degenerate. The unit endpoint slopes are what make each window join
    the identity core with a continuous Jacobian.
    """
    return _rqs_knot_derivatives(
        x_knots, y_knots, floor=_KNOT_DERIVATIVE_FLOOR, edge_slopes=(1.0, 1.0)
    )


def _make_calibration(
    lower_source: np.ndarray,
    lower_target: np.ndarray,
    upper_source: np.ndarray,
    upper_target: np.ndarray,
    source_samples: np.ndarray,
    target_samples: np.ndarray,
    core_quantiles: tuple[float, float],
    clamp_limits: tuple[Optional[float], Optional[float]],
    active_sides: tuple[bool, bool],
) -> tuple[_EdgeMarginalCalibration, tuple[float, float]]:
    """Construct compact lower/upper windows around calibration anchors."""
    q_low, q_high = core_quantiles
    lower_displacement = float(np.max(np.abs(lower_source - lower_target)))
    upper_displacement = float(np.max(np.abs(upper_source - upper_target)))
    lower_anchor_edge = max(float(np.max(lower_source)), float(np.max(lower_target)))
    upper_anchor_edge = min(float(np.min(upper_source)), float(np.min(upper_target)))
    if not lower_anchor_edge < upper_anchor_edge:
        raise ValueError(
            "The lower and upper calibration anchors cross, so no monotone "
            "calibration with an exact identity core exists."
        )
    available_core_width = upper_anchor_edge - lower_anchor_edge
    # Leave a resolvable shoulder between the last correction anchor and the
    # identity join without letting a large marginal shift erase the entire
    # core. The deep-tail guard below supplies the wider compensation interval.
    lower_width = min(max(0.10, 0.25 * lower_displacement), 0.20 * available_core_width)
    upper_width = min(max(0.10, 0.25 * upper_displacement), 0.20 * available_core_width)

    lower_join = max(
        float(np.quantile(source_samples, q_low)),
        float(np.quantile(target_samples, q_low)),
        lower_anchor_edge + lower_width,
    )
    upper_join = min(
        float(np.quantile(source_samples, q_high)),
        float(np.quantile(target_samples, q_high)),
        upper_anchor_edge - upper_width,
    )
    if not lower_join < upper_join:
        # The requested quantile joins can be incompatible with a large but
        # otherwise feasible edge correction. Fall back to an adaptive central
        # interval; the report exposes the actual identity limits.
        lower_join = lower_anchor_edge + 0.25 * available_core_width
        upper_join = upper_anchor_edge - 0.25 * available_core_width

    lower_anchor_start = min(float(np.min(lower_source)), float(np.min(lower_target)))
    upper_anchor_end = max(float(np.max(upper_source)), float(np.max(upper_target)))
    lower_guard = lower_anchor_start - max(0.5, 2.0 * lower_displacement)
    upper_guard = upper_anchor_end + max(0.5, 2.0 * upper_displacement)

    # `sample()` rejects rows whose smooth bounded marginal has reached its
    # float64 clamp. Keep the calibration identically inactive throughout that
    # region so pre- and post-calibration rejection sets are exactly the same.
    clamp_low, clamp_high = clamp_limits
    if clamp_low is not None:
        if not clamp_low < lower_anchor_start:
            raise ValueError("The lower calibration anchor lies inside the marginal clamp region.")
        clamp_margin = min(0.25, 0.1 * (lower_anchor_start - clamp_low))
        lower_guard = max(lower_guard, clamp_low + clamp_margin)
    if clamp_high is not None:
        if not upper_anchor_end < clamp_high:
            raise ValueError("The upper calibration anchor lies inside the marginal clamp region.")
        clamp_margin = min(0.25, 0.1 * (clamp_high - upper_anchor_end))
        upper_guard = min(upper_guard, clamp_high - clamp_margin)

    lower_x = np.concatenate([[lower_guard], lower_source, [lower_join]])
    lower_y = np.concatenate([[lower_guard], lower_target, [lower_join]])
    upper_x = np.concatenate([[upper_join], upper_source, [upper_guard]])
    upper_y = np.concatenate([[upper_join], upper_target, [upper_guard]])

    for name, knots in (
        ("lower source", lower_x),
        ("lower target", lower_y),
        ("upper source", upper_x),
        ("upper target", upper_y),
    ):
        if not np.all(np.diff(knots) > 0.0):
            raise ValueError(
                f"{name} calibration knots are not strictly increasing. "
                "This usually means an edge bin is empty or the requested "
                "calibration is too large for an identity-core map."
            )

    calibration = _EdgeMarginalCalibration(
        lower_x,
        lower_y,
        _window_derivatives(lower_x, lower_y),
        upper_x,
        upper_y,
        _window_derivatives(upper_x, upper_y),
        lower_enabled=active_sides[0],
        upper_enabled=active_sides[1],
    )
    return calibration, (lower_join, upper_join)


def _normalise_bin_edges(
    bin_edges: Sequence[np.ndarray] | np.ndarray,
    n_dims: int,
) -> list[np.ndarray]:
    """Return one validated edge array per dimension."""
    if isinstance(bin_edges, np.ndarray) and bin_edges.ndim == 1:
        if n_dims != 1:
            raise ValueError(
                "A one-dimensional bin_edges array is only valid for a 1D flow; "
                "pass one edge array per dimension."
            )
        edges = [np.asarray(bin_edges, dtype=float)]
    else:
        edges = [np.asarray(edge, dtype=float) for edge in bin_edges]
        if len(edges) != n_dims:
            raise ValueError(f"bin_edges has {len(edges)} dimensions, but the flow has {n_dims}.")

    for dim, edge in enumerate(edges):
        if edge.ndim != 1 or len(edge) < 4:
            raise ValueError(f"bin_edges[{dim}] must be a 1D array with at least 4 edges.")
        if not np.all(np.isfinite(edge)) or not np.all(np.diff(edge) > 0.0):
            raise ValueError(f"bin_edges[{dim}] must be finite and strictly increasing.")
    return edges


def _is_edge_calibrated(flow: Transformed) -> bool:
    inner = flow.base_dist
    if not isinstance(inner, Transformed) or not isinstance(inner.bijection, Stack):
        return False
    return all(
        isinstance(unwrap(bijection), _EdgeMarginalCalibration)
        for bijection in inner.bijection.bijections
    )


def _normalise_reference_ess(
    reference_ess: Optional[float | Sequence[float]],
    n_dims: int,
    n_reference: int,
) -> np.ndarray:
    if reference_ess is None:
        return np.full(n_dims, float(n_reference))
    values = np.asarray(reference_ess, dtype=float)
    if values.ndim == 0:
        values = np.full(n_dims, float(values))
    if values.shape != (n_dims,):
        raise ValueError(f"reference_ess must be a scalar or have shape ({n_dims},).")
    if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("reference_ess values must be finite and positive.")
    if np.any(values > n_reference):
        raise ValueError("reference_ess cannot exceed the number of reference rows.")
    return values


def _gated_probabilities(
    target: np.ndarray,
    before: np.ndarray,
    effective_n: float,
    confidence_level: float,
    force: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return minimum-change targets plus active masks and score intervals."""
    lower, upper = wilson_interval(target, effective_n, confidence_level)
    if force:
        return target.copy(), np.ones_like(target, dtype=bool), lower, upper
    active_low = before < lower
    active_high = before > upper
    active = active_low | active_high
    fitted = np.where(active_low, lower, np.where(active_high, upper, before))
    return fitted, active, lower, upper


class _CalibrationSettings(NamedTuple):
    """The validated scalar knobs every dimension's calibration shares.

    Bundling them keeps the per-dimension helpers from re-declaring the public
    signature; the public function still takes them one by one.
    """

    edge_bins: int
    core_quantiles: tuple[float, float]
    n_calibration_samples: int
    confidence_level: float
    min_reference_bin_count: int
    min_calibration_tail_count: int
    force: bool


def _flow_marginals(flow: Transformed) -> list[SmoothMarginalToGaussian]:
    """Return the flow's smooth marginal transforms, rejecting anything else.

    Notes
    -----
    The ``FleetFlow`` import is deferred: ``coppuccino.fleet`` imports this
    package's fitting and IO modules, so importing it at module scope would
    close an import cycle. ``coppuccino.copula_flows`` defers it for the same
    reason.
    """
    if not isinstance(flow, Transformed):
        from coppuccino.fleet import FleetFlow

        if isinstance(flow, FleetFlow):
            raise TypeError(
                "calibrate_edge_bins() takes a single fitted Transformed flow; a "
                f"FleetFlow of {flow.n_sources} sources carries per-source cube "
                "marginals and no Gaussian marginal stack to calibrate. Refit the "
                "source of interest with normalizing_flows_fit(chain, "
                "prior_bounds=...) and calibrate that flow."
            )
        raise TypeError("calibrate_edge_bins requires a fitted Transformed flow.")
    if getattr(flow, "arch", None) is not None:
        raise ValueError("calibrate_edge_bins currently supports Gaussian-space flows only.")
    if _is_edge_calibrated(flow):
        raise ValueError("This flow already has an edge calibration layer.")
    if not isinstance(flow.bijection, Stack):
        raise ValueError("Expected the flow's outer marginal transform to be a Stack.")

    marginals = [unwrap(bijection) for bijection in flow.bijection.bijections]
    for dim, marginal in enumerate(marginals):
        if not isinstance(marginal, SmoothMarginalToGaussian):
            raise ValueError(
                "calibrate_edge_bins requires every marginal to be a "
                "SmoothMarginalToGaussian, as produced by normalizing_flows_fit("
                "..., marginal='smooth') -- the default. Dimension "
                f"{dim} carries {type(marginal).__name__}."
            )
    return marginals


def _clean_reference(reference_samples: np.ndarray, n_dims: int) -> np.ndarray:
    """Coerce the reference sample and drop non-finite rows, with a count."""
    reference = np.asarray(reference_samples, dtype=float)
    if reference.ndim != 2 or reference.shape[1] != n_dims:
        raise ValueError(f"reference_samples must have shape (n, {n_dims}); got {reference.shape}.")
    finite = np.isfinite(reference).all(axis=1)
    if not finite.all():
        warnings.warn(
            f"calibrate_edge_bins dropped {int((~finite).sum())} non-finite rows.",
            stacklevel=3,
        )
        reference = reference[finite]
    if len(reference) < 20:
        raise ValueError("calibrate_edge_bins needs at least 20 finite reference rows.")
    return reference


def _validated_settings(
    edge_bins: int,
    core_quantiles: tuple[float, float],
    n_calibration_samples: int,
    confidence_level: float,
    min_reference_bin_count: int,
    min_calibration_tail_count: int,
    force: bool,
) -> _CalibrationSettings:
    """Range-check the scalar knobs, naming the offending one."""
    if not isinstance(edge_bins, int) or edge_bins < 1:
        raise ValueError("edge_bins must be a positive integer.")
    q_low, q_high = map(float, core_quantiles)
    if not 0.0 < q_low < q_high < 1.0:
        raise ValueError("core_quantiles must satisfy 0 < low < high < 1.")
    if n_calibration_samples < 1_000:
        raise ValueError("n_calibration_samples must be at least 1,000.")
    confidence_level = float(confidence_level)
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between 0 and 1.")
    if not isinstance(min_reference_bin_count, int) or min_reference_bin_count < 1:
        raise ValueError("min_reference_bin_count must be a positive integer.")
    if not isinstance(min_calibration_tail_count, int) or min_calibration_tail_count < 1:
        raise ValueError("min_calibration_tail_count must be a positive integer.")
    return _CalibrationSettings(
        edge_bins=edge_bins,
        core_quantiles=(q_low, q_high),
        n_calibration_samples=n_calibration_samples,
        confidence_level=confidence_level,
        min_reference_bin_count=min_reference_bin_count,
        min_calibration_tail_count=min_calibration_tail_count,
        force=force,
    )


def _edge_cdf_targets(
    reference_column: np.ndarray,
    lower_boundaries: np.ndarray,
    upper_boundaries: np.ndarray,
    dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Empirical CDF of the reference at each requested edge boundary."""
    sorted_reference = np.sort(reference_column)
    lower_counts = np.searchsorted(sorted_reference, lower_boundaries, side="left")
    upper_counts = np.searchsorted(sorted_reference, upper_boundaries, side="left")
    lower_probabilities = lower_counts / len(reference_column)
    upper_probabilities = upper_counts / len(reference_column)

    probabilities = np.concatenate([lower_probabilities, upper_probabilities])
    if np.any(probabilities <= 0.0) or np.any(probabilities >= 1.0):
        raise ValueError(
            f"Dimension {dim} has a requested edge boundary with empirical "
            "probability 0 or 1; widen/aggregate the edge bins."
        )
    if not np.all(np.diff(lower_probabilities) > 0.0):
        raise ValueError(f"Dimension {dim} has an empty lower edge bin.")
    if not np.all(np.diff(upper_probabilities) > 0.0):
        raise ValueError(f"Dimension {dim} has an empty upper edge bin.")
    if lower_probabilities[-1] >= upper_probabilities[0]:
        raise ValueError(f"Dimension {dim} edge calibration regions overlap.")
    return lower_probabilities, upper_probabilities


def _reference_interval_counts(
    lower_probabilities: np.ndarray,
    upper_probabilities: np.ndarray,
    effective_size: float,
    settings: _CalibrationSettings,
    dim: int,
) -> np.ndarray:
    """Effective reference counts in the selected tail-inclusive intervals.

    Unlike ``np.histogram`` counts these include under/overflow on a cropped
    axis, which is what the CDF anchors are defined against.
    """
    interval_probabilities = np.concatenate(
        [
            np.diff(np.concatenate([[0.0], lower_probabilities])),
            np.diff(np.concatenate([upper_probabilities, [1.0]])),
        ]
    )
    effective_interval_counts = interval_probabilities * effective_size
    if np.any(effective_interval_counts < settings.min_reference_bin_count):
        message = (
            f"Dimension {dim} has a selected tail interval with effective "
            f"count {effective_interval_counts.min():.1f}, below "
            f"min_reference_bin_count={settings.min_reference_bin_count}; aggregate "
            "the bins or provide a larger effective reference sample."
        )
        if settings.force:
            warnings.warn(message, stacklevel=4)
        else:
            raise ValueError(message)
    return effective_interval_counts


def _check_calibration_resolution(
    fitted_lower: np.ndarray,
    fitted_upper: np.ndarray,
    active_lower: np.ndarray,
    active_upper: np.ndarray,
    settings: _CalibrationSettings,
    dim: int,
) -> None:
    """Require enough calibration draws to resolve each fitted quantile.

    An inactive anchor is still a knot of the calibration spline, so the
    intervals a fitted quantile has to resolve are the gaps between *all*
    consecutive anchors, not only the active ones. A side with no active anchor
    is disabled and estimates no quantile at all, so only the gaps touching an
    active anchor are checked.
    """
    lower_gaps = np.diff(np.concatenate([[0.0], fitted_lower]))
    upper_gaps = np.diff(np.concatenate([fitted_upper, [1.0]]))
    lower_checked = active_lower.copy()
    lower_checked[1:] |= active_lower[:-1]
    upper_checked = active_upper.copy()
    upper_checked[:-1] |= active_upper[1:]
    active_probability_intervals = np.concatenate(
        [lower_gaps[lower_checked], upper_gaps[upper_checked]]
    )
    if not active_probability_intervals.size:
        return
    calibration_counts = active_probability_intervals * settings.n_calibration_samples
    if np.any(calibration_counts < settings.min_calibration_tail_count):
        message = (
            f"Dimension {dim} has an active calibration interval with only "
            f"{calibration_counts.min():.1f} expected calibration draws, "
            f"below min_calibration_tail_count={settings.min_calibration_tail_count}; "
            "increase n_calibration_samples or aggregate the bins."
        )
        if settings.force:
            warnings.warn(message, stacklevel=4)
        else:
            raise ValueError(message)


def _marginal_clamp_limits(
    marginal: SmoothMarginalToGaussian,
) -> tuple[Optional[float], Optional[float]]:
    """Latent positions of the marginal's float64 clamps, where it has them."""
    return (
        (
            float(marginal.z_of_theta(jnp.asarray(marginal.x_clamp_lo)))
            if marginal.x_clamp_lo is not None
            else None
        ),
        (
            float(marginal.z_of_theta(jnp.asarray(marginal.x_clamp_hi)))
            if marginal.x_clamp_hi is not None
            else None
        ),
    )


def _calibrate_dimension(
    dim: int,
    marginal: SmoothMarginalToGaussian,
    edge: np.ndarray,
    reference_column: np.ndarray,
    physical_column: np.ndarray,
    latent_column: np.ndarray,
    effective_size: float,
    settings: _CalibrationSettings,
) -> tuple[_EdgeMarginalCalibration, dict[str, Any]]:
    """Build one coordinate's calibration window pair and its report entry."""
    edge_bins = settings.edge_bins
    lower_boundaries = edge[1 : edge_bins + 1]
    upper_boundaries = edge[-edge_bins - 1 : -1]
    lower_probabilities, upper_probabilities = _edge_cdf_targets(
        reference_column, lower_boundaries, upper_boundaries, dim
    )
    effective_interval_counts = _reference_interval_counts(
        lower_probabilities, upper_probabilities, effective_size, settings, dim
    )

    target = np.asarray(marginal.z_of_theta(jnp.asarray(reference_column)))
    lower_target = np.asarray(marginal.z_of_theta(jnp.asarray(lower_boundaries)))
    upper_target = np.asarray(marginal.z_of_theta(jnp.asarray(upper_boundaries)))
    before_lower = np.asarray(
        [np.mean(physical_column < boundary) for boundary in lower_boundaries]
    )
    before_upper = np.asarray(
        [np.mean(physical_column < boundary) for boundary in upper_boundaries]
    )
    fitted_lower, active_lower, lower_ci, lower_ci_high = _gated_probabilities(
        lower_probabilities,
        before_lower,
        effective_size,
        settings.confidence_level,
        settings.force,
    )
    fitted_upper, active_upper, upper_ci, upper_ci_high = _gated_probabilities(
        upper_probabilities,
        before_upper,
        effective_size,
        settings.confidence_level,
        settings.force,
    )
    _check_calibration_resolution(
        fitted_lower, fitted_upper, active_lower, active_upper, settings, dim
    )

    # Compatible boundaries are literal identity anchors. Active boundaries
    # use the nearest Wilson edge (or the raw ECDF under force=True), making
    # the statistically smallest correction that is supported by the data.
    lower_source = lower_target.copy()
    upper_source = upper_target.copy()
    if active_lower.any():
        lower_source[active_lower] = np.quantile(latent_column, fitted_lower[active_lower])
    if active_upper.any():
        upper_source[active_upper] = np.quantile(latent_column, fitted_upper[active_upper])

    calibration, identity_core = _make_calibration(
        np.atleast_1d(lower_source),
        np.atleast_1d(lower_target),
        np.atleast_1d(upper_source),
        np.atleast_1d(upper_target),
        latent_column,
        target,
        settings.core_quantiles,
        _marginal_clamp_limits(marginal),
        (bool(active_lower.any()), bool(active_upper.any())),
    )
    report = {
        "bin_edges": edge.copy(),
        "lower_boundaries": lower_boundaries.copy(),
        "upper_boundaries": upper_boundaries.copy(),
        "target_lower_cdf": lower_probabilities.copy(),
        "target_upper_cdf": upper_probabilities.copy(),
        "before_lower_cdf": before_lower.copy(),
        "before_upper_cdf": before_upper.copy(),
        "fitted_lower_cdf": fitted_lower.copy(),
        "fitted_upper_cdf": fitted_upper.copy(),
        "lower_confidence_interval": np.column_stack([lower_ci, lower_ci_high]),
        "upper_confidence_interval": np.column_stack([upper_ci, upper_ci_high]),
        "active_lower": active_lower.copy(),
        "active_upper": active_upper.copy(),
        "reference_ess": float(effective_size),
        "effective_interval_counts": effective_interval_counts.copy(),
        "lower_source_knots": np.atleast_1d(lower_source).copy(),
        "lower_target_knots": np.atleast_1d(lower_target).copy(),
        "upper_source_knots": np.atleast_1d(upper_source).copy(),
        "upper_target_knots": np.atleast_1d(upper_target).copy(),
        "identity_core_latent": identity_core,
    }
    return calibration, report


def calibrate_edge_bins(
    flow: Transformed,
    reference_samples: np.ndarray,
    bin_edges: Sequence[np.ndarray] | np.ndarray,
    *,
    edge_bins: int = 1,
    core_quantiles: tuple[float, float] = (0.05, 0.95),
    n_calibration_samples: int = 100_000,
    rng_seed: Optional[int] = 999,
    confidence_level: float = 0.95,
    reference_ess: Optional[float | Sequence[float]] = None,
    min_reference_bin_count: int = 20,
    min_calibration_tail_count: int = 50,
    force: bool = False,
    return_report: bool = False,
) -> Transformed | tuple[Transformed, dict[str, Any]]:
    """Calibrate selected marginal edge-bin masses without retraining a flow.

    The function is intentionally limited to Gaussian-space flows using the
    clip-free ``marginal="smooth"`` transform.  It draws from the fitted flow,
    estimates each trained latent marginal CDF, and inserts a fixed monotone
    calibration before the existing physical marginal transform.

    ``edge_bins=1`` targets the cumulative mass below ``bin_edges[j][1]`` and
    above ``bin_edges[j][-2]`` in every dimension. By default only statistically
    supported minimum corrections are applied; ``force=True`` hard-fits the raw
    empirical masses. With known prior bounds as the outer histogram edges,
    these are the first and last bin masses. On an unbounded side they are tail-
    inclusive masses, so under/overflow must be reported separately when
    evaluating a cropped histogram.

    Parameters
    ----------
    flow : flowjax.distributions.Transformed
        A fitted Gaussian-space coppuccino flow with smooth marginals.
    reference_samples : np.ndarray
        Posterior/reference samples, shape ``(n_samples, n_dims)``.
    bin_edges : sequence of np.ndarray or np.ndarray
        Shared, fixed histogram edges for each dimension. A single 1D array is
        accepted for a one-dimensional flow.
    edge_bins : int, optional
        Number of cumulative edge-bin boundaries to calibrate on each side.
        Default is 1.
    core_quantiles : (float, float), optional
        Approximate lower/upper quantiles at which each calibration returns to
        the exact identity. Default is ``(0.05, 0.95)``; joins move inward if
        needed to contain the requested anchors and a smooth transition.
    n_calibration_samples : int, optional
        Number of fitted-flow draws used to estimate latent marginal CDFs.
        Default is 100,000.
    rng_seed : int or None, optional
        Seed for the calibration draw. Default is 999.
    confidence_level : float, optional
        Wilson interval level used by the minimum-change gate. A boundary is
        left exactly unchanged when the fitted mass is already compatible with
        the reference; otherwise it moves only to the nearest interval edge.
        Default is 0.95.
    reference_ess : float or sequence of float, optional
        Effective reference sample size per dimension for the Wilson intervals
        and sparse-bin guard. Defaults to the row count (independent samples).
        MCMC callers should provide an indicator/tail effective sample size.
    min_reference_bin_count : int, optional
        Minimum effective count in every selected tail-inclusive interval.
        Default is 20.
    min_calibration_tail_count : int, optional
        Minimum expected calibration draws resolving each active tail quantile
        interval. Default is 50.
    force : bool, optional
        If True, hard-fit raw empirical probabilities and downgrade sparse-bin
        and calibration-resolution failures to warnings. Default is False.
    return_report : bool, optional
        If True, return ``(calibrated_flow, report)`` with anchors, target
        probabilities, pre-calibration probabilities, and identity-core limits.

    Returns
    -------
    Transformed or (Transformed, dict)
        A new flow. The input flow and its fitted parameters are not modified.

    Notes
    -----
    Componentwise monotone calibration preserves the fitted copula/ranks. It
    corrects marginal edge bins, not multivariate corner dependence. Distinct
    boundaries with the same empirical CDF (an empty reference bin) cannot be
    represented by a strictly increasing continuous density and are rejected.
    Periodic coordinates must be recentered/unwrapped before calibration; a
    monotone linear-coordinate correction cannot cross a circular seam.
    """
    marginals = _flow_marginals(flow)
    n_dims = len(marginals)
    reference = _clean_reference(reference_samples, n_dims)
    settings = _validated_settings(
        edge_bins,
        core_quantiles,
        n_calibration_samples,
        confidence_level,
        min_reference_bin_count,
        min_calibration_tail_count,
        force,
    )

    edges = _normalise_bin_edges(bin_edges, n_dims)
    effective_sizes = _normalise_reference_ess(reference_ess, n_dims, len(reference))
    for dim, edge in enumerate(edges):
        if len(edge) - 1 <= 2 * edge_bins:
            raise ValueError(
                f"bin_edges[{dim}] needs more than {2 * edge_bins} bins to leave an identity core."
            )

    # Draw through the public sampler so smooth-tail clamp rejection is already
    # applied, then invert only the unchanged physical marginals. These are the
    # latent values whose empirical CDF governs emitted edge-bin occupancy.
    from coppuccino.copula_flows import sample

    physical_draws = sample(flow, n_samples=n_calibration_samples, rng_seed=rng_seed)
    inverse_marginals = jax.jit(jax.vmap(flow.bijection.inverse_and_log_det))
    latent_draws, _ = inverse_marginals(jnp.asarray(physical_draws))
    latent_draws = np.asarray(latent_draws)

    calibrations = []
    dimension_reports = []
    for dim, (marginal, edge) in enumerate(zip(marginals, edges)):
        calibration, dimension_report = _calibrate_dimension(
            dim,
            marginal,
            edge,
            reference[:, dim],
            physical_draws[:, dim],
            latent_draws[:, dim],
            float(effective_sizes[dim]),
            settings,
        )
        calibrations.append(non_trainable(calibration))
        dimension_reports.append(dimension_report)

    calibration_stack = Stack(calibrations)
    calibrated_inner = Transformed(flow.base_dist, calibration_stack)
    calibrated_flow = Transformed(calibrated_inner, flow.bijection)

    if return_report:
        return calibrated_flow, {
            "n_reference_samples": len(reference),
            "n_calibration_samples": n_calibration_samples,
            "edge_bins": edge_bins,
            "core_quantiles": settings.core_quantiles,
            "confidence_level": settings.confidence_level,
            "force": force,
            "min_reference_bin_count": min_reference_bin_count,
            "min_calibration_tail_count": min_calibration_tail_count,
            "dimensions": dimension_reports,
        }
    return calibrated_flow
