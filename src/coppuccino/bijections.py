from typing import Any, Callable, ClassVar, NamedTuple, Optional

import jax.numpy as jnp
import numpy as np
from flowjax.bijections import AbstractBijection
from jax import Array
from jax.scipy.special import log_ndtr, ndtri
from jax.typing import ArrayLike

__all__ = [
    "make_smooth_marginal",
    "SmoothMarginalToGaussian",
]

# Numerical floors, named so the intent is explicit at each use site.
_POS_FLOOR = 1e-30    # keep quantities that sit in denominators strictly positive
_NDTRI_FLOOR = 1e-300  # smallest ndtri argument that stays inside float64 range
_LOG_SQRT_2PI = float(0.5 * np.log(2.0 * np.pi))


def _ndtri_floor(x: ArrayLike) -> float:
    """Smallest usable ``ndtri`` argument in ``x``'s floating dtype.

    ``_NDTRI_FLOOR`` underflows to zero below float64, and ``ndtri(0)`` is
    ``-inf``, so the tail inversion needs the per-dtype limit rather than the
    float64 constant. The package enables x64 on import, but that is a
    process-wide setting a caller may turn back off (see the package
    docstring), and the marginals must stay finite when they do.
    """
    return max(_NDTRI_FLOOR, float(jnp.finfo(jnp.asarray(x).dtype).tiny))


# ----------------------------------------------------------------------
# Monotone rational-quadratic spline kernel (Gregory & Delbourgo 1982;
# Durkan et al. 2019), shared by every RQS in the package: the bulk
# marginal splines below and the fleet's stacked marginals
# (:mod:`coppuccino.fleet`). This module is the dependency leaf, so the fleet
# imports the kernel from here rather than restating it.
# ----------------------------------------------------------------------


def _rqs_bin(index_knots: Array, query: Array, xk: Array, yk: Array,
             dk: Array) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
    """Locate ``query`` in the RQS bins and gather that bin's parameters.

    ``index_knots`` is ``xk`` for the forward and derivative maps and ``yk``
    for the inverse. The bin index is clamped to the interior bins, so a
    query outside the knot range is evaluated on the nearest end bin.

    Parameters
    ----------
    index_knots : jax.Array
        1-D strictly increasing knots to search in.
    query : jax.Array
        Points to locate; any shape.
    xk, yk, dk : jax.Array
        1-D knot locations, knot values and knot derivatives.

    Returns
    -------
    x0, w, y0, dy, s, d0, d1 : jax.Array
        Left knot, bin width, left value, bin height, secant slope and the
        two knot derivatives of the containing bin, shaped like ``query``.
    """
    n_bins = index_knots.shape[0] - 1
    k = jnp.clip(jnp.searchsorted(index_knots, query, side="right") - 1, 0, n_bins - 1)
    x0, x1 = xk[k], xk[k + 1]
    y0, y1 = yk[k], yk[k + 1]
    w = x1 - x0
    dy = y1 - y0
    return x0, w, y0, dy, dy / w, dk[k], dk[k + 1]


def _rqs_forward(x: ArrayLike, xk: Array, yk: Array, dk: Array) -> Array:
    """Evaluate the monotone RQS: knot domain -> knot range.

    ``x`` outside the knot range is evaluated at the nearest end knot, so
    the map saturates at ``yk[0]`` / ``yk[-1]``. The rational-quadratic form
    is monotone only for ``xi`` in ``[0, 1]``: extrapolated it turns over,
    returning values that leave ``[yk[0], yk[-1]]`` and rise again as ``x``
    moves further out — and, in :func:`_rqs_derivative`, a negative density.
    Saturating instead is what the fleet's stacked marginals have always
    done (:mod:`coppuccino.fleet` layers its own out-of-box support test on
    top); the marginal call sites clip their own inputs first, so for them
    the clamp is a no-op that closes the mode off for good.
    """
    x = jnp.clip(jnp.asarray(x), xk[0], xk[-1])
    x0, w, y0, dy, s, d0, d1 = _rqs_bin(xk, x, xk, yk, dk)
    xi = (x - x0) / w
    num = dy * (s * xi**2 + d0 * xi * (1 - xi))
    den = s + (d1 + d0 - 2 * s) * xi * (1 - xi)
    return y0 + num / den


def _rqs_derivative(x: ArrayLike, xk: Array, yk: Array, dk: Array,
                    floor: Optional[float] = None) -> Array:
    """Closed-form derivative of :func:`_rqs_forward` (the density).

    Strictly positive for strictly positive knot derivatives, and constant
    beyond either end knot, where the query saturates exactly as in
    :func:`_rqs_forward`. ``floor``, when given, is a lower clamp for
    callers that feed the result straight to ``log``; the default leaves
    the raw value alone.
    """
    x = jnp.clip(jnp.asarray(x), xk[0], xk[-1])
    x0, w, y0, dy, s, d0, d1 = _rqs_bin(xk, x, xk, yk, dk)
    xi = (x - x0) / w
    den = s + (d1 + d0 - 2 * s) * xi * (1 - xi)
    num = s**2 * (d1 * xi**2 + 2 * s * xi * (1 - xi) + d0 * (1 - xi) ** 2)
    derivative = num / den**2
    return derivative if floor is None else jnp.maximum(derivative, floor)


def _rqs_inverse(u: ArrayLike, xk: Array, yk: Array, dk: Array) -> Array:
    """Invert the monotone RQS: knot range -> knot domain.

    The exact inverse of :func:`_rqs_forward` on the knot range, via the
    numerically stable root of the bin's quadratic (the form that stays in
    ``[0, 1]`` as the bin approaches linear). Clamping that root is what
    saturates ``u`` outside the knot range, so no separate guard on ``u``
    is needed here.
    """
    u = jnp.asarray(u)
    x0, w, y0, dy, s, d0, d1 = _rqs_bin(yk, u, xk, yk, dk)
    alpha = u - y0
    coef = d1 + d0 - 2 * s
    a = dy * (s - d0) + alpha * coef
    b = dy * d0 - alpha * coef
    c = -s * alpha
    disc = jnp.maximum(b**2 - 4 * a * c, 0.0)
    xi = jnp.clip(2 * c / (-b - jnp.sqrt(disc)), 0.0, 1.0)
    return x0 + xi * w


def _rqs_knot_derivatives(
    xk: ArrayLike,
    yk: ArrayLike,
    floor: float = _POS_FLOOR,
    edge_slope_floor: float = 0.0,
) -> np.ndarray:
    """Monotone PCHIP knot slopes for an RQS, floored strictly positive.

    The RQS reproduces the shape-preserving PCHIP slopes at its knots while
    admitting a closed-form inverse, so every RQS in the package takes its
    knot derivatives from here.

    Parameters
    ----------
    xk : array_like
        Strictly increasing knot locations.
    yk : array_like
        Strictly increasing knot values.
    floor : float, optional
        Lower clamp on every knot derivative. Slopes sit in RQS
        denominators, so this must stay strictly positive; the default
        ``_POS_FLOOR`` is the generic "not exactly zero" guard; a caller
        whose map must stay resolvably invertible wants something far
        larger.
    edge_slope_floor : float, optional
        If positive, floor the two *endpoint* derivatives at this fraction
        of their end-bin secant slope. PCHIP's endpoint rule can return
        exactly 0 there (a one-sided estimate clamped on sign change),
        which is fatal for callers that extend tails from the boundary
        slope. The default 0.0 disables it.

    Returns
    -------
    numpy.ndarray
        Knot derivatives, one per knot.
    """
    from scipy.interpolate import PchipInterpolator as _ScipyPchip
    xk = np.asarray(xk, dtype=float)
    yk = np.asarray(yk, dtype=float)
    dk = np.maximum(
        np.asarray(_ScipyPchip(xk, yk).derivative()(xk), dtype=float), floor)
    if edge_slope_floor > 0.0:
        sec = np.diff(yk) / np.diff(xk)
        dk[0] = max(dk[0], edge_slope_floor * sec[0])
        dk[-1] = max(dk[-1], edge_slope_floor * sec[-1])
    return dk


def _rqs_bulk(
    xg: ArrayLike,
    cdf_vals: ArrayLike,
    edge_slope_floor: float = 0.0,
) -> tuple[Callable[[ArrayLike], Array], Callable[[ArrayLike], Array],
           Callable[[ArrayLike], Array]]:
    """Build closed-form monotone rational-quadratic spline (RQS) callables.

    Returns ``(cdf, inverse, deriv)`` callables over the bulk knots. The
    forward map, its inverse and its derivative all come from a single
    parameterization, so the transform is self-consistent to machine
    precision.

    Knot derivatives come from :func:`_rqs_knot_derivatives` and the three
    callables from the shared kernel (:func:`_rqs_forward`,
    :func:`_rqs_inverse`, :func:`_rqs_derivative`); this function only
    prepares the knots.

    Parameters
    ----------
    xg : array_like
        Strictly sorted bulk knot locations (CDF domain).
    cdf_vals : array_like
        Strictly increasing CDF values at ``xg``.
    edge_slope_floor : float, optional
        Endpoint knot-derivative floor, as a fraction of the end-bin secant
        slope; see :func:`_rqs_knot_derivatives`. Rescuing PCHIP's zeroed
        endpoint derivative matters for callers that extend tails from the
        boundary slope (:func:`make_smooth_marginal`): the C^1-matched tail
        rate would otherwise collapse to the generic ``_POS_FLOOR`` and the
        tail map degenerates. The default 0.0 leaves PCHIP's endpoint slopes
        alone.

    Returns
    -------
    cdf, inverse, deriv : callables
        ``cdf(x)`` maps x -> CDF, ``inverse(u)`` maps CDF -> x, ``deriv(x)``
        is the closed-form CDF derivative (density). All are jnp-vectorized.
    """
    xg = np.asarray(xg, dtype=float)
    yg = np.asarray(cdf_vals, dtype=float)
    # RQS needs strictly increasing x knots; drop any ties (cdf_vals is already
    # strictly increasing thanks to the upstream monotonicity bump).
    keep = np.concatenate([[True], np.diff(xg) > 0])
    xk, yk = xg[keep], yg[keep].copy()
    # the tie-drop keeps the FIRST knot of a tied block; the top endpoint's
    # CDF value must survive on the kept boundary knot or the spline's range
    # silently tops out below it and the discarded mass reappears as an
    # exact-at-bound sampling atom
    yk[-1] = yg[-1]
    dk = _rqs_knot_derivatives(xk, yk, edge_slope_floor=edge_slope_floor)

    xk_j, yk_j, dk_j = jnp.asarray(xk), jnp.asarray(yk), jnp.asarray(dk)

    def cdf(x: ArrayLike) -> Array:
        return _rqs_forward(x, xk_j, yk_j, dk_j)

    def deriv(x: ArrayLike) -> Array:
        return _rqs_derivative(x, xk_j, yk_j, dk_j)

    def inverse(u: ArrayLike) -> Array:
        return _rqs_inverse(u, xk_j, yk_j, dk_j)

    return cdf, inverse, deriv


# ----------------------------------------------------------------------
# Smooth-tail marginals: a true diffeomorphism z in R  <->  theta in (lo, hi)
# ----------------------------------------------------------------------

# Both tails of a smooth marginal are the same map seen from opposite ends,
# so they are written once in each side's own *outward* coordinates: on the
# left ``u = z`` and ``y = theta``, on the right ``u = -z`` and
# ``y = -theta``. In those coordinates every tail is increasing in ``u`` and
# runs outward as ``u`` decreases, so one implementation serves both sides
# and the mirroring costs only exact sign flips.

_EXP_MAX = 700.0  # exp overflow guard for values that get where()-selected away
# Conditioning guard on the wall shape's join-side derivative. It is a ratio
# of densities, so a collapsed endpoint-slope estimate can send it to 1e30 or
# beyond, where the spline kernel's quadratic squares it. Past this value the
# shape is numerically its own limit anyway.
_WALL_BETA_MAX = 1e12
# How far past its own outermost knot, in z, a bounded tail may run before it
# saturates. It is a *margin*, not an absolute |z| cap, because the quantity
# that has to be bounded is extrapolation distance: the outermost bulk knot
# sits on the most extreme sample, so a tail's ``u_edge`` is exactly the
# deepest z the copula flow downstream was ever trained on in this dimension,
# and everything below it is extrapolation. Measured on U(0, 1)^2, n = 1500,
# 80 epochs, seeds 0-3, the fitted copula's excess over its own N(0, 1) base
# at (z, 0) ran -0.0 to +0.2 nats at |z| = 3 (inside training, |u_edge| =
# 3.40), +0.0 to +0.8 at 3.4, -0.3 to +1.9 at 4, +0.5 to +4.5 at 5 and +1.6 to
# +14.9 at 8 -- roughly 2 nats per unit z beyond the edge. An absolute cap
# cannot express that: |u_edge| is ndtri(0.5 / n), so a fixed 4.0 is 0.6 past
# the edge at n = 1500, 0.24 past it at n = 6000 and *inside* it from
# n = 15790 on. Well before that the half-gap guard below takes over and the
# constant stops meaning anything: with a fixed 4.0 the z actually reported at
# the bound of a U(0, 1) fit was 4.09 at n = 6000, 4.29 at 30000 and 4.51 at
# 100000 -- the guard's number, not the cap's.
# The margin is the whole trade. Everything below the cap collapses onto one
# theta, so a smaller margin buys a smaller copula error at a wall and pays in
# wall resolution: the saturation point moves out to about
# Phi(u_edge - margin) / Phi(u_edge) of the knot-to-bound distance, which at
# n = 1500 is 11% of it at margin 0.6 (and hits the half-gap guard on 2 of 4
# U(0, 1) seeds), 1.7% at 1.0 and 0.14% at 1.5. 1.5 is the smallest margin
# measured at which the marginal's own wall-density error is indistinguishable
# from the uncapped tail's (median 1.09 / worst 2.03 nats against the analytic
# truth on truncated N(0, 1), 8 seeds x both walls x {0, 1e-9, 1e-6} of the
# span, versus 1.09 / 1.97 uncapped and 1.13 / 2.92 at margin 1.0).
#
# It also brings the at-bound density back down near the bulk of the fitted
# distribution, where the uncapped tail put it clearly above: comparing a box
# mid-edge against the maximum log-density over 20000 draws, the uncapped tail
# ran 2.9 to 9.9 nats *above* on U(0, 1), truncated N(0, 1) and Beta(0.5, 3),
# and margin 1.5 brings it to roughly the same order as that maximum. Treat
# that comparison as indicative only -- the maximum over a finite sample is
# itself random, and holding one fit and varying only the draw seed moves it
# by a couple of nats in either direction, so it is not a bound. Box *corners*,
# where both dimensions sit at the cap at once, stay worse than the mid-edges
# (+0.5 to +11.9 against +0.1 to +4.6 on U(0, 1)^2 at the same settings)
# -- see SmoothMarginalToGaussian.
_WALL_Z_MARGIN = 1.5
# Largest |knot value| the spline kernel can carry in float64 (its quadratic
# squares theta-scale products, so the headroom is the fourth root of the
# float64 range) and largest usable unbounded-tail growth rate.
_MAX_ABS_KNOT = 1e75
_TAIL_ALPHA_MAX = 10.0


def _log_phi(u: Array) -> Array:
    """Standard normal log-density, finite for every representable ``u``."""
    return -0.5 * u * u - _LOG_SQRT_2PI


class _MarginalTail(NamedTuple):
    """One analytic tail of :func:`make_smooth_marginal`, in ``(u, y)`` coords.

    Two shapes share this record. ``wall`` selects between them, and it is a
    build-time Python bool, so each closure below traces only one branch.

    * ``wall=True`` -- *bounded side*. The CDF has to reach 0 at the bound,
      which in Gaussian space happens only as ``u -> -inf``, so the tail is
      written as the inverse of a CDF shape::

          Phi(u) = p_edge * S((y - y_bound) / gap)

      with ``S`` a monotone rational-quadratic spline on the unit interval
      pinned at ``S(0) = 0``, ``S(1) = 1``, ``S'(1) = beta`` (the bulk
      spline's own density at the join, so the density is continuous there)
      and ``S'(0) = 1``. That last pin is what the family exists for: the
      marginal density at the bound is ``p_edge / gap``, the mean density
      over the knot-to-bound interval -- finite, strictly positive, and
      reached smoothly. An exponential or power-law approach instead makes
      ``y - y_bound`` log-normal or a power law, whose density at the bound
      can only be 0 or infinity. It is also the estimator the fleet's
      endpoint knots (:mod:`coppuccino.fleet`) use, whose wall accuracy this
      reproduces.
    * ``wall=False`` -- *unbounded side*. ``dy/du = slope *
      exp(alpha |u - u_edge|)``: linear for ``alpha = 0``
      (Gaussian tails in theta), heavier above it. There is no saturation
      point on such a side, and ``tail_alpha`` is consumed only here.

    ``y_min`` is where a bounded forward map saturates: the bound plus the
    larger of the smallest offset that survives ``bound + offset`` in
    floating point and the offset whose ``u`` is ``_WALL_Z_MARGIN`` below
    ``u_edge``, so no finite ``u`` maps onto a bound exactly (``-inf`` on an
    unbounded side). ``r_floor`` is the same limit as a fractional wall
    offset; the inverse and the log-derivative clamp on it too, so all three
    directions agree there. It is also capped at half the knot-to-bound
    distance, which binds only when ``beta`` is large enough that the wall
    shape puts the margin's quantile past that point.
    """
    u_edge: float
    y_edge: float
    slope: float
    alpha: float
    y_min: float
    wall: bool
    y_bound: float
    gap: float
    beta: float
    log_p_edge: float
    r_floor: float
    shape_x: jnp.ndarray
    shape_y: jnp.ndarray
    shape_d: jnp.ndarray


def _inset_wall_atoms(y: np.ndarray, bound: float, eps0: float,
                      max_inset: float) -> np.ndarray:
    """Move samples sitting on ``bound`` (approached from above) just inside it.

    Values within ``eps0`` of the bound -- the reflection / in-bounds
    artifacts of a bounded MCMC kernel, which arrive as an exact atom -- are
    placed at half the gap to the first strictly interior value rather than
    at the raw ``eps0`` margin. The atom then becomes a *resolvable* shoulder
    whose knot offset is comparable to the first bin width, instead of a
    sub-ULP spike that the outermost knot slope cannot express. When every
    sample sits in the window there is no interior value to measure the gap
    against, so they are only pushed off the bound.

    The mirrored side calls this on ``-y`` with ``-bound``; sign flips are
    exact, so both walls get bit-for-bit the same treatment.
    """
    at = y <= bound + eps0
    if not (at.any() and (~at).any()):
        return np.maximum(y, bound + eps0)
    gap = float(np.min(y[~at])) - bound
    y[at] = bound + float(np.clip(0.5 * gap, eps0, max_inset))
    return y


def _make_tail(u_edge: float, y_edge: float, slope: float, alpha: float,
               bound: Optional[float], span: float) -> _MarginalTail:
    """Fit one tail beyond an outermost bulk knot; see :class:`_MarginalTail`.

    Parameters
    ----------
    u_edge, y_edge : float
        Outermost knot of the bulk spline in this side's outward coordinates.
    slope : float
        ``dy/du`` of the bulk spline at that knot (strictly positive).
    alpha : float
        ``tail_alpha``; only used on an unbounded side.
    bound : float or None
        The prior bound in ``y`` coordinates, or None for an unbounded side.
    span : float
        Support span, used only for the float-resolution floors.
    """
    unit = jnp.asarray([0.0, 1.0])
    free = _MarginalTail(
        u_edge=u_edge, y_edge=y_edge, slope=slope, alpha=alpha, y_min=-np.inf,
        wall=False, y_bound=-np.inf, gap=0.0, beta=1.0, log_p_edge=0.0,
        r_floor=0.0, shape_x=unit, shape_y=unit, shape_d=unit)
    if bound is None:
        return free
    # Distance from the outermost knot to the bound. It rounds to exactly
    # zero when the bound's magnitude dwarfs the data span, which would
    # divide by zero below; one ULP at the bound is the smallest gap
    # representable there.
    gap = max(y_edge - bound, float(np.spacing(abs(bound))))
    # Smallest wall offset the tail may produce or resolve. Below ~ULP(bound)
    # the float addition ``bound + tiny`` absorbs to the bound exactly -- an
    # exact-wall sample atom. Saturating there instead relocates that
    # unrepresentable mass to a strictly interior point. Half the knot offset
    # caps it: an offset at or beyond the outermost knot would make the tail
    # decreasing.
    eps = float(np.clip(4.0 * np.spacing(abs(bound)) + 1e-30 * span,
                        1e-30 * span, min(0.01 * span, 0.5 * gap)))
    from scipy.special import log_ndtr as _log_ndtr_np
    log_p_edge = float(_log_ndtr_np(u_edge))
    # S'(1): the bulk's density at the join knot divided by the mean density
    # over the knot-to-bound interval. 1 means the data approach the bound at
    # constant density; larger means the density falls off toward it.
    beta = float(np.clip((gap / slope) * np.exp(_log_phi(jnp.asarray(u_edge))
                                                - log_p_edge),
                         _POS_FLOOR, _WALL_BETA_MAX))
    shape_d = jnp.asarray([1.0, beta])
    # Saturation point: the later of the float-resolution limit above and the
    # wall offset whose u is one margin below the outermost knot. That stops
    # the tail handing the copula flow a z it has no basis for -- extrapolation
    # starts at ``u_edge``, not at some deep absolute |z| -- and it is what
    # keeps the joint density at a bound comparable to the interior: on
    # U(0, 1)^2, n = 1500, 80 epochs, seeds 0-3 the reported log-density on the
    # four box mid-edges was +1.8 to +15.5 nats above the flat truth when the
    # tail ran to |z| = 8 -- above the maximum over 20000 of the fit's own
    # draws in every seed -- and +0.1 to +4.6 at this margin, below that
    # maximum by 0.7 to 3.6 in every seed.
    r_z = float(_rqs_inverse(jnp.asarray(
        np.exp(float(_log_ndtr_np(u_edge - _WALL_Z_MARGIN)) - log_p_edge)),
        unit, unit, shape_d))
    r_floor = min(max(eps / gap, r_z), 0.5)
    return _MarginalTail(
        u_edge=u_edge, y_edge=y_edge, slope=slope, alpha=alpha,
        y_min=bound + gap * r_floor, wall=True, y_bound=bound, gap=gap,
        beta=beta, log_p_edge=log_p_edge, r_floor=r_floor,
        shape_x=unit, shape_y=unit, shape_d=shape_d)


def _tail_r(u: Array, t: _MarginalTail) -> Array:
    """Fractional wall offset ``S^-1(Phi(u) / p_edge)``, floored at the clamp."""
    q = jnp.exp(jnp.minimum(log_ndtr(u) - t.log_p_edge, 0.0))
    return jnp.maximum(_rqs_inverse(q, t.shape_x, t.shape_y, t.shape_d),
                       t.r_floor)


def _tail_y(u: Array, t: _MarginalTail) -> Array:
    """Tail value ``y(u)``, saturating at ``t.y_min``."""
    if t.wall:
        y = t.y_bound + t.gap * _tail_r(u, t)
        # ``y_min`` is computed in float64 at build time. Under a coarser
        # runtime dtype (x64 turned back off) it rounds onto the bound
        # itself, so the saturation guard has to be restated in the dtype
        # actually in use or the map emits exact-wall values.
        return jnp.maximum(y, jnp.nextafter(
            jnp.asarray(t.y_bound, dtype=y.dtype), jnp.inf))
    du = jnp.minimum(u - t.u_edge, 0.0)
    if t.alpha > 0.0:
        return t.y_edge - (t.slope / t.alpha) * jnp.expm1(
            jnp.minimum(-t.alpha * du, _EXP_MAX))
    return t.y_edge + t.slope * du


def _tail_u(y: Array, t: _MarginalTail) -> Array:
    """Exact inverse of :func:`_tail_y` above the saturation point."""
    if t.wall:
        r = jnp.maximum((jnp.maximum(y, t.y_min) - t.y_bound) / t.gap,
                        t.r_floor)
        s = _rqs_forward(r, t.shape_x, t.shape_y, t.shape_d)
        return ndtri(jnp.maximum(jnp.exp(t.log_p_edge) * s, _ndtri_floor(y)))
    dy = y - t.y_edge
    if t.alpha > 0.0:
        return t.u_edge - jnp.log(
            jnp.maximum(1.0 - t.alpha * dy / t.slope, _NDTRI_FLOOR)) / t.alpha
    return t.u_edge + dy / t.slope


def _tail_log_deriv(u: Array, t: _MarginalTail) -> Array:
    """``log |dtheta/dz|`` on the tail; orientation-free, so it needs no sign."""
    if not t.wall:
        return np.log(t.slope) - t.alpha * jnp.minimum(u - t.u_edge, 0.0)
    # dtheta/dz = phi(u) / f(y), with f = p_edge S'(r) / gap the tail density
    s_deriv = _rqs_derivative(_tail_r(u, t), t.shape_x, t.shape_y, t.shape_d)
    return (_log_phi(u) - t.log_p_edge + np.log(t.gap)
            - jnp.log(jnp.maximum(s_deriv, _POS_FLOOR)))


def make_smooth_marginal(samples: np.ndarray,
                         num_points: int = 200,
                         prior_low: Optional[float] = None,
                         prior_high: Optional[float] = None,
                         tail_alpha: float = 0.0,
                         # The three closures are ``Any`` rather than
                         # ``Callable`` on purpose. Both callers splat this
                         # tuple into ``SmoothMarginalToGaussian(...)``, whose
                         # synthesised ``__init__`` mypy reads as starting
                         # with flowjax's ``AbstractVar`` shape/cond_shape
                         # (see the ClassVar note on that class); typing them
                         # turns that existing positional mismatch into four
                         # new arg-type errors in copula_flows and model_io,
                         # which already carry a call-arg ignore for it.
                         ) -> tuple[Any, Any, Any,
                                    tuple[Optional[float], Optional[float]]]:
    """Build a clip-free marginal map ``theta(z)``: all of R onto the support.

    The map is a single monotone C^1 object evaluated consistently in both
    directions (forward, inverse and log-derivative share one
    parameterization), so the flow's sample stream and its reported density
    agree everywhere -- including arbitrarily deep in the tails. There is no
    CDF clip, no quantile clamp (hence no boundary sample atom) and no floored
    wall density (hence no density cliff), and the density the map reports
    *at* a prior bound is finite and positive rather than 0 or infinite (see
    :class:`SmoothMarginalToGaussian` for what that costs and what it does not
    fix).

    Construction:

    * **Bulk**: knots at the data quantiles ``theta_j`` paired with
      ``z_j = ndtri(p_j)`` at Hazen plotting positions -- all finite -- joined
      by the closed-form monotone rational-quadratic spline (PCHIP slopes).
    * **Tails** beyond the outermost knots (see :class:`_MarginalTail`):
        - *bounded side* (prior bound given): a monotone spline in CDF space
          pinned to reach 0 exactly at the bound, with a *finite, strictly
          positive* wall density equal to the mean density between the
          outermost knot and the bound. The bound is still reached only as
          ``z -> -+inf``, so every finite flow output lands strictly inside
          the prior, but the density there no longer has to diverge or
          vanish.
        - *unbounded side*: ``dtheta/dz = m exp(alpha |z - z_edge|)`` --
          ``tail_alpha = 0`` gives a linear map (Gaussian tails in theta);
          ``tail_alpha > 0`` gives progressively heavier tails. Density
          queries and samples beyond the observed data range are therefore
          honest and smooth (outlier handling).

    Boundary semantics: samples sitting exactly on a prior bound (the
    documented recommended use of ``prior_bounds``, and what a reflecting or
    clipping MCMC kernel produces) are relocated just inside it before the
    knots are built (:func:`_inset_wall_atoms`), so the atom becomes a
    resolvable shoulder rather than a sub-ULP spike. Query points are *not*
    relocated: the map is one bijection in both directions, and a row on the
    bound reads the wall density off the tail like any other point. That
    density is the atom's mass over the inset width, so it is genuinely
    large -- on ``clip(N(0, 1), -1, 1)`` with 16-18% of 1500 rows on the
    lower bound it came out 4.9 to 8.4 nats above the interior median over
    three seeds. A tie block that large also pulls the outermost knot far in,
    to ``z = -0.9`` or so, and the tail is only allowed
    ``_WALL_Z_MARGIN`` past that, so such a row lands at ``|z| ~ 2.4``
    rather than deep in the copula flow's extrapolation region.

    Returns
    -------
    theta_of_z, z_of_theta, log_dtheta_dz : callables (jnp-vectorized)
    x_clamps : tuple of (float or None, float or None)
        Saturation x values of the bounded tails (None on unbounded sides),
        at the float64 wall-offset resolution limit or at the wall offset
        whose z is ``_WALL_Z_MARGIN`` past the outermost knot, whichever is
        further inside. ``theta_of_z``
        returns exactly these floats for z at or beyond the corresponding
        clamp; the sampling layer
        redraws such rows (see ``coppuccino.copula_flows.sample``) so the
        emitted stream matches the reported density everywhere.
    """
    from scipy.special import ndtri as _ndtri

    x = np.asarray(samples, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    num_points = max(20, min(num_points, n // 3))
    if not 0.0 <= tail_alpha <= _TAIL_ALPHA_MAX:
        raise ValueError(
            f"tail_alpha must be in [0, {_TAIL_ALPHA_MAX:g}], got {tail_alpha}. "
            "It is a growth rate per unit z on an unbounded tail, so the map "
            "spans exp(tail_alpha * |z|) decades of theta over the range of z "
            "the base actually visits; at tail_alpha = 1e6 the emitted samples "
            "reached 1.8e298.")
    # a non-finite bound means "unbounded on that side"
    if prior_low is not None and not np.isfinite(prior_low):
        prior_low = None
    if prior_high is not None and not np.isfinite(prior_high):
        prior_high = None

    # The rational-quadratic kernel forms products of two theta-scale terms
    # and then squares them (``b ** 2`` in :func:`_rqs_inverse`), so knot
    # values above ~1e77 overflow float64 and the map silently goes
    # non-finite and non-monotone: standard-normal data scaled by 1e80 gives
    # a NaN round trip, and by 1e120 an all-NaN theta_of_z.
    scale = max(float(np.max(np.abs(x))) if n else 0.0,
                abs(prior_low) if prior_low is not None else 0.0,
                abs(prior_high) if prior_high is not None else 0.0)
    if scale > _MAX_ABS_KNOT:
        raise ValueError(
            f"make_smooth_marginal: largest |sample or prior bound| is "
            f"{scale:.3g}, above the {_MAX_ABS_KNOT:.0e} limit where the "
            "spline kernel's quadratic overflows float64 and the map stops "
            "being finite and monotone. Rescale this parameter (work in "
            "units of its own scale) before fitting.")

    # scale for the strictly-inside boundary clip
    lo_ref = prior_low if prior_low is not None else float(np.min(x))
    hi_ref = prior_high if prior_high is not None else float(np.max(x))
    span = max(hi_ref - lo_ref, _POS_FLOOR)
    # Boundary-atom inset window. Scale it by whichever span is *smaller*. A
    # prior-relative window alone is wrong whenever the prior is much wider
    # than the posterior: under prior_bounds=[[0, 1e12]] on a chain spanning
    # [4e2, 4e3] it reaches to x = 1000 and silently relocates a large
    # fraction of the chain onto one point. What counts as "sitting on the
    # bound" is set by the resolution of the samples, not by how
    # uninformative the prior is. The ULP floor keeps ``bound + eps0``
    # representable when the data span underflows it.
    data_span = max(float(np.max(x)) - float(np.min(x)), _POS_FLOOR)
    eps0 = max(1e-9 * min(span, data_span),
               4.0 * float(np.spacing(max(abs(lo_ref), abs(hi_ref)))))
    if prior_low is not None:
        x = _inset_wall_atoms(x, prior_low, eps0, 0.01 * span)
    if prior_high is not None:
        x = -_inset_wall_atoms(-x, -prior_high, eps0, 0.01 * span)
    xs = np.sort(x)

    # theta knots at *distinct* data quantiles. Tied values (MCMC-rejection
    # duplicates, boundary reflection atoms, ...) must NOT become multiple
    # knots separated by artificial bumps: that makes theta(z) effectively
    # constant across the tie block's whole z-span -- a non-invertible
    # plateau where the sampled point can no longer report its own density.
    # One knot per distinct value, carrying the full Hazen mass of its ties,
    # keeps the map a true diffeomorphism: a data atom becomes a steep but
    # smooth (and exactly invertible) segment over the real gap to its
    # neighbors, and a boundary tie block's mass flows into the analytic
    # tail -- the closest absolutely-continuous statement of an atom.
    thq = np.quantile(xs, np.linspace(0.0, 1.0, num_points))
    # one knot per distinct Hazen count: drops both tied-value knots and
    # interpolated knots that fall between the same adjacent data values
    counts, keep = np.unique(np.searchsorted(xs, thq, side="right"),
                             return_index=True)
    thq = thq[keep]
    if len(thq) < 2:
        raise ValueError(
            "make_smooth_marginal needs at least two distinct sample values.")
    p = np.clip((counts - 0.5) / n, 0.5 / n, 1.0 - 0.5 / n)
    zq = _ndtri(p)

    # bulk: closed-form monotone RQS on (z, theta) knots -- one
    # parameterization supplies forward, inverse and derivative. The edge
    # slope floor keeps the join slopes m1/mK (and hence the tail rates)
    # non-degenerate when PCHIP's endpoint rule returns 0.
    bulk_theta, bulk_z, bulk_deriv = _rqs_bulk(zq, thq, edge_slope_floor=0.05)

    z1, zK = float(zq[0]), float(zq[-1])
    th1, thK = float(thq[0]), float(thq[-1])
    m1 = float(np.maximum(np.asarray(bulk_deriv(jnp.asarray(z1))), _POS_FLOOR))
    mK = float(np.maximum(np.asarray(bulk_deriv(jnp.asarray(zK))), _POS_FLOOR))

    # Analytic tails beyond the outermost knots, each built in its own side's
    # outward coordinates (see :class:`_MarginalTail`), so the right tail is
    # the left one seen through an exact sign flip.
    tail_lo = _make_tail(z1, th1, m1, float(tail_alpha), prior_low, span)
    tail_hi = _make_tail(-zK, -thK, mK, float(tail_alpha),
                         None if prior_high is None else -prior_high, span)

    # x values the tails saturate at (metadata for the sampling-layer
    # rejection): the forward map returns exactly these floats for z at or
    # beyond the clamp, so `x <= x_clamp_lo` / `x >= x_clamp_hi` identifies
    # clamped draws by comparison alone.
    x_clamp_lo = float(tail_lo.y_min) if prior_low is not None else None
    x_clamp_hi = -float(tail_hi.y_min) if prior_high is not None else None
    # z beyond these build-time constants maps onto the tail clamp, so the
    # claimed log-derivative is evaluated at the clamp (a function of the
    # *output point*, matching what z_of_theta reconstructs there)
    z_clamp_lo = (float(_tail_u(jnp.asarray(tail_lo.y_min), tail_lo))
                  if prior_low is not None else -np.inf)
    z_clamp_hi = (-float(_tail_u(jnp.asarray(tail_hi.y_min), tail_hi))
                  if prior_high is not None else np.inf)

    def theta_of_z(z: ArrayLike) -> Array:
        z = jnp.asarray(z)
        bulk = bulk_theta(jnp.clip(z, z1, zK))
        return jnp.where(z < z1, _tail_y(z, tail_lo),
                         jnp.where(z > zK, -_tail_y(-z, tail_hi), bulk))

    def log_dtheta_dz(z: ArrayLike) -> Array:
        z = jnp.clip(jnp.asarray(z), z_clamp_lo, z_clamp_hi)
        bulk = jnp.log(jnp.maximum(bulk_deriv(jnp.clip(z, z1, zK)), _POS_FLOOR))
        return jnp.where(z < z1, _tail_log_deriv(z, tail_lo),
                         jnp.where(z > zK, _tail_log_deriv(-z, tail_hi), bulk))

    def z_of_theta(theta: ArrayLike) -> Array:
        theta = jnp.asarray(theta)
        bulk = bulk_z(jnp.clip(theta, th1, thK))
        return jnp.where(theta < th1, _tail_u(theta, tail_lo),
                         jnp.where(theta > thK, -_tail_u(-theta, tail_hi), bulk))

    return theta_of_z, z_of_theta, log_dtheta_dz, (x_clamp_lo, x_clamp_hi)


class SmoothMarginalToGaussian(AbstractBijection):
    """Clip-free marginal bijection between data space and Gaussian space.

    Wraps the closures from :func:`make_smooth_marginal`. There is no
    probability-integral / inverse-normal composition with clipped endpoints:
    the map ``theta(z)`` is one analytic monotone object (RQS bulk + analytic
    tails), so the sampling direction and the density direction agree
    everywhere.
    Consequences:

    * no boundary sample atoms: every finite ``z`` maps strictly inside the
      support;
    * a finite, strictly positive marginal density *at* a prior bound,
      approached smoothly. The bounded tail carries the CDF to exactly 0 at
      the bound with a non-zero slope (see :class:`_MarginalTail`), so the
      wall density it reports is the mean density between the outermost knot
      and the bound. Measured against the analytic truth at the bound and at
      1e-9 / 1e-6 of the span inside it, 8 seeds (``default_rng(0..7)``) x
      both walls x n = 1500: median 1.09 nats and worst 2.03 on ``N(0, 1)``
      truncated to ``[-1, 1]``, median 0.71 / worst 5.50 on ``U(0, 1)`` over
      ``[0, 1]`` (all three worst cases are the lower wall of one seed whose
      smallest order statistic landed far inside its expectation; the worst
      on any other seed is 2.10).
      The residual is the outermost order statistic: the mass the fit assigns
      between it and the bound is ``0.5 / n`` where the truth is
      ``Exp(1) / n``, and that ratio is the whole error. Where the true
      density is 0 or infinite at the bound (``Beta(0.5, 3)``: infinite at 0,
      zero at 1) no finite-density family can follow it -- the reported value
      stays the interval mean, e.g. -5.25 to +1.25 nats against the truth at
      ``1e-9`` from the lower wall over 8 seeds, and far above it arbitrarily
      close to the upper one;
    * a chain with rows sitting exactly *on* a bound (what a clipping or
      reflecting kernel produces) reports a wall density near the interior
      one rather than a spike: on ``clip(N(0.9, 0.25), -1, 1)`` with 32-36%
      of 1500 rows on the upper bound, at-bound minus interior-median
      log-density was -0.37, -0.16, +0.91, +0.64 over four seeds. Those rows
      are inset before the knots are built, so the density does vary steeply
      across the inset window (up to 6.4 nats per decade of wall offset in
      those fits) before settling to its wall value;
    * honest densities for outliers / out-of-data-range queries, with
      Gaussian (``tail_alpha = 0``) or heavier (``tail_alpha > 0``) tails on
      unbounded sides.

    Two resolution caveats. The forward tail saturates at ``x_clamp_lo`` /
    ``x_clamp_hi``, the point whose ``z`` is ``_WALL_Z_MARGIN`` past the
    outermost knot, or the nearest representable offset from the bound,
    whichever is further inside. The first term governs: it is about
    ``Phi(u_edge - margin) / Phi(u_edge)`` of the knot-to-bound distance,
    0.14% of it for an outermost knot at the ``0.5 / n`` plotting position
    with n = 1500; in absolute x that was 5.9e-8 to 2.6e-6 from either bound
    over three seeds of ``U(0, 1)`` on ``[0, 1]``, n = 1500. Queries below
    that report the clamp's density,
    so the reported density is bitwise constant over that last sliver, and
    the sampling entry points (:func:`coppuccino.copula_flows.sample` and
    ``sample_and_log_prob``) redraw any row that lands on a clamp value. The
    emitted stream is then the reported density truncated to the clamp-free
    region and renormalized; that is a constant factor, and it cancels in
    Metropolis-Hastings ratios anyway. Under a standard normal base the
    truncated mass would be ``Phi(u_edge - margin)``, 4.7e-7 per bounded wall
    at n = 1500; the copula's own tail is heavier than that, so the measured
    redraw rate over 20000 draws was 1e-4 to 8e-4 on ``U(0, 1)``, truncated
    ``N(0, 1)`` and ``Beta(0.5, 3)`` fits (3 seeds each, against 0 with the
    uncapped tail). On a chain with wall atoms the float-resolution clamp
    already dominated: ``clip(N(0.5, 0.6), 0, 1)`` redrew 26-30% of raw draws
    before this margin existed and 28-33% with it. The *reported*
    density is not truncated -- the plateau slab carries the clamp's density
    across the last sliver -- so it integrates to 1 only up to the difference
    between that slab and the tail mass it stands in for. That difference is
    second order because the wall shape is pinned to ``S'(0) = 1``: over the
    marginal alone, n = 1500, 4 seeds, the mass came to 1 within 2e-8 on
    ``U(0, 1)``, ``Beta(0.5, 3)`` and ``clip(N(0.9, 0.25), -1, 1)``. It is
    largest where a *lower*-wall tie block pulls the outermost knot in to
    ``z = -0.85`` and the cap with it: on ``clip(N(0.5, 0.6), 0, 1)``, 20% of
    rows on each bound, the mass came to 1 + 3.7e-4 to 1 + 5.6e-4 (against
    1 + 3.5e-3 to 1 + 4.7e-3 at margin 1.0). Second, the deep round trip
    ``z -> theta -> z`` is not exact: it is limited both by ``ULP(bound)``,
    which is all the resolution ``|x - bound|`` has near a bound of large
    magnitude, and by the wall shape's own conditioning where ``beta`` is
    extreme. Measured max ``|dz|`` 2.7e-7 over the whole resolvable ``z``
    range, ``[-2.78, 4.90]``, on a both-walls-spiked ``Beta(0.5, 0.5)`` fit
    with 10% of 1500 rows on each bound; the *density* the two directions
    report at the same point agrees to 1.4e-6 there, which is what the
    sampling contract needs.

    What this class does *not* fix: the copula flow downstream consumes ``z``,
    and a query at a bound hands it a ``z`` one ``_WALL_Z_MARGIN`` past the
    deepest it was trained on, where it extrapolates linearly. That
    extrapolation, not the marginal, dominates the joint density reported
    exactly at a wall. On ``U[0, 1]^2``, 1500 rows, 80 epochs, seeds 0-3, the
    four box mid-edges came out +0.1 to +4.6 nats above the flat truth, where
    the marginals alone put them at -1.9 to +1.7; the four corners, where
    both dimensions sit at their cap at once, ran +0.5 to +11.9. Shrinking the
    margin further trades that against wall resolution and against the
    marginal's own wall-density accuracy above, so bounding it properly
    needs a change to the copula, not to this map.

    Trailing attributes are reconstruction metadata for
    :mod:`coppuccino.model_io`.
    """
    samples: np.ndarray
    theta_of_z: Callable
    z_of_theta: Callable
    log_dtheta_dz_fn: Callable
    num_points: int = 200
    prior_low: Optional[float] = None
    prior_high: Optional[float] = None
    tail_alpha: float = 0.0
    x_clamp_lo: Optional[float] = None
    x_clamp_hi: Optional[float] = None
    # flowjax declares shape/cond_shape as equinox ``AbstractVar``, which mypy
    # reads as instance variables of the base; a concrete ``ClassVar`` is the
    # intended implementation here (both are per-class constants).
    cond_shape: ClassVar[None] = None  # type: ignore[misc]
    shape: ClassVar[tuple[int, ...]] = ()  # type: ignore[misc]

    def transform_and_log_det(self, z: ArrayLike,
                              condition: Optional[ArrayLike] = None) -> tuple[Array, Array]:
        """Gaussian space -> data space (sampling direction)."""
        x = self.theta_of_z(z)
        return x, jnp.sum(self.log_dtheta_dz_fn(z))

    def inverse_and_log_det(self, x: ArrayLike,
                            condition: Optional[ArrayLike] = None) -> tuple[Array, Array]:
        """Data space -> Gaussian space (training/density direction)."""
        z = self.z_of_theta(x)
        return z, -jnp.sum(self.log_dtheta_dz_fn(z))
