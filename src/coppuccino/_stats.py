"""Small statistical helpers shared by the fitting and calibration modules.

This module is deliberately a leaf: it imports nothing from ``coppuccino``, so
both :mod:`coppuccino.copula_flows` and :mod:`coppuccino.calibration` can
depend on it without depending on each other.  That matters because
``calibration`` keeps its single ``copula_flows`` dependency deferred inside a
function body; importing ``copula_flows`` at module scope merely to share a
formula would undo a decoupling that was put there on purpose.

Nothing here is part of the public API.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtri

__all__ = ["wilson_interval"]


def wilson_interval(
    probability: np.ndarray | float,
    effective_n: float,
    confidence_level: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Wilson score interval for a binomial proportion.

    ``effective_n`` is a float rather than an integer count because the
    proportions interval-ed here are measured on correlated MCMC draws.  The
    caller passes an effective sample size instead of a row count, which widens
    the interval to reflect the information actually present.

    Parameters
    ----------
    probability : np.ndarray or float
        Observed proportion(s), each in ``[0, 1]``.  Arrays are handled
        elementwise, so one call can interval a whole vector of bin masses.
    effective_n : float
        Positive effective sample size behind ``probability``.
    confidence_level : float
        Two-sided coverage in ``(0, 1)``; ``0.95`` requests a 95% interval.

    Returns
    -------
    lower : np.ndarray
        Lower endpoint(s), clipped to ``[0, 1]``.
    upper : np.ndarray
        Upper endpoint(s), clipped to ``[0, 1]``.

    Notes
    -----
    The Wilson interval is analytically contained in ``[0, 1]``, so the clip
    only repairs rounding: at ``probability`` of exactly 0 or 1 the endpoints
    can land a few ulps outside (measured worst cases ``-2.8e-17`` and
    ``1.0000000000000002``).  No continuity correction is applied, which is why
    the interval stays non-degenerate at those endpoints — the property the
    sparse-bin scorers rely on.
    """
    z = float(ndtri(0.5 + confidence_level / 2.0))
    z2_over_n = z**2 / effective_n
    denominator = 1.0 + z2_over_n
    center = (probability + 0.5 * z2_over_n) / denominator
    half_width = (
        z
        * np.sqrt(probability * (1.0 - probability) / effective_n + z**2 / (4.0 * effective_n**2))
        / denominator
    )
    return (
        np.clip(center - half_width, 0.0, 1.0),
        np.clip(center + half_width, 0.0, 1.0),
    )
